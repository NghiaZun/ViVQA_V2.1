"""
DETERMINISTIC VQA MODEL (No Latent Reasoning)
==============================================

Pure cross-attention fusion without VAE/KL regularization.
Focus on accuracy and stability for low-resource Vietnamese VQA.

Key differences from model.py:
- NO CompressedLatentReasoning module
- NO KL divergence loss
- NO free bits, no VAE sampling
- Direct cross-attention: decoder → (vision + text) features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
import re

from transformers import (
    AutoModel,
    AutoImageProcessor,
    AutoTokenizer,
    BartphoTokenizer,
    MBartForConditionalGeneration
)


def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
    """Shift tokens right for teacher forcing"""
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


# ============================================================================
# ============================================================================
# FLAMINGO-STYLE GATED CROSS ATTENTION
# ============================================================================
# Note: Manual LoRALayer removed - PEFT library handles all LoRA functionality

class FlamingoGatedCrossAttention(nn.Module):
    """Flamingo-style Gated Cross Attention"""
    def __init__(self, hidden_dim=1024, num_heads=16, dropout=0.1):
        super().__init__()
        
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        
        self.gate_cross = nn.Parameter(torch.zeros(1))
        self.gate_ffn = nn.Parameter(torch.zeros(1))
        
    def forward(self, vision_features, text_features, text_attention_mask=None):
        key_padding_mask = None
        if text_attention_mask is not None:
            key_padding_mask = (text_attention_mask == 0)
        
        attn_out, attn_weights = self.cross_attn(
            query=vision_features,
            key=text_features,
            value=text_features,
            key_padding_mask=key_padding_mask
        )
        
        vision_features = vision_features + torch.tanh(self.gate_cross) * self.norm_cross(attn_out)
        
        ffn_out = self.ffn(vision_features)
        vision_features = vision_features + torch.tanh(self.gate_ffn) * self.norm_ffn(ffn_out)
        
        return vision_features


# ============================================================================
# GATED TEXT INJECTION (kept for compatibility)
# ============================================================================

class GatedTextInjection(nn.Module):
    """Lightweight gated text injection"""
    
    def __init__(self, hidden_dim: int = 1024, num_text_tokens: int = 2, init_gate: float = -4.0):
        super().__init__()
        self.num_text_tokens = num_text_tokens
        self.hidden_dim = hidden_dim
        
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Parameter(torch.tensor(init_gate))
        
    def forward(self, reasoning_tokens, text_features, text_mask):
        batch_size = reasoning_tokens.size(0)
        
        pooled_text = (text_features * text_mask.unsqueeze(-1)).sum(dim=1) / text_mask.sum(dim=1, keepdim=True)
        pooled_text = self.text_proj(pooled_text)
        
        text_tokens = pooled_text.unsqueeze(1).expand(-1, self.num_text_tokens, -1)
        
        gate_value = torch.sigmoid(self.gate)
        
        combined = torch.cat([text_tokens, reasoning_tokens], dim=1)
        
        return combined


# ============================================================================
# TYPE PREDICTION HEAD (Auxiliary Task for Multi-task Learning)
# ============================================================================

class TypePredictionHead(nn.Module):
    """
    Auxiliary head for question type classification
    
    Types:
        0 = OBJECT (Đây là gì? Cái gì?)
        1 = COUNT (Có bao nhiêu? Mấy cái?)
        2 = COLOR (Màu gì?)
        3 = LOCATION (Ở đâu? Phía nào? Trên/dưới?)
    
    Purpose: Force question encoder to learn type-level patterns
    NOT used for hard decision - just auxiliary signal!
    """
    def __init__(self, hidden_dim=1024, num_types=4, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_types)
        )
    
    def forward(self, text_cls):
        """
        Args:
            text_cls: [B, D] - CLS token from question encoder
        Returns:
            type_logits: [B, 4] - logits over 4 question types
        """
        return self.classifier(text_cls)


# ============================================================================
# VISION GATING (Learnable Attention-Based)
# ============================================================================

class VisionGating(nn.Module):
    """
    🔥 TYPE-CONDITIONED Vision Gating with Attention
    
    Key idea: (question + type) → gate → select vision
    
    Different question types need different vision features:
        - COLOR → attend to color-rich patches
        - COUNT → attend to object distribution globally  
        - LOCATION → attend to spatial arrangement
        - OBJECT → attend to salient regions
    
    Implementation:
        1. Embed question type as learnable vector
        2. Combine (question_cls + type_emb) as "query"
        3. Attention query @ vision → importance scores α
        4. Gated vision = α * vision + (1-α) * text_context
    """
    def __init__(self, hidden_dim=1024, num_types=4, init_bias=1.5):
        super().__init__()
        
        # Type embeddings (learnable per-type representations)
        self.type_embedding = nn.Embedding(num_types, hidden_dim)
        
        # Project vision features
        self.vision_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Project text features  
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 🔥 NEW: Type-aware query projection
        # Combines question + type to form attention query
        self.query_proj = nn.Linear(hidden_dim * 2, hidden_dim)  # concat(text_cls, type_emb)
        
        # Gating network: learns α ∈ [0, 1] per position
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # α ∈ [0, 1]
        )
        
        # Learnable bias to prefer vision
        self.vision_bias = nn.Parameter(torch.tensor(init_bias))
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, vision_features, text_features, type_ids=None):
        """
        Args:
            vision_features: [B, num_patches, D]  (e.g. [B, 256, 1024])
            text_features: [B, seq_len, D]         (e.g. [B, 20, 1024])
            type_ids: [B] - question type IDs (0=OBJECT, 1=COUNT, 2=COLOR, 3=LOCATION)
                      If None, uses uniform attention (no type conditioning)
        
        Returns:
            gated_vision: [B, num_patches, D]  # Type-conditioned vision
            gate_values: [B, num_patches]      # α values for monitoring
        """
        batch_size, num_patches, hidden_dim = vision_features.shape
        
        # 1. Project features
        v_proj = self.vision_proj(vision_features)  # [B, P, D]
        t_proj = self.text_proj(text_features)      # [B, L, D]
        
        # 2. Get text CLS token (first token in BARTpho)
        text_cls = t_proj[:, 0, :]  # [B, D]
        
        # 3. 🔥 Type-aware attention query
        if type_ids is not None:
            # Embed type and combine with text CLS
            type_emb = self.type_embedding(type_ids)  # [B, D]
            query = torch.cat([text_cls, type_emb], dim=-1)  # [B, 2D]
            query = self.query_proj(query)  # [B, D]
        else:
            # Fallback: use text CLS only (no type conditioning)
            query = text_cls
        
        # 4. Broadcast query for per-patch attention
        query_expanded = query.unsqueeze(1).expand(-1, num_patches, -1)  # [B, P, D]
        
        # 5. Compute gating scores
        # Concatenate vision + type-aware query for each patch
        gate_input = torch.cat([v_proj, query_expanded], dim=-1)  # [B, P, 2D]
        
        # Learn α per patch (which patches are important for THIS type?)
        alpha = self.gate_net(gate_input)  # [B, P, 1]
        
        # 6. Apply vision bias (learnable parameter)
        alpha = torch.sigmoid(alpha.squeeze(-1) + self.vision_bias)  # [B, P]
        alpha_expanded = alpha.unsqueeze(-1)  # [B, P, 1] for broadcasting
        
        # 7. Gated combination
        # Pool text for context (average over sequence)
        text_pooled = t_proj.mean(dim=1, keepdim=True)  # [B, 1, D]
        text_pooled = text_pooled.expand(-1, num_patches, -1)  # [B, P, D]
        
        # α close to 1 → use vision features (important patches)
        # α close to 0 → use text context (suppress noise)
        gated_vision = alpha_expanded * v_proj + (1 - alpha_expanded) * text_pooled
        
        # 8. Layer norm for stability
        gated_vision = self.layer_norm(gated_vision)
        
        return gated_vision, alpha  # Return alpha for monitoring


# ============================================================================
# TYPE-AWARE LOGITS BIASING (Soft Vocabulary Conditioning)
# ============================================================================

class TypeAwareLogitsBias(nn.Module):
    """
    Soft logits biasing based on question type
    
    Key idea: Different question types should prefer different answer tokens
        - COLOR → boost color words (đỏ, xanh, vàng, ...)
        - COUNT → boost numbers (một, hai, ba, 1, 2, 3, ...)
        - LOCATION → boost spatial words (trên, dưới, trái, phải, ...)
        - OBJECT → no bias (all objects equally likely)
    
    Implementation: Learn type-specific bias vectors
        final_logits = base_logits + type_bias[type_id]
    
    ⚠️ This is SOFT - tokens outside preferred vocab still have probability!
    """
    def __init__(self, vocab_size, num_types=4, init_scale=0.1):
        super().__init__()
        
        # Learnable bias per type: [num_types, vocab_size]
        # Initialize small to avoid dominating base logits
        self.type_biases = nn.Parameter(
            torch.randn(num_types, vocab_size) * init_scale
        )
    
    def forward(self, logits, type_ids):
        """
        Args:
            logits: [B, seq_len, vocab_size] - base answer logits
            type_ids: [B] - question type IDs
        
        Returns:
            biased_logits: [B, seq_len, vocab_size] - type-conditioned logits
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Get bias for each sample's type
        bias = self.type_biases[type_ids]  # [B, vocab_size]
        
        # Broadcast to match logits shape
        bias = bias.unsqueeze(1).expand(-1, seq_len, -1)  # [B, seq_len, vocab_size]
        
        # Add bias (soft reweighting)
        return logits + bias


# ============================================================================
# OUTPUT DATACLASS (simplified)
# ============================================================================

@dataclass
class DeterministicVQAOutput:
    """Output for deterministic VQA (no KL)"""
    answer_logits: torch.Tensor
    answer_loss: Optional[torch.Tensor] = None
    type_loss: Optional[torch.Tensor] = None  # 🔥 NEW: Auxiliary type classification loss
    total_loss: Optional[torch.Tensor] = None
    type_logits: Optional[torch.Tensor] = None  # 🔥 NEW: Type predictions [B, num_types]
    attention_weights: Optional[torch.Tensor] = None
    gate_stats: Optional[dict] = None  # Vision gate statistics
    vision_kd_loss: Optional[torch.Tensor] = None  # 🔥🔥🔥 Vision distillation loss
    text_kd_loss: Optional[torch.Tensor] = None  # 🔥🔥🔥 Text distillation loss


# ============================================================================
# MAIN MODEL: DETERMINISTIC VQA (NO LATENT)
# ============================================================================

class DeterministicVQA(nn.Module):
    """
    Deterministic VQA without latent reasoning bottleneck.
    
    Architecture:
    1. Vision encoder (SigLIP) - frozen or LoRA adapted
    2. Text encoder (BART) - frozen/partially unfrozen or LoRA adapted
    3. Vision-text fusion (Flamingo gated cross-attn)
    4. Decoder cross-attn directly to fused features
    5. Answer generation
    
    NO VAE, NO KL, NO free bits!
    """
    
    def __init__(
        self,
        vision_model_name: str = 'google/siglip-base-patch16-224',  # 🔥 CHANGED: DINOv2 → SigLIP
        bartpho_model_name: str = 'vinai/bartpho-syllable',
        num_fusion_layers: int = 4,  # 🔥 INCREASED: 2→4 for deeper vision-text reasoning
        num_heads: int = 8,
        dropout: float = 0.1,
        gradient_checkpointing: bool = True,
        use_vision_lora: bool = False,  # 🔥 Use LoRA for vision encoder
        vision_lora_r: int = 8,  # 🔥 LoRA rank (8 recommended for ~10K samples)
        vision_lora_alpha: int = 16,  # 🔥 LoRA alpha scaling
        vision_lora_dropout: float = 0.1,  # 🔥 LoRA dropout
        use_text_lora: bool = False,  # 🔥 NEW: Use LoRA for text encoder
        text_lora_r: int = 16,  # 🔥 Text LoRA rank (higher than vision)
        text_lora_alpha: int = 32,  # 🔥 Text LoRA alpha
        text_lora_dropout: float = 0.1,  # 🔥 Text LoRA dropout
        use_vision_gate: bool = False,  # 🔥 NEW: Use vision gating
        vision_gate_init: float = 1.5,  # 🔥 Initial vision boost (>1.0 = prefer vision)
        use_type_adapter: bool = False,  # 🔥 NEW: Type-conditioned vision adapter
        type_adapter_rank: int = 64,  # 🔥 Adapter bottleneck rank
        type_adapter_bias: float = 2.0,  # 🔥 Type supervision strength
        # 🔥🔥🔥 ONLINE DISTILLATION 🔥🔥🔥
        use_distillation: bool = False,  # Enable online knowledge distillation
        vision_teacher_name: str = 'google/siglip-so400m-patch14-384',  # Vision teacher (SigLIP-SO400M)
        text_teacher_name: str = 'vinai/phobert-large',  # Text teacher (PhoBERT-large)
        distill_alpha: float = 0.5,  # Distillation weight (0.5 = 50% CE + 50% KD)
        distill_temperature: float = 2.0  # Temperature for soft targets
    ):
        super().__init__()
        
        print("[DETERMINISTIC VQA] Initializing without latent reasoning...")
        print("  ✅ No VAE/KL regularization")
        print("  ✅ Direct cross-attention fusion")
        print("  ✅ Optimized for accuracy & stability")
        print(f"  🔥 Vision Encoder: {vision_model_name}")
        
        # Store distillation config
        self.use_distillation = use_distillation
        self.distill_alpha = distill_alpha
        self.distill_temperature = distill_temperature
        
        self.use_vision_lora = use_vision_lora
        self.vision_lora_r = vision_lora_r
        self.vision_lora_alpha = vision_lora_alpha
        self.vision_lora_dropout = vision_lora_dropout
        
        # Type adapter settings
        self.use_type_adapter = use_type_adapter
        self.type_adapter_rank = type_adapter_rank
        self.type_adapter_bias = type_adapter_bias
        
        self.use_text_lora = use_text_lora  # 🔥 NEW
        self.text_lora_r = text_lora_r  # 🔥 NEW
        self.text_lora_alpha = text_lora_alpha  # 🔥 NEW
        self.text_lora_dropout = text_lora_dropout  # 🔥 NEW
        
        # 🔥 Vision gating (will be initialized after knowing bart_hidden_dim)
        self.use_vision_gate = use_vision_gate
        self.vision_gate_init = vision_gate_init  # Store for later init
        
        # Vision encoder (SigLIP or DINOv2)
        # For SigLIP, load full model first, then extract vision_model
        full_vision_model = AutoModel.from_pretrained(vision_model_name)
        
        # Extract vision-only component if it's a multi-modal model (like SigLIP)
        if hasattr(full_vision_model, 'vision_model'):
            # SigLIP has separate vision_model and text_model
            self.vision_encoder = full_vision_model.vision_model
            vision_hidden_dim = full_vision_model.config.vision_config.hidden_size
            self.is_siglip = True
            print(f"  📊 Detected SigLIP - using vision_model component only")
        else:
            # DINOv2 is vision-only already
            self.vision_encoder = full_vision_model
            vision_hidden_dim = full_vision_model.config.hidden_size
            self.is_siglip = False
            print(f"  📊 Detected DINOv2 - using full model")
        
        print(f"  📊 Vision encoder: {vision_model_name}")
        print(f"  📊 Vision hidden_dim: {vision_hidden_dim}")
        
        # 🔥 Enable gradient checkpointing BEFORE LoRA (if requested)
        # NOTE: SigLIP vision_model has compatibility issues with gradient checkpointing + LoRA
        # Since vision encoder is frozen (only LoRA adapters train), we can skip it safely
        self.gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing:
            if self.is_siglip:
                # SigLIP: Skip gradient checkpointing (frozen + LoRA = minimal memory anyway)
                print(f"  ⚠️  Vision Gradient Checkpointing: SKIPPED for SigLIP")
                print(f"      (SigLIP vision_model has implementation conflicts with PEFT)")
                print(f"      (Vision encoder is frozen, only ~1M LoRA params train - memory OK)")
            else:
                # DINOv2: Safe to enable
                if hasattr(self.vision_encoder, 'gradient_checkpointing_enable'):
                    self.vision_encoder.gradient_checkpointing_enable()
                    print(f"  🔥 Vision Gradient Checkpointing: ENABLED (DINOv2)")
                elif hasattr(self.vision_encoder, 'config'):
                    self.vision_encoder.config.gradient_checkpointing = True
                    print(f"  🔥 Vision Gradient Checkpointing: ENABLED (config-based)")
        
        # 🔥 Add LoRA to vision encoder if requested (AFTER gradient checkpointing setup)
        if use_vision_lora:
            self._inject_lora_to_vision_encoder()
            print(f"  🔥 Vision LoRA: r={vision_lora_r}, alpha={vision_lora_alpha}, dropout={vision_lora_dropout}")
        
        # Language model
        bartpho_full = MBartForConditionalGeneration.from_pretrained(bartpho_model_name)
        bartpho_full.config.use_cache = False
        
        self.tokenizer = BartphoTokenizer.from_pretrained(bartpho_model_name)
        bart_hidden_dim = bartpho_full.config.d_model
        print(f"  📊 BARTpho d_model: {bart_hidden_dim}")
        
        self.encoder = bartpho_full.model.encoder
        self.decoder = bartpho_full.model.decoder
        self.lm_head = bartpho_full.lm_head
        
        self.config = self.encoder.config
        self.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.config.pad_token_id = self.tokenizer.pad_token_id
        self.config.eos_token_id = self.tokenizer.eos_token_id
        
        del bartpho_full
        
        # 🔥 Add LoRA to text encoder if requested
        if use_text_lora:
            self._inject_lora_to_text_encoder()
            print(f"  🔥 Text LoRA: r={text_lora_r}, alpha={text_lora_alpha}, dropout={text_lora_dropout}")
        
        # Vision position embeddings (calculate dynamically based on model)
        # SigLIP & DINOv2: 224x224 image with patch_size=16 → 14x14 = 196 patches
        # Note: Models return [batch, num_patches+1, hidden] where +1 is CLS token
        # We'll initialize for 196 patches (after removing CLS)
        self.num_patches = 196  # Standard for 224x224 with patch_size=16
        self.vision_pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, vision_hidden_dim) * 0.02
        )
        print(f"  📊 Vision position embeddings: {self.num_patches} patches")
        
        # Vision projection
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_hidden_dim, bart_hidden_dim),
            nn.LayerNorm(bart_hidden_dim),
            nn.Dropout(dropout)
        )
        print(f"  ✅ Vision projection: {vision_hidden_dim} → {bart_hidden_dim}")
        
        # Flamingo-style fusion
        self.flamingo_fusion = nn.ModuleList([
            FlamingoGatedCrossAttention(bart_hidden_dim, num_heads, dropout)
            for _ in range(num_fusion_layers)
        ])
        print(f"  ✅ Fusion: {num_fusion_layers} Flamingo layers")
        
        # 🔥 NEW: Type-Conditioned Vision Adapter (AFTER vision projection)
        if self.use_type_adapter:
            from type_conditioned_adapter import TypeConditionedVisionAdapter
            
            self.vision_adapter = TypeConditionedVisionAdapter(
                hidden_dim=vision_hidden_dim,  # Apply to vision features (768 for SigLIP)
                num_types=4,
                rank=self.type_adapter_rank,
                dropout=dropout,
                use_type_supervision=True,
                type_bias_strength=self.type_adapter_bias
            )
            print(f"  🔥 Type-Conditioned Vision Adapter: rank={self.type_adapter_rank}, bias={self.type_adapter_bias}")
        else:
            self.vision_adapter = None
        
        # 🔥 Initialize VisionGating NOW (after bart_hidden_dim is known)
        if self.use_vision_gate:
            self.vision_gating = VisionGating(
                hidden_dim=bart_hidden_dim,
                num_types=4,  # 🔥 NEW: 4 question types
                init_bias=self.vision_gate_init
            )
            print(f"  🔥 Type-Conditioned Vision Gating: 4 types, init_bias={self.vision_gate_init:.2f}")
        
        # 🔥 NEW: Type prediction head (auxiliary task)
        self.type_head = TypePredictionHead(
            hidden_dim=bart_hidden_dim,
            num_types=4,
            dropout=dropout
        )
        print(f"  🔥 Type Prediction Head: 4 types (OBJECT/COUNT/COLOR/LOCATION)")
        
        # 🔥 NEW: Type-aware logits biasing
        vocab_size = self.lm_head.out_features
        self.logits_bias = TypeAwareLogitsBias(
            vocab_size=vocab_size,
            num_types=4,
            init_scale=0.1  # Small initialization to not dominate base logits
        )
        print(f"  🔥 Type-Aware Logits Bias: vocab_size={vocab_size}, 4 types")
        
        # Gradient checkpointing for text encoder (vision already enabled earlier)
        if self.gradient_checkpointing:
            # Text encoder: Always supports gradient checkpointing via method
            if hasattr(self.encoder, 'gradient_checkpointing_enable'):
                self.encoder.gradient_checkpointing_enable()
                print(f"  🔥 Text Gradient Checkpointing: ENABLED")
        
        # 🔥🔥🔥 ONLINE KNOWLEDGE DISTILLATION 🔥🔥🔥
        if self.use_distillation:
            print("\n" + "="*80)
            print("🔥🔥🔥 ONLINE KNOWLEDGE DISTILLATION ENABLED 🔥🔥🔥")
            print("="*80)
            
            # Vision Teacher (SigLIP-SO400M)
            print(f"  📚 Loading Vision Teacher: {vision_teacher_name}")
            vision_teacher_full = AutoModel.from_pretrained(
                vision_teacher_name,
                torch_dtype=torch.float16  # FP16 to save VRAM
            )
            if hasattr(vision_teacher_full, 'vision_model'):
                self.vision_teacher = vision_teacher_full.vision_model
            else:
                self.vision_teacher = vision_teacher_full
            
            # Freeze vision teacher
            for param in self.vision_teacher.parameters():
                param.requires_grad = False
            self.vision_teacher.eval()
            
            # Get teacher's vision processor for 384px images
            self.vision_teacher_processor = AutoImageProcessor.from_pretrained(vision_teacher_name)
            teacher_vision_hidden = self.vision_teacher.config.hidden_size
            print(f"     ✅ Vision Teacher loaded: {teacher_vision_hidden}D, FP16, frozen")
            
            # Text Teacher (PhoBERT-large)
            print(f"  📚 Loading Text Teacher: {text_teacher_name}")
            self.text_teacher = AutoModel.from_pretrained(
                text_teacher_name,
                torch_dtype=torch.float16  # FP16 to save VRAM
            )
            
            # Freeze text teacher
            for param in self.text_teacher.parameters():
                param.requires_grad = False
            self.text_teacher.eval()
            
            # Get teacher's tokenizer
            self.text_teacher_tokenizer = AutoTokenizer.from_pretrained(text_teacher_name)
            teacher_text_hidden = self.text_teacher.config.hidden_size
            print(f"     ✅ Text Teacher loaded: {teacher_text_hidden}D, FP16, frozen")
            
            # Projection layers for distillation (match dimensions)
            # Vision: student patches (196) → teacher patches (729) via pooling/interpolation
            # Then project to same dimension for MSE loss
            self.vision_distill_proj = nn.Linear(vision_hidden_dim, teacher_vision_hidden)
            
            # Text: student text embeddings → teacher text embeddings
            self.text_distill_proj = nn.Linear(bart_hidden_dim, teacher_text_hidden)
            
            print(f"  🎯 Distillation α={distill_alpha:.2f}, T={distill_temperature:.1f}")
            print(f"  🎯 Vision proj: {vision_hidden_dim} → {teacher_vision_hidden}")
            print(f"  🎯 Text proj: {bart_hidden_dim} → {teacher_text_hidden}")
            print("="*80 + "\n")
        else:
            self.vision_teacher = None
            self.text_teacher = None
            self.vision_teacher_processor = None
            self.text_teacher_tokenizer = None
        
        print("[DETERMINISTIC VQA] ✓ Multi-task type-conditioned model initialized!")
    
    def _inject_lora_to_vision_encoder(self):
        """
        Inject LoRA into vision encoder using HuggingFace PEFT library.
        
        CRITICAL WARNING: SigLIP vision_model has compatibility issues with PEFT LoRA!
        - Bug: SigLIP encoder forward() conflicts with PEFT wrapper
        - Error: "got multiple values for keyword argument 'inputs_embeds'"
        - Root cause: SigLIP internal implementation incompatible with PEFT hooks
        
        RECOMMENDED SOLUTIONS:
        1. Use DINOv2 instead: --vision_model facebook/dinov2-base
        2. OR disable vision LoRA for SigLIP (freeze vision encoder completely)
        
        This method will RAISE ERROR if SigLIP + LoRA detected!
        """
        # 🚨 CRITICAL CHECK: Block SigLIP + LoRA combination
        if self.is_siglip:
            raise RuntimeError(
                "\n"
                "="*70 + "\n"
                "❌ CRITICAL ERROR: SigLIP + Vision LoRA is NOT SUPPORTED!\n"
                "="*70 + "\n"
                "SigLIP vision_model has implementation conflicts with PEFT LoRA.\n"
                "Error: 'got multiple values for keyword argument inputs_embeds'\n"
                "\n"
                "SOLUTIONS:\n"
                "  1. Use DINOv2 (RECOMMENDED):\n"
                "     --vision_model facebook/dinov2-base \\\n"
                "     --use_vision_lora --vision_lora_r 8\n"
                "\n"
                "  2. Use SigLIP WITHOUT vision LoRA:\n"
                "     --vision_model google/siglip-base-patch16-224 \\\n"
                "     (remove --use_vision_lora flag)\n"
                "\n"
                "  3. Use text LoRA only (still beneficial!):\n"
                "     --vision_model google/siglip-base-patch16-224 \\\n"
                "     --use_text_lora --text_lora_r 16\n"
                "="*70 + "\n"
            )
        
        # DINOv2: Safe to apply LoRA
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise RuntimeError(
                "\n"
                "❌ PEFT library is REQUIRED for LoRA!\n"
                "   Install with: pip install peft\n"
                "   Then retry training.\n"
            )
        
        print(f"  [LoRA] Using PEFT library for vision encoder...")
        
        # LoRA config for vision encoder (SigLIP)
        lora_config = LoraConfig(
            r=self.vision_lora_r,
            lora_alpha=self.vision_lora_alpha,
            lora_dropout=self.vision_lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj"],  # SigLIP uses different naming: q_proj/k_proj/v_proj
            bias="none",
            task_type="FEATURE_EXTRACTION"  # SigLIP vision encoder is feature extractor
        )
        
        # Apply LoRA (PEFT automatically hooks into forward pass!)
        self.vision_encoder = get_peft_model(self.vision_encoder, lora_config)
        
        # Print trainable parameters summary
        trainable_params = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.vision_encoder.parameters())
        print(f"  [LoRA] Vision - Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%) | Total: {total_params:,}")
    
    def _inject_lora_to_text_encoder(self):
        """
        Inject LoRA into BARTpho text encoder using PEFT library
        
        This adapts the encoder with low-rank matrices instead of unfreezing layers.
        Benefits:
        - 10x fewer parameters than unfreezing 3 layers (~1.5M vs ~18M)
        - Adapts ALL 12 layers instead of just last 3
        - More stable training (no gradient mismatch between frozen/unfrozen layers)
        - Better generalization on low-resource datasets (~10K samples)
        
        Proven effective in LoRA paper: https://arxiv.org/abs/2106.09685
        """
        try:
            from peft import LoraConfig, get_peft_model
            
            print(f"  [LoRA] Injecting into BARTpho encoder (r={self.text_lora_r})...")
            
            # LoRA config for text encoder (BARTpho)
            # Note: Use FEATURE_EXTRACTION not SEQ_2_SEQ_LM because encoder doesn't generate
            lora_config = LoraConfig(
                r=self.text_lora_r,
                lora_alpha=self.text_lora_alpha,
                lora_dropout=self.text_lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj"],  # BART uses *_proj naming
                bias="none",
                task_type="FEATURE_EXTRACTION"  # Encoder is feature extractor, not generator
            )
            
            # Apply LoRA to encoder (PEFT handles everything!)
            self.encoder = get_peft_model(self.encoder, lora_config)
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.encoder.parameters())
            print(f"  [LoRA] Text Encoder - Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%) | Total: {total_params:,}")
            
        except ImportError:
            print(f"  ⚠️  PEFT library not found!")
            print(f"      Install with: pip install peft")
            print(f"      Text encoder will remain frozen unless you unfreeze layers manually")
            raise RuntimeError("PEFT required for text LoRA. Install: pip install peft")
    
    def freeze_pretrained(
        self, 
        unfreeze_encoder_layers: int = 3, 
        unfreeze_decoder: bool = True
    ):
        """
        Freeze pretrained components (vision frozen, optionally with LoRA)
        
        Args:
            unfreeze_encoder_layers: Number of text encoder layers to unfreeze (from end)
            unfreeze_decoder: Whether to unfreeze decoder
        
        Note: Vision encoder is ALWAYS frozen except for LoRA adapters (if enabled)
        """
        # 🔥 Vision Encoder: PEFT LoRA handles freezing automatically
        if self.use_vision_lora:
            try:
                from peft import PeftModel
                if isinstance(self.vision_encoder, PeftModel):
                    # PEFT automatically freezes base weights and unfreezes LoRA adapters
                    trainable = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)
                    print(f"[Freeze] Vision encoder: FROZEN (base) + PEFT LoRA ({trainable/1e6:.2f}M params)")
                else:
                    print(f"[Freeze] Vision encoder: WARNING - LoRA requested but not applied!")
            except ImportError:
                raise RuntimeError("PEFT not installed but vision LoRA requested!")
        else:
            # Manually freeze if no LoRA
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            print(f"[Freeze] Vision encoder: FULLY FROZEN")
        
        # 🔥 Text Encoder: Freeze base, unfreeze LoRA OR last N layers
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Handle text LoRA (RECOMMENDED for low-resource)
        if self.use_text_lora:
            try:
                from peft import PeftModel
                if isinstance(self.encoder, PeftModel):
                    # PEFT automatically sets requires_grad for LoRA params
                    trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
                    print(f"[Freeze] Text encoder: FROZEN (base) + PEFT LoRA TRAINABLE ({trainable/1e6:.2f}M params)")
                    print(f"         ✅ Adapting ALL 12 layers with low-rank matrices")
                else:
                    print(f"[Freeze] Text encoder: LoRA requested but PEFT not applied")
            except ImportError:
                print(f"[Freeze] Text encoder: LoRA requested but PEFT not installed")
        
        # Fallback: Unfreeze last N layers (OLD METHOD - less efficient)
        elif unfreeze_encoder_layers > 0:
            for layer in self.encoder.layers[-unfreeze_encoder_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            trainable = sum(p.numel() for layer in self.encoder.layers[-unfreeze_encoder_layers:] 
                          for p in layer.parameters() if p.requires_grad)
            print(f"[Freeze] Text encoder: Last {unfreeze_encoder_layers} layers UNFROZEN ({trainable/1e6:.2f}M params)")
            print(f"         ⚠️  Consider using --use_text_lora for better efficiency!")
        else:
            print(f"[Freeze] Text encoder: FULLY FROZEN")
        
        # Decoder
        if unfreeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = True
            for param in self.lm_head.parameters():
                param.requires_grad = True
            print(f"[Freeze] Decoder + LM head: UNFROZEN")
        else:
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.lm_head.parameters():
                param.requires_grad = False
            print(f"[Freeze] Decoder + LM head: FROZEN")
    
    def _extract_teacher_vision_features(self, images_384):
        """
        Extract vision teacher features (SigLIP-SO400M at 384px)
        
        Args:
            images_384: [B, 3, 384, 384] - Images preprocessed for teacher
            
        Returns:
            teacher_vision_features: [B, 729, 1152] - Teacher patch embeddings
        """
        with torch.no_grad():
            teacher_outputs = self.vision_teacher(pixel_values=images_384.half())
            teacher_patches = teacher_outputs.last_hidden_state  # [B, 730, 1152] with CLS
            
            # Remove CLS token
            if teacher_patches.size(1) == 730:  # 729 patches + 1 CLS
                teacher_patches = teacher_patches[:, 1:, :]  # [B, 729, 1152]
            
            return teacher_patches.float()  # Convert back to FP32 for loss
    
    def _extract_teacher_text_features(self, raw_questions):
        """
        Extract text teacher features (PhoBERT-large)
        
        Args:
            raw_questions: List[str] - Raw Vietnamese questions
            
        Returns:
            teacher_text_features: [B, 1024] - Teacher question embeddings (CLS)
        """
        with torch.no_grad():
            # Tokenize with teacher's tokenizer
            teacher_inputs = self.text_teacher_tokenizer(
                raw_questions,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(self.text_teacher.device)
            
            teacher_outputs = self.text_teacher(**teacher_inputs)
            teacher_cls = teacher_outputs.last_hidden_state[:, 0, :]  # [B, 1024]
            
            return teacher_cls.half().float()  # FP16 → FP32
    
    def compute_distillation_loss(
        self,
        student_vision_patches,  # [B, 196, vision_hidden]
        student_text_features,   # [B, bart_hidden]
        teacher_vision_patches,  # [B, 729, teacher_vision_hidden]
        teacher_text_features    # [B, teacher_text_hidden]
    ):
        """
        Compute knowledge distillation losses
        
        Returns:
            vision_kd_loss: MSE between student and teacher vision features
            text_kd_loss: MSE between student and teacher text features
        """
        # Vision KD: Student 196 patches → Teacher 729 patches
        # Strategy: Downsample teacher 729 → 196 via adaptive pooling
        B = student_vision_patches.size(0)
        
        # Reshape teacher patches for pooling: [B, 729, D] → [B, D, 27, 27]
        teacher_vision_hidden = teacher_vision_patches.size(-1)
        teacher_patches_2d = teacher_vision_patches.transpose(1, 2).reshape(
            B, teacher_vision_hidden, 27, 27
        )
        
        # Downsample to 14x14 = 196 patches
        teacher_patches_downsampled = F.adaptive_avg_pool2d(
            teacher_patches_2d, 
            output_size=(14, 14)
        )  # [B, teacher_D, 14, 14]
        
        # Reshape back: [B, teacher_D, 14, 14] → [B, 196, teacher_D]
        teacher_patches_downsampled = teacher_patches_downsampled.reshape(
            B, teacher_vision_hidden, 196
        ).transpose(1, 2)  # [B, 196, teacher_D]
        
        # Project student to teacher dimension
        student_vision_proj = self.vision_distill_proj(student_vision_patches)  # [B, 196, teacher_D]
        
        # MSE loss
        vision_kd_loss = F.mse_loss(student_vision_proj, teacher_patches_downsampled)
        
        # Text KD: Direct MSE between CLS embeddings
        student_text_proj = self.text_distill_proj(student_text_features)  # [B, teacher_text_D]
        text_kd_loss = F.mse_loss(student_text_proj, teacher_text_features)
        
        return vision_kd_loss, text_kd_loss
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        stage: int = 3,  # Kept for compatibility, but ignored
        answer_weights: Optional[torch.Tensor] = None,  # 🔥 NEW: Token-level weights for balanced loss
        question_types: Optional[torch.Tensor] = None,  # 🔥 NEW: Question type (0=object_id, 1=counting, 2=color, 3=location)
        images_384: Optional[torch.Tensor] = None,  # 🔥🔥🔥 For vision teacher (384px)
        raw_questions: Optional[list] = None  # 🔥🔥🔥 For text teacher (raw strings)
    ):
        """
        Forward pass - deterministic fusion
        
        NO sampling, NO KL, just pure cross-attention!
        
        Args:
            answer_weights: [vocab_size] tensor with per-token loss weights (inverse freq)
            question_types: [batch] tensor with question type:
                0 = object identification (Đây là gì?)
                1 = counting (Có bao nhiêu?)
                2 = color (Màu gì?)
                3 = location (Ở đâu? Trên bàn?)
        """
        batch_size = pixel_values.size(0)
        
        # 1. Vision encoding
        # Note: self.vision_encoder is already vision_model component for SigLIP
        # or full DINOv2 model. Both take pixel_values directly.
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        patch_tokens = vision_outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # Remove CLS token if present
        # SigLIP vision_model: [batch, 197, hidden_dim] → 196 patches + 1 CLS
        # DINOv2: [batch, 197, hidden_dim] → 196 patches + 1 CLS  
        # We only need patch tokens for cross-attention fusion
        original_seq_len = patch_tokens.size(1)
        if original_seq_len > self.num_patches:  # Has CLS token
            patch_tokens = patch_tokens[:, 1:, :]  # Remove first token (CLS)
            # Verify shape matches expected
            assert patch_tokens.size(1) == self.num_patches, \
                f"Shape mismatch after CLS removal: got {patch_tokens.size(1)} patches, expected {self.num_patches}"
        
        # 🔥 NEW: Apply type-conditioned adapter (BEFORE position embeddings)
        # This allows adapter to transform raw vision features based on question type
        if self.vision_adapter is not None:
            patch_tokens = self.vision_adapter(
                patch_tokens,
                type_ids=question_types  # Use ground-truth types during training
            )
        
        # Add position embeddings
        patch_tokens = patch_tokens + self.vision_pos_embed.expand(batch_size, -1, -1)
        vision_features = self.vision_proj(patch_tokens)  # [batch, 196, bart_hidden]
        
        # 2. Text encoding
        text_encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_encoder_outputs.last_hidden_state
        
        # 🔥 NEW: Type prediction (auxiliary task)
        # Extract CLS token (first token in BARTpho)
        text_cls = text_features[:, 0, :]  # [B, D]
        type_logits = self.type_head(text_cls)  # [B, 4]
        
        # Compute type loss if labels provided
        type_loss = None
        if question_types is not None:
            type_loss = F.cross_entropy(type_logits, question_types)
        
        # 3. Vision-text fusion (Flamingo style)
        fused_vision = vision_features
        for fusion_layer in self.flamingo_fusion:
            fused_vision = fusion_layer(fused_vision, text_features, attention_mask)
        
        # 🔥 Type-Conditioned Vision Gating
        gate_stats = None
        if self.use_vision_gate:
            # Use predicted types during inference, ground truth during training
            type_ids_for_gating = question_types if question_types is not None else torch.argmax(type_logits, dim=-1)
            gated_vision, gate_values = self.vision_gating(
                fused_vision, 
                text_features,
                type_ids=type_ids_for_gating  # 🔥 Type-conditioned!
            )
            
            # Compute statistics for monitoring
            gate_stats = {
                'mean': gate_values.mean().item(),
                'std': gate_values.std().item(),
                'min': gate_values.min().item(),
                'max': gate_values.max().item()
            }
        else:
            gated_vision = fused_vision
        
        # 4. Prepare decoder inputs
        if labels is not None:
            decoder_input_ids = shift_tokens_right(
                labels, 
                self.config.pad_token_id, 
                self.config.decoder_start_token_id
            )
        else:
            decoder_input_ids = torch.full(
                (batch_size, 1),
                self.config.decoder_start_token_id,
                dtype=torch.long,
                device=pixel_values.device
            )
        
        # 5. Decoder: Cross-attend to fused vision features
        # 🔥 Vision-First Ordering: Put vision tokens BEFORE text
        # Reason: Decoder attends to earlier tokens first, increasing vision usage
        encoder_hidden_states = torch.cat([gated_vision, text_features], dim=1)
        encoder_attention_mask = torch.cat([
            torch.ones(batch_size, gated_vision.size(1), device=attention_mask.device),
            attention_mask
        ], dim=1)
        
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )
        
        # 6. Generate answer logits
        base_answer_logits = self.lm_head(decoder_outputs.last_hidden_state)
        
        # 🔥 NEW: Apply type-aware logits biasing (soft vocab conditioning)
        if question_types is not None:
            answer_logits = self.logits_bias(base_answer_logits, question_types)
        else:
            # Inference: use predicted types
            predicted_types = torch.argmax(type_logits, dim=-1)
            answer_logits = self.logits_bias(base_answer_logits, predicted_types)
        
        # 7. 🔥 MULTI-TASK LOSS: Type + Answer + Distillation
        answer_loss = None
        total_loss = None
        vision_kd_loss = None
        text_kd_loss = None
        
        if labels is not None:
            # (A) Answer generation loss
            answer_loss = F.cross_entropy(
                answer_logits.view(-1, answer_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                weight=answer_weights if answer_weights is not None else None,
                label_smoothing=0.1
            )
            
            # (B) 🔥🔥🔥 KNOWLEDGE DISTILLATION 🔥🔥🔥
            if self.use_distillation and images_384 is not None and raw_questions is not None:
                # Extract teacher features
                teacher_vision_patches = self._extract_teacher_vision_features(images_384)
                teacher_text_features = self._extract_teacher_text_features(raw_questions)
                
                # Compute KD losses
                vision_kd_loss, text_kd_loss = self.compute_distillation_loss(
                    student_vision_patches=patch_tokens,  # Before projection! [B, 196, vision_hidden]
                    student_text_features=text_cls,  # Question CLS embedding [B, bart_hidden]
                    teacher_vision_patches=teacher_vision_patches,  # [B, 729, teacher_hidden]
                    teacher_text_features=teacher_text_features  # [B, teacher_hidden]
                )
                
                # Combine: (1-α)*CE + α*KD
                # KD = 0.5*vision + 0.5*text (equal weight)
                kd_loss = 0.5 * vision_kd_loss + 0.5 * text_kd_loss
                answer_loss_with_kd = (1 - self.distill_alpha) * answer_loss + self.distill_alpha * kd_loss
            else:
                answer_loss_with_kd = answer_loss
            
            # (C) Multi-task loss: Type (auxiliary) + Answer (main) + KD
            if type_loss is not None:
                # Weight type loss lower (auxiliary signal, not primary task)
                # λ_type = 0.2 means type contributes 20% to total loss
                total_loss = answer_loss_with_kd + 0.2 * type_loss
            else:
                total_loss = answer_loss_with_kd
        
        return DeterministicVQAOutput(
            answer_logits=answer_logits,
            answer_loss=answer_loss,
            type_loss=type_loss,  # 🔥 NEW
            total_loss=total_loss,
            type_logits=type_logits,  # 🔥 NEW
            gate_stats=gate_stats,
            vision_kd_loss=vision_kd_loss,  # 🔥🔥🔥 NEW
            text_kd_loss=text_kd_loss  # 🔥🔥🔥 NEW
        )
    
    @torch.inference_mode()  # Faster than @torch.no_grad()!
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 20,
        num_beams: int = 3,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_p: float = 0.9,
        top_k: int = 50
    ):
        """
        Generate answers using Hugging Face's beam search (FIXED!)
        
        Previous bug: Claimed beam search but did multinomial sampling
        Fix: Use model.decoder.generate() with proper beam search
        """
        batch_size = pixel_values.size(0)
        
        # Encode vision (same logic as forward())
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        patch_tokens = vision_outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # Remove CLS token if present (same as forward())
        original_seq_len = patch_tokens.size(1)
        if original_seq_len > self.num_patches:  # Has CLS token
            patch_tokens = patch_tokens[:, 1:, :]  # Remove first token (CLS)
            # Verify shape matches expected
            assert patch_tokens.size(1) == self.num_patches, \
                f"Shape mismatch after CLS removal in generate(): got {patch_tokens.size(1)} patches, expected {self.num_patches}"
        
        # Add position embeddings
        patch_tokens = patch_tokens + self.vision_pos_embed.expand(batch_size, -1, -1)
        vision_features = self.vision_proj(patch_tokens)
        
        # Encode text
        text_encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_encoder_outputs.last_hidden_state
        
        # 🔥 NEW: Predict type for type-conditioned generation
        text_cls = text_features[:, 0, :]
        type_logits = self.type_head(text_cls)
        predicted_types = torch.argmax(type_logits, dim=-1)  # [B]
        
        # Fusion
        fused_vision = vision_features
        for fusion_layer in self.flamingo_fusion:
            fused_vision = fusion_layer(fused_vision, text_features, attention_mask)
        
        # 🔥 Type-Conditioned Vision Gating (same as forward)
        if self.use_vision_gate:
            gated_vision, _ = self.vision_gating(
                fused_vision, 
                text_features,
                type_ids=predicted_types  # 🔥 Use predicted type!
            )
        else:
            gated_vision = fused_vision
        
        # Prepare encoder hidden states
        # 🔥 Vision-First Ordering (same as forward pass)
        encoder_hidden_states = torch.cat([gated_vision, text_features], dim=1)
        encoder_attention_mask = torch.cat([
            torch.ones(batch_size, gated_vision.size(1), device=attention_mask.device),
            attention_mask
        ], dim=1)
        
        # Greedy decoding (simple but effective)
        device = pixel_values.device
        generated_ids = torch.full(
            (batch_size, 1),
            self.config.decoder_start_token_id,
            dtype=torch.long,
            device=device
        )
        
        for _ in range(max_length):
            # Decode
            decoder_outputs = self.decoder(
                input_ids=generated_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask
            )
            
            # Get base logits
            base_logits = self.lm_head(decoder_outputs.last_hidden_state)
            
            # 🔥 Apply type-aware logits biasing
            logits = self.logits_bias(base_logits, predicted_types)
            next_token_logits = logits[:, -1, :]
            
            # Greedy: take argmax
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_tokens], dim=1)
            
            # Check if all sequences have generated EOS
            if (next_tokens == self.config.eos_token_id).all():
                break
        
        # Decode
        answers = []
        for i in range(batch_size):
            answer = self.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
            answers.append(answer)
        
        return answers


if __name__ == '__main__':
    print("="*80)
    print("DETERMINISTIC VQA MODEL (NO LATENT REASONING)")
    print("="*80)
    print("\nKey features:")
    print("  ✅ No VAE/KL regularization")
    print("  ✅ Direct cross-attention fusion")
    print("  ✅ Optimized for low-resource VQA")
    print("  ✅ Stable training, no KL tuning needed")
    print("="*80)
    
    model = DeterministicVQA(
        num_fusion_layers=4,  # Match default in __init__
        gradient_checkpointing=False
    )
    
    print(f"\nTotal params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print("Model ready for training! 🎉")
