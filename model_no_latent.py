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

from transformers import (
    AutoModel,
    AutoImageProcessor,
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
# FLAMINGO-STYLE GATED CROSS ATTENTION (kept from original)
# ============================================================================

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
# OUTPUT DATACLASS (simplified)
# ============================================================================

@dataclass
class DeterministicVQAOutput:
    """Output for deterministic VQA (no KL)"""
    answer_logits: torch.Tensor
    answer_loss: Optional[torch.Tensor] = None
    total_loss: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None


# ============================================================================
# MAIN MODEL: DETERMINISTIC VQA (NO LATENT)
# ============================================================================

class DeterministicVQA(nn.Module):
    """
    Deterministic VQA without latent reasoning bottleneck.
    
    Architecture:
    1. Vision encoder (DINOv2) - frozen
    2. Text encoder (BART) - frozen/partially unfrozen
    3. Vision-text fusion (Flamingo gated cross-attn)
    4. Decoder cross-attn directly to fused features
    5. Answer generation
    
    NO VAE, NO KL, NO free bits!
    """
    
    def __init__(
        self,
        dinov2_model_name: str = 'facebook/dinov2-base',
        bartpho_model_name: str = 'vinai/bartpho-syllable',
        num_fusion_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        gradient_checkpointing: bool = True
    ):
        super().__init__()
        
        print("[DETERMINISTIC VQA] Initializing without latent reasoning...")
        print("  ✅ No VAE/KL regularization")
        print("  ✅ Direct cross-attention fusion")
        print("  ✅ Optimized for accuracy & stability")
        
        # Vision encoder
        self.vision_encoder = AutoModel.from_pretrained(dinov2_model_name)
        vision_hidden_dim = self.vision_encoder.config.hidden_size
        print(f"  📊 DINOv2 hidden_dim: {vision_hidden_dim}")
        
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
        
        # Vision position embeddings
        self.num_patches = 256
        self.vision_pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, vision_hidden_dim) * 0.02
        )
        
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
        
        # Gradient checkpointing
        if gradient_checkpointing:
            self.vision_encoder.gradient_checkpointing_enable()
            self.encoder.gradient_checkpointing_enable()
        
        print("[DETERMINISTIC VQA] ✓ Initialization complete (NO latent module!)")
    
    def freeze_pretrained(self, unfreeze_encoder_layers: int = 3, unfreeze_decoder: bool = True):
        """Freeze pretrained components"""
        # Freeze vision
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        print(f"[Freeze] Vision encoder: FROZEN")
        
        # Freeze text encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        if unfreeze_encoder_layers > 0:
            for layer in self.encoder.layers[-unfreeze_encoder_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"[Freeze] Text encoder: Last {unfreeze_encoder_layers} layers UNFROZEN")
        else:
            print(f"[Freeze] Text encoder: FROZEN")
        
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
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        stage: int = 3  # Kept for compatibility, but ignored
    ):
        """
        Forward pass - deterministic fusion
        
        NO sampling, NO KL, just pure cross-attention!
        """
        batch_size = pixel_values.size(0)
        
        # 1. Vision encoding
        vision_outputs = self.vision_encoder(pixel_values)
        patch_tokens = vision_outputs.last_hidden_state
        
        # Add position embeddings
        patch_tokens = patch_tokens + self.vision_pos_embed.expand(batch_size, -1, -1)
        vision_features = self.vision_proj(patch_tokens)
        
        # 2. Text encoding
        text_encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_encoder_outputs.last_hidden_state
        
        # 3. Vision-text fusion (Flamingo style)
        fused_vision = vision_features
        for fusion_layer in self.flamingo_fusion:
            fused_vision = fusion_layer(fused_vision, text_features, attention_mask)
        
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
        # Create encoder_hidden_states by concatenating text + vision
        encoder_hidden_states = torch.cat([text_features, fused_vision], dim=1)
        encoder_attention_mask = torch.cat([
            attention_mask,
            torch.ones(batch_size, fused_vision.size(1), device=attention_mask.device)
        ], dim=1)
        
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )
        
        # 6. Generate answer logits
        answer_logits = self.lm_head(decoder_outputs.last_hidden_state)
        
        # 7. Compute loss
        answer_loss = None
        total_loss = None
        
        if labels is not None:
            answer_loss = F.cross_entropy(
                answer_logits.view(-1, answer_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                label_smoothing=0.1  # 🔥 Prevent overfitting on small dataset!
            )
            total_loss = answer_loss  # Only answer loss, no KL!
        
        return DeterministicVQAOutput(
            answer_logits=answer_logits,
            answer_loss=answer_loss,
            total_loss=total_loss
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
        
        # Encode vision
        vision_outputs = self.vision_encoder(pixel_values)
        patch_tokens = vision_outputs.last_hidden_state
        patch_tokens = patch_tokens + self.vision_pos_embed.expand(batch_size, -1, -1)
        vision_features = self.vision_proj(patch_tokens)
        
        # Encode text
        text_encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_encoder_outputs.last_hidden_state
        
        # Fusion
        fused_vision = vision_features
        for fusion_layer in self.flamingo_fusion:
            fused_vision = fusion_layer(fused_vision, text_features, attention_mask)
        
        # Prepare encoder hidden states
        encoder_hidden_states = torch.cat([text_features, fused_vision], dim=1)
        encoder_attention_mask = torch.cat([
            attention_mask,
            torch.ones(batch_size, fused_vision.size(1), device=attention_mask.device)
        ], dim=1)
        
        # 🔥 FIX: Use Hugging Face's generate() for PROPER beam search!
        # Create a wrapper for encoder outputs
        from transformers.modeling_outputs import BaseModelOutput
        
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_hidden_states
        )
        
        # Generate with REAL beam search
        generated_ids = self.decoder.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            top_k=top_k if do_sample else 50,
            early_stopping=True,
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id,
            bos_token_id=self.config.decoder_start_token_id,
            use_cache=True  # Speed up generation
        )
        
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
        num_fusion_layers=2,
        gradient_checkpointing=False
    )
    
    print(f"\nTotal params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print("Model ready for training! 🎉")
