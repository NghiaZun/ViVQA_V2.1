# 🚀 CHIẾN LƯỢC CẢI THIỆN SIGLIP - GIẢI THÍCH CHI TIẾT

## 📊 HIỆN TRẠNG

Bạn đang dùng **SigLIP (frozen, no LoRA)** vì:
- ✅ SigLIP cho kết quả tốt hơn DINOv2 (+1.34% EM)
- ❌ Không thể dùng vision LoRA (PEFT conflict)
- ⚠️ Vision encoder **frozen hoàn toàn** → không adapt được

**Vấn đề:** Không tận dụng được điểm mạnh của cả 2 models!

---

# 🔥 CHIẾN LƯỢC 1: HYBRID VISION ENCODER (Dual-Encoder)

## 💡 Ý TƯỞNG CỐT LÕI

**"Sử dụng 2 vision encoders cùng lúc, mỗi cái làm việc khác nhau!"**

```
Input Image (224x224)
       │
       ├──────────────────┬──────────────────┐
       │                  │                  │
   ┌───▼────┐      ┌─────▼──────┐    ┌─────▼──────┐
   │ SigLIP │      │  DINOv2    │    │  DINOv2    │
   │(frozen)│      │  (frozen)  │    │  (LoRA)    │
   └───┬────┘      └─────┬──────┘    └─────┬──────┘
       │                  │                  │
       │ [B,196,768]      │ [B,256,768]     │ [B,256,768]
       │                  │                  │
       │                  │  Interpolate     │
       │                  │  256→196         │
       │                  └─────┬────────────┘
       │                        │
       │                   [B,196,768]
       │                        │
       └────────────────┬───────┘
                        │
                    Concat
                        │
                   [B,196,1536]
                        │
                  Fusion MLP
                  768×2 → 1024
                        │
                   [B,196,1024]
                        │
                   (To BART)
```

## 🎯 TẠI SAO LÀM VẬY?

### Phân công công việc:

| Model | Vai trò | Tại sao? |
|-------|---------|----------|
| **SigLIP (frozen)** | Language-vision alignment specialist | - Pretrained với image-text pairs<br>- Hiểu mối quan hệ question↔image<br>- Mạnh ở COUNT (+12.96%), COLOR (+4.48%) |
| **DINOv2 (LoRA)** | Visual details specialist | - Self-supervised DINO objective<br>- Mạnh về objects, spatial structure<br>- Mạnh ở OBJECT (67.77%), LOCATION (64.08%)<br>- **LoRA adaptable** → học Vietnamese patterns |

### Analogy:
- **SigLIP** = Phiên dịch viên (hiểu ngôn ngữ + hình ảnh)
- **DINOv2** = Chuyên gia thị giác (thấy chi tiết vật thể, vị trí)
- **Fusion** = Kết hợp 2 góc nhìn → câu trả lời tốt nhất

## 📝 IMPLEMENTATION CHI TIẾT

### Bước 1: Tạo Hybrid Encoder Module

```python
# File: hybrid_vision_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


class HybridVisionEncoder(nn.Module):
    """
    Dual-encoder approach combining:
    - SigLIP: For language-vision alignment (frozen)
    - DINOv2: For visual details (LoRA adapted)
    
    Expected improvement: +3-4.5% EM (64-66% total)
    """
    
    def __init__(
        self,
        siglip_model_name: str = 'google/siglip-base-patch16-224',
        dinov2_model_name: str = 'facebook/dinov2-base',
        output_dim: int = 1024,  # Đầu ra cho BART
        use_dinov2_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        
        print("[Hybrid Vision Encoder] Initializing...")
        
        # ============================================================
        # ENCODER 1: SigLIP (FROZEN - for alignment)
        # ============================================================
        print(f"  [1/2] Loading SigLIP (frozen)...")
        siglip_full = AutoModel.from_pretrained(siglip_model_name)
        
        # SigLIP structure: SigLIPModel → vision_model → SigLIPVisionModel
        if hasattr(siglip_full, 'vision_model'):
            self.siglip = siglip_full.vision_model
        else:
            self.siglip = siglip_full
        
        # Freeze SigLIP hoàn toàn
        for param in self.siglip.parameters():
            param.requires_grad = False
        
        siglip_dim = self.siglip.config.hidden_size  # 768
        print(f"      ✅ SigLIP loaded: {siglip_dim}D, frozen")
        
        # ============================================================
        # ENCODER 2: DINOv2 (LoRA - for visual details)
        # ============================================================
        print(f"  [2/2] Loading DINOv2 (LoRA adaptable)...")
        self.dinov2 = AutoModel.from_pretrained(dinov2_model_name)
        
        dinov2_dim = self.dinov2.config.hidden_size  # 768
        
        # Apply LoRA to DINOv2
        if use_dinov2_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["query", "key", "value"],  # DINOv2 naming
                bias="none",
                task_type="FEATURE_EXTRACTION"
            )
            self.dinov2 = get_peft_model(self.dinov2, lora_config)
            
            trainable = sum(p.numel() for p in self.dinov2.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.dinov2.parameters())
            print(f"      ✅ DINOv2 LoRA: {trainable:,}/{total:,} ({trainable/total*100:.2f}%)")
        else:
            # Freeze DINOv2 nếu không dùng LoRA
            for param in self.dinov2.parameters():
                param.requires_grad = False
            print(f"      ✅ DINOv2 loaded: {dinov2_dim}D, frozen")
        
        # ============================================================
        # FUSION: Combine 2 encoders
        # ============================================================
        # SigLIP: 196 patches (14×14, patch_size=16)
        # DINOv2: 256 patches (16×16, patch_size=14) → cần interpolate!
        
        self.fusion = nn.Sequential(
            nn.Linear(siglip_dim + dinov2_dim, output_dim),  # 768+768 → 1024
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        print(f"      ✅ Fusion: {siglip_dim}+{dinov2_dim} → {output_dim}")
        print("[Hybrid Vision Encoder] ✓ Initialized!")
        
        self.siglip_patches = 196  # 14×14
        self.dinov2_patches = 256  # 16×16
    
    def forward(self, pixel_values):
        """
        Args:
            pixel_values: [B, 3, 224, 224] - input images
        
        Returns:
            fused_features: [B, 196, 1024] - hybrid vision features
        """
        batch_size = pixel_values.size(0)
        
        # ============================================================
        # PATH 1: SigLIP features (frozen)
        # ============================================================
        with torch.no_grad():  # No gradients for SigLIP
            siglip_output = self.siglip(pixel_values)
            siglip_features = siglip_output.last_hidden_state  # [B, 197, 768]
            
            # Remove CLS token (first token)
            if siglip_features.size(1) > 196:
                siglip_features = siglip_features[:, 1:, :]  # [B, 196, 768]
        
        # ============================================================
        # PATH 2: DINOv2 features (LoRA trainable)
        # ============================================================
        dinov2_output = self.dinov2(pixel_values)
        dinov2_features = dinov2_output.last_hidden_state  # [B, 257, 768]
        
        # Remove CLS token
        if dinov2_features.size(1) > 256:
            dinov2_features = dinov2_features[:, 1:, :]  # [B, 256, 768]
        
        # 🔥 CRITICAL: Interpolate DINOv2 từ 256 patches → 196 patches
        # Reshape: [B, 256, 768] → [B, 768, 256] (for interpolation)
        dinov2_features = dinov2_features.transpose(1, 2)  # [B, 768, 256]
        
        # Interpolate along patch dimension (256 → 196)
        dinov2_features = F.interpolate(
            dinov2_features,
            size=196,
            mode='linear',
            align_corners=False
        )  # [B, 768, 196]
        
        # Reshape back: [B, 768, 196] → [B, 196, 768]
        dinov2_features = dinov2_features.transpose(1, 2)  # [B, 196, 768]
        
        # ============================================================
        # FUSION: Concatenate and fuse
        # ============================================================
        # Concat along feature dimension
        combined = torch.cat([siglip_features, dinov2_features], dim=-1)  # [B, 196, 1536]
        
        # Fuse through MLP
        fused_features = self.fusion(combined)  # [B, 196, 1024]
        
        return fused_features
```

### Bước 2: Tích hợp vào DeterministicVQA

```python
# File: model_no_latent.py

# Thêm import
from hybrid_vision_encoder import HybridVisionEncoder

class DeterministicVQA(nn.Module):
    def __init__(
        self,
        vision_model_name: str = 'google/siglip-base-patch16-224',
        bartpho_model_name: str = 'vinai/bartpho-syllable',
        use_hybrid_vision: bool = False,  # 🔥 NEW FLAG
        # ...existing args...
    ):
        super().__init__()
        
        # ============================================================
        # VISION ENCODER: Hybrid hoặc Single
        # ============================================================
        if use_hybrid_vision:
            print("[MODEL] Using HYBRID vision encoder (SigLIP + DINOv2)")
            
            # Sử dụng Hybrid encoder
            self.vision_encoder = HybridVisionEncoder(
                siglip_model_name='google/siglip-base-patch16-224',
                dinov2_model_name='facebook/dinov2-base',
                output_dim=1024,  # Match BART hidden_dim
                use_dinov2_lora=True,
                lora_r=8,
                lora_alpha=16
            )
            
            vision_hidden_dim = 1024  # Hybrid output dimension
            self.is_hybrid = True
            
        else:
            print("[MODEL] Using SINGLE vision encoder (SigLIP only)")
            
            # Existing single encoder logic
            full_vision_model = AutoModel.from_pretrained(vision_model_name)
            
            if hasattr(full_vision_model, 'vision_model'):
                self.vision_encoder = full_vision_model.vision_model
                vision_hidden_dim = self.vision_encoder.config.hidden_size
                self.is_siglip = True
            else:
                self.vision_encoder = full_vision_model
                vision_hidden_dim = full_vision_model.config.hidden_size
                self.is_siglip = False
            
            self.is_hybrid = False
        
        # ============================================================
        # Nếu dùng hybrid → SKIP vision projection (already 1024)
        # ============================================================
        if use_hybrid_vision:
            # Hybrid đã output 1024, không cần project
            self.vision_proj = nn.Identity()  # Pass-through
        else:
            # Single encoder cần project 768 → 1024
            self.vision_proj = nn.Sequential(
                nn.Linear(vision_hidden_dim, 1024),
                nn.LayerNorm(1024),
                nn.Dropout(dropout)
            )
        
        # ...rest of model initialization...
    
    def forward(self, pixel_values, input_ids, attention_mask, type_ids=None):
        batch_size = pixel_values.size(0)
        
        # ============================================================
        # VISION ENCODING
        # ============================================================
        if self.is_hybrid:
            # Hybrid encoder returns [B, 196, 1024] directly
            vision_features = self.vision_encoder(pixel_values)
            
            # Add position embeddings
            vision_features = vision_features + self.vision_pos_embed
            
        else:
            # Single encoder logic (existing code)
            vision_output = self.vision_encoder(pixel_values)
            patch_tokens = vision_output.last_hidden_state
            
            # Remove CLS if exists
            if patch_tokens.size(1) > 196:
                patch_tokens = patch_tokens[:, 1:, :]
            
            # Add position embeddings
            patch_tokens = patch_tokens + self.vision_pos_embed
            
            # Project to BART dim
            vision_features = self.vision_proj(patch_tokens)
        
        # ...rest of forward pass unchanged...
```

### Bước 3: Training Command

```bash
python train_no_latent.py \
    --use_hybrid_vision \
    --use_text_lora \
    --text_lora_r 16 \
    --use_type_loss \
    --use_vision_gate \
    --batch_size 12 \
    --epochs 30 \
    --lr 5e-5 \
    --output_dir checkpoints/hybrid_siglip_dinov2
```

## 🎯 KỲ VỌNG KẾT QUẢ

### Breakdown theo loại câu hỏi:

| Type | SigLIP hiện tại | Hybrid dự đoán | Cải thiện | Giải thích |
|------|----------------|----------------|-----------|-----------|
| **COUNT** | 59.33% | **~59-60%** | ~0% | SigLIP đã tối ưu (frozen OK) |
| **COLOR** | 60.30% | **~62-63%** | +2-3% | SigLIP + DINOv2 visual details |
| **OBJECT** | 64.95% | **~67-68%** | +2-3% | 🔥 DINOv2 strength kicks in! |
| **LOCATION** | 59.14% | **~63-64%** | +4-5% | 🔥 DINOv2 spatial understanding! |
| **OVERALL** | 61.45% | **~64-66%** | **+3-4.5%** | Best of both worlds! |

### Tại sao cải thiện?

1. **COUNT/COLOR:** SigLIP đã mạnh → giữ nguyên (frozen OK)
2. **OBJECT:** DINOv2 LoRA học object features cho Vietnamese → bù lại điểm yếu
3. **LOCATION:** DINOv2 spatial bias → boost mạnh

## ⚙️ TECHNICAL DETAILS

### Memory Usage:

```
SigLIP (frozen):        ~86M params × 4 bytes = 344 MB
DINOv2 (frozen base):   ~86M params × 4 bytes = 344 MB
DINOv2 LoRA adapters:   ~0.44M params        = 2 MB (trainable!)
Fusion MLP:             ~1.5M params          = 6 MB
------------------------------------------------------------
TOTAL:                  ~173.5M params        = 696 MB
TRAINABLE:              ~2M params (1.15%)
```

**So sánh với hiện tại:**
- Hiện tại: 86M params (SigLIP only)
- Hybrid: 173.5M params (2× vision encoders)
- **Trade-off:** 2× memory nhưng chỉ +2M trainable params!

### Computational Cost:

- **Forward pass:** 2× vision encoder calls (parallelizable nếu có 2 GPU)
- **Backward pass:** Chỉ qua DINOv2 LoRA (SigLIP frozen → no gradients)
- **Training time:** ~1.5× slower (acceptable cho +3-4.5% EM!)

---

# 🎨 CHIẾN LƯỢC 2: TYPE-CONDITIONED VISION ADAPTER

## 💡 Ý TƯỞNG CỐT LÕI

**"Học cách biến đổi SigLIP features khác nhau cho từng loại câu hỏi!"**

Không dùng 2 encoders, mà dùng **4 adapter networks** (1 cho mỗi loại):

```
Input Image
    │
    ▼
┌─────────────┐
│   SigLIP    │ (frozen)
│  (frozen)   │
└──────┬──────┘
       │ [B, 196, 768]
       │
       ├──────────┬──────────┬──────────┬──────────┐
       │          │          │          │          │
   ┌───▼───┐  ┌──▼───┐  ┌───▼───┐  ┌──▼───┐  ┌───▼───┐
   │Expert │  │Expert│  │Expert │  │Expert│  │ Gate  │
   │ OBJECT│  │COUNT │  │ COLOR │  │LOCATI│  │Network│
   │  MLP  │  │ MLP  │  │  MLP  │  │ON MLP│  │       │
   └───┬───┘  └──┬───┘  └───┬───┘  └──┬───┘  └───┬───┘
       │          │          │          │          │
       │ [B,196,768] × 4 experts         │ [B, 4] weights
       └──────────┴──────────┴───────────┴──────────┘
                           │
                    Weighted Sum
                           │
                      [B, 196, 768]
                           │
                      Project 768→1024
                           │
                      (To BART)
```

## 🧠 LOGIC

### Mỗi expert học một cách nhìn khác nhau:

| Expert | Học gì? |Ví dụ |
|--------|---------|-------|
| **OBJECT Expert** | Enhance object boundaries, salient regions | "Đây là **xe lửa**" → focus vào train patches |
| **COUNT Expert** | Global context, object distribution | "Có **mấy** người?" → spread attention toàn ảnh |
| **COLOR Expert** | Color-sensitive features, texture | "Màu **gì**?" → enhance color information |
| **LOCATION Expert** | Spatial relationships, positional encoding | "Ở **đâu**?" → boost spatial structure |

### Gating Network:

Học cách "route" input đến expert nào:

```python
# Ví dụ: Question = "Có bao nhiêu người?"
# → Gate network outputs: [0.05, 0.85, 0.05, 0.05]
#   (85% COUNT expert, 5% mỗi cái khác)

# Question = "Cái gì bên trái?"
# → Gate network outputs: [0.45, 0.0, 0.05, 0.50]
#   (45% OBJECT, 50% LOCATION, 5% COLOR)
```

## 📝 IMPLEMENTATION CHI TIẾT

### Bước 1: Tạo Adapter Module

```python
# File: type_conditioned_adapter.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class TypeConditionedVisionAdapter(nn.Module):
    """
    Mixture-of-Experts style adapter for SigLIP features.
    
    Each question type gets a specialized expert network that
    transforms vision features differently.
    
    Expected improvement: +1.5-2% EM (63-64% total)
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_types: int = 4,
        intermediate_dim: int = 768,  # Expert hidden size
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_types = num_types
        
        print(f"[Type-Conditioned Adapter] Initializing...")
        print(f"  Input dim: {hidden_dim}")
        print(f"  Num experts: {num_types} (OBJECT, COUNT, COLOR, LOCATION)")
        
        # ============================================================
        # EXPERT NETWORKS (4 specialized MLPs)
        # ============================================================
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, intermediate_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_types)
        ])
        
        expert_params = sum(p.numel() for p in self.experts.parameters())
        print(f"  Expert networks: {expert_params:,} params total")
        
        # ============================================================
        # GATING NETWORK (learns routing)
        # ============================================================
        # Input: pooled vision features
        # Output: weights over 4 experts
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_types),
            nn.Softmax(dim=-1)  # Output: [B, 4] probabilities
        )
        
        gate_params = sum(p.numel() for p in self.gate.parameters())
        print(f"  Gating network: {gate_params:,} params")
        
        # ============================================================
        # TYPE-CONDITIONED GATING (optional enhancement)
        # ============================================================
        # If type_ids provided, bias gating toward that type
        self.type_bias = nn.Parameter(torch.zeros(num_types, num_types))
        # type_bias[i, j] = bias for expert j when question type is i
        
        print(f"[Type-Conditioned Adapter] ✓ Total: {expert_params + gate_params:,} params")
    
    def forward(
        self,
        vision_features,
        type_ids=None,
        return_gate_weights=False
    ):
        """
        Args:
            vision_features: [B, num_patches, hidden_dim] - SigLIP features
            type_ids: [B] - question type IDs (0-3), optional
            return_gate_weights: bool - whether to return gating weights
        
        Returns:
            adapted_features: [B, num_patches, hidden_dim] - type-adapted features
            (gate_weights): [B, num_types] - routing weights (if requested)
        """
        batch_size, num_patches, hidden_dim = vision_features.shape
        
        # ============================================================
        # STEP 1: Compute gating weights
        # ============================================================
        # Pool vision features (mean pooling across patches)
        pooled = vision_features.mean(dim=1)  # [B, hidden_dim]
        
        # Gate network: pooled → routing weights
        gate_weights = self.gate(pooled)  # [B, num_types]
        
        # If type_ids provided, bias gating (supervised routing)
        if type_ids is not None:
            # Get bias for each sample's type
            type_bias = self.type_bias[type_ids]  # [B, num_types]
            
            # Add bias and re-normalize
            gate_weights = gate_weights + type_bias
            gate_weights = F.softmax(gate_weights, dim=-1)
        
        # ============================================================
        # STEP 2: Apply expert networks
        # ============================================================
        # Process vision features through ALL experts
        expert_outputs = []
        for expert in self.experts:
            # Expert transforms features: [B, num_patches, hidden_dim]
            expert_out = expert(vision_features)
            expert_outputs.append(expert_out)
        
        # Stack: [num_types, B, num_patches, hidden_dim]
        expert_outputs = torch.stack(expert_outputs, dim=0)
        
        # Permute to [B, num_types, num_patches, hidden_dim]
        expert_outputs = expert_outputs.permute(1, 0, 2, 3)
        
        # ============================================================
        # STEP 3: Weighted combination (soft routing)
        # ============================================================
        # Reshape gate_weights for broadcasting: [B, num_types, 1, 1]
        gate_weights = gate_weights.unsqueeze(2).unsqueeze(3)
        
        # Weighted sum: [B, num_types, num_patches, hidden_dim] → [B, num_patches, hidden_dim]
        adapted_features = (expert_outputs * gate_weights).sum(dim=1)
        
        # ============================================================
        # STEP 4: Residual connection (stabilize training)
        # ============================================================
        # Add residual from original features
        adapted_features = adapted_features + vision_features
        
        if return_gate_weights:
            return adapted_features, gate_weights.squeeze(2).squeeze(2)
        else:
            return adapted_features
```

### Bước 2: Tích hợp vào Model

```python
# File: model_no_latent.py

from type_conditioned_adapter import TypeConditionedVisionAdapter

class DeterministicVQA(nn.Module):
    def __init__(
        self,
        # ...existing args...
        use_type_adapter: bool = False,  # 🔥 NEW FLAG
    ):
        super().__init__()
        
        # ...existing vision encoder initialization...
        
        # ============================================================
        # TYPE-CONDITIONED ADAPTER (AFTER vision encoder)
        # ============================================================
        if use_type_adapter:
            self.vision_adapter = TypeConditionedVisionAdapter(
                hidden_dim=vision_hidden_dim,  # 768 for SigLIP
                num_types=4,
                intermediate_dim=768,
                dropout=dropout
            )
            print("[MODEL] Type-Conditioned Adapter: ENABLED")
        else:
            self.vision_adapter = None
        
        # ...rest of initialization...
    
    def forward(self, pixel_values, input_ids, attention_mask, type_ids=None):
        # ...existing vision encoding...
        
        # Vision features: [B, 196, 768]
        vision_output = self.vision_encoder(pixel_values)
        patch_tokens = vision_output.last_hidden_state[:, 1:, :]
        
        # ============================================================
        # 🔥 APPLY TYPE-CONDITIONED ADAPTER (if enabled)
        # ============================================================
        if self.vision_adapter is not None:
            patch_tokens = self.vision_adapter(
                patch_tokens,
                type_ids=type_ids  # Use ground-truth types during training
            )
        
        # Add position embeddings
        patch_tokens = patch_tokens + self.vision_pos_embed
        
        # Project to BART dim
        vision_features = self.vision_proj(patch_tokens)
        
        # ...rest of forward pass...
```

### Bước 3: Training Command

```bash
python train_no_latent.py \
    --vision_model google/siglip-base-patch16-224 \
    --use_type_adapter \
    --use_text_lora \
    --text_lora_r 16 \
    --use_type_loss \
    --use_vision_gate \
    --batch_size 12 \
    --epochs 30 \
    --lr 5e-5 \
    --output_dir checkpoints/siglip_type_adapter
```

## 🎯 KỲ VỌNG KẾT QUẢ

| Type | SigLIP hiện tại | Adapter dự đoán | Cải thiện | Giải thích |
|------|----------------|----------------|-----------|-----------|
| **COUNT** | 59.33% | **~60-61%** | +1-2% | COUNT expert học global attention |
| **COLOR** | 60.30% | **~61-62%** | +1-2% | COLOR expert enhance texture |
| **OBJECT** | 64.95% | **~66-67%** | +1-2% | OBJECT expert learn saliency |
| **LOCATION** | 59.14% | **~61-62%** | +2-3% | LOCATION expert boost spatial |
| **OVERALL** | 61.45% | **~63-64%** | **+1.5-2%** | Adaptive features! |

### Ưu điểm so với Hybrid:

- ✅ **Nhẹ hơn:** ~4.7M trainable (vs 2M hybrid, nhưng no 2× encoders)
- ✅ **Nhanh hơn:** 1× forward pass (vs 2× for hybrid)
- ✅ **Đơn giản hơn:** Chỉ thêm adapter, không cần 2 vision encoders
- ⚠️ **Cải thiện ít hơn:** +1.5-2% (vs +3-4.5% hybrid)

---

# 📊 SO SÁNH 2 CHIẾN LƯỢC

| Tiêu chí | **Hybrid Encoder** | **Type Adapter** |
|----------|-------------------|------------------|
| **Ý tưởng** | 2 vision encoders (SigLIP + DINOv2) | 4 expert networks for SigLIP features |
| **Trainable params** | ~2M (DINOv2 LoRA) | ~4.7M (4 experts + gate) |
| **Memory** | ~696 MB (2× encoders) | ~350 MB (1× encoder) |
| **Forward time** | 2× vision calls | 1× vision + 4× small MLPs |
| **Cải thiện dự đoán** | **+3-4.5% EM** | **+1.5-2% EM** |
| **Target EM** | **64-66%** | **63-64%** |
| **Phức tạp** | Trung bình (handle 2 encoders) | Đơn giản (just add adapter) |
| **Khi nào dùng?** | Muốn max performance | Muốn balance performance/cost |

---

# 🚀 KHUYẾN NGHỊ

## 🥇 **Nếu có GPU/Memory tốt:**
→ **Dùng Hybrid Encoder** (+3-4.5% EM)
- Kết quả tốt nhất
- Trade-off memory OK (chỉ 2× encoders)

## 🥈 **Nếu resource hạn chế:**
→ **Dùng Type Adapter** (+1.5-2% EM)
- Nhẹ hơn, nhanh hơn
- Vẫn cải thiện đáng kể

## 🥉 **Nếu muốn thử nhanh:**
→ **Text LoRA only** (code đã có, +0.5-1% EM)
- 0 effort
- Quick win

---

**Bạn muốn implement cái nào? Tôi có thể code full ngay! 🚀**
