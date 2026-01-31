# 🔬 Vision Encoder Comparison: SigLIP vs DINOv2

## 📊 Model Specs

| Feature | **SigLIP-base** | **DINOv2-base** |
|---------|----------------|-----------------|
| **Parameters** | ~87M | ~86M |
| **Hidden dim** | 768 | 768 |
| **Image resolution** | 224×224 | 224×224 |
| **Training data** | WebLI (12B images) | LVD-142M |
| **Training objective** | Sigmoid loss (language-aligned) | Self-supervised (no language) |
| **Strengths** | Better text-vision alignment | Better visual representations |

---

## �️ Architecture Differences (CRITICAL!)

### **SigLIP Structure:**
```python
SiglipModel (full model from AutoModel.from_pretrained())
├── vision_model (SiglipVisionModel)  ← WE NEED THIS!
│   ├── embeddings (SiglipVisionEmbeddings)
│   ├── encoder (SiglipEncoder)
│   │   └── layers[0-11] (SiglipEncoderLayer)
│   │       ├── self_attn (SiglipAttention)
│   │       │   ├── q_proj ← LoRA target
│   │       │   ├── k_proj ← LoRA target
│   │       │   └── v_proj ← LoRA target
│   │       └── mlp (SiglipMLP)
│   └── post_layernorm
└── text_model (SiglipTextModel)  ← NOT NEEDED for VQA
```

**Key Points:**
- Must extract `vision_model` component only!
- LoRA target modules: `q_proj`, `k_proj`, `v_proj`
- Config attribute: `vision_config.hidden_size` (NOT `hidden_size`)
- Gradient checkpointing: Use `config.gradient_checkpointing = True` (NO method!)

### **DINOv2 Structure:**
```python
Dinov2Model (vision-only from AutoModel.from_pretrained())
├── embeddings (Dinov2Embeddings)
├── encoder (Dinov2Encoder)
│   └── layer[0-11] (Dinov2Layer)
│       ├── attention (Dinov2Attention)
│       │   ├── attention.query ← LoRA target
│       │   ├── attention.key ← LoRA target
│       │   └── attention.value ← LoRA target
│       └── mlp
└── layernorm
```

**Key Points:**
- Already vision-only, use directly
- LoRA target modules: `query`, `key`, `value`
- Config attribute: `hidden_size` directly
- Gradient checkpointing: Has `gradient_checkpointing_enable()` method

---

## 🔧 Code Implementation

### **1. Model Loading:**
```python
# Load full model
full_model = AutoModel.from_pretrained(model_name)

# Extract vision component for SigLIP
if hasattr(full_model, 'vision_model'):
    vision_encoder = full_model.vision_model  # SigLIP
    hidden_dim = full_model.config.vision_config.hidden_size
else:
    vision_encoder = full_model  # DINOv2
    hidden_dim = full_model.config.hidden_size
```

### **2. Gradient Checkpointing:**
```python
# MUST enable BEFORE LoRA injection!
if hasattr(vision_encoder, 'config'):
    # SigLIP: config-based
    vision_encoder.config.gradient_checkpointing = True
elif hasattr(vision_encoder, 'gradient_checkpointing_enable'):
    # DINOv2: method-based
    vision_encoder.gradient_checkpointing_enable()
```

### **3. LoRA Injection:**
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj"],  # SigLIP
    # target_modules=["query", "key", "value"],  # DINOv2
    bias="none",
    task_type="FEATURE_EXTRACTION"
)

vision_encoder = get_peft_model(vision_encoder, lora_config)
```

### **4. Forward Pass:**
```python
# Both work the same after extraction
outputs = vision_encoder(pixel_values=pixel_values)
patch_tokens = outputs.last_hidden_state  # [batch, 197, 768]

# Remove CLS token (first token)
patch_tokens = patch_tokens[:, 1:, :]  # [batch, 196, 768]
```

---

## �🎯 Which to Use?

### **SigLIP** (RECOMMENDED for VQA)
✅ **Pros:**
- Trained with **language-image pairs** → Better for VQA tasks
- Sigmoid loss → Better fine-grained alignment
- Larger training data (12B images)
- Better zero-shot transfer for vision-language tasks

⚠️ **Cons:**
- More complex structure (need extraction)
- Gradient checkpointing via config only
- Slightly more memory usage

### **DINOv2**
✅ **Pros:**
- Pure visual features → Better object detection
- Self-supervised → More general representations
- Simpler structure (vision-only)
- Easier gradient checkpointing

⚠️ **Cons:**
- Not trained with language → Need more fusion layers
- May struggle with language-specific concepts

---

## 🚀 Usage

### Switch to SigLIP (default):
```bash
python train_no_latent.py \
    --vision_model google/siglip-base-patch16-224 \
    --train_csv data/train.csv \
    --image_dir data/images \
    --use_vision_lora \
    --vision_lora_r 8 \
    ...
```

### Back to DINOv2:
```bash
python train_no_latent.py \
    --vision_model facebook/dinov2-base \
    --train_csv data/train.csv \
    --image_dir data/images \
    --use_vision_lora \
    --vision_lora_r 8 \
    ...
```

### Try SigLIP-large (more capacity):
```bash
python train_no_latent.py \
    --vision_model google/siglip-large-patch16-256 \
    --train_csv data/train.csv \
    --image_dir data/images \
    --batch_size 8 \  # Reduce batch size!
    ...
```

---

## 📈 Expected Performance

Based on similar VQA benchmarks:

| Model | Expected EM | Expected F1 | Training Speed |
|-------|-------------|-------------|----------------|
| **SigLIP-base** | ~45-50% | ~60-65% | 1.0x (baseline) |
| **DINOv2-base** | ~42-47% | ~57-62% | 1.1x faster |
| **SigLIP-large** | ~48-53% | ~63-68% | 0.7x (slower) |

*Note: Actual performance depends on your dataset quality and hyperparameters*

---

## 🐛 Common Issues & Fixes

### Issue 1: `AttributeError: 'SiglipConfig' object has no attribute 'hidden_size'`
**Fix:** Use `vision_config.hidden_size` instead
```python
hidden_dim = model.config.vision_config.hidden_size  # SigLIP
```

### Issue 2: `'SiglipVisionTransformer' object has no attribute 'gradient_checkpointing_enable'`
**Fix:** Use config-based approach BEFORE LoRA
```python
model.config.gradient_checkpointing = True  # SigLIP
```

### Issue 3: `TypeError: got multiple values for keyword argument 'inputs_embeds'`
**Fix:** Extract `vision_model` component only, not full SigLIP model
```python
vision_encoder = full_model.vision_model  # Extract component
```

---

## 🧪 Experiment Tracking

When comparing models, use different output directories:
```bash
# SigLIP experiment
--output_dir checkpoints/siglip_base_lora

# DINOv2 experiment  
--output_dir checkpoints/dinov2_base_lora
```

Then compare training curves in `training_metrics.csv`!
