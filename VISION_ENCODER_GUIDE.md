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

## 🎯 Which to Use?

### **SigLIP** (RECOMMENDED for VQA)
✅ **Pros:**
- Trained with **language-image pairs** → Better for VQA tasks
- Sigmoid loss → Better fine-grained alignment
- Larger training data (12B images)
- Better zero-shot transfer for vision-language tasks

⚠️ **Cons:**
- Slightly slower inference (attention-based)
- May overfit to WebLI domain

### **DINOv2**
✅ **Pros:**
- Pure visual features → Better object detection
- Self-supervised → More general representations
- Faster inference
- Better for spatial reasoning

⚠️ **Cons:**
- Not trained with language → Need more fusion layers
- May struggle with language-specific concepts

---

## 🚀 Usage

### Switch to SigLIP (default now):
```bash
python train_no_latent.py \
    --vision_model google/siglip-base-patch16-224 \
    --train_csv data/train.csv \
    --image_dir data/images \
    ...
```

### Back to DINOv2:
```bash
python train_no_latent.py \
    --vision_model facebook/dinov2-base \
    --train_csv data/train.csv \
    --image_dir data/images \
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

## 🔧 Code Changes

The model now **auto-detects** vision encoder architecture:
```python
# In model_no_latent.py
if hasattr(self.vision_encoder.config, 'hidden_size'):
    vision_hidden_dim = self.vision_encoder.config.hidden_size  # DINOv2
elif hasattr(self.vision_encoder.config, 'vision_config'):
    vision_hidden_dim = self.vision_encoder.config.vision_config.hidden_size  # SigLIP
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

## 🧪 Experiment Tracking

When comparing models, use different output directories:
```bash
# SigLIP experiment
--output_dir checkpoints/siglip_base_lora

# DINOv2 experiment  
--output_dir checkpoints/dinov2_base_lora
```

Then compare training curves in `training_metrics.csv`!
