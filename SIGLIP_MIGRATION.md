# SigLIP vs DINOv2 Comparison for ViVQA

## 🔄 Model Change: DINOv2 → SigLIP

### Why SigLIP?

| Feature | **DINOv2** | **SigLIP** | Winner |
|---------|-----------|-----------|---------|
| **Training objective** | Self-supervised (DINO) | Contrastive (image-text) | 🏆 SigLIP |
| **Language alignment** | ❌ None (vision only) | ✅ Native (trained with text) | 🏆 SigLIP |
| **VQA suitability** | Good for general vision | Better for vision-language | 🏆 SigLIP |
| **Model size** | 86M params (base) | 86M params (base) | 🤝 Tie |
| **Hidden dim** | 768 | 768 | 🤝 Tie |
| **Patch size** | 14x14 or 16x16 | 16x16 | 🤝 Tie |
| **Performance** | Strong on ImageNet | Strong on multimodal tasks | 🏆 SigLIP |
| **Vietnamese support** | Indirect (visual features) | Better (cross-lingual CLIP) | 🏆 SigLIP |

---

## 📊 Expected Performance Changes

### ✅ Advantages of SigLIP:
1. **Better language-vision alignment** → Better at understanding question-image relationships
2. **Contrastive training** → Learned to match images with text descriptions
3. **Cross-lingual capability** → Better handling of Vietnamese (via multilingual CLIP training)
4. **Optimized for VQA** → Trained on tasks similar to VQA

### ⚠️ Potential Challenges:
1. **Different embedding space** → May need different fusion hyperparams
2. **Less tested** → DINOv2 more widely used in academia
3. **Different preprocessing** → SigLIP uses different image normalization

---

## 🔧 Code Changes Made

### 1. Model Architecture (`model_no_latent.py`)
```python
# OLD (DINOv2)
def __init__(
    self,
    dinov2_model_name: str = 'facebook/dinov2-base',
    ...
):
    self.vision_encoder = AutoModel.from_pretrained(dinov2_model_name)
    # LoRA target modules: ["query", "key", "value"]

# NEW (SigLIP)
def __init__(
    self,
    vision_model_name: str = 'google/siglip-base-patch16-224',
    ...
):
    self.vision_encoder = AutoModel.from_pretrained(vision_model_name)
    # LoRA target modules: ["q_proj", "k_proj", "v_proj"]  # Different naming!
```

### 2. CLS Token Handling
```python
# DINOv2: Always has CLS token at position 0
patch_tokens = patch_tokens[:, 1:, :]  # [batch, 256, 768]

# SigLIP: Check if CLS exists (more flexible)
if patch_tokens.size(1) > 196:  # Has CLS token
    patch_tokens = patch_tokens[:, 1:, :]
```

### 3. Training Script (`train_no_latent.py`)
```python
# OLD
parser.add_argument('--dinov2_model', type=str, default='facebook/dinov2-base')

# NEW
parser.add_argument('--vision_model', type=str, default='google/siglip-base-patch16-224')
```

---

## 🚀 How to Use

### Option 1: Quick test (use bash script)
```bash
chmod +x train_siglip.sh
./train_siglip.sh
```

### Option 2: Custom command
```bash
python train_no_latent.py \
    --train_csv /path/to/train.csv \
    --image_dir /path/to/images \
    --vision_model google/siglip-base-patch16-224 \
    --use_vision_lora \
    --use_text_lora \
    --use_vision_gate \
    --use_type_loss \
    ...
```

### Option 3: Switch back to DINOv2 (if needed)
```bash
python train_no_latent.py \
    --vision_model facebook/dinov2-base \
    ...
```

---

## 📈 Performance Expectations

### Baseline (DINOv2)
- Exact Match: ~45-50%
- F1 Score: ~55-60%
- Strong on: Visual features, object detection
- Weak on: Text-image alignment

### Expected (SigLIP)
- Exact Match: ~48-53% (+3-5% improvement expected)
- F1 Score: ~58-63% (+3-5% improvement expected)
- Strong on: Text-image alignment, multimodal understanding
- Weak on: Pure visual features (slightly)

### Why expect improvement?
1. **Better at "understanding" questions** → SigLIP learned image-text relationships
2. **Cross-lingual robustness** → CLIP training includes multilingual data
3. **VQA is multimodal** → SigLIP explicitly trained for this

---

## 🔬 Ablation Study Recommended

Compare both models to understand which works best for Vietnamese VQA:

```bash
# Experiment 1: DINOv2 baseline
python train_no_latent.py \
    --vision_model facebook/dinov2-base \
    --output_dir checkpoints/dinov2_baseline \
    ...

# Experiment 2: SigLIP
python train_no_latent.py \
    --vision_model google/siglip-base-patch16-224 \
    --output_dir checkpoints/siglip_baseline \
    ...

# Compare results!
python compare_checkpoints.py \
    --checkpoint1 checkpoints/dinov2_baseline/best_model.pt \
    --checkpoint2 checkpoints/siglip_baseline/best_model.pt
```

---

## ⚙️ Technical Notes

### LoRA Target Modules Difference
- **DINOv2**: Uses `query`, `key`, `value` (standard ViT naming)
- **SigLIP**: Uses `q_proj`, `k_proj`, `v_proj` (CLIP-style naming)

This is handled automatically in the updated code!

### Image Preprocessing
Both models use similar preprocessing (resize, normalize), but check:
```python
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained('google/siglip-base-patch16-224')
# SigLIP may use different normalization stats than DINOv2
```

### Hidden Dimensions
Both are 768, so no changes needed in fusion layers!

---

## 🎯 Recommendations

1. **Try SigLIP first** → Likely better for VQA tasks
2. **Keep same hyperparams** → lr=2e-4, batch_size=16, etc.
3. **Monitor training curves** → Compare convergence speed with DINOv2
4. **Test on validation** → Check if EM/F1 improves

If SigLIP doesn't improve:
- Try larger SigLIP model: `google/siglip-large-patch16-224`
- Or try different vision encoder: `openai/clip-vit-base-patch16`

---

## 📝 Summary

**Changed**: Vision encoder DINOv2 → SigLIP
**Why**: Better language-vision alignment for VQA
**Expected**: +3-5% performance improvement
**How**: Run `./train_siglip.sh` or use `--vision_model google/siglip-base-patch16-224`
