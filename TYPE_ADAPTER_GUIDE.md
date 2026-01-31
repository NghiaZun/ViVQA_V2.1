# 🎯 Type-Conditioned Vision Adapter - Complete Guide

## 📊 Overview

Lightweight adapter that transforms SigLIP vision features based on question type.

**Expected improvement:** +1.5-2% EM (from 61.45% → 63.0-63.5%)

### Key Features:
- ✅ 4 expert networks (OBJECT, COUNT, COLOR, LOCATION)
- ✅ Low-rank bottleneck (768 → 64 → 768) for efficiency
- ✅ Gating network for soft routing
- ✅ Type supervision during training
- ✅ Residual connection to prevent collapse
- ✅ Only ~600K trainable params

---

## 🚀 Quick Start

### Step 1: Test the adapter module

```bash
python test_type_adapter.py
```

This will run 5 tests:
1. ✅ Forward pass
2. ✅ Gating specialization
3. ✅ Gradient flow
4. ✅ Inference mode
5. ✅ Parameter count

Expected output:
```
🎉 ALL TESTS PASSED!
Adapter is ready for training!
```

### Step 2: Train with adapter

```bash
chmod +x train_type_adapter.sh
./train_type_adapter.sh
```

Or custom command:
```bash
python train_no_latent.py \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --image_dir data/images \
    --vision_model google/siglip-base-patch16-224 \
    --use_type_adapter \
    --type_adapter_rank 64 \
    --type_adapter_bias 2.0 \
    --use_text_lora \
    --text_lora_r 16 \
    --use_type_loss \
    --type_loss_weight 0.5 \
    --use_vision_gate \
    --batch_size 12 \
    --epochs 30 \
    --lr 5e-5 \
    --output_dir checkpoints/type_adapter
```

### Step 3: Monitor training

Watch for gating specialization in logs:
```
[Type-Conditioned Adapter] Initializing...
  Input dim: 768
  Num experts: 4 (OBJECT, COUNT, COLOR, LOCATION)
  Rank: 64
  Expert networks: 393,216 params
  Gating network: 148,740 params
  Type bias: 2.0 (diagonal)
```

---

## 📂 File Structure

```
ViVQA_V2.1/
├── type_conditioned_adapter.py    # Core adapter module
├── model_no_latent.py             # Updated with adapter integration
├── train_no_latent.py             # Updated training script
├── test_type_adapter.py           # Test suite
├── train_type_adapter.sh          # Training script
└── TYPE_ADAPTER_GUIDE.md          # This file
```

---

## 🧠 How It Works

### Architecture

```
Input Image (224×224)
        ↓
   SigLIP Encoder (frozen)
        ↓
   Vision Features [B, 196, 768]
        ↓
   ┌─────────────────────────────┐
   │  Type-Conditioned Adapter   │
   │                             │
   │  ┌────────┐   ┌──────────┐ │
   │  │ Expert │   │  Gating  │ │
   │  │   0    │   │ Network  │ │
   │  │OBJECT  │   │          │ │
   │  └───┬────┘   │  Pool    │ │
   │      │        │    ↓     │ │
   │  ┌───┴────┐   │ Weights  │ │
   │  │ Expert │   │ [B, 4]   │ │
   │  │   1    │   └─────┬────┘ │
   │  │ COUNT  │         │      │
   │  └───┬────┘         │      │
   │      │              │      │
   │  ┌───┴────┐         │      │
   │  │ Expert │         │      │
   │  │   2    │    Weighted    │
   │  │ COLOR  │   Combination  │
   │  └───┬────┘         │      │
   │      │              │      │
   │  ┌───┴────┐         │      │
   │  │ Expert │         │      │
   │  │   3    │         │      │
   │  │LOCATION│         │      │
   │  └────────┘         │      │
   │                     ↓      │
   │   Adapted Features [B,196,768]
   │         + Residual         │
   └─────────────────────────────┘
        ↓
   Position Embeddings
        ↓
   Vision Projection (768 → 1024)
        ↓
   Flamingo Fusion
        ↓
   BART Decoder
        ↓
   Answer Generation
```

### Expert Specialization

Each expert learns to transform features differently:

| Expert | Transformation | Example |
|--------|----------------|---------|
| **OBJECT** | Enhance object boundaries, salient regions | "Đây là **xe lửa**" → focus on train patches |
| **COUNT** | Global context, object distribution | "Có **mấy** người?" → spread attention across image |
| **COLOR** | Color-sensitive features, texture | "Màu **gì**?" → enhance color information |
| **LOCATION** | Spatial relationships, positional encoding | "Ở **đâu**?" → boost spatial structure |

### Gating Mechanism

The gating network learns to route inputs to appropriate experts:

**Example 1:** Question = "Có bao nhiêu người?"
```
Gate weights: [0.05, 0.85, 0.05, 0.05]
              OBJECT COUNT COLOR LOCATION
→ Routes 85% to COUNT expert ✅
```

**Example 2:** Question = "Cái gì bên trái?"
```
Gate weights: [0.45, 0.0, 0.05, 0.50]
              OBJECT COUNT COLOR LOCATION
→ Routes 45% OBJECT + 50% LOCATION ✅
```

---

## 📈 Expected Results

### Overall Improvement

| Metric | Baseline (SigLIP) | With Adapter | Gain |
|--------|------------------|--------------|------|
| **Loss** | 1.8390 | ~1.80-1.82 | -0.02 |
| **Exact Match** | 61.45% | **63.0-63.5%** | **+1.5-2%** |
| **F1 Score** | 67.75% | **68.5-69.0%** | **+0.75-1.25%** |

### Per-Type Breakdown

| Type | Baseline | Expected | Gain | Reason |
|------|----------|----------|------|--------|
| COUNT | 59.33% | 60.0-60.5% | +0.5-1% | COUNT expert learns global attention |
| COLOR | 60.30% | 61.0-61.5% | +0.5-1% | COLOR expert enhances texture |
| OBJECT | 64.95% | 66.0-66.5% | +1-1.5% | OBJECT expert learns saliency |
| LOCATION | 59.14% | 61.0-62.0% | +2-3% | 🔥 LOCATION expert boosts spatial features |

**Best improvement:** LOCATION (+2-3%) because adapter can learn spatial bias.

---

## 🛠️ Configuration Options

### Adapter Parameters

```bash
--use_type_adapter              # Enable adapter (default: False)
--type_adapter_rank 64          # Bottleneck rank (default: 64)
                                # Lower = fewer params, higher = more capacity
                                # Recommended: 32-128
--type_adapter_bias 2.0         # Type supervision strength (default: 2.0)
                                # Higher = stronger bias toward correct expert
                                # Recommended: 1.5-3.0
```

### Recommended Combinations

**Option 1: Adapter Only** (Minimal)
```bash
python train_no_latent.py \
    --use_type_adapter \
    --use_type_loss \
    --use_vision_gate
```

**Option 2: Adapter + Text LoRA** (Recommended! ⭐)
```bash
python train_no_latent.py \
    --use_type_adapter \
    --use_text_lora \
    --text_lora_r 16 \
    --use_type_loss \
    --use_vision_gate
```

**Option 3: Full Features** (Maximum Performance)
```bash
python train_no_latent.py \
    --use_type_adapter \
    --type_adapter_rank 64 \
    --type_adapter_bias 2.0 \
    --use_text_lora \
    --text_lora_r 16 \
    --use_type_loss \
    --type_loss_weight 0.5 \
    --use_vision_gate \
    --vision_gate_init 1.5 \
    --batch_size 12 \
    --lr 5e-5 \
    --epochs 30 \
    --gradient_checkpointing
```

---

## 🔍 Debugging & Analysis

### Check Gating Specialization

After training, check if experts specialize correctly:

```python
from type_conditioned_adapter import TypeConditionedVisionAdapter, print_specialization_matrix

# Load trained model
model = torch.load('checkpoints/type_adapter/best_model.pt')
adapter = model.vision_adapter

# Analyze on validation set
specialization = adapter.get_expert_specialization(val_loader, device)

# Print matrix
print_specialization_matrix(specialization)
```

Expected output (good specialization):
```
Expert Specialization Matrix:
Type      OBJECT    COUNT     COLOR     LOCATION  
--------------------------------------------------
OBJECT     0.650     0.150     0.100     0.100
COUNT      0.100     0.750     0.100     0.050
COLOR      0.100     0.100     0.700     0.100
LOCATION   0.100     0.050     0.100     0.750

✅ Diagonal values > 0.6 = Good specialization!
```

Bad specialization (collapse):
```
OBJECT     0.250     0.250     0.250     0.250
COUNT      0.250     0.250     0.250     0.250
...

❌ Uniform distribution = Experts collapsed!
```

### Visualize Gating Distribution

```python
from type_conditioned_adapter import visualize_gating_distribution

# Collect gating weights during validation
gate_weights_list = []
type_ids_list = []

for batch in val_loader:
    with torch.no_grad():
        _, gate_info = adapter(
            batch['vision_features'],
            return_gate_info=True
        )
        gate_weights_list.append(gate_info['weights'])
        type_ids_list.append(batch['type_ids'])

gate_weights = torch.cat(gate_weights_list)
type_ids = torch.cat(type_ids_list)

# Plot
visualize_gating_distribution(
    gate_weights, 
    type_ids, 
    save_path='gating_distribution.png'
)
```

---

## 🎓 Tips & Best Practices

### 1. **Type Supervision Strategy**

**Stage 1 (Epochs 1-15):** Strong supervision
```python
type_adapter_bias = 2.0  # Force experts to specialize
```

**Stage 2 (Epochs 16-30):** Weaker supervision
```python
type_adapter_bias = 1.0  # Let gating network learn naturally
```

Or use curriculum learning:
```python
# In training loop
current_bias = 2.0 * (1 - epoch / total_epochs)  # 2.0 → 0.0
```

### 2. **Rank Selection**

| Rank | Params | When to Use |
|------|--------|-------------|
| 32 | ~200K | Small dataset (<5K samples) |
| 64 | ~400K | Medium dataset (5-15K samples) ⭐ |
| 128 | ~800K | Large dataset (>15K samples) |

### 3. **Combining with Other Features**

✅ **Good combinations:**
- Adapter + Text LoRA (complementary)
- Adapter + Vision Gating (both type-conditioned)
- Adapter + Type Loss (reinforcing)

❌ **Bad combinations:**
- Adapter + Vision LoRA (conflicts - SigLIP can't use vision LoRA)
- Adapter + Vision Dropout (might hurt specialization)

### 4. **Monitoring Training**

Watch these metrics:
- **Gating entropy:** Should decrease (experts specializing)
- **Per-expert usage:** Should be balanced (~25% each)
- **Gradient norm:** Adapter grads should be non-zero

---

## ❓ FAQ

**Q: Why not just use a single big MLP?**
A: Single MLP learns average transformation. Experts learn specialized transformations per type, leading to better accuracy.

**Q: Why low-rank (768→64→768)?**
A: Efficiency! Low-rank = fewer params = less overfitting on small datasets. Still effective because we have 4 experts.

**Q: Can I use this with DINOv2?**
A: Yes! Just change `--vision_model facebook/dinov2-base`. Works with any vision encoder.

**Q: What if experts don't specialize?**
A: Try:
1. Increase `type_adapter_bias` (2.0 → 3.0)
2. Increase `type_loss_weight` (0.5 → 1.0)
3. Use stronger supervision in early epochs

**Q: Can I freeze adapter after training?**
A: No! Adapter needs to adapt to each input. But you can freeze vision encoder.

---

## 🔄 Next Steps

After successful training with Type Adapter:

1. **Analyze results:** Check per-type improvements
2. **Visualize gating:** Understand expert specialization
3. **Ablation study:** Try removing adapter to validate gain
4. **Consider Hybrid:** If ceiling reached, try Hybrid Vision Encoder (SigLIP + DINOv2)

---

## 📚 References

- **Mixture of Experts:** Shazeer et al., 2017
- **Vision Transformers:** Dosovitskiy et al., 2020
- **SigLIP:** Zhai et al., 2023
- **Type-Conditioned Models:** Task-specific conditioning for multimodal learning

---

**Questions? Check `REALISTIC_IMPROVEMENT_STRATEGY.md` for full context!**
