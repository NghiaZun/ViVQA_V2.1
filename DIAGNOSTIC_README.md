# 🔬 VQA Model Diagnostic Tools

## Triết lý: Evidence-Based Debugging

**KHÔNG BAO GIỜ** thay đổi model dựa trên triệu chứng (symptoms)!

**LUÔN LUÔN** chạy diagnostics để tìm root cause trước khi đổi architecture.

---

## 📋 Quick Start

```bash
# Chạy với validation set
python diagnostic_tools.py \
    --checkpoint checkpoints/best_model.pt \
    --csv data/val.csv \
    --image_dir data/images \
    --batch_size 16

# Hoặc chạy với test set (nếu không có val)
python diagnostic_tools.py \
    --checkpoint checkpoints/best_model.pt \
    --csv data/test.csv \
    --image_dir data/images \
    --batch_size 16
```

**Thời gian chạy:** ~5-10 phút (tùy dataset size)

**Output:** 
- ✅ Clear recommendation (FIX GATE / CHANGE ENCODER / FIX DATA)
- 📊 Detailed statistics for each test
- 🎯 Actionable next steps

---

## 🧪 Test Suite Overview

### Test A: Gate Behavior Analysis
**Mục đích:** Kiểm tra xem gating có đang suppress vision không

**Metric:** Mean gate value across all patches

**Thresholds:**
- `< 0.3` → ❌ Vision suppressed (FIX GATE!)
- `0.3-0.5` → ⚠️  Low vision usage
- `0.5-0.7` → ✅ Balanced fusion
- `> 0.7` → ✅ Strong vision reliance

**Recommendation if fail:**
1. Increase `vision_gate_init` (3.0 thay vì 1.5)
2. Add gate regularization loss
3. Remove gating (`use_vision_gate=False`)

---

### Test B: Vision Dependency (Ablation)
**Mục đích:** Đo lường model phụ thuộc vision bao nhiêu

**Method:** Compare accuracy:
- Real images vs Blank images
- Real images vs Noise images

**Metric:** Accuracy drop khi remove vision

**Thresholds:**
- `< 10%` → ❌ Model không cần vision (BAD!)
- `10-25%` → ⚠️  Moderate vision usage
- `> 25%` → ✅ Strong vision dependency

**Diagnosis tree:**
```
Drop < 10%
  ├─ IF feature quality GOOD (Test C pass)
  │  └─> PROBLEM: Dataset bias
  │     ACTION: Fix data, không đổi encoder
  │
  └─ IF feature quality BAD (Test C fail)
     └─> PROBLEM: DINOv2 không extract tốt
        ACTION: Đổi sang CLIP/SigLIP
```

---

### Test C: Vision Feature Quality
**Mục đích:** Kiểm tra DINOv2 có extract features tốt không

**Metrics:**
- Feature std (variance across features)
- Feature diversity (pairwise distance across samples)
- NaN/Inf detection

**Thresholds:**
- Std `< 0.1` → ❌ Feature collapse
- Diversity `< 1.0` → ⚠️  Low diversity
- Has NaN/Inf → ❌ CRITICAL BUG

**Recommendation if fail:**
1. Fine-tune vision encoder (LoRA)
2. Check preprocessing (normalization)
3. Consider different encoder (CLIP, SigLIP)

---

### Test D: Per-Type Breakdown
**Mục đích:** Identify which question types fail

**Output:** Accuracy for:
- OBJECT (Đây là gì?)
- COUNT (Có bao nhiêu?)
- COLOR (Màu gì?)
- LOCATION (Ở đâu?)

**Use case:** 
- If COUNT fails → Need better global features
- If COLOR fails → Need semantic understanding
- If LOCATION fails → Need spatial reasoning

---

## 🎯 Decision Tree (Tóm tắt)

```
1. Run diagnostic_tools.py

2. Check output recommendation:

   ┌─ "FIX GATING" 
   │  └─> Increase vision_gate_init
   │     Hoặc remove gating
   │     ❌ KHÔNG đổi encoder
   │
   ├─ "CHANGE ENCODER"
   │  └─> DINOv2 features xấu
   │     → Đổi sang CLIP/SigLIP
   │     ✅ Có evidence rõ ràng
   │
   ├─ "FIX DATA"
   │  └─> Dataset bias (text-only đủ mạnh)
   │     → Harder questions
   │     → Data augmentation
   │     ❌ KHÔNG đổi encoder
   │
   └─ "DINOv2 WORKING WELL"
      └─> Encoder tốt rồi
          → Fix fusion/decoder
          ❌ KHÔNG đổi encoder
```

---

## 📊 Example Output

```
================================================================================
🔬 VQA MODEL DIAGNOSTIC SUITE
================================================================================
Checkpoint: checkpoints/best_model.pt
Eval CSV: data/test.csv
Image Dir: data/images
================================================================================

📥 Loading model...
📥 Loading evaluation dataset...
✅ Loaded 1000 evaluation samples

================================================================================
TEST A: VISION GATE BEHAVIOR ANALYSIS
================================================================================
📊 Gate Statistics (across 256000 patch-level gates):
   Mean:   0.234
   Median: 0.189
   Std:    0.156
   Min:    0.012
   Max:    0.892
   P25:    0.134
   P75:    0.298
   P90:    0.445

🔍 Diagnosis:
   ❌ PROBLEM: Vision is heavily suppressed (mean < 0.3)
   → Model is NOT using vision features effectively
   → ACTION: Fix gating mechanism or remove gate

================================================================================
TEST B: VISION DEPENDENCY (ABLATION TEST)
================================================================================
📊 Vision Dependency Results:
   Accuracy with REAL images:  45.2%
   Accuracy with BLANK images: 42.1%
   Accuracy with NOISE images: 41.8%
   
   Drop when removing vision (blank): 3.1%
   Drop when adding noise:            3.4%

🔍 Diagnosis:
   ❌ PROBLEM: Model doesn't rely on vision (drop < 10%)
   → Text-only is sufficient for high accuracy
   → Possible causes:
      1. Dataset bias (questions answerable from text)
      2. Vision features not informative
      3. Gating suppressing vision
   → ACTION: Run Test C to check feature quality

================================================================================
🎯 FINAL RECOMMENDATION
================================================================================
❌ DON'T change vision encoder yet!
✅ ACTION: Fix gating mechanism
   Reason: Gating is suppressing vision features
   Try:
      1. Increase vision_gate_init (e.g., 3.0 instead of 1.5)
      2. Add gate regularization loss
      3. Remove gating entirely (use_vision_gate=False)
================================================================================
```

---

## 🛠️ Advanced Usage

### Test individual components

```python
from diagnostic_tools import (
    analyze_gate_statistics_v2,
    test_vision_dependency,
    check_vision_feature_quality,
    analyze_per_type_performance
)

# Load your model and dataloader
model = ...
dataloader = ...

# Run individual tests
gate_stats = analyze_gate_statistics_v2(model, dataloader)
ablation = test_vision_dependency(model, dataloader)
features = check_vision_feature_quality(model, dataloader)
per_type = analyze_per_type_performance(model, dataloader)
```

### Custom analysis

```python
# Example: Track gate evolution during training
gates_per_epoch = []

for epoch in range(num_epochs):
    train(...)
    
    gate_stats, gate_values = analyze_gate_statistics_v2(model, val_loader)
    gates_per_epoch.append(gate_stats['mean'])
    
    print(f"Epoch {epoch}: Mean gate = {gate_stats['mean']:.3f}")

# Plot gate evolution
import matplotlib.pyplot as plt
plt.plot(gates_per_epoch)
plt.xlabel('Epoch')
plt.ylabel('Mean Gate Value')
plt.title('Gate Evolution During Training')
plt.savefig('gate_evolution.png')
```

---

## 🚨 Common Issues & Solutions

### Issue 1: Gate mean < 0.3
**Root cause:** Gating initialized too low or learned to suppress

**Solution:**
```python
model = DeterministicVQA(
    use_vision_gate=True,
    vision_gate_init=3.0,  # 🔥 Increase from 1.5
    ...
)
```

Or add gate regularization:
```python
# In training loop
gate_penalty = -torch.log(gate_values.mean() + 1e-8)
loss = answer_loss + 0.1 * gate_penalty
```

### Issue 2: Vision drop < 10% but features good
**Root cause:** Dataset bias (text cues too strong)

**Solution:**
1. Filter text-biased questions
2. Add adversarial augmentation
3. Use contrastive VQA (positive/negative image pairs)

### Issue 3: Feature std < 0.1
**Root cause:** DINOv2 frozen and not adapted to domain

**Solution:**
```python
model = DeterministicVQA(
    use_vision_lora=True,  # 🔥 Enable LoRA fine-tuning
    vision_lora_r=8,
    vision_lora_alpha=16,
    ...
)
```

Or switch encoder:
```python
model = DeterministicVQA(
    dinov2_model_name='openai/clip-vit-large-patch14',  # 🔥 Use CLIP
    ...
)
```

---

## 📖 References

**Gating mechanisms:**
- Flamingo paper: https://arxiv.org/abs/2204.14198
- Vision-language fusion: https://arxiv.org/abs/2301.12597

**Vision encoders:**
- DINOv2: https://arxiv.org/abs/2304.07193
- CLIP: https://arxiv.org/abs/2103.00020
- SigLIP: https://arxiv.org/abs/2303.15343

**Ablation studies:**
- Vision ablation in VQA: https://arxiv.org/abs/1606.00061
- Modality dropout: https://arxiv.org/abs/1911.12782

---

## ✅ Checklist Trước Khi Đổi Model

- [ ] Đã chạy `diagnostic_tools.py`
- [ ] Đã xem gate statistics (Test A)
- [ ] Đã test vision dependency (Test B)
- [ ] Đã check feature quality (Test C)
- [ ] Đã phân tích per-type performance (Test D)
- [ ] Đã thử fix gating trước (nếu gate < 0.3)
- [ ] Đã thử fix data bias (nếu drop < 10% + features good)
- [ ] Có evidence rõ ràng cho việc đổi encoder

**Chỉ đổi encoder KHI:**
- ✅ Drop < 10% (không dùng vision)
- ✅ Feature quality BAD (std < 0.1 hoặc diversity < 1.0)
- ✅ Đã thử fix gate và data nhưng không improve

---

**💡 Remember:** 

> "Premature optimization is the root of all evil" - Donald Knuth

> "Evidence over intuition" - Engineering principle

Chạy tests → Có data → Make decision 🎯
