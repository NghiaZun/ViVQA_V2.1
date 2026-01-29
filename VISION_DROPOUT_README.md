# 🔥 VISION DROPOUT + GATE REGULARIZATION

## ✅ STATUS: FIXED (v2)

**Date:** January 29, 2026  
**Version:** 2.0 - Gate Regularization Bug Fixed

### What Changed

**v1 (BROKEN):**
```python
# ❌ No gradient flow!
gate_mean = outputs.fusion_weights.mean().item()  # Detaches gradient
batch_penalty += -torch.log(torch.tensor(gate_mean, ...))  # New detached tensor
```

**v2 (FIXED):**
```python
# ✅ Gradient flows correctly!
fusion_weights = outputs.fusion_weights  # Keep gradient
for i, qtype in enumerate(question_types):
    sample_gate = fusion_weights[i]  # Per-sample, keeps gradient
    batch_penalty += -torch.log(sample_gate + 1e-8) * weight
```

**Key fixes:**
1. ✅ No `.item()` call - keeps gradient
2. ✅ Per-sample instead of batch average
3. ✅ No `torch.tensor()` reconstruction
4. ✅ Correct type-to-sample matching

---

## Tổng quan

Đã implement **COMBINED approach** để tăng vision dependency:

1. **Vision Dropout Augmentation (Hard Constraint)**
   - Randomly zero out vision features 30% of time during training
   - Type-specific dropout rates:
     - COUNT: 40% (yếu nhất - 42.6%)
     - COLOR: 35% (yếu - 49.9%)
     - OBJECT: 20% (khá tốt - 61.5%)
     - LOCATION: 20% (tốt nhất - 65%)

2. **Gate Regularization (Soft Constraint)**
   - Add penalty `-log(gate_mean)` to loss
   - Type-specific penalties:
     - COUNT: 0.15 (mạnh nhất)
     - COLOR: 0.10
     - OBJECT: 0.05
     - LOCATION: 0.05

---

## Kết quả kỳ vọng

| Metric | Before (fusion=4) | After Combined | Improvement |
|--------|-------------------|----------------|-------------|
| **Vision Drop** | 14.9% | **22-25%** | +7-10% ✅ |
| **COUNT Acc** | 42.6% | **52-56%** | +10-13% ✅ |
| **COLOR Acc** | 49.9% | **56-60%** | +6-10% ✅ |
| **OBJECT Acc** | 61.5% | **64-66%** | +2-4% ✅ |
| **LOCATION Acc** | 65.0% | **67-70%** | +2-5% ✅ |
| **Overall** | 57.4% | **62-65%** | +4-7% ✅ |

---

## Cách chạy

### Training với Vision Dropout (Default: ON)

```bash
python train_no_latent.py \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --image_dir data/images \
    --output_dir checkpoints/with_vision_dropout \
    --epochs 50 \
    --batch_size 16 \
    --use_type_loss  # Enable type-conditioned loss
```

### Disable Vision Dropout (nếu cần)

Modify trong `train_no_latent.py` line ~1050:

```python
train_metrics = run_one_epoch_deterministic(
    ...
    use_vision_dropout=False,  # 🔥 TẮT vision dropout
    vision_dropout_prob=0.3    # Hoặc adjust tỷ lệ
)
```

---

## Monitoring

Trong progress bar sẽ thấy:

```
Train Stage 3: 100%|██| 675/675 [06:56<00:00, 1.62it/s, 
    loss=1.85, 
    ans=1.70, 
    type=0.15, 
    g_pen=0.05,  ← Gate penalty
    α_mean=0.88, ← Gate mean
    α_std=0.04]
```

**Indicators:**
- `g_pen` (gate penalty): Nên giảm dần qua epochs (model học sử dụng vision)
- `α_mean`: Nên tăng hoặc giữ cao (>0.7)
- Nếu α_mean giảm → tăng gate penalty weight

---

## Diagnostic sau training

Sau khi train xong, chạy lại diagnostic:

```bash
python diagnostic_tools.py \
    --checkpoint checkpoints/with_vision_dropout/best_model.pt \
    --csv data/test.csv \
    --image_dir data/images
```

**Expected improvements:**
- Vision Drop: 14.9% → **22-25%** ✅
- COUNT accuracy: 42.6% → **52-56%** ✅
- Gate mean: Stable hoặc cao hơn

---

## Troubleshooting

### Issue 1: Loss tăng đột ngột

**Nguyên nhân:** Vision dropout quá aggressive

**Giải pháp:**
```python
# Giảm dropout rate
vision_dropout_prob=0.2  # Thay vì 0.3
```

### Issue 2: Gate mean giảm quá nhanh

**Nguyên nhân:** Gate penalty quá mạnh

**Giải pháp:**
```python
# Trong run_one_epoch_deterministic, giảm penalty weights:
type_penalties = {
    0: 0.03,  # Giảm từ 0.05
    1: 0.10,  # Giảm từ 0.15
    2: 0.07,  # Giảm từ 0.10
    3: 0.03,  # Giảm từ 0.05
}
```

### Issue 3: COUNT accuracy vẫn không cải thiện

**Nguyên nhân:** Cần specialized counting module

**Giải pháp:** Implement CountingModule (see TODO below)

---

## TODO (nếu cần tiếp)

Nếu combined approach chưa đủ để đạt >55% COUNT accuracy:

1. **CountingModule** - Explicit density estimation
2. **ColorDiscriminator** - Color-specific attention
3. **Vision Shortcuts** - Residual connections giữ raw vision

---

## Technical Details

### Vision Dropout Implementation

```python
class VisionDropoutAugmentation:
    def __call__(self, vision_features, question_types, training=True):
        # Type-specific dropout
        mask = torch.ones(batch_size, 1, 1)
        for i, qtype in enumerate(question_types):
            prob = self.type_probs[qtype]
            if random() < prob:
                mask[i] = 0.0  # Zero out vision
        
        return vision_features * mask
```

### Gate Regularization

```python
gate_penalty = 0.0
for qtype in question_types:
    weight = type_penalties[qtype]
    gate_penalty += -log(gate_mean + 1e-8) * weight

total_loss = answer_loss + gate_penalty
```

### Why it works

1. **Vision Dropout:**
   - 30% of batches → vision = zero
   - Model MUST learn: "No vision → FAIL"
   - Forces vision dependency

2. **Gate Regularization:**
   - Penalty for low gate values
   - Encourages model to USE vision
   - Type-specific → stronger for weak types

3. **Combined:**
   - Hard + Soft constraints
   - Model learns vision is ESSENTIAL (dropout)
   - AND uses it INTELLIGENTLY (gate reg)

---

**Author:** GitHub Copilot  
**Date:** January 29, 2026  
**Based on:** Diagnostic results showing vision drop=14.9%, COUNT=42.6%
