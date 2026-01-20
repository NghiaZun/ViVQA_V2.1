# TRAINING MONITORING CHECKLIST 📋
**Date**: January 20, 2026  
**Purpose**: Real-time checklist for monitoring and tuning during training

---

## ✅ PRE-TRAINING VERIFICATION

- [ ] `free_bits = 0.02` in model.py, train_utils.py
- [ ] `stage1_epochs = 0` (skip Stage 1)
- [ ] `stage2_epochs = 15` (KL warmup)
- [ ] `max_kl_weight = 15.0` (aggressive but tunable)
- [ ] `teacher_weight = 0.3` (conservative start)
- [ ] `teacher_type = rule_based` (fast baseline)
- [ ] Dataset loaded correctly (train/val split)
- [ ] GPU available and CUDA working

**Command**:
```bash
python train.py \
    --csv_path data/ViVQA_train.csv \
    --image_folder data/images \
    --stage1_epochs 0 \
    --stage2_epochs 15 \
    --max_kl_weight 15.0 \
    --teacher_type rule_based \
    --early_stopping_patience 10
```

---

## 📊 PHASE 1: INITIAL VALIDATION (Epochs 0-5)

### Epoch 1 Checks
- [ ] Training starts without errors
- [ ] Progress bar shows: `L`, `A`, `KLr`, `KLa`, `fb%`, `T`, `KLw`
- [ ] KL raw appears (>0.005) - not stuck at 0
- [ ] No NaN or Inf in losses

**Expected first epoch**:
```
Train: 100%|████| L:0.520, A:0.420, KLr:0.008, KLa:0.000, fb%:100%, T:0.000, KLw:0.00
Val:   100%|████| L:0.680, A:0.580, KLr:0.010, KLa:0.000, fb%:100%, T:0.000, KLw:0.00

Epoch 1 Summary:
  Train - Total: 0.520, Answer: 0.420, KL: 0.000
  Val   - Total: 0.680, Answer: 0.580, KL: 0.000
  🔍 KL Diagnostics: raw=0.008, after_free_bits=0.000, penalty_reduction=100%
     ⚠️  FREE BITS TOO HIGH! All KL becomes free (kl_raw=0.008). Reduce from 0.02 to 0.01!
```

**Action**: ⚠️ **EXPECTED** - KL raw very low early. Wait for epoch 3-5.

---

### Epoch 3 Checks
- [ ] KL raw growing (>0.015)
- [ ] Vision encoder unfrozen (announced in logs)
- [ ] Val loss decreasing (not stuck)

**Expected epoch 3**:
```
🔓 UNFREEZING VISION ENCODER (Stage 2, Epoch 3)

Train: L:0.450, A:0.360, KLr:0.025, KLa:0.005, fb%:80%, T:0.000, KLw:0.60

Epoch 3 Summary:
  🔍 KL Diagnostics: raw=0.025, after_free_bits=0.005, penalty_reduction=80%
     🟡 Free bits generous (>80% reduction). Consider lowering.
```

**Action**: 🟡 If penalty_red >80% for 3 consecutive epochs → Reduce free_bits to 0.015

---

### Epoch 5 Decision Point ⚠️

**CRITICAL CHECKPOINT** - Decide if free_bits needs adjustment!

**Collect stats**:
```python
# From epochs 3-5:
avg_kl_raw = (epoch3_kl + epoch4_kl + epoch5_kl) / 3
avg_penalty_red = (epoch3_pr + epoch4_pr + epoch5_pr) / 3
```

**Decision tree**:

```python
if avg_kl_raw < 0.02:
    # Case 1: KL too low
    print("❌ KL raw < 0.02 - FREE BITS TOO HIGH!")
    print("   Action: Reduce free_bits from 0.02 to 0.01")
    print("   Restart training or adjust config")
    # Expected after fix: kl_after will be higher

elif avg_kl_raw > 0.12:
    # Case 2: KL too high
    print("❌ KL raw > 0.12 - OVER-REGULARIZATION!")
    print("   Option A: Increase free_bits to 0.03 (more forgiving)")
    print("   Option B: Reduce max_kl_weight to 10.0 (less aggressive)")
    print("   Restart training")

elif avg_penalty_red > 75%:
    # Case 3: Free bits too generous
    print("🟡 Penalty reduction > 75% - FREE BITS GENEROUS")
    print("   Action: Reduce free_bits from 0.02 to 0.015")
    print("   Continue training (minor adjustment)")

elif avg_penalty_red < 15%:
    # Case 4: Free bits too tight
    print("🟡 Penalty reduction < 15% - FREE BITS TIGHT")
    print("   Action: Increase free_bits from 0.02 to 0.025")
    print("   Continue training")

elif 0.03 <= avg_kl_raw <= 0.08 and 20 <= avg_penalty_red <= 50:
    # Case 5: Goldilocks zone!
    print("✅ KL HEALTHY! Continue without changes")
    print(f"   kl_raw: {avg_kl_raw:.3f} (target: 0.03-0.08)")
    print(f"   penalty_red: {avg_penalty_red:.1f}% (target: 20-50%)")

else:
    # Case 6: Borderline - continue monitoring
    print("🟡 KL dynamics borderline - continue monitoring")
```

**Record decision**:
- [ ] free_bits optimal → Continue
- [ ] free_bits adjusted to: _____ → Restart/Continue
- [ ] KL weight adjusted to: _____ → Restart

---

## 📈 PHASE 2: STAGE 2 MONITORING (Epochs 5-15)

### Continuous Monitoring (Every Epoch)

**Check in progress bar**:
- [ ] KL raw growing smoothly (0.03 → 0.08 by epoch 15)
- [ ] Penalty reduction stable (not wildly varying ±20%)
- [ ] Val loss decreasing (answer loss primary metric)
- [ ] No overfitting explosion (ratio <3x)

**Expected progression**:
```
Epoch 5:  KLr:0.032, fb%:38% → baseline
Epoch 8:  KLr:0.048, fb%:35% → growing
Epoch 10: KLr:0.058, fb%:32% → stable growth
Epoch 12: KLr:0.065, fb%:30% → approaching target
Epoch 15: KLr:0.072, fb%:28% → ready for Stage 3
```

**Health check warnings**:
- [ ] If KL raw drops suddenly (>30% drop) → Check for collapse
- [ ] If penalty_red spikes >80% → Free bits may need adjustment
- [ ] If val loss increases >20% → Possible overfitting or instability

---

### Epoch 10 Mid-Stage Review

**Analyze trends (epochs 5-10)**:

1. **KL growth rate**:
   ```python
   kl_growth = (epoch10_kl - epoch5_kl) / 5
   # Target: ~0.005 per epoch (0.03 → 0.055 in 5 epochs)
   ```
   - [ ] Growth too slow (<0.003/epoch) → May need longer Stage 2
   - [ ] Growth too fast (>0.010/epoch) → May overfit, reduce KL weight

2. **Val loss trajectory**:
   ```python
   val_improvement = (epoch5_val - epoch10_val) / epoch5_val
   # Target: >10% improvement in 5 epochs
   ```
   - [ ] No improvement → Model may be stuck, check learning rate
   - [ ] Improvement <5% → Slow progress, may need more epochs

3. **Overfitting check**:
   ```python
   overfitting_ratio = val_loss / train_loss
   # Target: <2.5x at this stage
   ```
   - [ ] Ratio >3x → Early overfitting, may need regularization
   - [ ] Ratio <1.5x → Good generalization

**Record mid-stage status**:
- [ ] KL dynamics: ✅ Healthy / 🟡 Borderline / ❌ Problematic
- [ ] Val loss trend: ✅ Decreasing / 🟡 Plateau / ❌ Increasing
- [ ] Decision: Continue / Adjust / Restart

---

### Epoch 15 Stage 2 Completion

**Final Stage 2 assessment**:

1. **KL final state**:
   - [ ] kl_raw in range 0.05-0.10 (healthy)
   - [ ] penalty_reduction 20-40% (optimal)
   - [ ] KL warmup complete (KL weight = 15.0)

2. **Model readiness for Stage 3**:
   - [ ] Val answer loss <0.50 (reasonable baseline)
   - [ ] Val exact match >0.20 (beating random)
   - [ ] No collapse (kl_raw not stuck at 0)
   - [ ] No explosion (losses not diverging)

**Calibrate empirical thresholds**:
```python
# Collect statistics from epochs 5-15
kl_raw_min = _____ (observed)
kl_raw_max = _____ (observed)
kl_raw_mean = _____ (observed)
penalty_red_mean = _____ (observed)

# Compare with theory
Theory: kl_raw = 0.03-0.08, penalty_red = 20-40%
Actual: kl_raw = ___-___, penalty_red = ___%

# Update expectations for Stage 3
```

**Record Stage 2 completion**:
- [ ] Best val_answer_loss: _____ (epoch _____)
- [ ] Empirical KL range: _____ to _____
- [ ] Free_bits final value: _____ (if adjusted)
- [ ] Ready for Stage 3: ✅ Yes / ❌ Need more epochs

---

## 🎓 PHASE 3: TEACHER VALIDATION (Epochs 15-20)

### Epoch 16 Checks (First Stage 3 Epoch)

**Verify teacher activation**:
- [ ] Progress bar shows teacher loss: `T:0.XXX` (not 0.000)
- [ ] Teacher diagnostics printed in summary
- [ ] Batch expansion working (2x faster than loop version)

**Expected output**:
```
🟢 STAGE 3: FULL (Complete + Teacher)

Train: L:0.380, A:0.310, KLr:0.072, KLa:0.052, fb%:28%, T:0.018, KLw:3.00

Epoch 16 Summary:
  Train - Total: 0.380, Answer: 0.310, KL: 0.052, Teacher: 0.018
  🎓 Teacher Diagnostics: loss=0.018, weight=0.3, contribution=1.4%
     🟡 Teacher loss weak (<0.05). Contribution only 1.4%.
        💡 Suggestion: Increase teacher_weight from 0.3 to 0.5
```

**Action**: 🟡 Expected - teacher loss may be weak initially. Monitor for 5 epochs.

---

### Epoch 20 Teacher Effectiveness Review ⚠️

**CRITICAL CHECKPOINT** - Decide if teacher_weight needs adjustment!

**Compare before/after Stage 3**:
```python
# Best from Stage 2
stage2_best_answer = _____ (epoch 15)
stage2_best_exact = _____ (epoch 15)

# Current Stage 3
stage3_current_answer = _____ (epoch 20)
stage3_current_exact = _____ (epoch 20)

# Calculate improvement
answer_improvement = stage2_best_answer - stage3_current_answer
exact_improvement = stage3_current_exact - stage2_best_exact
```

**Decision tree**:

```python
if exact_improvement <= 0:
    # Case 1: No improvement - teacher not helping!
    print("❌ TEACHER NOT EFFECTIVE!")
    print(f"   Stage 2 best: {stage2_best_exact:.3f}")
    print(f"   Stage 3 current: {stage3_current_exact:.3f}")
    print("   Action: Increase teacher_weight from 0.3 to 0.5")
    # Restart from epoch 15 checkpoint with new weight
    
elif exact_improvement < 0.02:
    # Case 2: Marginal improvement - teacher weak
    print("🟡 TEACHER MARGINALLY EFFECTIVE")
    print(f"   Improvement: +{exact_improvement:.3f} (target: >0.02)")
    print("   Action: Increase teacher_weight from 0.3 to 0.4")
    print("   Continue training")

elif exact_improvement >= 0.03:
    # Case 3: Good improvement - teacher working!
    print("✅ TEACHER EFFECTIVE!")
    print(f"   Improvement: +{exact_improvement:.3f}")
    print("   Continue with current teacher_weight=0.3")

else:
    # Case 4: Moderate improvement - borderline
    print("🟡 Teacher showing some effect")
    print(f"   Improvement: +{exact_improvement:.3f}")
    print("   Continue monitoring")
```

**Analyze teacher contribution**:
```python
avg_teacher_loss = _____ (epochs 16-20)
avg_teacher_contrib = _____ % (epochs 16-20)

# Target: 3-10% contribution
if avg_teacher_contrib < 2%:
    print("⚠️ Teacher contribution too low!")
elif avg_teacher_contrib > 15%:
    print("⚠️ Teacher dominating - may need to reduce weight!")
else:
    print("✅ Teacher contribution healthy")
```

**Record teacher assessment**:
- [ ] Teacher effective: ✅ Yes / 🟡 Marginal / ❌ No
- [ ] Teacher weight adjusted to: _____ (if needed)
- [ ] Continue Stage 3: ✅ Yes / ❌ Restart with new weight

---

## 🎯 PHASE 4: FINAL OPTIMIZATION (Epochs 20+)

### Continuous Monitoring (Every 5 Epochs)

**Target metrics**:
- [ ] Val answer loss < 0.80 (primary target)
- [ ] Val exact match > 0.30 (secondary target)
- [ ] Overfitting ratio < 2.5x (generalization check)
- [ ] Early stopping triggered appropriately

**Check every 5 epochs**:

**Epoch 25**:
- [ ] Val answer: _____ (target: <0.80)
- [ ] Val exact: _____ (target: >0.30)
- [ ] Overfitting: _____ x (target: <2.5x)
- [ ] Patience counter: _____ / 10

**Epoch 30**:
- [ ] Val answer: _____ 
- [ ] Val exact: _____
- [ ] Overfitting: _____ x
- [ ] Patience counter: _____ / 10

**Epoch 35**:
- [ ] Val answer: _____ 
- [ ] Val exact: _____
- [ ] Best epoch so far: _____
- [ ] Decision: Continue / Trigger early stop

---

### Success Criteria

**Training successful if**:
- [ ] Val answer loss < 0.80 ✅
- [ ] Val exact match > 0.30 ✅
- [ ] Model converged (early stopping triggered) ✅
- [ ] No major instabilities (no divergence) ✅

**Expected best model**:
```
Best Model (Epoch XX):
  Val answer loss: 0.7X (target: <0.80) ✅
  Val exact match: 0.3X (target: >0.30) ✅
  Train/Val ratio: 2.XX (target: <2.5x) ✅
  KL dynamics: Healthy (0.05-0.08) ✅
  Teacher contribution: Effective (3-8%) ✅
```

---

### Failure Analysis (If val_loss > 0.80 at convergence)

**Possible causes**:

1. **Free bits still wrong**:
   - [ ] Check final penalty_reduction: _____
   - [ ] If >70%: KL too free, reduce free_bits
   - [ ] If <10%: KL too tight, increase free_bits

2. **Teacher ineffective**:
   - [ ] Check teacher contribution: _____%
   - [ ] If <2%: Teacher too weak, increase weight
   - [ ] If <5%: Teacher marginal, consider VLM teacher

3. **KL weight imbalance**:
   - [ ] Check KL vs answer loss ratio
   - [ ] If KL dominates (>60%): Reduce max_kl_weight to 10.0
   - [ ] If KL too weak (<20%): Increase max_kl_weight to 20.0

4. **Architecture limitations**:
   - [ ] Check exact match: If <0.25, model may need more capacity
   - [ ] Consider: More reasoning tokens (6→8), larger latent (256→384)

**Fallback action plan**:
```python
# Priority order:
1. Adjust free_bits based on penalty_reduction
2. Increase teacher_weight if contribution <3%
3. Tune KL weight if ratio imbalanced
4. Architecture changes (last resort)
```

---

## 📝 TRAINING LOG SUMMARY

### Configuration Used
```
free_bits: _____ (initial: 0.02)
max_kl_weight: _____ (initial: 15.0)
teacher_weight: _____ (initial: 0.3)
stage1_epochs: 0
stage2_epochs: 15
```

### Adjustments Made
1. Epoch _____: free_bits adjusted to _____ (reason: _______)
2. Epoch _____: teacher_weight adjusted to _____ (reason: _______)
3. Epoch _____: Other: _____________________________________

### Final Results
```
Best Epoch: _____
Val Answer Loss: _____
Val Exact Match: _____
Train/Val Ratio: _____
KL Raw (final): _____
Penalty Reduction (final): _____%
Teacher Contribution (final): _____%
```

### Lessons Learned
1. Free bits optimal value: _____ (theory: 0.02, actual: _____)
2. Teacher weight optimal: _____ (theory: 0.3, actual: _____)
3. KL dynamics: _____ (describe observed behavior)
4. Other insights: _____________________________________

---

## 🚀 NEXT STEPS

If training successful (val < 0.80):
- [ ] Save best model checkpoint
- [ ] Run full evaluation on test set
- [ ] Document final hyperparameters
- [ ] Consider ablation studies

If training needs tuning:
- [ ] Apply adjustments from failure analysis
- [ ] Restart from appropriate checkpoint
- [ ] Monitor with updated expectations
- [ ] Document changes for reproducibility

---

**Good luck with training! 🍀**
