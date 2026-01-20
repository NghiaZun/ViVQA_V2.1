# QUICK REFERENCE CARD 🎴
**Purpose**: Fast lookup during training - no need to read full docs!

---

## 🔢 EXPECTED VALUES (Healthy Ranges)

### Stage 2 (Epochs 0-15)
```
KL Raw:              0.03 → 0.08  (growing smoothly)
KL After:            0.02 → 0.06  (with free_bits=0.02)
Penalty Reduction:   20% → 40%    (sweet spot)
Answer Loss:         0.50 → 0.35  (decreasing)
Val/Train Ratio:     1.5x → 2.5x  (some overfitting OK)
```

### Stage 3 (Epochs 15+)
```
KL Raw:              0.06 → 0.10  (stable)
Teacher Loss:        0.05 → 0.20  (active)
Teacher Contrib:     3% → 10%     (visible but not dominant)
Answer Loss:         0.35 → 0.25  (continuing to decrease)
Val Exact Match:     0.20 → 0.35  (improving)
```

---

## 🚨 WARNING SIGNS & QUICK FIXES

### KL Collapse (kl_raw < 0.02)
```
🚨 CRITICAL: Model ignoring latent!
Fix: Reduce free_bits from 0.02 → 0.01
When: Immediately (restart recommended)
```

### KL Explosion (kl_raw > 0.15)
```
⚠️ WARNING: Over-regularization!
Fix: Increase free_bits to 0.03 OR reduce max_kl_weight to 10.0
When: After 3 consecutive high epochs
```

### Free Bits Too Generous (penalty_red > 70%)
```
🟡 MINOR: Most KL becomes free
Fix: Reduce free_bits from 0.02 → 0.015
When: After epoch 10 if persists
```

### Free Bits Too Tight (penalty_red < 15%)
```
🟡 MINOR: Too much penalty
Fix: Increase free_bits from 0.02 → 0.025
When: After epoch 10 if persists
```

### Teacher Ineffective (no improvement Stage 2→3)
```
⚠️ WARNING: Teacher not helping!
Fix: Increase teacher_weight from 0.3 → 0.5
When: Epoch 20 if exact_match not improving
```

### Overfitting Explosion (val/train > 3.5x)
```
⚠️ WARNING: Severe overfitting!
Fix: Increase dropout, reduce learning rate, or early stop
When: After epoch 20
```

---

## 📊 PROGRESS BAR DECODER

### What You See
```
Train: 100%|████| L:0.412, A:0.305, KLr:0.048, KLa:0.032, fb%:33%, T:0.015, KLw:3.00
```

### What It Means
```
L:     Total loss (all components combined)
A:     Answer loss (primary metric for quality)
KLr:   KL raw (before free bits) - should be 0.03-0.08
KLa:   KL after (after free bits) - should be 0.02-0.06
fb%:   Free bits reduction % - target 20-40%
T:     Teacher loss (Stage 3 only) - should be 0.05-0.20
KLw:   Effective KL weight (max × 0.2) - grows 0→3.0 in Stage 2
```

---

## ⏱️ CHECKPOINT SCHEDULE

### Mandatory Checks

**Epoch 1**: Smoke test
- Training starts without errors?
- Progress bar shows all metrics?
- No NaN/Inf?

**Epoch 5**: Free bits validation
- Is kl_raw > 0.02? (Yes = good)
- Is penalty_red 15-70%? (Yes = continue)
- Adjust free_bits if needed

**Epoch 10**: Mid-Stage 2 review
- Is KL growing smoothly?
- Is val loss decreasing?
- Overfitting under control?

**Epoch 15**: Stage 2→3 transition
- KL warmed up? (kl_raw = 0.05-0.10)
- Val answer < 0.50?
- Ready for teacher?

**Epoch 20**: Teacher effectiveness
- Exact match improved from epoch 15?
- Teacher contribution 3-10%?
- Adjust teacher_weight if needed

**Epoch 30**: Mid-training check
- On track for val < 0.80?
- Early stopping working?
- Final tuning needed?

---

## 🎯 DECISION TREES

### At Epoch 5: Free Bits Tuning
```
avg_kl_raw from epochs 3-5
├─ < 0.02?
│  └─ ❌ Reduce free_bits to 0.01 (RESTART)
├─ > 0.12?
│  └─ ❌ Increase free_bits to 0.03 or reduce KL weight (RESTART)
├─ penalty_red > 75%?
│  └─ 🟡 Reduce free_bits to 0.015 (CONTINUE)
├─ penalty_red < 15%?
│  └─ 🟡 Increase free_bits to 0.025 (CONTINUE)
└─ 0.03-0.08 and penalty 20-70%?
   └─ ✅ Perfect! Continue
```

### At Epoch 20: Teacher Tuning
```
exact_match improvement from epoch 15
├─ None or negative?
│  └─ ❌ Increase teacher_weight to 0.5 (RESTART from epoch 15)
├─ < 0.02?
│  └─ 🟡 Increase teacher_weight to 0.4 (CONTINUE)
├─ >= 0.03?
│  └─ ✅ Teacher working! Continue
└─ Teacher contribution < 2%?
   └─ ⚠️ Increase weight regardless of improvement
```

---

## 🔧 TUNING RANGES

### Free Bits
```
Conservative:  0.01  (tight, high penalty)
Default:       0.02  (balanced - START HERE)
Generous:      0.03  (loose, low penalty)

Range: 0.01 - 0.03 (tune in steps of 0.005)
```

### Teacher Weight
```
Weak:       0.2  (subtle guidance)
Default:    0.3  (balanced - START HERE)
Strong:     0.5  (clear signal)
Dominant:   0.8  (risk: over-reliance)

Range: 0.2 - 0.5 (tune in steps of 0.1)
```

### Max KL Weight
```
Conservative: 10.0  (gentle regularization)
Default:      15.0  (balanced - START HERE)
Aggressive:   20.0  (strong bottleneck)

Range: 10.0 - 20.0 (tune in steps of 2.5)
```

---

## 📈 SUCCESS INDICATORS

### Stage 2 Success (By Epoch 15)
- ✅ kl_raw reached 0.05-0.10
- ✅ penalty_reduction stabilized 20-40%
- ✅ val_answer dropped below 0.50
- ✅ No collapse (kl_raw not stuck at 0)
- ✅ Smooth KL warmup (no spikes)

### Stage 3 Success (By Epoch 25)
- ✅ exact_match improved by 0.02+ from Stage 2
- ✅ teacher_contribution 3-10%
- ✅ val_answer continuing to decrease
- ✅ overfitting under control (<2.5x)

### Final Success (Training Complete)
- ✅ val_answer < 0.80 (PRIMARY TARGET!)
- ✅ val_exact > 0.30
- ✅ Early stopping triggered appropriately
- ✅ Best model saved

---

## 🆘 EMERGENCY PROCEDURES

### Training Diverged (Loss Exploding)
```
1. STOP training immediately
2. Check last stable epoch
3. Restart from checkpoint
4. Reduce learning rate by 2x
5. Increase free_bits by 0.01
```

### Training Stuck (Val Plateau >10 Epochs)
```
1. Check overfitting ratio
2. If >3x: Model overfit, early stop
3. If <2x: Increase learning rate or extend training
4. Consider teacher weight increase (Stage 3)
```

### KL Collapsed (kl_raw = 0 for 5+ Epochs)
```
1. Model ignoring latent completely
2. Reduce free_bits to 0.005 (very tight)
3. Increase max_kl_weight to 20.0
4. Restart training from scratch
```

---

## 💾 SAVE THESE COMMANDS

### Quick Check Config
```bash
grep -n "free_bits" model.py train_utils.py
grep -n "teacher_weight" train_utils.py
grep -n "max_kl_weight" train.py
```

### Monitor Training in Real-Time
```bash
tail -f training_log.csv | grep "kl_raw"
```

### Plot Mid-Training
```bash
python plot_curves.py training_log.csv
```

### Emergency Stop
```bash
# Ctrl+C in training terminal
# Or: kill $(ps aux | grep train.py | awk '{print $2}')
```

---

## 🎓 RULE OF THUMB

**80% of issues come from 2 parameters:**
1. **free_bits** (controls KL penalty strength)
2. **teacher_weight** (controls teacher influence)

**Monitor these closely!**

**Good defaults:**
- free_bits = 0.02 (adjust ±0.005 based on penalty_reduction)
- teacher_weight = 0.3 (double to 0.5 if no improvement)

**When in doubt:**
- Be conservative (smaller adjustments)
- Monitor for 5 epochs before deciding
- Document every change!

---

**Print this card and keep it visible during training! 📌**
