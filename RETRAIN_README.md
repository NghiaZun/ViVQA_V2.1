# ViVQA Training - Improved Version
## 🔥 New Features (Ready for Retrain)

### 1️⃣ Counting Loss Penalty ⭐⭐
**Problem:** Model sai hoàn toàn câu hỏi "có bao nhiêu..."
- "Pred: ba | GT: hai" ❌
- "Pred: bốn | GT: hai" ❌

**Solution:** Apply 2x loss weight cho counting questions
```bash
--use_counting_penalty \
--counting_weight 2.0
```

**How it works:**
- Detect questions with: "có bao nhiêu", "bao nhiêu", "số lượng", "mấy"
- Apply 2x weight to loss for these samples
- Forces model to pay more attention to counting

**Expected improvement:** +5-8% accuracy on counting questions

---

### 2️⃣ Unfreeze Vision Encoder Last Layers ⭐⭐⭐ (NEW!)
**Problem:** DINOv2 fully frozen → không học được:
- Counting-specific features (phân biệt 2 vs 3 objects)
- Color fine-grained features (phân biệt xám vs xanh dương)

**Solution:** Unfreeze last 2 layers của vision encoder
```bash
--unfreeze_vision_layers 2
```

**How it works:**
- DINOv2 có 12 layers (base model)
- Freeze layers 0-9 (general features)
- Unfreeze layers 10-11 (task-specific adaptation)

**Trade-offs:**
- ✅ Better counting accuracy (+5-10%)
- ✅ Better color fine-grained (+3-5%)
- ⚠️ More memory (~2GB extra)
- ⚠️ Slower training (~15% slower)
- ⚠️ Risk of overfitting (need monitoring)

**Options:**
- `--unfreeze_vision_layers 0` (default) - Fully frozen, safe
- `--unfreeze_vision_layers 2` (recommended) - Good balance
- `--unfreeze_vision_layers 4` (aggressive) - Max performance, risk overfit

**Expected improvement:** +5-10% on counting, +3-5% on color questions

---

### 3️⃣ Higher Learning Rate ⭐⭐⭐
**Problem:** LR=5e-5 quá thấp → converge chậm
- Train loss: 3.05 → 1.95 → 2.06 (dao động)
- Best val loss tại epoch 10 (muộn)

**Solution:** LR=1e-4
```bash
--lr 1e-4  # Instead of 5e-5
```

**Expected improvement:** 
- Best val loss: 1.90 → **1.85-1.87**
- EM: 57.5% → **60-62%**
- Converge faster: epoch 10 → **epoch 6-8**

---

### 4️⃣ Label Smoothing ✅ (Already Enabled!)
**Feature:** Prevent overfitting với label_smoothing=0.1
- Already in code (model_no_latent.py line 361)
- No changes needed!

---

## 🚀 Recommended Training Commands

### Option A: Conservative (Safe, Kaggle 9h timeout):
**Recommend:** Không unfreeze vision, chỉ dùng counting penalty + LR cao
```bash
python train_no_latent.py \
  --train_csv /kaggle/input/vivqa/data/train.csv \
  --image_dir /kaggle/input/vivqa/data/images/train \
  --val_split 0.1 \
  --batch_size 16 \
  --gradient_accumulation_steps 2 \
  --num_workers 2 \
  --epochs 25 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --dropout 0.1 \
  --unfreeze_encoder_layers 3 \
  --scheduler plateau \
  --scheduler_patience 3 \
  --scheduler_factor 0.5 \
  --early_stopping \
  --early_stopping_patience 6 \
  --use_counting_penalty \
  --sample_every 5 \
  --output_dir /kaggle/working/checkpoints
```

### Option B: Aggressive (Max Performance, cần theo dõi overfit):
**Recommend:** Unfreeze 2 vision layers + counting penalty
```bash
python train_no_latent.py \
  --train_csv /kaggle/input/vivqa/data/train.csv \
  --image_dir /kaggle/input/vivqa/data/images/train \
  --val_split 0.1 \
  --batch_size 16 \
  --gradient_accumulation_steps 2 \
  --num_workers 2 \
  --epochs 25 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --dropout 0.1 \
  --unfreeze_encoder_layers 3 \
  --unfreeze_vision_layers 2 \
  --scheduler plateau \
  --scheduler_patience 3 \
  --scheduler_factor 0.5 \
  --early_stopping \
  --early_stopping_patience 6 \
  --use_counting_penalty \
  --sample_every 5 \
  --output_dir /kaggle/working/checkpoints
```

### For Kaggle (9-hour timeout) - DEPRECATED:
```bash
python train_no_latent.py \
  --train_csv /kaggle/input/vivqa/data/train.csv \
  --image_dir /kaggle/input/vivqa/data/images/train \
  --val_split 0.1 \
  --batch_size 16 \
  --gradient_accumulation_steps 2 \
  --num_workers 2 \
  --epochs 25 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --dropout 0.1 \
  --unfreeze_encoder_layers 3 \
  --scheduler plateau \
  --scheduler_patience 3 \
  --scheduler_factor 0.5 \
  --early_stopping \
  --early_stopping_patience 6 \
  --use_counting_penalty \
  --sample_every 5 \
  --output_dir /kaggle/working/checkpoints
```

### For Local (with more resources):
```bash
python train_no_latent.py \
  --train_csv ./train.csv \
  --image_dir ./images \
  --val_split 0.1 \
  --batch_size 32 \
  --gradient_accumulation_steps 1 \
  --num_workers 4 \
  --epochs 30 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --dropout 0.1 \
  --unfreeze_encoder_layers 3 \
  --scheduler plateau \
  --scheduler_patience 3 \
  --scheduler_factor 0.5 \
  --early_stopping \
  --early_stopping_patience 6 \
  --use_counting_penalty \
  --counting_weight 2.0 \
  --sample_every 3 \
  --output_dir ./checkpoints_improved
```

---

## 📊 Expected Results

### Previous Run (LR=5e-5, no counting penalty):
```
Best Epoch: 10
Val Loss: 1.8997
EM: 57.5%
F1: 69.1%
ROUGE-L: 71.3%
```

### Expected with Improvements:
```
Best Epoch: 6-8 (faster!)
Val Loss: 1.85-1.87 (↓ 2-3%)
EM: 60-62% (↑ 4-7%)
F1: 71-73% (↑ 2-4%)
ROUGE-L: 73-75% (↑ 2-4%)

Counting Questions EM: +5-8% improvement
Color Questions: Similar (need data balancing for more gains)
```

---

## 🔧 What Changed in Code

### model_no_latent.py:
1. ✅ Added `is_counting_question()` helper function
2. ✅ Added `is_color_question()` helper function  
3. ✅ Modified `forward()` to support `use_counting_penalty` parameter
4. ✅ Implement weighted loss computation for counting questions

### train_no_latent.py:
1. ✅ Added `--use_counting_penalty` argument
2. ✅ Added `--counting_weight` argument (default: 2.0)
3. ✅ Pass counting penalty to `run_one_epoch_deterministic()`
4. ✅ Display counting penalty in training config

---

## 🎯 Next Phase (After This Retrain)

### Phase 2: Data Improvements
- Color-balanced sampling (reduce "màu xanh dương" bias)
- Rare color augmentation
- Counting question augmentation

### Phase 3: Advanced
- Beam search for inference (+2-3% EM)
- Relaxed metrics (synonym-aware EM)
- Unfreeze vision encoder last 2 layers

---

## 📝 Training Checklist

Before starting training:
- [ ] Check GPU memory (need ~16GB for batch_size=16)
- [ ] Verify dataset paths
- [ ] Ensure checkpoints directory exists
- [ ] Monitor disk space on Kaggle (~6GB needed)

During training:
- [ ] Watch for Epoch 3 spike (if happens again, note it)
- [ ] Check if counting questions improve
- [ ] Monitor LR schedule (should reduce at epoch ~6-8)
- [ ] Watch early stopping (should trigger at epoch ~12-14)

After training:
- [ ] Download best_model.pt
- [ ] Check training_curves.png
- [ ] Analyze training_metrics.csv
- [ ] Compare EM/F1 on counting questions specifically

---

## 🐛 Troubleshooting

### If loss spikes at epoch 3 again:
```bash
# Reduce gradient clipping
--max_norm 0.5  # Instead of 1.0
```

### If overfitting too early:
```bash
# Increase regularization
--weight_decay 0.02  # Instead of 0.01
--dropout 0.15      # Instead of 0.1
```

### If underfitting:
```bash
# Increase capacity
--unfreeze_encoder_layers 4  # Instead of 3
```

---

**Status:** ✅ Ready to retrain!
**Estimated time:** ~3 hours on Kaggle GPU
**Expected improvement:** +4-7% EM, better counting accuracy
