# LỆNH CHẠY DETERMINISTIC VQA (No Latent) - VERSION 2.0

## 🎉 **NEW IN VERSION 2.0**

- ✅ **FIXED beam search** (was broken, now uses HuggingFace's proper implementation)
- ✅ **LR scheduler** (ReduceLROnPlateau hoặc Cosine)
- ✅ **Early stopping** (tự động dừng khi overfitting)
- ✅ **F1 score** (metrics tốt hơn exact match)
- ✅ **Label smoothing** (prevent overfitting)
- ✅ **Dataset analysis** (detect imbalance)

---

## 🚀 BASIC COMMANDS

### 1. Chạy training MẶC ĐỊNH (with improvements!)
```bash
python train_no_latent.py --early_stopping --scheduler plateau
```

### 2. Chạy training FULL FEATURES
```bash
python train_no_latent.py \
  --batch_size 12 \
  --epochs 30 \
  --lr 5e-5 \
  --scheduler plateau \
  --early_stopping \
  --analyze_dataset \
  --sample_every 3
```

### 3. Chạy training nền (background)
```bash
nohup python train_no_latent.py > logs/train.log 2>&1 &
```

### 4. Chạy training với nhiều options
```bash
python train_no_latent.py \
  --data_dir ./data \
  --batch_size 12 \
  --epochs 30 \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --num_fusion_layers 2 \
  --output_dir ./checkpoints_no_latent \
  --sample_every 3
```

---

## 📋 TẤT CẢ ARGUMENTS

### **Data Arguments**
```bash
--data_dir PATH           # Thư mục data (default: ./data)
--batch_size INT          # Batch size (default: 12)
--num_workers INT         # Số workers cho dataloader (default: 4)
```

### **Model Arguments**
```bash
--dinov2_model NAME       # DINOv2 model (default: facebook/dinov2-base)
--bartpho_model NAME      # BARTpho model (default: vinai/bartpho-syllable)
--num_fusion_layers INT   # Số Flamingo layers (default: 2)
--num_heads INT           # Số attention heads (default: 8)
--dropout FLOAT           # Dropout rate (default: 0.1)
```

### **Training Arguments**
```bash
--epochs INT              # Số epochs (default: 30)
--lr FLOAT                # Learning rate (default: 5e-5)
--weight_decay FLOAT      # Weight decay (default: 0.01)
--max_norm FLOAT          # Gradient clipping (default: 1.0)
--no_amp                  # Tắt mixed precision
```

### **🔥 NEW: LR Scheduler & Early Stopping**
```bash
--scheduler {none,plateau,cosine}  # LR scheduler (default: plateau)
--scheduler_patience INT           # Patience cho plateau (default: 3)
--scheduler_factor FLOAT           # Factor cho plateau (default: 0.5)
--early_stopping                   # Enable early stopping
--early_stopping_patience INT      # Early stop patience (default: 5)
```

### **Freezing Arguments**
```bash
--unfreeze_encoder_layers INT  # Số text encoder layers unfreeze (default: 3)
--freeze_decoder              # Freeze decoder (default: unfrozen)
```

### **Checkpoint Arguments**
```bash
--output_dir PATH         # Thư mục lưu checkpoints (default: ./checkpoints_no_latent)
--resume PATH             # Resume từ checkpoint
--save_every INT          # Lưu checkpoint mỗi N epochs (default: 1)
--sample_every INT        # Sample predictions mỗi N epochs (default: 3)
```

### **Misc Arguments**
```bash
--seed INT                # Random seed (default: 42)
--no_gradient_checkpointing  # Tắt gradient checkpointing
--analyze_dataset         # 🔥 NEW: Analyze dataset trước training (detect imbalance!)
```

---

## 🎯 CÁC LỆNH PHỔ BIẾN (V2.0)

### 🔥 RECOMMENDED: Training với tất cả improvements
```bash
python train_no_latent.py \
  --batch_size 12 \
  --epochs 30 \
  --lr 5e-5 \
  --scheduler plateau \
  --scheduler_patience 3 \
  --early_stopping \
  --early_stopping_patience 5 \
  --analyze_dataset \
  --sample_every 3
```

### Training nhanh (test run)
```bash
python train_no_latent.py \
  --epochs 10 \
  --batch_size 8 \
  --scheduler plateau \
  --early_stopping \
  --sample_every 2
```

### Training với learning rate cao hơn
```bash
python train_no_latent.py --lr 1e-4
```

### Training với nhiều fusion layers
```bash
python train_no_latent.py --num_fusion_layers 4
```

### Resume training từ checkpoint
```bash
python train_no_latent.py --resume checkpoints_no_latent/checkpoint_stage3_epoch10.pt
```

### Training để debug (save ít hơn)
```bash
python train_no_latent.py \
  --epochs 5 \
  --batch_size 8 \
  --save_every 2 \
  --sample_every 1
```

### Training full power (max settings)
```bash
python train_no_latent.py \
  --batch_size 20 \
  --epochs 50 \
  --lr 1e-4 \
  --num_fusion_layers 4 \
  --unfreeze_encoder_layers 6
```

### Training cho low memory (16GB GPU)
```bash
python train_no_latent.py \
  --batch_size 8 \
  --num_workers 2
```

---

## 📊 EVALUATION COMMANDS

### Eval validation set
```bash
python eval_no_latent.py \
  --checkpoint checkpoints_no_latent/best_model.pt \
  --split val
```

### Eval test set
```bash
python eval_no_latent.py \
  --checkpoint checkpoints_no_latent/best_model.pt \
  --split test
```

### Eval với nhiều samples
```bash
python eval_no_latent.py \
  --checkpoint checkpoints_no_latent/best_model.pt \
  --split val \
  --num_samples 50 \
  --batch_size 16
```

---

## 🔍 MONITORING COMMANDS

### Xem help (tất cả arguments)
```bash
python train_no_latent.py --help
```

### Kiểm tra GPU
```bash
nvidia-smi
watch -n 1 nvidia-smi  # Auto refresh mỗi 1s
```

### Xem log realtime
```bash
tail -f logs/train.log
```

### Kiểm tra training đang chạy
```bash
ps aux | grep train_no_latent
```

### Kill training
```bash
pkill -f train_no_latent.py
```

---

## 🆚 SO SÁNH KẾT QUẢ

### Compare models
```bash
python compare_models.py
```

---

## 💡 EXAMPLES THỰC TẾ

### Example 1: Quick test (5 epochs)
```bash
python train_no_latent.py \
  --epochs 5 \
  --batch_size 8 \
  --sample_every 1 \
  --output_dir ./test_checkpoints
```

### Example 2: Full training
```bash
mkdir -p logs checkpoints_no_latent

nohup python train_no_latent.py \
  --epochs 30 \
  --batch_size 12 \
  --lr 5e-5 \
  --num_fusion_layers 2 \
  --output_dir ./checkpoints_no_latent \
  > logs/train_no_latent.log 2>&1 &

# Xem log
tail -f logs/train_no_latent.log
```

### Example 3: Resume sau khi bị crash
```bash
python train_no_latent.py \
  --resume checkpoints_no_latent/checkpoint_stage3_epoch15.pt \
  --epochs 30
```

### Example 4: Experiment với config mới
```bash
python train_no_latent.py \
  --epochs 25 \
  --batch_size 16 \
  --lr 8e-5 \
  --num_fusion_layers 3 \
  --dropout 0.15 \
  --output_dir ./checkpoints_experiment1
```

---

## 🎓 TIPS

1. **Chạy song song nhiều experiments:**
   ```bash
   # Experiment 1
   python train_no_latent.py --lr 5e-5 --output_dir ./exp1 &
   
   # Experiment 2
   python train_no_latent.py --lr 1e-4 --output_dir ./exp2 &
   ```

2. **Monitor GPU usage:**
   ```bash
   watch -n 1 'nvidia-smi && echo && ps aux | grep train'
   ```

3. **Auto restart nếu crash:**
   ```bash
   while true; do
       python train_no_latent.py || sleep 10
   done
   ```

---

## ❓ HELP

Để xem tất cả options:
```bash
python train_no_latent.py -h
```

Output:
```
usage: train_no_latent.py [-h] [--data_dir DATA_DIR] [--batch_size BATCH_SIZE]
                          [--num_workers NUM_WORKERS] [--dinov2_model DINOV2_MODEL]
                          [--bartpho_model BARTPHO_MODEL]
                          [--num_fusion_layers NUM_FUSION_LAYERS] [--num_heads NUM_HEADS]
                          [--dropout DROPOUT] [--epochs EPOCHS] [--lr LR]
                          [--weight_decay WEIGHT_DECAY] [--max_norm MAX_NORM] [--no_amp]
                          [--unfreeze_encoder_layers UNFREEZE_ENCODER_LAYERS]
                          [--freeze_decoder] [--output_dir OUTPUT_DIR] [--resume RESUME]
                          [--save_every SAVE_EVERY] [--sample_every SAMPLE_EVERY]
                          [--seed SEED] [--no_gradient_checkpointing]

Train Deterministic VQA (No Latent)
...
```
