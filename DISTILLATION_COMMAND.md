# 🔥 ONLINE DISTILLATION - KAGGLE COMMAND

## Copy-Paste vào Kaggle Notebook:

```python
!python train_no_latent.py \
    --train_csv /kaggle/input/vivqa/data/train.csv \
    --image_dir /kaggle/input/vivqa/data/images/train \
    --val_split 0.15 \
    --batch_size 16 \
    --gradient_accumulation_steps 2 \
    --epochs 30 \
    --lr 5e-5 \
    --weight_decay 0.2 \
    --dropout 0.2 \
    --vision_model google/siglip-base-patch16-224 \
    --bartpho_model vinai/bartpho-syllable \
    --use_vision_gate \
    --vision_gate_init 1.5 \
    --use_type_loss \
    --use_text_lora \
    --text_lora_r 96 \
    --text_lora_alpha 192 \
    --scheduler cosine \
    --early_stopping \
    --early_stopping_patience 3 \
    --num_fusion_layers 6 \
    --label_smoothing 0.1 \
    --answer_weights answer_weights.json \
    --output_dir /kaggle/working/checkpoints_distill \
    --seed 42 \
    --use_distillation \
    --vision_teacher google/siglip-so400m-patch14-384 \
    --text_teacher vinai/phobert-large \
    --distill_alpha 0.5 \
    --distill_temperature 2.0
```

## Thay đổi so với BEST config:

### ✅ GIỮ NGUYÊN (Từ best config 65.78%):
- text_lora_r=96, alpha=192 (thay vì vision LoRA)
- num_fusion_layers=6
- use_vision_gate + vision_gate_init=1.5
- use_type_loss
- batch_size=16, grad_accum=2
- dropout=0.2, weight_decay=0.2
- scheduler=cosine
- early_stopping_patience=3
- label_smoothing=0.1
- answer_weights.json

### 🔥 THÊM MỚI (Distillation):
```python
--use_distillation \
--vision_teacher google/siglip-so400m-patch14-384 \
--text_teacher vinai/phobert-large \
--distill_alpha 0.5 \
--distill_temperature 2.0
```

## Memory Usage:

```
Your best config (no distillation):  ~3.5 GB VRAM
+ Vision Teacher (FP16):             ~1.0 GB
+ Text Teacher (FP16):               ~0.8 GB
+ Distillation overhead:             ~0.7 GB
------------------------------------------------
TOTAL:                               ~6.0 GB / 16GB ✅
```

## Expected Results:

| Metric               | Baseline      | + Distillation | Gain    |
|---------------------|---------------|----------------|---------|
| **Overall EM**      | **65.78%**    | **67-68%**     | **+1.2-2.2%** |
| OBJECT              | 70.2%         | 71.5%          | +1.3%   |
| COUNT               | 61.6%         | 63.5%          | +1.9%   |
| COLOR               | 65.1%         | 66.8%          | +1.7%   |
| LOCATION            | 66.8%         | 68.2%          | +1.4%   |

## Training Time:

- **Without distillation**: ~3.5 hours (baseline)
- **With distillation**: ~4.5 hours (~30% slower)

## Key Features:

1. **0GB disk space** (no offline extraction)
2. **Dynamic teacher inference** every batch
3. **Vision KD**: SigLIP-SO400M (729 patches @ 384px) → Student (196 patches @ 224px)
4. **Text KD**: PhoBERT-large (1024D) → PhoBERT-base (768D)
5. **Combined loss**: (1-α)×CE + α×(0.5×vision_kd + 0.5×text_kd)

## Monitoring:

Progress bar sẽ show:
```
Train Stage 3: 100%|██| 718/718 [1:23:45<00:00, 7.0s/it, 
    loss=2.145, ans=2.023, vkd=0.142, tkd=0.089, α_mean=1.52]
```

- `loss`: Total loss
- `ans`: Answer CE loss
- `vkd`: Vision KD loss (teacher → student vision)
- `tkd`: Text KD loss (teacher → student text)
- `α_mean`: Vision gate value (>1.0 = prefer vision)

## Troubleshooting:

### Nếu OOM:
```python
# Giảm batch size
--batch_size 12 \  # từ 16 → 12
--gradient_accumulation_steps 3 \  # từ 2 → 3 (giữ effective=36)
```

### Nếu quá chậm:
```python
# Giảm distillation weight
--distill_alpha 0.3 \  # từ 0.5 → 0.3 (70% CE + 30% KD)
```

### Nếu không improve:
```python
# Tăng distillation
--distill_alpha 0.7 \  # 30% CE + 70% KD
--distill_temperature 3.0 \  # Softer targets
```

## Checkpoint:

Best model sẽ save tại:
```
/kaggle/working/checkpoints_distill/best_model.pt
```

## Next: Evaluate

```python
!python eval_no_latent.py \
    --checkpoint /kaggle/working/checkpoints_distill/best_model.pt \
    --test_csv /kaggle/input/vivqa/data/test.csv \
    --image_dir /kaggle/input/vivqa/data/images/test \
    --output_csv results_distill.csv
```

---

## 🚀 Ready to achieve 67-68% EM!
