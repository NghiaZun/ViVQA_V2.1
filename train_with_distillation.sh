#!/bin/bash

# ============================================================================
# 🔥🔥🔥 ONLINE KNOWLEDGE DISTILLATION TRAINING 🔥🔥🔥
# ============================================================================
# 
# Based on your BEST config (65.78% EM) + Online Distillation
# 
# Teachers (FP16, frozen, online):
#   - Vision Teacher: SigLIP-SO400M-384 (~430M params)
#   - Text Teacher: PhoBERT-large (307M params)
#
# VRAM Usage: ~5-6GB / 16GB (safe for Kaggle T4)
# Training Time: ~20-30% slower than baseline
# Expected: 65.78% → 67-68% EM (+1.2-2.2%)
#
# Your best settings preserved:
#   - text_lora_r=96, fusion_layers=6, vision_gate, type_loss
#   - batch=16, grad_accum=2, dropout=0.2, weight_decay=0.2
# ============================================================================

echo "🔥🔥🔥 ONLINE KNOWLEDGE DISTILLATION TRAINING 🔥🔥🔥"
echo "Starting from your BEST config (65.78% EM)"
echo "Vision Teacher: SigLIP-SO400M-384 (FP16, online)"
echo "Text Teacher: PhoBERT-large (FP16, online)"
echo "Expected: 65.78% → 67-68% EM"
echo ""

python train_no_latent.py \
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

echo ""
echo "============================================================================"
echo "✅ Training complete! Check /kaggle/working/checkpoints_distill/"
echo ""
echo "Expected improvement:"
echo "  Baseline:        65.78% EM (your best)"
echo "  + Distillation:  67-68% EM (+1.2-2.2%)"
echo "  COUNT type:      63.5-65.0% EM (from 61.57%)"
echo "============================================================================"
