#!/bin/bash
# ============================================================================
# TRAINING SCRIPT: Type-Conditioned Vision Adapter
# ============================================================================
# Expected improvement: +1.5-2% EM (63.0-63.5% total)
#
# Architecture:
#   - SigLIP vision encoder (frozen)
#   - Type-conditioned adapter (4 experts, rank=64)
#   - Text LoRA (optional, for extra +0.3-0.8%)
#   - Type prediction head (auxiliary task)
#   - Vision gating (type-conditioned attention)
# ============================================================================

echo "========================================================================"
echo "TRAINING: Type-Conditioned Vision Adapter + Text LoRA"
echo "========================================================================"

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
    --text_lora_alpha 32 \
    --use_type_loss \
    --type_loss_weight 0.5 \
    --use_vision_gate \
    --vision_gate_init 1.5 \
    --batch_size 12 \
    --epochs 30 \
    --lr 5e-5 \
    --weight_decay 0.01 \
    --scheduler plateau \
    --scheduler_patience 3 \
    --early_stopping \
    --early_stopping_patience 5 \
    --gradient_checkpointing \
    --output_dir checkpoints/type_adapter_siglip \
    --save_every 1 \
    --sample_every 3

echo ""
echo "========================================================================"
echo "Training complete!"
echo "Expected results:"
echo "  - EM: 63.0-63.5% (+1.5-2% from 61.45% baseline)"
echo "  - Improved LOCATION (+2-3%) and OBJECT (+1-1.5%)"
echo "========================================================================"
