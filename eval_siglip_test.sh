#!/bin/bash
################################################################################
# EVALUATE SIGLIP CHECKPOINT ON TEST SET
################################################################################

python eval_siglip.py \
    --checkpoint /kaggle/input/sigclip-v1/transformers/default/1/best_model.pt \
    --csv_path /kaggle/input/vivqa/data/test.csv \
    --image_folder /kaggle/input/vivqa/data/images/test \
    --vision_model_name google/siglip-base-patch16-224 \
    --batch_size 16 \
    --num_samples 30 \
    --output_csv results_siglip_test.csv

################################################################################
# For DINOv2 checkpoint, use:
# --vision_model_name facebook/dinov2-base
################################################################################
