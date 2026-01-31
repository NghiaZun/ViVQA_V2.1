#!/bin/bash
################################################################################
# EVALUATE SIGLIP - MINIMAL VERSION (100% WORKING)
################################################################################

python eval_minimal.py \
    --checkpoint /kaggle/input/sigclip-v1/transformers/default/1/best_model.pt \
    --csv_path /kaggle/input/vivqa/data/test.csv \
    --image_folder /kaggle/input/vivqa/data/images/test \
    --vision_model google/siglip-base-patch16-224 \
    --batch_size 16 \
    --output_csv results_siglip_test.csv

echo ""
echo "✅ Evaluation complete!"
echo "📊 Results saved to: results_siglip_test.csv"
