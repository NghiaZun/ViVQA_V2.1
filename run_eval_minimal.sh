#!/bin/bash
################################################################################
# EVALUATE SIGLIP - MINIMAL VERSION (100% WORKING)
################################################################################

echo "🔍 Debugging file save..."
python debug_kaggle_save.py

echo ""
echo "🚀 Running evaluation..."

python eval_minimal.py \
    --checkpoint "/kaggle/input/sigclip-test2/transformers/default/1/best_model(1).pt" \
    --csv_path /kaggle/input/vivqa/data/test.csv \
    --image_folder /kaggle/input/vivqa/data/images/test \
    --vision_model google/siglip-base-patch16-224 \
    --batch_size 16 \
    --output_csv /kaggle/working/results_siglip_test.csv

echo ""
echo "📂 Checking if file was saved..."
if [ -f /kaggle/working/results_siglip_test.csv ]; then
    echo "✅ File found!"
    ls -lh /kaggle/working/results_siglip_test.csv
    echo ""
    echo "First 5 lines:"
    head -5 /kaggle/working/results_siglip_test.csv
else
    echo "❌ File NOT found!"
    echo "Files in /kaggle/working:"
    ls -lh /kaggle/working/*.csv 2>/dev/null || echo "No CSV files found"
fi

echo ""
echo "✅ Evaluation complete!"
