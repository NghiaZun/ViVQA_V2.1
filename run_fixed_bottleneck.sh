#!/bin/bash

################################################################################
# FIXED BOTTLENECK TRAINING
# ==========================
# Root cause: Model learns shortcuts ("màu đỏ" instead of "xe lửa")
# 
# COMPREHENSIVE FIX:
# 1. TIGHTER BOTTLENECK: 6×512=3072 → 4×384=1536 (50% reduction)
#    - Force model to SELECT essential info only
#    - No room for shortcuts like "màu đỏ"
# 
# 2. DEEPER REASONING: 2→4 layers
#    - Enable multi-hop: "đường ray" → "phương tiện" → "xe lửa"
#    - 2 layers only learn surface → shortcuts win
#    - 4 layers can chain concepts → semantics possible
# 
# 3. BALANCED FREE BITS: 0.27
#    - Not too low (0.23 → high KL pressure → lose info)
#    - Not too high (0.35 → posterior collapse)
#    - Balanced: KL after = 0.10-0.15
# 
# 4. TEACHER DISTILLATION (Stage 3):
#    - Penalize shortcuts ("màu đỏ" gets score 0.0)
#    - Reward semantics ("xe lửa" gets score 1.0)
#    - 8 reasoning samples for diversity
################################################################################

echo "================================================================================"
echo "🔥 FIXED BOTTLENECK TRAINING - ROOT CAUSE FIX"
echo "================================================================================"
echo ""
echo "Changes from previous run:"
echo "  - Bottleneck: 6×512=3072 → 4×384=1536 (50% smaller, FORCE selection!)"
echo "  - Depth: 2→4 layers (multi-hop reasoning capability)"
echo "  - Free bits: 0.27 (balanced KL pressure)"
echo "  - Teacher: rule_based with 8 samples (Stage 3)"
echo ""
echo "Expected improvements:"
echo "  - Reduce shortcuts: 'màu đỏ' → 'xe lửa' (semantic understanding)"
echo "  - Val answer: 0.73 → 0.60-0.68 (10-18% improvement)"
echo "  - KL after: 0.20 → 0.10-0.15 (healthier latent space)"
echo "================================================================================"
echo ""

# Kaggle paths (update if needed)
CSV_PATH="/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"
IMAGE_FOLDER="/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"

# Check if paths exist
if [ ! -f "$CSV_PATH" ]; then
    echo "❌ ERROR: CSV file not found at $CSV_PATH"
    echo "   Please update CSV_PATH in this script"
    exit 1
fi

if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "❌ ERROR: Image folder not found at $IMAGE_FOLDER"
    echo "   Please update IMAGE_FOLDER in this script"
    exit 1
fi

echo "✅ Data paths validated"
echo ""

# Run training
python train.py \
  --csv_path "$CSV_PATH" \
  --image_folder "$IMAGE_FOLDER" \
  --stage1_epochs 0 \
  --stage2_epochs 10 \
  --stage3_epochs 25 \
  --teacher_type rule_based \
  --batch_size 2 \
  --early_stopping_patience 4

echo ""
echo "================================================================================"
echo "✅ Training completed!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Check training_log.csv for metrics"
echo "  2. Look at sample predictions (every 5 epochs)"
echo "  3. Compare with previous run:"
echo "     - Old: 'màu đỏ' shortcuts, val=0.73"
echo "     - New: 'xe lửa' semantics, val=0.60-0.68 (expected)"
echo ""
echo "If still shortcut issues:"
echo "  - Consider removing VAE (use simple deterministic bottleneck)"
echo "  - Try VLM teacher (--teacher_type vlm, slower but better)"
echo "================================================================================"
