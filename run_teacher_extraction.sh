#!/bin/bash
################################################################################
# TEACHER EXTRACTION PIPELINE (CORRECTED)
################################################################################
# Extract teacher features for knowledge distillation
#
# CORRECTED TEACHERS:
#   - Vision: SigLIP-SO400M (~400-430M params, NOT 878M!)
#   - Text: PhoBERT-large (307M, REPRESENTATION teacher, NOT generative!)
#   - Answer: Trained student r=96 checkpoint (self-distillation)
#
# What is saved:
#   Vision: patch_emb [729,hidden], cls_emb [hidden], intermediate [4,729,hidden]
#   Text: token_emb [seq,1024], cls_emb [1024], attention [heads,seq,seq], qa_similarity [1]
#   Answer: logits [max_len,vocab] from trained student
#
# Estimated time: 2-3 hours for 12K samples (train + val)
# Estimated storage: ~3.5GB .npy files
################################################################################

set -e  # Exit on error

# Path to trained r=96 checkpoint (REQUIRED!)
CHECKPOINT_PATH="/kaggle/working/checkpoints/best_model.pt"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ ERROR: r=96 checkpoint not found at $CHECKPOINT_PATH"
    echo "   Please train r=96 model first or update CHECKPOINT_PATH"
    exit 1
fi

echo "========================================================================"
echo "STEP 1: Extract TRAINING set teacher features"
echo "========================================================================"

python extract_teacher_logits.py \
    --csv_path train.csv \
    --image_folder vivqa/images \
    --output_dir /kaggle/working/teacher_cache \
    --student_checkpoint "$CHECKPOINT_PATH" \
    --batch_size 16 \
    --device cuda

echo ""
echo "========================================================================"
echo "STEP 2: Extract VALIDATION set teacher features"
echo "========================================================================"

python extract_teacher_logits.py \
    --csv_path OpenViVQA/dev.json \
    --image_folder vivqa/images \
    --output_dir /kaggle/working/teacher_cache \
    --student_checkpoint "$CHECKPOINT_PATH" \
    --batch_size 16 \
    --device cuda

echo ""
echo "========================================================================"
echo "✅ TEACHER EXTRACTION COMPLETE!"
echo "========================================================================"
echo "Files created:"
ls -lh /kaggle/working/teacher_cache/
echo ""
echo "Total storage:"
du -sh /kaggle/working/teacher_cache/
echo ""
echo "========================================================================"
echo "WHAT WAS SAVED:"
echo "========================================================================"
echo "Vision (SigLIP-SO400M ~400-430M params):"
echo "  - vision_patch_emb_*.npy: [N, 729, hidden] - Spatial features"
echo "  - vision_cls_emb_*.npy: [N, hidden] - Global scene"
echo "  - vision_intermediate_*.npy: [N, 4, 729, hidden] - Multi-scale"
echo ""
echo "Text (PhoBERT-large 307M, representation teacher):"
echo "  - text_token_emb_*.npy: [N, seq, 1024] - Contextual embeddings"
echo "  - text_cls_emb_*.npy: [N, 1024] - Question representation"
echo "  - text_attention_*.npy: [N, heads, seq, seq] - Attention patterns"
echo "  - text_qa_similarity_*.npy: [N] - Q-A semantic alignment"
echo ""
echo "Answer (r=96 student self-distillation):"
echo "  - answer_logits_*.npy: [N, max_len, vocab] - Soft labels"
echo ""
echo "========================================================================"
echo "NEXT STEP: Implement distillation training"
echo "========================================================================"
echo "Modify train_no_latent.py to add:"
echo "  1. TeacherDistillationDataset (loads .npy with mmap)"
echo "  2. Vision KD: MSE(student_patch, teacher_patch_downsampled)"
echo "  3. Text KD: MSE(student_question_emb, teacher_question_emb)"
echo "  4. Answer KD: KL(student_answer, teacher_answer_from_r96)"
echo "  5. Combined loss: (1-α)*CE + α*(0.4*vision + 0.3*text + 0.3*answer)"
echo "========================================================================"
