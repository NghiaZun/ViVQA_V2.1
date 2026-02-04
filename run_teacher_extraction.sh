#!/bin/bash
################################################################################
# TEACHER REPRESENTATIONS EXTRACTION (VISION + TEXT ONLY)
################################################################################
# Extract teacher representations for knowledge distillation
#
# TEACHERS:
#   - Vision: SigLIP-SO400M (~400-430M params)
#   - Text: PhoBERT-large (307M, representation teacher)
#   - Answer: Ground truth labels ONLY (no distillation)
#
# What is saved:
#   Vision: patch_emb [729,hidden], cls_emb [hidden], intermediate [4,729,hidden]
#   Text: token_emb [seq,1024], cls_emb [1024], attention [heads,seq,seq], qa_similarity [1]
#
# Estimated time: ~1.5 hours for 12K samples (train + val)
# Estimated storage: ~2GB .npy files
################################################################################

set -e  # Exit on error

echo "========================================================================"
echo "STEP 1: Extract TRAINING set teacher representations"
echo "========================================================================"

python extract_teacher_logits.py \
    --csv_path train.csv \
    --image_folder vivqa/images \
    --output_dir /kaggle/working/teacher_cache \
    --batch_size 8 \
    --device cuda

echo ""
echo "========================================================================"
echo "STEP 2: Extract VALIDATION set teacher representations"
echo "========================================================================"

python extract_teacher_logits.py \
    --csv_path OpenViVQA/dev.json \
    --image_folder vivqa/images \
    --output_dir /kaggle/working/teacher_cache \
    --batch_size 8 \
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
echo "Answer supervision: Ground truth labels ONLY (no distillation)"
echo ""
echo "========================================================================"
echo "NEXT STEP: Implement distillation training"
echo "========================================================================"
echo "Modify train_no_latent.py to add:"
echo "  1. TeacherDistillationDataset (loads .npy with mmap)"
echo "  2. Vision KD: MSE(student_patch, teacher_patch_downsampled)"
echo "  3. Text KD: MSE(student_question_emb, teacher_question_emb)"
echo "  4. Combined loss: (1-α)*CE + α*(0.5*vision_kd + 0.5*text_kd)"
echo "========================================================================"
