"""
🔍 DATASET BIAS ANALYZER
========================

Phân tích dataset bias để tìm nguyên nhân text-only accuracy quá cao.

Kiểm tra:
    1. Text-only accuracy PER QUESTION TYPE
    2. Answer distribution (có bias về "2", "đỏ" không?)
    3. Question patterns (template-based hay diverse?)
    4. Biased questions cần filter

Usage:
    python analyze_dataset_bias.py --checkpoint <path> --test_csv <path> --image_dir <path>

Output:
    - Per-type text-only accuracy
    - Most frequent answers per type
    - Biased question examples
    - Filtering recommendations
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from collections import defaultdict, Counter
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer

from dataset import VQAGenDataset
from model_no_latent import DeterministicVQA, shift_tokens_right


# ============================================================================
# QUESTION TYPE AUTO-DETECTION (từ diagnostic_tools.py)
# ============================================================================

def detect_question_type(question_text):
    """
    Auto-detect question type từ text
    
    Returns:
        0: OBJECT (cái gì, con gì, vật gì)
        1: COUNT (bao nhiêu, mấy)
        2: COLOR (màu gì, màu sắc)
        3: LOCATION (đâu, ở đâu, bên nào)
    """
    q = question_text.lower()
    
    # COUNT patterns
    if any(word in q for word in ['bao nhiêu', 'mấy', 'có', 'số lượng']):
        return 1
    
    # COLOR patterns
    if any(word in q for word in ['màu', 'màu sắc', 'sắc']):
        return 2
    
    # LOCATION patterns
    if any(word in q for word in ['đâu', 'ở đâu', 'chỗ nào', 'vị trí', 'bên', 'phía']):
        return 3
    
    # Default: OBJECT
    return 0


TYPE_NAMES = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}


# ============================================================================
# TEXT-ONLY ACCURACY TEST
# ============================================================================

def test_text_only_accuracy(model, dataloader, device='cuda'):
    """
    Test accuracy khi MASK vision features (text-only)
    
    Returns:
        dict: {
            'overall_acc': float,
            'per_type_acc': {type_id: accuracy},
            'per_sample_results': [
                {
                    'question': str,
                    'answer': str,
                    'prediction': str,
                    'correct': bool,
                    'type': int
                }
            ]
        }
    """
    print("\n" + "="*80)
    print("🧪 TESTING TEXT-ONLY ACCURACY (Vision Masked)")
    print("="*80)
    
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('vinai/bartpho-syllable')
    
    all_results = []
    type_correct = defaultdict(int)
    type_total = defaultdict(int)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing text-only"):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            batch_size = pixel_values.size(0)
            
            # Manual forward with MASKED vision
            # 1. Vision encoding (then MASK to zero)
            vision_outputs = model.vision_encoder(pixel_values=pixel_values)
            patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]
            patch_tokens = patch_tokens + model.vision_pos_embed.expand(batch_size, -1, -1)
            vision_features_masked = torch.zeros_like(model.vision_proj(patch_tokens))  # 🔥 Zero!
            
            # 2. Text encoding
            text_outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_features = text_outputs.last_hidden_state
            
            # 3. Fusion (with zero vision)
            fused_vision_masked = vision_features_masked
            for fusion_layer in model.flamingo_fusion:
                fused_vision_masked = fusion_layer(fused_vision_masked, text_features, attention_mask)
            
            # 4. Gating (if enabled)
            gated_vision_masked = fused_vision_masked
            if model.use_vision_gate:
                text_cls = text_features[:, 0, :]
                type_logits = model.type_head(text_cls)
                predicted_types = torch.argmax(type_logits, dim=-1)
                gated_vision_masked, _ = model.vision_gating(
                    fused_vision_masked,
                    text_features,
                    type_ids=predicted_types
                )
            
            # 5. Prepare decoder inputs
            decoder_input_ids = shift_tokens_right(
                labels,
                model.config.pad_token_id,
                model.config.decoder_start_token_id
            )
            
            # 6. Decoder cross-attention
            encoder_hidden_states = torch.cat([gated_vision_masked, text_features], dim=1)
            encoder_attention_mask = torch.cat([
                torch.ones(batch_size, gated_vision_masked.size(1), device=attention_mask.device),
                attention_mask
            ], dim=1)
            
            decoder_outputs = model.decoder(
                input_ids=decoder_input_ids,
                attention_mask=None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask
            )
            
            # 7. Generate logits
            answer_logits = model.lm_head(decoder_outputs.last_hidden_state)
            preds = answer_logits.argmax(dim=-1)
            
            # Decode predictions and answers
            for i in range(batch_size):
                # Get question text
                question_ids = input_ids[i].cpu().tolist()
                question_text = tokenizer.decode(question_ids, skip_special_tokens=True)
                
                # Get answer text
                answer_ids = labels[i].cpu().tolist()
                answer_ids = [aid for aid in answer_ids if aid != -100]
                answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
                
                # Get prediction
                pred_ids = preds[i].cpu().tolist()
                # Lấy đến khi gặp EOS hoặc PAD
                pred_ids_clean = []
                for pid in pred_ids:
                    if pid in [model.config.eos_token_id, model.config.pad_token_id]:
                        break
                    pred_ids_clean.append(pid)
                pred_text = tokenizer.decode(pred_ids_clean, skip_special_tokens=True)
                
                # Detect question type
                q_type = detect_question_type(question_text)
                
                # Check correctness (exact match)
                correct = (answer_text.strip().lower() == pred_text.strip().lower())
                
                all_results.append({
                    'question': question_text,
                    'answer': answer_text,
                    'prediction': pred_text,
                    'correct': correct,
                    'type': q_type,
                    'type_name': TYPE_NAMES[q_type]
                })
                
                type_correct[q_type] += int(correct)
                type_total[q_type] += 1
    
    # Compute per-type accuracy
    per_type_acc = {}
    for q_type in type_total:
        per_type_acc[q_type] = type_correct[q_type] / type_total[q_type] if type_total[q_type] > 0 else 0
    
    # Overall accuracy
    overall_acc = sum(r['correct'] for r in all_results) / len(all_results) if all_results else 0
    
    print(f"\n📊 Text-Only Accuracy Results:")
    print(f"   Overall: {overall_acc:.1%}")
    print(f"\n   Per Type:")
    for q_type in sorted(per_type_acc.keys()):
        acc = per_type_acc[q_type]
        total = type_total[q_type]
        type_name = TYPE_NAMES[q_type]
        print(f"      {type_name:<12} {acc:<8.1%}  ({total} samples)")
    
    return {
        'overall_acc': overall_acc,
        'per_type_acc': per_type_acc,
        'per_sample_results': all_results
    }


# ============================================================================
# ANSWER DISTRIBUTION ANALYSIS
# ============================================================================

def analyze_answer_distribution(results):
    """
    Phân tích distribution của answers per type
    Detect bias (vd: 90% COUNT answers là "2")
    """
    print("\n" + "="*80)
    print("📈 ANSWER DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Group by type
    type_answers = defaultdict(list)
    for r in results['per_sample_results']:
        type_answers[r['type']].append(r['answer'])
    
    # Analyze each type
    for q_type in sorted(type_answers.keys()):
        type_name = TYPE_NAMES[q_type]
        answers = type_answers[q_type]
        
        # Count frequency
        counter = Counter(answers)
        total = len(answers)
        
        print(f"\n🔹 {type_name} ({total} samples):")
        print(f"   Top 10 most frequent answers:")
        
        for i, (answer, count) in enumerate(counter.most_common(10), 1):
            freq = count / total
            print(f"      {i:2d}. '{answer:<20}' : {count:4d} ({freq:.1%})")
        
        # Detect bias
        most_common_freq = counter.most_common(1)[0][1] / total if counter else 0
        if most_common_freq > 0.3:
            print(f"   ⚠️  BIAS DETECTED: Top answer chiếm {most_common_freq:.1%} (> 30%)")
            print(f"      → Model có thể học bias này từ text pattern!")
        
        # Diversity score
        unique_ratio = len(counter) / total
        print(f"   Diversity: {len(counter)} unique answers / {total} total = {unique_ratio:.1%}")
        if unique_ratio < 0.2:
            print(f"   ⚠️  LOW DIVERSITY: Chỉ {unique_ratio:.1%} unique answers")


# ============================================================================
# QUESTION PATTERN ANALYSIS
# ============================================================================

def analyze_question_patterns(results):
    """
    Phân tích question patterns để detect template-based questions
    """
    print("\n" + "="*80)
    print("🔤 QUESTION PATTERN ANALYSIS")
    print("="*80)
    
    # Group by type
    type_questions = defaultdict(list)
    for r in results['per_sample_results']:
        type_questions[r['type']].append(r['question'])
    
    for q_type in sorted(type_questions.keys()):
        type_name = TYPE_NAMES[q_type]
        questions = type_questions[q_type]
        
        print(f"\n🔹 {type_name} ({len(questions)} samples):")
        
        # Extract question patterns (first 3-4 words)
        patterns = []
        for q in questions:
            words = q.split()[:4]  # First 4 words
            pattern = ' '.join(words)
            patterns.append(pattern)
        
        pattern_counter = Counter(patterns)
        
        print(f"   Top 5 question patterns:")
        for i, (pattern, count) in enumerate(pattern_counter.most_common(5), 1):
            freq = count / len(questions)
            print(f"      {i}. '{pattern}...' : {count} ({freq:.1%})")
        
        # Diversity
        unique_ratio = len(pattern_counter) / len(questions)
        print(f"   Pattern diversity: {len(pattern_counter)} unique / {len(questions)} total = {unique_ratio:.1%}")
        
        if unique_ratio < 0.3:
            print(f"   ⚠️  TEMPLATE-BASED: Patterns lặp lại nhiều (diversity < 30%)")
            print(f"      → Dễ bị text bias, model học pattern thay vì vision!")


# ============================================================================
# BIASED QUESTIONS DETECTION
# ============================================================================

def identify_biased_questions(results, threshold=0.8):
    """
    Identify questions mà text-only đoán đúng với high confidence
    → Đây là biased questions cần filter!
    """
    print("\n" + "="*80)
    print(f"🎯 BIASED QUESTIONS DETECTION (threshold={threshold:.0%})")
    print("="*80)
    
    # Group by (question_pattern, answer)
    pattern_answer_correct = defaultdict(lambda: {'correct': 0, 'total': 0, 'examples': []})
    
    for r in results['per_sample_results']:
        # Extract pattern (first 3 words)
        words = r['question'].split()[:3]
        pattern = ' '.join(words)
        
        key = (pattern, r['answer'], r['type'])
        pattern_answer_correct[key]['total'] += 1
        pattern_answer_correct[key]['correct'] += int(r['correct'])
        
        if len(pattern_answer_correct[key]['examples']) < 3:
            pattern_answer_correct[key]['examples'].append({
                'question': r['question'],
                'answer': r['answer'],
                'prediction': r['prediction']
            })
    
    # Find biased patterns
    biased_patterns = []
    for (pattern, answer, q_type), stats in pattern_answer_correct.items():
        if stats['total'] >= 3:  # At least 3 samples
            acc = stats['correct'] / stats['total']
            if acc >= threshold:
                biased_patterns.append({
                    'pattern': pattern,
                    'answer': answer,
                    'type': TYPE_NAMES[q_type],
                    'accuracy': acc,
                    'count': stats['total'],
                    'examples': stats['examples']
                })
    
    # Sort by count (most frequent first)
    biased_patterns.sort(key=lambda x: x['count'], reverse=True)
    
    print(f"\n📋 Found {len(biased_patterns)} biased patterns:")
    print(f"   (Pattern that text-only guesses correctly ≥ {threshold:.0%})")
    print()
    
    for i, bp in enumerate(biased_patterns[:20], 1):  # Top 20
        print(f"{i:2d}. Pattern: '{bp['pattern']}...'")
        print(f"    Type: {bp['type']}")
        print(f"    Text-only accuracy: {bp['accuracy']:.1%} ({bp['count']} samples)")
        print(f"    Typical answer: '{bp['answer']}'")
        print(f"    Example: Q: {bp['examples'][0]['question']}")
        print(f"             A: {bp['examples'][0]['answer']}")
        print()
    
    return biased_patterns


# ============================================================================
# FILTERING RECOMMENDATIONS
# ============================================================================

def recommend_filtering(results, biased_patterns):
    """
    Đề xuất cách filter dataset để giảm bias
    """
    print("\n" + "="*80)
    print("💡 FILTERING RECOMMENDATIONS")
    print("="*80)
    
    total_samples = len(results['per_sample_results'])
    
    # Count biased samples
    biased_samples = 0
    for bp in biased_patterns:
        biased_samples += bp['count']
    
    bias_ratio = biased_samples / total_samples if total_samples > 0 else 0
    
    print(f"\n📊 Current Dataset Status:")
    print(f"   Total samples: {total_samples}")
    print(f"   Biased samples (text-only acc > 80%): {biased_samples} ({bias_ratio:.1%})")
    print(f"   Clean samples: {total_samples - biased_samples} ({1-bias_ratio:.1%})")
    
    print(f"\n💡 Recommendations:")
    
    if bias_ratio > 0.5:
        print("   ❌ CRITICAL: > 50% dataset is biased!")
        print("   Actions:")
        print("      1. Filter out biased patterns (giảm xuống còn 30%)")
        print("      2. Add harder visual questions")
        print("      3. Re-balance dataset")
    elif bias_ratio > 0.3:
        print("   ⚠️  WARNING: 30-50% dataset is biased")
        print("   Actions:")
        print("      1. Consider filtering top biased patterns")
        print("      2. Add data augmentation (paraphrase questions)")
    else:
        print("   ✅ ACCEPTABLE: < 30% bias")
        print("   Actions:")
        print("      1. Current dataset is reasonable")
        print("      2. Can improve by filtering top 10 biased patterns")
    
    # Specific filtering code
    print(f"\n🔧 Sample Filtering Code:")
    print("   ```python")
    print("   # Filter out biased patterns")
    print("   biased_patterns = [")
    for bp in biased_patterns[:5]:
        print(f"       '{bp['pattern']}',")
    print("       # ... (add more)")
    print("   ]")
    print()
    print("   df_filtered = df[~df['question'].str.startswith(tuple(biased_patterns))]")
    print("   ```")


# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_full_analysis(checkpoint_path, test_csv, image_dir, batch_size=16, device='cuda'):
    """
    Chạy toàn bộ dataset bias analysis
    """
    print("\n" + "="*80)
    print("🔍 DATASET BIAS ANALYSIS")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test CSV: {test_csv}")
    print(f"Image Dir: {image_dir}")
    print("="*80)
    
    # Load model
    print("\n📥 Loading model...")
    model = DeterministicVQA(
        use_vision_gate=True,
        vision_gate_init=1.5,
        use_vision_lora=False,
        use_text_lora=False
    )
    
    raw_ckpt = torch.load(checkpoint_path, map_location=device)
    state = raw_ckpt.get('model_state_dict', raw_ckpt)
    
    # Remap checkpoint keys
    adapted_state = {}
    skipped_keys = []
    for k, v in state.items():
        new_k = k
        new_k = new_k.replace('vision_encoder.base_model.model.', 'vision_encoder.')
        new_k = new_k.replace('encoder.base_model.model.', 'encoder.')
        new_k = new_k.replace('vision_encoder.base_model.', 'vision_encoder.')
        new_k = new_k.replace('encoder.base_model.', 'encoder.')
        
        if 'lora' in new_k or '.lora_' in new_k:
            skipped_keys.append(new_k)
            continue
        
        adapted_state[new_k] = v
    
    model.load_state_dict(adapted_state, strict=False)
    model = model.to(device)
    model.eval()
    
    # Load dataset
    print("📥 Loading test dataset...")
    vision_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    
    test_dataset = VQAGenDataset(
        csv_path=test_csv,
        image_folder=image_dir,
        vision_processor=vision_processor,
        tokenizer_name='vinai/bartpho-syllable',
        max_q_len=32,
        max_a_len=20,
        include_question_type=False,  # We auto-detect
        auto_detect_type=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✅ Loaded {len(test_dataset)} test samples")
    
    # Run analyses
    results = test_text_only_accuracy(model, test_loader, device)
    analyze_answer_distribution(results)
    analyze_question_patterns(results)
    biased_patterns = identify_biased_questions(results, threshold=0.8)
    recommend_filtering(results, biased_patterns)
    
    # Save results
    print("\n💾 Saving results to dataset_bias_analysis.txt...")
    with open('dataset_bias_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("DATASET BIAS ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Overall text-only accuracy: {results['overall_acc']:.1%}\n\n")
        
        f.write("Per-type accuracy:\n")
        for q_type, acc in results['per_type_acc'].items():
            f.write(f"  {TYPE_NAMES[q_type]}: {acc:.1%}\n")
        
        f.write(f"\nTop {min(20, len(biased_patterns))} biased patterns:\n")
        for i, bp in enumerate(biased_patterns[:20], 1):
            f.write(f"\n{i}. {bp['pattern']}... ({bp['type']})\n")
            f.write(f"   Acc: {bp['accuracy']:.1%}, Count: {bp['count']}, Answer: '{bp['answer']}'\n")
    
    print("✅ Report saved!")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze dataset bias')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test.csv')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    run_full_analysis(
        checkpoint_path=args.checkpoint,
        test_csv=args.test_csv,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        device=args.device
    )
