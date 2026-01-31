"""
EVALUATION SCRIPT FOR DETERMINISTIC VQA WITH SIGLIP
====================================================

SigLIP-compatible evaluation with:
- Auto-detection of vision encoder (SigLIP vs DINOv2)
- Multiple metrics: EM, F1, Per-Type Breakdown
- CSV export support

Version 3.0 - SigLIP Ready! 🚀
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from collections import Counter, defaultdict

from dataset import VQAGenDataset
from model_no_latent import DeterministicVQA


def detect_question_type(question_text: str) -> int:
    """
    Auto-detect question type from text
    
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


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match (0 or 1)"""
    return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0


def compute_f1_score(prediction: str, ground_truth: str) -> float:
    """
    Compute F1 score for partial credit
    
    F1 is better than EM for VQA evaluation!
    """
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1


def evaluate(model, dataloader, device, tokenizer):
    """Evaluate model on validation set with EM + F1 + Per-Type breakdown"""
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    all_questions = []
    all_question_types = []
    
    total_loss = 0.0
    num_batches = 0
    
    exact_matches = []
    f1_scores = []
    
    # Per-type tracking
    from collections import defaultdict
    type_exact_matches = defaultdict(list)
    type_f1_scores = defaultdict(list)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass for loss
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            if outputs.total_loss is not None:
                total_loss += outputs.total_loss.item()
                num_batches += 1
            
            # Generate predictions with beam search
            predictions = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=20,
                num_beams=3
            )
            
            # Decode ground truths
            for label in labels:
                label_tokens = label[label != -100].cpu().tolist()
                gt_text = tokenizer.decode(label_tokens, skip_special_tokens=True)
                all_ground_truths.append(gt_text)
            
            # Decode questions and detect types
            for inp in input_ids:
                question_text = tokenizer.decode(inp, skip_special_tokens=True)
                all_questions.append(question_text)
                q_type = detect_question_type(question_text)
                all_question_types.append(q_type)
            
            all_predictions.extend(predictions)
            
            # Compute metrics for this batch
            batch_start_idx = len(all_ground_truths) - len(predictions)
            for i, (pred, gt) in enumerate(zip(predictions, all_ground_truths[-len(predictions):])):
                em = compute_exact_match(pred, gt)
                f1 = compute_f1_score(pred, gt)
                q_type = all_question_types[batch_start_idx + i]
                
                exact_matches.append(em)
                f1_scores.append(f1)
                
                # Track per-type
                type_exact_matches[q_type].append(em)
                type_f1_scores[q_type].append(f1)
            
            # Update progress
            current_em = sum(exact_matches) / len(exact_matches) * 100
            current_f1 = sum(f1_scores) / len(f1_scores) * 100
            
            pbar.set_postfix({
                'loss': f"{total_loss/num_batches:.3f}",
                'EM': f"{current_em:.1f}%",
                'F1': f"{current_f1:.1f}%"
            })
    
    # Compute final metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    exact_match_acc = sum(exact_matches) / len(exact_matches) * 100
    f1_score_avg = sum(f1_scores) / len(f1_scores) * 100
    
    # Compute per-type metrics
    per_type_results = {}
    for q_type in sorted(type_exact_matches.keys()):
        type_em = sum(type_exact_matches[q_type]) / len(type_exact_matches[q_type]) * 100 if type_exact_matches[q_type] else 0
        type_f1 = sum(type_f1_scores[q_type]) / len(type_f1_scores[q_type]) * 100 if type_f1_scores[q_type] else 0
        per_type_results[q_type] = {
            'exact_match': type_em,
            'f1_score': type_f1,
            'count': len(type_exact_matches[q_type])
        }
    
    results = {
        'loss': avg_loss,
        'exact_match': exact_match_acc,
        'f1_score': f1_score_avg,
        'per_type': per_type_results,
        'predictions': all_predictions,
        'ground_truths': all_ground_truths,
        'questions': all_questions,
        'question_types': all_question_types
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Deterministic VQA (SigLIP/DINOv2)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file (val/test)')
    parser.add_argument('--image_folder', type=str, required=True, help='Path to image folder')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to print')
    parser.add_argument('--max_q_len', type=int, default=32, help='Max question length')
    parser.add_argument('--max_a_len', type=int, default=10, help='Max answer length')
    parser.add_argument('--tokenizer_name', type=str, default='vinai/bartpho-syllable', help='Tokenizer name')
    parser.add_argument('--vision_model_name', type=str, default='google/siglip-base-patch16-224', 
                        help='Vision encoder name (default: SigLIP, use "facebook/dinov2-base" for DINOv2)')
    parser.add_argument('--include_question_type', action='store_true', help='Include question type if available')
    parser.add_argument('--output_csv', type=str, default=None, help='Path to save results CSV file')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using: {device}")

    # Load checkpoint first to detect vision encoder
    print(f"\n[Checkpoint] Loading: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Detect features from checkpoint state_dict
    state_dict_keys = checkpoint['model_state_dict'].keys()
    
    # Determine vision model name
    # Since checkpoint doesn't save vision_model_name metadata yet,
    # we rely on user input (default: SigLIP)
    vision_model_name = args.vision_model_name
    print(f"[Vision] Using vision encoder: {vision_model_name}")
    
    # Determine vision processor name
    if 'siglip' in vision_model_name.lower():
        vision_processor_name = vision_model_name
        is_siglip = True
    else:
        vision_processor_name = 'facebook/dinov2-base'
        is_siglip = False
    
    print(f"[Vision] Processor: {vision_processor_name}")
    print(f"[Vision] Type: {'SigLIP 🚀' if is_siglip else 'DINOv2'}")

    # Load vision processor
    from transformers import AutoImageProcessor
    vision_processor = AutoImageProcessor.from_pretrained(vision_processor_name)

    # Load dataset
    print(f"\n[Data] Loading dataset from {args.csv_path} ...")
    dataset = VQAGenDataset(
        csv_path=args.csv_path,
        image_folder=args.image_folder,
        vision_processor=vision_processor,
        tokenizer_name=args.tokenizer_name,
        max_q_len=args.max_q_len,
        max_a_len=args.max_a_len,
        include_question_type=args.include_question_type
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"[Data] Loaded {len(dataset)} samples")
    
    # Detect model features from checkpoint
    has_vision_lora = any('lora_A' in k or 'lora_B' in k for k in state_dict_keys if 'vision' in k)
    has_text_lora = any('encoder.base_model.model' in k for k in state_dict_keys)
    
    # Detect vision gating (2 versions):
    # - OLD version: simple scalar "vision_gate" 
    # - NEW version: attention-based "vision_gating" (VisionGating class)
    has_vision_gate_old = any(k == 'vision_gate' for k in state_dict_keys)
    has_vision_gate_new = any('vision_gating' in k for k in state_dict_keys)
    has_vision_gate = has_vision_gate_old or has_vision_gate_new
    
    # Detect number of fusion layers from checkpoint
    fusion_layer_indices = set()
    for key in state_dict_keys:
        if key.startswith('flamingo_fusion.'):
            parts = key.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                fusion_layer_indices.add(int(parts[1]))
    num_fusion_layers = max(fusion_layer_indices) + 1 if fusion_layer_indices else 4
    
    print(f"\n[Model] Checkpoint features detected:")
    print(f"  • Vision Encoder: {vision_model_name}")
    print(f"  • Vision LoRA: {'YES' if has_vision_lora else 'NO ❌ (frozen)' if is_siglip else 'NO'}")
    print(f"  • Text LoRA: {'YES' if has_text_lora else 'NO'}")
    if has_vision_gate_old:
        print(f"  • Vision Gating: YES (OLD scalar version)")
    elif has_vision_gate_new:
        print(f"  • Vision Gating: YES (NEW attention-based)")
    else:
        print(f"  • Vision Gating: NO")
    print(f"  • Fusion Layers: {num_fusion_layers}")
    
    if is_siglip and has_vision_lora:
        print(f"\n⚠️  WARNING: SigLIP + Vision LoRA detected!")
        print(f"    This combination may have compatibility issues.")
        print(f"    Recommended: Use SigLIP frozen (no vision LoRA)")
    
    # Build model matching checkpoint configuration
    print(f"\n[Model] Building Deterministic VQA (matching checkpoint)...")
    model = DeterministicVQA(
        vision_model_name=vision_model_name,  # 🔥 Auto-detected
        bartpho_model_name='vinai/bartpho-syllable',  # 🔥 FIXED: was text_model_name
        num_fusion_layers=num_fusion_layers,
        num_heads=8,
        dropout=0.1,
        gradient_checkpointing=False,
        use_vision_lora=has_vision_lora,
        vision_lora_r=8,
        vision_lora_alpha=16,
        vision_lora_dropout=0.1,
        use_text_lora=has_text_lora,
        text_lora_r=16,
        text_lora_alpha=32,
        text_lora_dropout=0.1,
        use_vision_gate=has_vision_gate_new  # Only enable NEW version
    ).to(device)
    
    # Load state dict with strict=False to handle version differences
    if has_vision_gate_old:
        print(f"[Model] ⚠️  Skipping old 'vision_gate' scalar (not supported in current model)")
        missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if unexpected:
            skipped = [k for k in unexpected if 'vision_gate' in k]
            if skipped:
                print(f"[Model] Skipped keys: {skipped}")
    else:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    print(f"\n[Checkpoint] Info:")
    print(f"  • Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  • Stage: {checkpoint.get('stage', 'N/A')}")
    if 'val_loss' in checkpoint:
        print(f"  • Val loss: {checkpoint['val_loss']:.4f}")
    if 'val_exact_match' in checkpoint:
        print(f"  • Val EM: {checkpoint['val_exact_match']:.2f}%")
    if 'val_f1' in checkpoint:
        print(f"  • Val F1: {checkpoint['val_f1']:.2f}%")
    
    # Evaluate
    print(f"\n[Eval] Running evaluation...")
    results = evaluate(model, dataloader, device, model.tokenizer)
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nOverall Metrics:")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  Exact Match: {results['exact_match']:.2f}%")
    print(f"  F1 Score: {results['f1_score']:.2f}%")
    
    if results['per_type']:
        print(f"\nPer Question Type:")
        print(f"  {'Type':<12} {'EM %':<8} {'F1 %':<8} {'Count':<8}")
        print(f"  {'-'*40}")
        for q_type in sorted(results['per_type'].keys()):
            type_name = TYPE_NAMES[q_type]
            type_data = results['per_type'][q_type]
            print(f"  {type_name:<12} {type_data['exact_match']:<8.2f} {type_data['f1_score']:<8.2f} {type_data['count']:<8}")
    
    print("="*80)
    
    # Print sample predictions
    print(f"\n[Sample Predictions] (first {args.num_samples})")
    print("="*80)
    
    for i in range(min(args.num_samples, len(results['predictions']))):
        q = results['questions'][i]
        pred = results['predictions'][i]
        gt = results['ground_truths'][i]
        q_type = results['question_types'][i]
        type_name = TYPE_NAMES[q_type]
        
        em = compute_exact_match(pred, gt)
        f1 = compute_f1_score(pred, gt)
        match = "✓" if em == 1.0 else "✗"
        
        print(f"\n{i+1}. {match} [{type_name}] Q: {q}")
        print(f"   Pred: {pred}")
        print(f"   GT:   {gt}")
        print(f"   F1:   {f1:.2f}")
    
    print("\n" + "="*80)
    
    # Save results to CSV if specified
    if args.output_csv:
        import pandas as pd
        
        print(f"\n[Saving] Writing results to {args.output_csv}")
        
        results_data = []
        for i in range(len(results['predictions'])):
            q = results['questions'][i]
            pred = results['predictions'][i]
            gt = results['ground_truths'][i]
            q_type = results['question_types'][i]
            type_name = TYPE_NAMES[q_type]
            
            em = compute_exact_match(pred, gt)
            f1 = compute_f1_score(pred, gt)
            
            results_data.append({
                'question': q,
                'prediction': pred,
                'ground_truth': gt,
                'question_type': type_name,
                'exact_match': em,
                'f1_score': f1
            })
        
        df = pd.DataFrame(results_data)
        df.to_csv(args.output_csv, index=False)
        print(f"[Saved] {len(results_data)} results saved to {args.output_csv}")


if __name__ == '__main__':
    main()
