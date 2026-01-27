"""
EVALUATION SCRIPT FOR DETERMINISTIC VQA (No Latent)
====================================================

Simple evaluation with multiple metrics:
- Exact Match (EM)
- F1 Score (partial credit)
- ROUGE-1 (unigram overlap)
- ROUGE-L (longest common subsequence)

Version 2.1 with ROUGE metrics!
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from collections import Counter

from dataset import VQAGenDataset
from model_no_latent import DeterministicVQA

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠️  Warning: rouge_score not installed. ROUGE metrics will be skipped.")
    print("   Install with: pip install rouge-score")


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


def compute_rouge_scores(prediction: str, ground_truth: str) -> dict:
    """
    Compute ROUGE-1 and ROUGE-L scores
    
    ROUGE-1: Unigram overlap (word-level similarity)
    ROUGE-L: Longest common subsequence (fluency/order)
    
    Returns:
        dict with 'rouge1' and 'rougeL' F1 scores (0-1 range)
    """
    if not ROUGE_AVAILABLE:
        return {'rouge1': 0.0, 'rougeL': 0.0}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)
    scores = scorer.score(ground_truth, prediction)
    
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }


def evaluate(model, dataloader, device, tokenizer):
    """Evaluate model on validation set with EM + F1 + ROUGE metrics"""
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    all_questions = []
    
    total_loss = 0.0
    num_batches = 0
    
    exact_matches = []
    f1_scores = []
    rouge1_scores = []
    rougeL_scores = []
    
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
            
            # Generate predictions (now with REAL beam search!)
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
            
            # Decode questions
            for inp in input_ids:
                question_text = tokenizer.decode(inp, skip_special_tokens=True)
                all_questions.append(question_text)
            
            all_predictions.extend(predictions)
            
            # Compute metrics for this batch
            for pred, gt in zip(predictions, all_ground_truths[-len(predictions):]):
                em = compute_exact_match(pred, gt)
                f1 = compute_f1_score(pred, gt)
                rouge_scores = compute_rouge_scores(pred, gt)
                
                exact_matches.append(em)
                f1_scores.append(f1)
                rouge1_scores.append(rouge_scores['rouge1'])
                rougeL_scores.append(rouge_scores['rougeL'])
            
            # Update progress
            current_em = sum(exact_matches) / len(exact_matches) * 100
            current_f1 = sum(f1_scores) / len(f1_scores) * 100
            
            pbar_dict = {
                'loss': f"{total_loss/num_batches:.3f}",
                'EM': f"{current_em:.1f}%",
                'F1': f"{current_f1:.1f}%"
            }
            
            if ROUGE_AVAILABLE and rouge1_scores:
                current_r1 = sum(rouge1_scores) / len(rouge1_scores) * 100
                current_rl = sum(rougeL_scores) / len(rougeL_scores) * 100
                pbar_dict['R1'] = f"{current_r1:.1f}%"
                pbar_dict['RL'] = f"{current_rl:.1f}%"
            
            pbar.set_postfix(pbar_dict)
    
    # Compute final metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    exact_match_acc = sum(exact_matches) / len(exact_matches) * 100
    f1_score_avg = sum(f1_scores) / len(f1_scores) * 100
    
    results = {
        'loss': avg_loss,
        'exact_match': exact_match_acc,
        'f1_score': f1_score_avg,
        'predictions': all_predictions,
        'ground_truths': all_ground_truths,
        'questions': all_questions
    }
    
    if ROUGE_AVAILABLE and rouge1_scores:
        results['rouge1'] = sum(rouge1_scores) / len(rouge1_scores) * 100
        results['rougeL'] = sum(rougeL_scores) / len(rougeL_scores) * 100
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Deterministic VQA')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file (val/test)')
    parser.add_argument('--image_folder', type=str, required=True, help='Path to image folder')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to print')
    parser.add_argument('--max_q_len', type=int, default=32, help='Max question length')
    parser.add_argument('--max_a_len', type=int, default=10, help='Max answer length')
    parser.add_argument('--tokenizer_name', type=str, default='vinai/bartpho-syllable', help='Tokenizer name')
    parser.add_argument('--vision_processor_name', type=str, default='facebook/dinov2-base', help='Vision processor name')
    parser.add_argument('--include_question_type', action='store_true', help='Include question type if available')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using: {device}")

    from transformers import AutoImageProcessor
    vision_processor = AutoImageProcessor.from_pretrained(args.vision_processor_name)

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
    
    # Load checkpoint first to check configuration
    print(f"\n[Model] Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Detect if checkpoint has LoRA/vision gating
    has_vision_lora = any('lora_A' in k or 'lora_B' in k for k in checkpoint['model_state_dict'].keys())
    has_vision_gate = any('vision_gating' in k for k in checkpoint['model_state_dict'].keys())
    
    print(f"[Model] Checkpoint features detected:")
    print(f"  • Vision LoRA: {'YES' if has_vision_lora else 'NO'}")
    print(f"  • Vision Gating: {'YES' if has_vision_gate else 'NO'}")
    
    # Build model matching checkpoint configuration
    print(f"\n[Model] Building Deterministic VQA (matching checkpoint)...")
    model = DeterministicVQA(
        dinov2_model_name='facebook/dinov2-base',
        bartpho_model_name='vinai/bartpho-syllable',
        num_fusion_layers=2,
        num_heads=8,
        dropout=0.1,
        gradient_checkpointing=False,
        use_vision_lora=has_vision_lora,  # 🔥 Auto-detect LoRA
        vision_lora_r=8,
        vision_lora_alpha=16,
        vision_lora_dropout=0.1,
        use_vision_gate=has_vision_gate  # 🔥 Auto-detect vision gating
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"[Model] Checkpoint info:")
    print(f"  • Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  • Stage: {checkpoint.get('stage', 'N/A')}")
    print(f"  • Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    # Evaluate
    print(f"\n[Eval] Running evaluation...")
    results = evaluate(model, dataloader, device, model.tokenizer)
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  Exact Match: {results['exact_match']:.2f}%")
    print(f"  F1 Score: {results['f1_score']:.2f}%")
    
    if 'rouge1' in results:
        print(f"  ROUGE-1: {results['rouge1']:.2f}%")
        print(f"  ROUGE-L: {results['rougeL']:.2f}%")
    elif not ROUGE_AVAILABLE:
        print(f"  ROUGE metrics: N/A (install rouge-score)")
    
    print("="*80)
    
    # Print sample predictions
    print(f"\n[Sample Predictions] (first {args.num_samples})")
    print("="*80)
    
    for i in range(min(args.num_samples, len(results['predictions']))):
        q = results['questions'][i]
        pred = results['predictions'][i]
        gt = results['ground_truths'][i]
        
        em = compute_exact_match(pred, gt)
        f1 = compute_f1_score(pred, gt)
        match = "✓" if em == 1.0 else "✗"
        
        print(f"\n{i+1}. {match} Q: {q}")
        print(f"   Pred: {pred}")
        print(f"   GT:   {gt}")
        print(f"   F1:   {f1:.2f}", end="")
        
        if ROUGE_AVAILABLE:
            rouge_scores = compute_rouge_scores(pred, gt)
            print(f" | ROUGE-1: {rouge_scores['rouge1']:.2f} | ROUGE-L: {rouge_scores['rougeL']:.2f}")
        else:
            print()
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
