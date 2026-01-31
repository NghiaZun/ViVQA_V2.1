"""
MINIMAL EVAL FOR SIGLIP - KAGGLE COMPATIBLE
"""
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from collections import Counter, defaultdict

from dataset import VQAGenDataset
from model_no_latent import DeterministicVQA


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0


def compute_f1_score(prediction: str, ground_truth: str) -> float:
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
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    total_loss = 0.0
    num_batches = 0
    
    exact_matches = []
    f1_scores = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            if outputs.total_loss is not None:
                total_loss += outputs.total_loss.item()
                num_batches += 1
            
            # Generate
            predictions = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=20,
                num_beams=3
            )
            
            # Decode
            for label in labels:
                label_tokens = label[label != -100].cpu().tolist()
                gt_text = tokenizer.decode(label_tokens, skip_special_tokens=True)
                all_ground_truths.append(gt_text)
            
            all_predictions.extend(predictions)
            
            # Metrics
            for pred, gt in zip(predictions, all_ground_truths[-len(predictions):]):
                em = compute_exact_match(pred, gt)
                f1 = compute_f1_score(pred, gt)
                exact_matches.append(em)
                f1_scores.append(f1)
            
            # Progress
            current_em = sum(exact_matches) / len(exact_matches) * 100
            current_f1 = sum(f1_scores) / len(f1_scores) * 100
            
            pbar.set_postfix({
                'loss': f"{total_loss/num_batches:.3f}",
                'EM': f"{current_em:.1f}%",
                'F1': f"{current_f1:.1f}%"
            })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    exact_match_acc = sum(exact_matches) / len(exact_matches) * 100
    f1_score_avg = sum(f1_scores) / len(f1_scores) * 100
    
    return {
        'loss': avg_loss,
        'exact_match': exact_match_acc,
        'f1_score': f1_score_avg,
        'predictions': all_predictions,
        'ground_truths': all_ground_truths
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--vision_model', type=str, default='google/siglip-base-patch16-224')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_csv', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Vision: {args.vision_model}")

    # Vision processor
    from transformers import AutoImageProcessor
    vision_processor = AutoImageProcessor.from_pretrained(args.vision_model)

    # Dataset
    print(f"\nLoading {args.csv_path}...")
    dataset = VQAGenDataset(
        csv_path=args.csv_path,
        image_folder=args.image_folder,
        vision_processor=vision_processor,
        tokenizer_name='vinai/bartpho-syllable',
        max_q_len=32,
        max_a_len=10,
        include_question_type=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"Loaded {len(dataset)} samples")
    
    # Load checkpoint
    print(f"\nLoading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict_keys = checkpoint['model_state_dict'].keys()
    
    # Detect features
    has_vision_lora = any('lora_A' in k or 'lora_B' in k for k in state_dict_keys if 'vision' in k)
    has_text_lora = any('encoder.base_model.model' in k for k in state_dict_keys)
    has_vision_gate = any('vision_gating' in k for k in state_dict_keys)
    
    # Detect fusion layers
    fusion_layer_indices = set()
    for key in state_dict_keys:
        if key.startswith('flamingo_fusion.'):
            parts = key.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                fusion_layer_indices.add(int(parts[1]))
    num_fusion_layers = max(fusion_layer_indices) + 1 if fusion_layer_indices else 4
    
    print(f"\nCheckpoint features:")
    print(f"  Vision LoRA: {has_vision_lora}")
    print(f"  Text LoRA: {has_text_lora}")
    print(f"  Vision Gate: {has_vision_gate}")
    print(f"  Fusion Layers: {num_fusion_layers}")
    
    # Build model
    print(f"\nBuilding model...")
    model = DeterministicVQA(
        vision_model_name=args.vision_model,
        bartpho_model_name='vinai/bartpho-syllable',
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
        use_vision_gate=has_vision_gate
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"Loaded weights from epoch {checkpoint.get('epoch', 'N/A')}")
    
    # Evaluate
    print(f"\nEvaluating...")
    results = evaluate(model, dataloader, device, model.tokenizer)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Loss: {results['loss']:.4f}")
    print(f"Exact Match: {results['exact_match']:.2f}%")
    print(f"F1 Score: {results['f1_score']:.2f}%")
    print("="*80)
    
    # Save CSV
    if args.output_csv:
        import pandas as pd
        df = pd.DataFrame({
            'prediction': results['predictions'],
            'ground_truth': results['ground_truths'],
            'exact_match': [compute_exact_match(p, g) for p, g in zip(results['predictions'], results['ground_truths'])],
            'f1_score': [compute_f1_score(p, g) for p, g in zip(results['predictions'], results['ground_truths'])]
        })
        df.to_csv(args.output_csv, index=False)
        print(f"\nSaved to {args.output_csv}")


if __name__ == '__main__':
    main()
