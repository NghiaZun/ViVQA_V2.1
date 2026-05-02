"""
MINIMAL EVAL FOR SIGLIP - KAGGLE COMPATIBLE
"""
import os
import unicodedata
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from collections import Counter, defaultdict

from dataset import VQAGenDataset
from model import DeterministicVQA
from dataset import detect_question_type as _detect_type_int


# Map integer type → string label for display
_TYPE_NAMES = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}


def _normalize_vn(text: str) -> str:
    """NFC normalization cho tiếng Việt — tránh false negative do byte khác nhau."""
    return unicodedata.normalize('NFC', text).strip().lower()


def _decode_gt(tokenizer, label_token_ids: list) -> str:
    """Decode ground-truth labels, filtering BOS token manually.
    BARTpho's BOS is not in all_special_ids so skip_special_tokens won't remove it."""
    ids = [t for t in label_token_ids if t != tokenizer.bos_token_id]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if _normalize_vn(prediction) == _normalize_vn(ground_truth) else 0.0


def compute_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_vn(prediction).split()
    gt_tokens   = _normalize_vn(ground_truth).split()

    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0

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


def detect_question_type(question_text: str) -> str:
    """Wrapper around dataset.detect_question_type — same logic as training."""
    return _TYPE_NAMES[_detect_type_int(question_text)]


def evaluate(model, dataloader, device, tokenizer, num_beams=3):
    model.eval()

    _INT_TO_TYPE = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}

    all_predictions = []
    all_ground_truths = []
    all_questions = []
    all_question_types = []

    total_loss = 0.0
    num_batches = 0

    exact_matches = []
    f1_scores = []

    # Per-type tracking
    type_exact_matches = defaultdict(list)
    type_f1_scores = defaultdict(list)

    # Type prediction accuracy tracking (only when model has type_head)
    type_pred_correct = []
    type_pred_per_type = defaultdict(list)  # ground_type → [correct/incorrect]

    has_type_head = getattr(model, 'use_type_task', False) and model.type_head is not None

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")

        for batch in pbar:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass (for loss)
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
                max_length=10,
                num_beams=num_beams
            )

            # Decode questions + detect types via regex (ground-truth type)
            batch_gt_types = []
            for inp in input_ids:
                question_text = tokenizer.decode(inp, skip_special_tokens=True)
                all_questions.append(question_text)
                q_type = detect_question_type(question_text)
                all_question_types.append(q_type)
                batch_gt_types.append(q_type)

            # Type prediction accuracy: compare model's predicted type vs regex type
            if has_type_head:
                text_enc = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
                text_cls = text_enc.last_hidden_state[:, 0, :]
                type_logits = model.type_head(text_cls)
                pred_type_ids = torch.argmax(type_logits, dim=-1).cpu().tolist()
                pred_type_names = [_INT_TO_TYPE[t] for t in pred_type_ids]
                for pred_t, gt_t in zip(pred_type_names, batch_gt_types):
                    correct = int(pred_t == gt_t)
                    type_pred_correct.append(correct)
                    type_pred_per_type[gt_t].append(correct)

            # Decode ground truths
            for label in labels:
                label_tokens = label[label != -100].cpu().tolist()
                gt_text = _decode_gt(tokenizer, label_tokens)
                all_ground_truths.append(gt_text)

            all_predictions.extend(predictions)

            # Metrics (overall and per-type)
            batch_start_idx = len(all_ground_truths) - len(predictions)
            for i, (pred, gt) in enumerate(zip(predictions, all_ground_truths[-len(predictions):])):
                em = compute_exact_match(pred, gt)
                f1 = compute_f1_score(pred, gt)
                q_type = all_question_types[batch_start_idx + i]

                exact_matches.append(em)
                f1_scores.append(f1)

                type_exact_matches[q_type].append(em)
                type_f1_scores[q_type].append(f1)

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

    # Per-type metrics
    per_type_results = {}
    for q_type in sorted(type_exact_matches.keys()):
        type_em = sum(type_exact_matches[q_type]) / len(type_exact_matches[q_type]) * 100 if type_exact_matches[q_type] else 0
        type_f1 = sum(type_f1_scores[q_type]) / len(type_f1_scores[q_type]) * 100 if type_f1_scores[q_type] else 0
        per_type_results[q_type] = {
            'exact_match': type_em,
            'f1_score': type_f1,
            'count': len(type_exact_matches[q_type])
        }

    # Type prediction accuracy (if model has type_head)
    type_pred_accuracy = None
    if type_pred_correct:
        overall_acc = sum(type_pred_correct) / len(type_pred_correct) * 100
        per_type_acc = {
            t: sum(v) / len(v) * 100
            for t, v in type_pred_per_type.items() if v
        }
        type_pred_accuracy = {'overall': overall_acc, 'per_type': per_type_acc}

    return {
        'loss': avg_loss,
        'exact_match': exact_match_acc,
        'f1_score': f1_score_avg,
        'per_type': per_type_results,
        'type_pred_accuracy': type_pred_accuracy,
        'predictions': all_predictions,
        'ground_truths': all_ground_truths,
        'questions': all_questions,
        'question_types': all_question_types
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--vision_model', type=str, default='google/siglip-base-patch16-224')
    parser.add_argument('--fusion_type', type=str, default=None,
                       choices=['text2vision', 'vision2text', 'bidirectional'],
                       help='Fusion type (default: auto-detect from checkpoint args)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_beams', type=int, default=3,
                        help='Beam search width (default: 3, try 5 for +0.5-1%% EM)')
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
    has_type_adapter = any('vision_adapter' in k for k in state_dict_keys)  # TypeConditionedVisionAdapter
    has_type_task = any(k.startswith('type_head') for k in state_dict_keys)   # 🔥 Type prediction head
    has_logits_bias = any(k.startswith('logits_bias') for k in state_dict_keys)  # 🔥 Type-aware logits bias
    
    # Detect fusion layers
    fusion_layer_indices = set()
    for key in state_dict_keys:
        if key.startswith('flamingo_fusion.'):
            parts = key.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                fusion_layer_indices.add(int(parts[1]))
    num_fusion_layers = max(fusion_layer_indices) + 1 if fusion_layer_indices else 4
    
    # 🔥 Detect fusion_type: CLI arg > checkpoint['args'] > fallback 'text2vision'
    saved_args = checkpoint.get('args', {})
    fusion_type = args.fusion_type or saved_args.get('fusion_type', 'text2vision')
    
    # 🔥 Detect LoRA ranks from checkpoint weights
    text_lora_r = 16  # default
    vision_lora_r = 8  # default
    
    if has_text_lora:
        # Check shape of text LoRA weight to infer rank
        for key in state_dict_keys:
            if 'encoder.base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight' in key:
                shape = checkpoint['model_state_dict'][key].shape
                text_lora_r = shape[0]  # First dim is rank
                break
    
    if has_vision_lora:
        # Check shape of vision LoRA weight to infer rank
        for key in state_dict_keys:
            if 'vision_lora_A' in key:
                shape = checkpoint['model_state_dict'][key].shape
                vision_lora_r = shape[0]  # First dim is rank
                break
    
    print(f"\nCheckpoint features:")
    print(f"  Vision LoRA: {has_vision_lora}")
    if has_vision_lora:
        print(f"    └─ Rank: {vision_lora_r}")
    print(f"  Text LoRA: {has_text_lora}")
    if has_text_lora:
        print(f"    └─ Rank: {text_lora_r}")
    print(f"  Vision Gate: {has_vision_gate}")
    print(f"  Type Adapter: {has_type_adapter}")
    print(f"  Type Task Head: {has_type_task}")       # 🔥
    print(f"  Logits Bias: {has_logits_bias}")        # 🔥
    print(f"  Fusion Layers: {num_fusion_layers}")
    print(f"  Fusion Type: {fusion_type}")
    
    # Build model
    print(f"\nBuilding model...")
    model = DeterministicVQA(
        vision_model_name=args.vision_model,
        bartpho_model_name='vinai/bartpho-syllable',
        num_fusion_layers=num_fusion_layers,
        fusion_type=fusion_type,
        num_heads=8,
        dropout=0.1,
        gradient_checkpointing=False,
        use_vision_lora=has_vision_lora,
        vision_lora_r=vision_lora_r,
        vision_lora_alpha=16,
        vision_lora_dropout=0.1,
        use_text_lora=has_text_lora,
        text_lora_r=text_lora_r,
        text_lora_alpha=32,
        text_lora_dropout=0.1,
        use_vision_gate=has_vision_gate,
        use_type_task=has_type_task,          # 🔥 was missing
        use_logits_bias=has_logits_bias,      # 🔥 was missing
        use_type_adapter=has_type_adapter,  # 🔥 NEW
        type_adapter_rank=64,  # 🔥 NEW
        type_adapter_bias=2.0  # 🔥 NEW
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"Loaded weights from epoch {checkpoint.get('epoch', 'N/A')}")
    
    # Evaluate
    print(f"\nEvaluating... (num_beams={args.num_beams})")
    results = evaluate(model, dataloader, device, model.tokenizer, num_beams=args.num_beams)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Loss: {results['loss']:.4f}")
    print(f"Exact Match: {results['exact_match']:.2f}%")
    print(f"F1 Score: {results['f1_score']:.2f}%")
    
    # Per-type breakdown
    if results.get('per_type'):
        print(f"\nPer Question Type:")
        print(f"  {'Type':<12} {'EM':<8} {'F1':<8} {'Count':<8}")
        print(f"  {'-'*40}")
        for q_type in sorted(results['per_type'].keys()):
            type_data = results['per_type'][q_type]
            print(f"  {q_type:<12} {type_data['exact_match']:<8.2f} {type_data['f1_score']:<8.2f} {type_data['count']:<8}")

    # Type prediction accuracy (TCVG quality indicator)
    if results.get('type_pred_accuracy'):
        tpa = results['type_pred_accuracy']
        print(f"\nType Prediction Accuracy (TypePredictionHead → TCVG quality):")
        print(f"  Overall: {tpa['overall']:.2f}%")
        print(f"  {'Type':<12} {'Acc':<8}")
        print(f"  {'-'*20}")
        for t in sorted(tpa['per_type'].keys()):
            print(f"  {t:<12} {tpa['per_type'][t]:<8.2f}")

    print("="*80)
    
    # Save CSV
    if args.output_csv:
        try:
            import pandas as pd
            
            # Prepare data (include question_type)
            save_data = {
                'question': results['questions'],
                'prediction': results['predictions'],
                'ground_truth': results['ground_truths'],
                'question_type': results['question_types'],
                'exact_match': [compute_exact_match(p, g) for p, g in zip(results['predictions'], results['ground_truths'])],
                'f1_score': [compute_f1_score(p, g) for p, g in zip(results['predictions'], results['ground_truths'])]
            }
            
            df = pd.DataFrame(save_data)
            
            # Ensure directory exists
            import os
            output_dir = os.path.dirname(args.output_csv) or '.'
            os.makedirs(output_dir, exist_ok=True)
            
            # Save CSV
            df.to_csv(args.output_csv, index=False, encoding='utf-8')
            
            # Verify file was saved
            if os.path.exists(args.output_csv):
                file_size = os.path.getsize(args.output_csv)
                print(f"\n✅ Saved to {args.output_csv}")
                print(f"   File size: {file_size:,} bytes ({len(df)} rows)")
            else:
                print(f"\n❌ ERROR: File was not created at {args.output_csv}")
                
        except Exception as e:
            print(f"\n❌ ERROR saving CSV: {e}")
            print(f"   Attempted path: {args.output_csv}")
            
            # Fallback: save to current directory
            try:
                fallback_path = 'results_fallback.csv'
                df.to_csv(fallback_path, index=False, encoding='utf-8')
                print(f"   Saved to fallback location: {fallback_path}")
            except Exception as e2:
                print(f"   Fallback also failed: {e2}")


if __name__ == '__main__':
    main()
