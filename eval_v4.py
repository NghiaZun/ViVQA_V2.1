"""
eval_v4.py — Enhanced eval with two features:
  1. --type_specific_beams: per-type beam config (COUNT=greedy, LOC=wider)
  2. --vocab_rerank: snap COUNT/COLOR predictions to nearest training vocab answer

Both features use ground-truth question type from test.csv (already has 'type' column).
"""
import os
import unicodedata
import difflib
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from collections import Counter, defaultdict

from dataset import VQAGenDataset
from model import DeterministicVQA
from dataset import detect_question_type as _detect_type_int


_TYPE_NAMES = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}
_TYPE_IDS   = {'OBJECT': 0, 'COUNT': 1, 'COLOR': 2, 'LOCATION': 3}

# Type-specific beam configs (active when --type_specific_beams)
TYPE_BEAM_CONFIG = {
    'COUNT':    {'num_beams': 1, 'repetition_penalty': 1.0},   # greedy — 10 simple answers
    'COLOR':    {'num_beams': 3, 'repetition_penalty': 1.3},   # standard
    'LOCATION': {'num_beams': 5, 'repetition_penalty': 1.5},   # wider search + stricter
    'OBJECT':   {'num_beams': 3, 'repetition_penalty': 1.3},   # standard
}

_SYNONYM_MAP = {
    'ngựa rằn'         : 'ngựa vằn',
    'tủ đá'            : 'tủ lạnh',
    'máy tính'         : 'laptop',
    'máy vi tính'      : 'laptop',
    'vali'             : 'hành lý',
    'hươu cao cổ khắc' : 'hươu cao cổ',
    'màu nâua'         : 'màu nâu',
    'màu nâu gấu'      : 'màu nâu',
    'nhà vệ sinh'      : 'phòng tắm',
    'đĩa'              : 'đĩa ăn',
    'tủ đông'          : 'tủ lạnh',
    'đường phố'        : 'đường',
    'đường bộ'         : 'đường',
    'nón'              : 'mũ',
    'bữa ăn tối'       : 'bữa ăn',
    'cửa tiệm'         : 'cửa hàng',
    'bữa trưa'         : 'bữa ăn',
}


def _normalize_vn(text: str, use_synonyms: bool = False) -> str:
    t = unicodedata.normalize('NFC', text).strip().lower()
    if use_synonyms:
        t = _SYNONYM_MAP.get(t, t)
    return t


def _decode_gt(tokenizer, label_token_ids: list) -> str:
    ids = [t for t in label_token_ids if t != tokenizer.bos_token_id]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def compute_exact_match(prediction: str, ground_truth: str, use_synonyms: bool = False) -> float:
    return 1.0 if _normalize_vn(prediction, use_synonyms) == _normalize_vn(ground_truth, use_synonyms) else 0.0


def compute_f1_score(prediction: str, ground_truth: str, use_synonyms: bool = False) -> float:
    pred_tokens = _normalize_vn(prediction, use_synonyms).split()
    gt_tokens   = _normalize_vn(ground_truth, use_synonyms).split()
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
    return 2 * precision * recall / (precision + recall)


def build_type_vocab(train_csv: str) -> dict:
    """Build normalized answer vocab per type from training CSV."""
    import pandas as pd
    df = pd.read_csv(train_csv)
    vocab = {}
    for type_id, type_name in _TYPE_NAMES.items():
        answers = df[df['type'] == type_id]['answer'].dropna()
        vocab[type_name] = set(
            unicodedata.normalize('NFC', a.strip().lower()) for a in answers
        )
    for name, v in vocab.items():
        print(f"  {name}: {len(v)} unique answers in training vocab")
    return vocab


def snap_to_vocab(prediction: str, q_type: str, type_vocab: dict,
                  snap_types=('COUNT', 'COLOR')) -> str:
    """If prediction not in type vocab, find closest answer by string similarity."""
    if q_type not in snap_types or q_type not in type_vocab:
        return prediction
    vocab = type_vocab[q_type]
    norm_pred = unicodedata.normalize('NFC', prediction.strip().lower())
    if norm_pred in vocab:
        return prediction  # already valid, no change
    # Snap to closest by sequence match ratio
    best = max(vocab, key=lambda v: difflib.SequenceMatcher(None, norm_pred, v).ratio())
    return best


def evaluate(model, dataloader, device, tokenizer,
             num_beams=3, repetition_penalty=1.0, max_length=20,
             use_synonyms=False, type_specific_beams=False,
             vocab_rerank=False, type_vocab=None):
    model.eval()

    all_predictions = []
    all_ground_truths = []
    all_questions = []
    all_question_types = []
    exact_matches = []
    f1_scores = []
    type_exact_matches = defaultdict(list)
    type_f1_scores = defaultdict(list)

    snap_types = ('COUNT', 'COLOR') if vocab_rerank else ()

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")

        for batch in pbar:
            pixel_values   = batch['pixel_values'].to(device)
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)
            csv_type_ids   = batch.get('question_type')  # ground-truth type from CSV

            B = pixel_values.size(0)

            # Decode question types for this batch (use CSV type column)
            batch_types = []
            for i in range(B):
                if csv_type_ids is not None:
                    q_type = _TYPE_NAMES[int(csv_type_ids[i])]
                else:
                    q_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    q_type = _TYPE_NAMES[_detect_type_int(q_text)]
                batch_types.append(q_type)
                q_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                all_questions.append(q_text)
                all_question_types.append(q_type)

            # Generate predictions (per-type beams if enabled)
            predictions = [''] * B

            if type_specific_beams:
                # Group by type, generate per-group with type-specific config
                for type_name, cfg in TYPE_BEAM_CONFIG.items():
                    type_indices = [i for i, t in enumerate(batch_types) if t == type_name]
                    if not type_indices:
                        continue
                    idx_tensor = torch.tensor(type_indices, device=device)
                    type_preds = model.generate(
                        pixel_values=pixel_values[idx_tensor],
                        input_ids=input_ids[idx_tensor],
                        attention_mask=attention_mask[idx_tensor],
                        max_length=max_length,
                        num_beams=cfg['num_beams'],
                        repetition_penalty=cfg['repetition_penalty'],
                    )
                    for local_i, global_i in enumerate(type_indices):
                        predictions[global_i] = type_preds[local_i]
            else:
                # Uniform beam config
                preds = model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                )
                predictions = list(preds)

            # Vocab reranking: snap COUNT/COLOR to nearest training answer
            if vocab_rerank and type_vocab:
                predictions = [
                    snap_to_vocab(p, t, type_vocab, snap_types)
                    for p, t in zip(predictions, batch_types)
                ]

            # Decode ground truths
            for label in labels:
                label_tokens = label[label != -100].cpu().tolist()
                gt_text = _decode_gt(tokenizer, label_tokens)
                all_ground_truths.append(gt_text)

            all_predictions.extend(predictions)

            # Metrics
            batch_start = len(all_ground_truths) - B
            for i, (pred, gt) in enumerate(zip(predictions, all_ground_truths[batch_start:])):
                q_type = batch_types[i]
                em = compute_exact_match(pred, gt, use_synonyms)
                f1 = compute_f1_score(pred, gt, use_synonyms)
                exact_matches.append(em)
                f1_scores.append(f1)
                type_exact_matches[q_type].append(em)
                type_f1_scores[q_type].append(f1)

            pbar.set_postfix({
                'EM': f"{sum(exact_matches)/len(exact_matches)*100:.1f}%",
                'F1': f"{sum(f1_scores)/len(f1_scores)*100:.1f}%"
            })

    exact_match_acc = sum(exact_matches) / len(exact_matches) * 100
    f1_score_avg    = sum(f1_scores)    / len(f1_scores)    * 100

    per_type_results = {}
    for q_type in sorted(type_exact_matches.keys()):
        per_type_results[q_type] = {
            'exact_match': sum(type_exact_matches[q_type]) / len(type_exact_matches[q_type]) * 100,
            'f1_score':    sum(type_f1_scores[q_type])    / len(type_f1_scores[q_type])    * 100,
            'count':       len(type_exact_matches[q_type]),
        }

    return {
        'exact_match': exact_match_acc,
        'f1_score':    f1_score_avg,
        'per_type':    per_type_results,
        'predictions': all_predictions,
        'ground_truths': all_ground_truths,
        'questions':   all_questions,
        'question_types': all_question_types,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',  type=str, required=True)
    parser.add_argument('--csv_path',    type=str, required=True)
    parser.add_argument('--image_folder',type=str, required=True)
    parser.add_argument('--train_csv',   type=str, default=None,
                        help='Training CSV for building answer vocab (required for --vocab_rerank)')
    parser.add_argument('--vision_model',type=str, default='google/siglip-base-patch16-224')
    parser.add_argument('--batch_size',  type=int, default=16)
    parser.add_argument('--num_beams',   type=int, default=3)
    parser.add_argument('--max_length',  type=int, default=10)
    parser.add_argument('--repetition_penalty', type=float, default=1.3)
    parser.add_argument('--use_synonyms', action='store_true')
    parser.add_argument('--type_specific_beams', action='store_true',
                        help='Use per-type beam config: COUNT=greedy, LOC=beams5, etc.')
    parser.add_argument('--vocab_rerank', action='store_true',
                        help='Snap COUNT/COLOR predictions to nearest training vocab answer')
    parser.add_argument('--output_csv', type=str, default=None)
    args = parser.parse_args()

    if args.vocab_rerank and not args.train_csv:
        parser.error('--vocab_rerank requires --train_csv')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Type-specific beams: {args.type_specific_beams}")
    print(f"Vocab rerank: {args.vocab_rerank}")

    # Build answer vocab if needed
    type_vocab = None
    if args.vocab_rerank:
        print("\nBuilding answer vocab from training CSV...")
        type_vocab = build_type_vocab(args.train_csv)

    from transformers import AutoImageProcessor
    vision_processor = AutoImageProcessor.from_pretrained(args.vision_model)

    print(f"\nLoading {args.csv_path}...")
    dataset = VQAGenDataset(
        csv_path=args.csv_path,
        image_folder=args.image_folder,
        vision_processor=vision_processor,
        tokenizer_name='vinai/bartpho-syllable',
        max_q_len=32, max_a_len=10,
        include_question_type=True,
        auto_detect_type=False,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=1, pin_memory=True)
    print(f"Loaded {len(dataset)} samples")

    # Load checkpoint (same detection logic as eval.py)
    print(f"\nLoading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict_keys = checkpoint['model_state_dict'].keys()

    has_vision_lora = any('lora_A' in k or 'lora_B' in k for k in state_dict_keys if 'vision_encoder' in k)
    has_text_lora   = any(k.startswith('encoder.base_model.model') for k in state_dict_keys)
    has_vision_gate = any('vision_gating' in k for k in state_dict_keys)
    has_delta_gate  = any('vision_gating.orig_proj' in k for k in state_dict_keys)
    has_type_task   = any(k.startswith('type_head') for k in state_dict_keys)
    has_logits_bias = any(k.startswith('logits_bias') for k in state_dict_keys)

    fusion_layer_indices = set()
    for key in state_dict_keys:
        if key.startswith('flamingo_fusion.'):
            parts = key.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                fusion_layer_indices.add(int(parts[1]))
    num_fusion_layers = max(fusion_layer_indices) + 1 if fusion_layer_indices else 4

    saved_args  = checkpoint.get('args', {})
    fusion_type = saved_args.get('fusion_type', 'text2vision')
    text_lora_r = 16
    if has_text_lora:
        for key in state_dict_keys:
            if key.startswith('encoder.base_model.model.layers.0.self_attn.q_proj.lora_A'):
                text_lora_r = checkpoint['model_state_dict'][key].shape[0]
                break

    print(f"  Text LoRA r={text_lora_r}, Vision Gate={has_vision_gate}, "
          f"Type Head={has_type_task}, Fusion={fusion_type}×{num_fusion_layers}")

    model = DeterministicVQA(
        vision_model_name=args.vision_model,
        bartpho_model_name='vinai/bartpho-syllable',
        num_fusion_layers=num_fusion_layers,
        fusion_type=fusion_type,
        num_heads=8, dropout=0.1, gradient_checkpointing=False,
        use_text_lora=has_text_lora, text_lora_r=text_lora_r,
        text_lora_alpha=32, text_lora_dropout=0.1,
        use_vision_gate=has_vision_gate,
        vision_gate_init=saved_args.get('vision_gate_init', 1.5),
        vision_gate_min_alpha=saved_args.get('vision_gate_min_alpha', 0.35),
        use_delta_gate=has_delta_gate,
        use_type_task=has_type_task,
        use_logits_bias=has_logits_bias,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"Loaded weights from epoch {checkpoint.get('epoch', 'N/A')}")

    # Describe mode
    if args.type_specific_beams:
        print("\nType-specific beam config:")
        for t, cfg in TYPE_BEAM_CONFIG.items():
            print(f"  {t}: beams={cfg['num_beams']}, rep_penalty={cfg['repetition_penalty']}")
    else:
        print(f"\nUniform: beams={args.num_beams}, rep_penalty={args.repetition_penalty}")

    results = evaluate(
        model, dataloader, device, model.tokenizer,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        use_synonyms=args.use_synonyms,
        type_specific_beams=args.type_specific_beams,
        vocab_rerank=args.vocab_rerank,
        type_vocab=type_vocab,
    )

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Exact Match: {results['exact_match']:.2f}%")
    print(f"F1 Score:    {results['f1_score']:.2f}%")

    if results.get('per_type'):
        print(f"\nPer Question Type:")
        print(f"  {'Type':<12} {'EM':<8} {'F1':<8} {'Count':<8}  vs run25")
        print(f"  {'-'*50}")
        run25 = {'COLOR': 72.5, 'COUNT': 65.8, 'LOCATION': 69.3, 'OBJECT': 75.2}
        for q_type in sorted(results['per_type'].keys()):
            d = results['per_type'][q_type]
            diff = d['exact_match'] - run25.get(q_type, 0)
            sign = '+' if diff >= 0 else ''
            print(f"  {q_type:<12} {d['exact_match']:<8.2f} {d['f1_score']:<8.2f} "
                  f"{d['count']:<8} {sign}{diff:.2f}pp")
    print("="*80)

    if args.output_csv:
        import pandas as pd
        df = pd.DataFrame({
            'question':      results['questions'],
            'prediction':    results['predictions'],
            'ground_truth':  results['ground_truths'],
            'question_type': results['question_types'],
            'exact_match':   [compute_exact_match(p, g, args.use_synonyms)
                              for p, g in zip(results['predictions'], results['ground_truths'])],
        })
        os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
        df.to_csv(args.output_csv, index=False, encoding='utf-8')
        print(f"\n✅ Saved to {args.output_csv} ({len(df)} rows)")


if __name__ == '__main__':
    main()
