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


_SYNONYM_MAP = {
    # --- existing ---
    'ngựa rằn'         : 'ngựa vằn',
    'tủ đá'            : 'tủ lạnh',
    'máy tính'         : 'laptop',
    'máy vi tính'      : 'laptop',
    'vali'             : 'hành lý',
    # --- label noise / dataset inconsistency ---
    'hươu cao cổ khắc' : 'hươu cao cổ',   # 9x "khắc" thừa trong label
    'màu nâua'         : 'màu nâu',        # 2x typo
    'màu nâu gấu'      : 'màu nâu',        # 2x label noise
    # --- Vietnamese synonyms ---
    'nhà vệ sinh'      : 'phòng tắm',      # 9x toilet=bathroom
    'đĩa'              : 'đĩa ăn',         # 3x plate=dish
    'tủ đông'          : 'tủ lạnh',        # 3x freezer=fridge
    'đường phố'        : 'đường',          # 2x street=road
    'đường bộ'         : 'đường',          # 2x road=road
    # --- v3 additions (confirmed from run25 error analysis) ---
    'nón'              : 'mũ',             # 3x headwear synonyms
    'bữa ăn tối'       : 'bữa ăn',        # 2x dinner≈meal in VQA context
    'cửa tiệm'         : 'cửa hàng',       # same meaning: store/shop
    'bữa trưa'         : 'bữa ăn',        # 1x lunch≈meal
}


def _normalize_vn(text: str, use_synonyms: bool = False) -> str:
    """NFC normalization cho tiếng Việt — tránh false negative do byte khác nhau."""
    t = unicodedata.normalize('NFC', text).strip().lower()
    if use_synonyms:
        t = _SYNONYM_MAP.get(t, t)
    return t


def _decode_gt(tokenizer, label_token_ids: list) -> str:
    """Decode ground-truth labels, filtering BOS token manually.
    BARTpho's BOS is not in all_special_ids so skip_special_tokens won't remove it."""
    ids = [t for t in label_token_ids if t != tokenizer.bos_token_id]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def compute_exact_match(prediction: str, ground_truth: str, use_synonyms: bool = False) -> float:
    return 1.0 if _normalize_vn(prediction, use_synonyms) == _normalize_vn(ground_truth, use_synonyms) else 0.0


def compute_prf(prediction: str, ground_truth: str, use_synonyms: bool = False,
                token_mode: str = 'set') -> tuple:
    """Per-sample (precision, recall, f1) theo docs/vivqa_metrics.md.

    token_mode:
      'set'      — trùng token theo TẬP HỢP: |P ∩ G| / |P|, |P ∩ G| / |G|
                   (đúng ký hiệu ∩ trong paper). MẶC ĐỊNH.
      'multiset' — kiểu SQuAD-F1 (Counter, đếm cả token lặp) — số cũ tương thích ngược.

    Edge cases khớp spec: cả hai rỗng → (1,1,1); một bên rỗng → (0,0,0);
    không overlap → (0,0,0). F1 = 2PR/(P+R), average ở mức từng câu (macro).
    """
    pred_tokens = _normalize_vn(prediction, use_synonyms).split()
    gt_tokens   = _normalize_vn(ground_truth, use_synonyms).split()

    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0, 0.0, 0.0

    if token_mode == 'multiset':
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        denom_p, denom_g = len(pred_tokens), len(gt_tokens)
    else:  # 'set' — công thức paper
        pred_set, gt_set = set(pred_tokens), set(gt_tokens)
        num_same = len(pred_set & gt_set)
        denom_p, denom_g = len(pred_set), len(gt_set)

    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / denom_p
    recall = num_same / denom_g
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def compute_f1_score(prediction: str, ground_truth: str, use_synonyms: bool = False,
                     token_mode: str = 'set') -> float:
    """F1 per-sample. Wrapper mỏng quanh compute_prf để tương thích ngược."""
    return compute_prf(prediction, ground_truth, use_synonyms, token_mode)[2]


def detect_question_type(question_text: str) -> str:
    """Wrapper around dataset.detect_question_type — same logic as training."""
    return _TYPE_NAMES[_detect_type_int(question_text)]


def build_valid_answers_set(train_csv):
    """Build set of NFC-normalized lowercase valid answers (for snap post-processing)."""
    import pandas as pd
    df = pd.read_csv(train_csv)
    return {unicodedata.normalize('NFC', str(a).strip().lower()) for a in df['answer'].unique()}


def snap_to_valid_answer(pred, valid_answers_set):
    """Fix garbled constrained-decoding outputs: if pred not in trie vocab,
    snap to the longest valid answer that is a character-level prefix of pred.

    Root cause: BARTpho SentencePiece sometimes concatenates syllables without
    spaces at answer boundaries, producing 'cái ghếa', 'diềuván lướt sóng', etc.
    These are NOT in the training trie, but their valid prefix IS (e.g. 'cái ghế').
    """
    pred_n = unicodedata.normalize('NFC', pred.strip().lower())
    if pred_n in valid_answers_set:
        return pred.strip()
    candidates = [a for a in valid_answers_set if pred_n.startswith(a) and len(a) > 0]
    if candidates:
        return max(candidates, key=len)
    return pred.strip()


def build_answer_trie(train_csv, tokenizer):
    """Build prefix trie from all unique answers in training CSV.

    IMPORTANT: Dataset encodes answers WITH special tokens (add_special_tokens=True),
    so labels = [BOS=0, tokens..., EOS=2, PAD=-100...].
    After shift_tokens_right, model learns: [BOS] → BOS → actual_tokens → EOS.
    The trie must match this protocol: include BOS=0 as mandatory first token.
    """
    import pandas as pd
    df = pd.read_csv(train_csv)
    pad_id = tokenizer.pad_token_id
    trie = {}
    for answer in df['answer'].unique():
        # encode WITH special tokens to match training label format:
        # [BOS=0, answer_tokens..., EOS=2]
        tokens = tokenizer.encode(str(answer))
        tokens = [t for t in tokens if t != pad_id]  # strip padding just in case
        node = trie
        for t in tokens:
            if t not in node:
                node[t] = {}
            node = node[t]
    print(f"[Trie] Built from {len(df['answer'].unique())} unique answers ({len(df)} total rows)")
    return trie


def evaluate(model, dataloader, device, tokenizer, num_beams=3, repetition_penalty=1.0,
             max_length=20, use_synonyms=False, num_samples=1, vote_temp=0.8,
             prefix_trie=None, valid_answers_set=None, token_mode='set'):
    model.eval()

    _INT_TO_TYPE = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}

    all_predictions = []
    all_ground_truths = []
    all_questions = []
    all_question_types = []

    exact_matches = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # Per-type tracking
    type_exact_matches = defaultdict(list)
    type_precision_scores = defaultdict(list)
    type_recall_scores = defaultdict(list)
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

            # Generate (encoder runs once inside here)
            gen_out = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                return_type_preds=has_type_head,
                num_samples=num_samples,
                vote_temp=vote_temp,
                prefix_trie=prefix_trie,
            )
            if has_type_head:
                predictions, pred_type_ids = gen_out
            else:
                predictions = gen_out
                pred_type_ids = None

            # Ground-truth type: from CSV column (same source as training)
            batch_gt_types = []
            csv_type_ids = batch.get('question_type')
            for i, inp in enumerate(input_ids):
                question_text = tokenizer.decode(inp, skip_special_tokens=True)
                all_questions.append(question_text)
                if csv_type_ids is not None:
                    q_type = _INT_TO_TYPE[int(csv_type_ids[i])]
                else:
                    q_type = detect_question_type(question_text)
                all_question_types.append(q_type)
                batch_gt_types.append(q_type)

            # Type prediction accuracy: use type preds already computed in generate()
            if has_type_head and pred_type_ids is not None:
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

            # Snap garbled predictions back to nearest valid trie answer
            if valid_answers_set is not None:
                predictions = [snap_to_valid_answer(p, valid_answers_set) for p in predictions]

            all_predictions.extend(predictions)

            # Metrics (overall and per-type)
            batch_start_idx = len(all_ground_truths) - len(predictions)
            for i, (pred, gt) in enumerate(zip(predictions, all_ground_truths[-len(predictions):])):
                em = compute_exact_match(pred, gt, use_synonyms)
                precision, recall, f1 = compute_prf(pred, gt, use_synonyms, token_mode)
                q_type = all_question_types[batch_start_idx + i]

                exact_matches.append(em)
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)

                type_exact_matches[q_type].append(em)
                type_precision_scores[q_type].append(precision)
                type_recall_scores[q_type].append(recall)
                type_f1_scores[q_type].append(f1)

            current_em = sum(exact_matches) / len(exact_matches) * 100
            current_f1 = sum(f1_scores) / len(f1_scores) * 100

            pbar.set_postfix({
                'EM': f"{current_em:.1f}%",
                'F1': f"{current_f1:.1f}%"
            })

    exact_match_acc = sum(exact_matches) / len(exact_matches) * 100
    precision_avg = sum(precision_scores) / len(precision_scores) * 100
    recall_avg = sum(recall_scores) / len(recall_scores) * 100
    f1_score_avg = sum(f1_scores) / len(f1_scores) * 100

    # Per-type metrics
    per_type_results = {}
    for q_type in sorted(type_exact_matches.keys()):
        n_type = len(type_exact_matches[q_type])
        type_em = sum(type_exact_matches[q_type]) / n_type * 100 if n_type else 0
        type_p  = sum(type_precision_scores[q_type]) / n_type * 100 if n_type else 0
        type_r  = sum(type_recall_scores[q_type]) / n_type * 100 if n_type else 0
        type_f1 = sum(type_f1_scores[q_type]) / n_type * 100 if n_type else 0
        per_type_results[q_type] = {
            'exact_match': type_em,
            'precision': type_p,
            'recall': type_r,
            'f1_score': type_f1,
            'count': n_type
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
        'exact_match': exact_match_acc,
        'precision': precision_avg,
        'recall': recall_avg,
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
                        help='Beam search width (default: 3, beam=5 is slower and not better)')
    parser.add_argument('--max_length', type=int, default=20,
                        help='Max generated tokens (default: 20, matches older eval)')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help='Repetition penalty to suppress repeated tokens (default: 1.0)')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Majority vote: sample N sequences, pick most frequent. 1=off (default)')
    parser.add_argument('--vote_temp', type=float, default=0.8,
                        help='Temperature for majority vote sampling (default: 0.8)')
    parser.add_argument('--use_synonyms', action='store_true',
                        help='Apply synonym normalization before computing EM/F1')
    parser.add_argument('--token_mode', type=str, default='set', choices=['set', 'multiset'],
                        help="Token overlap cho Precision/Recall/F1: 'set'=công thức paper (mặc định), "
                             "'multiset'=SQuAD-style đếm token lặp (số cũ, tương thích ngược)")
    parser.add_argument('--use_constrained', action='store_true',
                        help='Constrained beam search: only generate tokens in training answer trie')
    parser.add_argument('--train_csv_for_trie', type=str, default='archive/train_split.csv',
                        help='Train CSV to build answer trie for constrained decoding')
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
        include_question_type=True,
        auto_detect_type=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    print(f"Loaded {len(dataset)} samples")
    
    # Load checkpoint
    print(f"\nLoading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict_keys = checkpoint['model_state_dict'].keys()
    
    # Detect features
    has_vision_lora = any('lora_A' in k or 'lora_B' in k for k in state_dict_keys if 'vision_encoder' in k)
    # Exclude vision_encoder keys to avoid false positive when vision LoRA key contains 'encoder.base_model.model'
    has_text_lora = any(k.startswith('encoder.base_model.model') for k in state_dict_keys)
    has_vision_gate = any('vision_gating' in k for k in state_dict_keys)
    has_delta_gate = any('vision_gating.orig_proj' in k for k in state_dict_keys)
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
    
    # 🔥 Detect LoRA ranks and alphas from checkpoint weights + saved_args
    text_lora_r = 16  # default
    vision_lora_r = 8  # default
    text_lora_alpha = saved_args.get('text_lora_alpha', 32)
    vision_lora_alpha = saved_args.get('vision_lora_alpha', 16)

    if has_text_lora:
        for key in state_dict_keys:
            # Must start with 'encoder.' to exclude vision_encoder.encoder.base_model.model... keys
            if key.startswith('encoder.base_model.model.layers.0.self_attn.q_proj.lora_A'):
                shape = checkpoint['model_state_dict'][key].shape
                text_lora_r = shape[0]
                break

    if has_vision_lora:
        # New format: vision_encoder.encoder.base_model.model.layers.X.self_attn.*.lora_A.*
        # Fallback to saved_args if key pattern not matched
        for key in state_dict_keys:
            if 'vision_encoder' in key and 'lora_A' in key:
                shape = checkpoint['model_state_dict'][key].shape
                vision_lora_r = shape[0]
                break
        else:
            vision_lora_r = saved_args.get('vision_lora_r', 8)
    
    print(f"\nCheckpoint features:")
    print(f"  Vision LoRA: {has_vision_lora}")
    if has_vision_lora:
        print(f"    └─ Rank: {vision_lora_r}")
    print(f"  Text LoRA: {has_text_lora}")
    if has_text_lora:
        print(f"    └─ Rank: {text_lora_r}")
    print(f"  Vision Gate: {has_vision_gate} (delta={has_delta_gate})")
    print(f"  Type Adapter: {has_type_adapter}")
    print(f"  Type Task Head: {has_type_task}")       # 🔥
    print(f"  Logits Bias: {has_logits_bias}")        # 🔥
    # Auto-detect architecture flags from saved checkpoint args
    use_siglip_pooler = saved_args.get('use_siglip_pooler', False)
    use_mean_pool_cls = saved_args.get('use_mean_pool_cls', False)
    use_attn_pool_cls = saved_args.get('use_attn_pool_cls', False)
    vision_gate_max_alpha = saved_args.get('vision_gate_max_alpha', 1.0)
    use_type_text_adapter = saved_args.get('use_type_text_adapter', False)
    type_text_adapter_bottleneck = saved_args.get('type_text_adapter_bottleneck', 64)

    print(f"  Fusion Layers: {num_fusion_layers}")
    print(f"  Fusion Type: {fusion_type}")
    print(f"  SigLIP pooler token: {use_siglip_pooler}")
    print(f"  Mean-pool text cls: {use_mean_pool_cls}")
    print(f"  Attn-pool text cls: {use_attn_pool_cls}")
    print(f"  Gate max alpha: {vision_gate_max_alpha}")
    print(f"  TypeSpecificTextAdapter: {use_type_text_adapter} (bottleneck={type_text_adapter_bottleneck})")
    print(f"  Text LoRA: r={text_lora_r}, alpha={text_lora_alpha}")
    print(f"  Vision LoRA: r={vision_lora_r}, alpha={vision_lora_alpha}")

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
        vision_lora_alpha=vision_lora_alpha,
        vision_lora_dropout=0.1,
        use_text_lora=has_text_lora,
        text_lora_r=text_lora_r,
        text_lora_alpha=text_lora_alpha,
        text_lora_dropout=0.1,
        use_vision_gate=has_vision_gate,
        vision_gate_init=saved_args.get('vision_gate_init', 1.5),
        vision_gate_min_alpha=saved_args.get('vision_gate_min_alpha', 0.35),
        vision_gate_max_alpha=vision_gate_max_alpha,
        use_delta_gate=has_delta_gate,
        use_type_task=has_type_task,
        use_logits_bias=has_logits_bias,
        use_type_adapter=has_type_adapter,
        type_adapter_rank=64,
        type_adapter_bias=2.0,
        use_siglip_pooler=use_siglip_pooler,
        use_mean_pool_cls=use_mean_pool_cls,
        use_attn_pool_cls=use_attn_pool_cls,
        use_type_text_adapter=use_type_text_adapter,
        type_text_adapter_bottleneck=type_text_adapter_bottleneck,
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"Loaded weights from epoch {checkpoint.get('epoch', 'N/A')}")
    
    # Evaluate
    mode_str = (f"majority_vote n={args.num_samples} temp={args.vote_temp}"
                if args.num_samples > 1 else f"num_beams={args.num_beams}")
    print(f"\nEvaluating... ({mode_str}, max_length={args.max_length}, "
          f"synonyms={'on' if args.use_synonyms else 'off'})")
    prefix_trie = None
    valid_answers_set = None
    if args.use_constrained:
        prefix_trie = build_answer_trie(args.train_csv_for_trie, model.tokenizer)
        valid_answers_set = build_valid_answers_set(args.train_csv_for_trie)

    results = evaluate(model, dataloader, device, model.tokenizer,
                       num_beams=args.num_beams, repetition_penalty=args.repetition_penalty,
                       max_length=args.max_length, use_synonyms=args.use_synonyms,
                       num_samples=args.num_samples, vote_temp=args.vote_temp,
                       prefix_trie=prefix_trie, valid_answers_set=valid_answers_set,
                       token_mode=args.token_mode)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    # NOTE: dòng "Exact Match:" và "F1 Score:" là CONTRACT được evaluator.py parse
    # bằng regex — giữ nguyên chuỗi/format. Precision/Recall thêm ở dòng riêng, an toàn.
    print(f"Exact Match: {results['exact_match']:.2f}%")
    print(f"Precision: {results['precision']:.2f}%")
    print(f"Recall: {results['recall']:.2f}%")
    print(f"F1 Score: {results['f1_score']:.2f}%")
    print(f"(token_mode={args.token_mode})")

    # Per-type breakdown — EM & F1 phải là 2 cột số ĐẦU TIÊN sau tên type
    # (evaluator.py regex đọc 2 số đầu = EM, F1). Prec/Rec đặt SAU F1.
    if results.get('per_type'):
        print(f"\nPer Question Type:")
        print(f"  {'Type':<12} {'EM':<8} {'F1':<8} {'Prec':<8} {'Rec':<8} {'Count':<8}")
        print(f"  {'-'*56}")
        for q_type in sorted(results['per_type'].keys()):
            type_data = results['per_type'][q_type]
            print(f"  {q_type:<12} {type_data['exact_match']:<8.2f} {type_data['f1_score']:<8.2f} "
                  f"{type_data['precision']:<8.2f} {type_data['recall']:<8.2f} {type_data['count']:<8}")

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
            
            # Prepare data (include question_type). Tính (P,R,F1) một lần mỗi dòng.
            _prf_rows = [compute_prf(p, g, use_synonyms=args.use_synonyms, token_mode=args.token_mode)
                         for p, g in zip(results['predictions'], results['ground_truths'])]
            save_data = {
                'question': results['questions'],
                'prediction': results['predictions'],
                'ground_truth': results['ground_truths'],
                'question_type': results['question_types'],
                'exact_match': [compute_exact_match(p, g, use_synonyms=args.use_synonyms) for p, g in zip(results['predictions'], results['ground_truths'])],
                'precision': [r[0] for r in _prf_rows],
                'recall': [r[1] for r in _prf_rows],
                'f1_score': [r[2] for r in _prf_rows],
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