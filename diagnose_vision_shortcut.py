"""
Chẩn đoán text shortcut: model có thực sự dùng vision không?

3 tests:
  1. Blank vision  — pixel_values = zeros
  2. Noise vision  — pixel_values = random noise
  3. Normal vision — pixel_values = thật

Nếu EM(blank) ≈ EM(normal) → model đang dùng text shortcut.
Nếu EM(blank) << EM(normal) → model dùng vision thật sự.

Usage:
    python diagnose_vision_shortcut.py \
        --checkpoint checkpoints_bidirectional/best_model.pt \
        --val_csv    data/val_split.csv \
        --image_dir  data/images \
        --num_samples 200
"""

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from dataset import VQAGenDataset
from model import DeterministicVQA
from train import _normalize_vn, compute_f1_score


def eval_with_vision_mode(model, dataloader, device, mode='normal', num_samples=200):
    """
    mode: 'normal' | 'blank' | 'noise'
    """
    model.eval()
    exact_matches = []
    f1_scores = []
    n_done = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Vision={mode}', leave=False):
            pixel_values  = batch['pixel_values'].to(device)
            input_ids     = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels        = batch['labels'].to(device)

            # ── Modify vision input ──────────────────────────────────────────
            if mode == 'blank':
                pixel_values = torch.zeros_like(pixel_values)
            elif mode == 'noise':
                pixel_values = torch.randn_like(pixel_values)
                # Normalize to same mean/std as ImageNet to avoid domain shift
                pixel_values = pixel_values * 0.229 + 0.485
            # 'normal' → no change

            pixel_values = pixel_values.to(memory_format=torch.channels_last)

            predictions = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=10,
                num_beams=3,
            )

            for pred, label in zip(predictions, labels):
                label_tokens = label[label != -100].cpu().tolist()
                gt = model.tokenizer.decode(label_tokens, skip_special_tokens=True)
                em = 1.0 if _normalize_vn(pred) == _normalize_vn(gt) else 0.0
                exact_matches.append(em)
                f1_scores.append(compute_f1_score(pred, gt))

            n_done += len(predictions)
            if n_done >= num_samples:
                break

    n = len(exact_matches)
    return {
        'exact_match': sum(exact_matches) / n * 100,
        'f1_score':    sum(f1_scores)     / n * 100,
        'n': n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',  type=str, required=True)
    parser.add_argument('--val_csv',     type=str, required=True)
    parser.add_argument('--image_dir',   type=str, required=True)
    parser.add_argument('--vision_model',type=str, default='google/siglip-base-patch16-224')
    parser.add_argument('--bartpho_model',type=str,default='vinai/bartpho-syllable')
    parser.add_argument('--batch_size',  type=int, default=16)
    parser.add_argument('--num_samples', type=int, default=200,
                        help='Số samples để test (default: 200, đủ để đánh giá nhanh)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Device] {device}')

    # ── Load checkpoint ──────────────────────────────────────────────────────
    print(f'\n[Checkpoint] Loading {args.checkpoint}...')
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt.get('args', {})

    # ── Build model từ saved args ────────────────────────────────────────────
    model = DeterministicVQA(
        vision_model_name  = saved_args.get('vision_model',  args.vision_model),
        bartpho_model_name = saved_args.get('bartpho_model', args.bartpho_model),
        num_fusion_layers  = saved_args.get('num_fusion_layers', 2),
        fusion_type        = saved_args.get('fusion_type', 'text2vision'),
        num_heads          = saved_args.get('num_heads', 8),
        dropout            = saved_args.get('dropout', 0.1),
        gradient_checkpointing = False,
        use_vision_lora    = saved_args.get('use_vision_lora', False),
        vision_lora_r      = saved_args.get('vision_lora_r', 8),
        vision_lora_alpha  = saved_args.get('vision_lora_alpha', 16),
        vision_lora_dropout= saved_args.get('vision_lora_dropout', 0.1),
        use_text_lora      = saved_args.get('use_text_lora', False),
        text_lora_r        = saved_args.get('text_lora_r', 16),
        text_lora_alpha    = saved_args.get('text_lora_alpha', 32),
        text_lora_dropout  = saved_args.get('text_lora_dropout', 0.1),
        use_type_task      = saved_args.get('use_type_loss', False),
        use_logits_bias    = saved_args.get('use_logits_bias', False),
        use_vision_gate    = saved_args.get('use_vision_gate', False),
        vision_gate_init   = saved_args.get('vision_gate_init', 1.5),
        use_type_adapter   = saved_args.get('use_type_adapter', False),
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(memory_format=torch.channels_last)
    print(f'  fusion_type : {saved_args.get("fusion_type", "?")}')
    print(f'  use_vision_gate: {saved_args.get("use_vision_gate", False)}')
    print(f'  Loaded from epoch {ckpt.get("epoch", "?")}')

    # ── Dataset ──────────────────────────────────────────────────────────────
    vision_processor = AutoProcessor.from_pretrained(
        saved_args.get('vision_model', args.vision_model)
    )
    val_dataset = VQAGenDataset(
        csv_path=args.val_csv,
        image_folder=args.image_dir,
        vision_processor=vision_processor,
        tokenizer_name=saved_args.get('bartpho_model', args.bartpho_model),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    # ── Run 3 tests ──────────────────────────────────────────────────────────
    print(f'\n[Test] Running vision ablation on {args.num_samples} samples...\n')

    results = {}
    for mode in ['normal', 'blank', 'noise']:
        results[mode] = eval_with_vision_mode(
            model, val_loader, device,
            mode=mode, num_samples=args.num_samples
        )

    # ── Report ───────────────────────────────────────────────────────────────
    print('\n' + '='*55)
    print('  VISION SHORTCUT DIAGNOSIS')
    print('='*55)
    print(f"  {'Mode':<10} {'EM%':>8} {'F1%':>8}  {'Δ EM vs normal':>16}")
    print(f"  {'-'*55}")
    normal_em = results['normal']['exact_match']
    for mode, r in results.items():
        delta = r['exact_match'] - normal_em
        delta_str = f"{delta:+.2f}%" if mode != 'normal' else '—'
        print(f"  {mode:<10} {r['exact_match']:>7.2f}% {r['f1_score']:>7.2f}%  {delta_str:>16}")

    print('='*55)

    blank_drop  = normal_em - results['blank']['exact_match']
    noise_drop  = normal_em - results['noise']['exact_match']
    avg_drop = (blank_drop + noise_drop) / 2

    print('\n  VERDICT:')
    if avg_drop < 3.0:
        print(f'  🚨 STRONG TEXT SHORTCUT — vision drop={avg_drop:.1f}%')
        print(f'     Model gần như không dùng vision. Cần kiểm tra fusion/gate.')
    elif avg_drop < 10.0:
        print(f'  ⚠️  MODERATE TEXT SHORTCUT — vision drop={avg_drop:.1f}%')
        print(f'     Model dùng vision một phần, nhưng text vẫn dominant.')
    else:
        print(f'  ✅ VISION IS USED — vision drop={avg_drop:.1f}%')
        print(f'     Model phụ thuộc đáng kể vào vision input.')
    print('='*55)


if __name__ == '__main__':
    main()
