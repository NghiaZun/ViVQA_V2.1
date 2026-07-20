#!/usr/bin/env python3
"""
probe.py — Deep-dive activation analysis for ViVQA model.

Traces how vectors change at every stage of the forward pass and compares
statistics between correct vs incorrect predictions per question type.

Capture points:
  [1] SigLIP patch tokens (raw, before projection)
  [2] SigLIP global token (pooler_output projected, if use_siglip_pooler)
  [3] After vision_proj  (patches in bart_hidden space)
  [4] After each Flamingo layer  (how much does cross-attention shift patches?)
  [5] Gate α per patch  (which patches are amplified / suppressed per type?)
  [6] text_cls  (mean-pool vs BOS norm and type separability)
  [7] vision↔text cosine alignment  (pre/post Flamingo)
  [8] Decoder first-token confidence

Usage:
    python3 probe.py \\
        --checkpoint checkpoints_run69/best_model.pt \\
        --csv_path archive/val_split.csv \\
        --image_folder archive/data/images/train \\
        --output_dir probe_run69/ \\
        [--n_samples 500] [--batch_size 8] [--device cuda]
"""

import argparse, sys, os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model import DeterministicVQA
from dataset import VQAGenDataset
from transformers import AutoProcessor

TYPE_NAMES = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}
TYPE_IDS   = {'object': 0, 'counting': 1, 'color': 2, 'location': 3}

# ─────────────────────────────────────────────────────────────────────────────
# Hook registry
# ─────────────────────────────────────────────────────────────────────────────

class ActivationStore:
    """Thread-safe (single-process) store for per-batch hook outputs."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.data = defaultdict(list)

    def record(self, key, tensor):
        # Detach, move to CPU, store as float32 numpy
        self.data[key].append(tensor.detach().float().cpu())

    def get(self, key):
        """Return stacked tensor for this key, or None."""
        vals = self.data.get(key)
        if not vals:
            return None
        return torch.cat(vals, dim=0)  # [total_samples, ...]


store = ActivationStore()


def register_hooks(model):
    """Register forward hooks on key modules. Returns list of hook handles."""
    handles = []

    # [1] SigLIP raw patch tokens (after CLS removal happens in forward(), so hook output
    #     of vision_encoder and strip CLS in post-processing)
    def hook_vision_encoder(module, inp, out):
        # out.last_hidden_state: [B, 197, 768]  (197 = 196 patches + 1 CLS)
        patches = out.last_hidden_state[:, 1:, :]   # remove CLS → [B, 196, 768]
        store.record('raw_patch_norm_mean', patches.norm(dim=-1).mean(dim=-1))   # [B]
        store.record('raw_patch_norm_std',  patches.norm(dim=-1).std(dim=-1))    # [B]
        # Global token: pooler_output [B, 768] if available
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            store.record('siglip_pooler_norm', out.pooler_output.norm(dim=-1))   # [B]
    handles.append(model.vision_encoder.register_forward_hook(hook_vision_encoder))

    # [2] Global token after projection to bart_hidden (only if siglip_pooler is used)
    if model.use_siglip_pooler and hasattr(model, 'siglip_global_proj'):
        def hook_global_proj(module, inp, out):
            # out: [B, 1024]
            store.record('global_proj_norm', out.norm(dim=-1))                   # [B]
        handles.append(model.siglip_global_proj.register_forward_hook(hook_global_proj))

    # [3] vision_proj output (patches in bart_hidden space, before Flamingo)
    def hook_vision_proj(module, inp, out):
        # out: [B, P, 1024]  P=196 or 197
        store.record('vision_proj_norm_mean', out.norm(dim=-1).mean(dim=-1))     # [B]
        store.record('vision_proj_input',     inp[0].detach().float().cpu())     # [B, P, ...]  full tensor
    handles.append(model.vision_proj.register_forward_hook(hook_vision_proj))

    # [4] Flamingo layers — capture fused_vision output of each layer
    for i, layer in enumerate(model.flamingo_fusion):
        def make_flamingo_hook(idx):
            def hook_flamingo(module, inp, out):
                # out: (fused_vision [B,P,D], fused_text [B,L,D])
                fv, ft = out
                store.record(f'flamingo_{idx}_vision_norm_mean', fv.norm(dim=-1).mean(dim=-1))  # [B]
                store.record(f'flamingo_{idx}_text_norm_mean',   ft.norm(dim=-1).mean(dim=-1))  # [B]
                # Δ magnitude: how much Flamingo changed patches (captured relative to vision_proj)
                store.record(f'flamingo_{idx}_vision_out', fv.detach().float().cpu())           # [B,P,D]
            return hook_flamingo
        handles.append(layer.register_forward_hook(make_flamingo_hook(i)))

    # [5] VisionGating — capture gate α per patch
    def hook_gate(module, inp, out):
        gated_vision, alpha = out   # alpha: [B, P]
        store.record('gate_alpha_mean',   alpha.mean(dim=-1))    # [B]
        store.record('gate_alpha_std',    alpha.std(dim=-1))     # [B]
        store.record('gate_alpha_max',    alpha.max(dim=-1).values)  # [B]
        store.record('gate_alpha_min',    alpha.min(dim=-1).values)  # [B]
        # Top-5 patch indices (spatial attention peaks)
        topk_idx = alpha.topk(5, dim=-1).indices.detach().float().cpu()  # [B, 5]
        store.record('gate_alpha_top5_idx', topk_idx)
        store.record('gate_alpha_all',      alpha.detach().float().cpu())  # [B, P]
    handles.append(model.vision_gating.register_forward_hook(hook_gate))

    # [6] Text encoder output — for text_cls analysis
    def hook_text_encoder(module, inp, out):
        hidden = out.last_hidden_state   # [B, L, D]
        # BOS token norm ([:, 0, :])
        bos_norm = hidden[:, 0, :].norm(dim=-1)  # [B]
        # Mean of content tokens (exclude padding via attention_mask)
        attn_mask = inp[1] if len(inp) > 1 else None
        if attn_mask is not None:
            mask = attn_mask.float().unsqueeze(-1)  # [B, L, 1]
            mean_pool = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)  # [B, D]
        else:
            mean_pool = hidden.mean(1)
        store.record('text_bos_norm',      bos_norm)                           # [B]
        store.record('text_meanpool_norm', mean_pool.norm(dim=-1))             # [B]
        # Cosine similarity between BOS and mean pool (how different are they?)
        cos = F.cosine_similarity(hidden[:, 0, :], mean_pool, dim=-1)         # [B]
        store.record('text_bos_meanpool_cos', cos)
        # Full mean-pool for later cross-modal alignment
        store.record('text_meanpool_vec', mean_pool.detach().float().cpu())    # [B, D]
    handles.append(model.encoder.register_forward_hook(hook_text_encoder))

    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Greedy forward-pass prediction (fast, no beam search)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def forward_with_probe(model, batch, device):
    """
    Run model.forward() with teacher-forced labels to get logits,
    then greedily decode first token as prediction proxy.
    Returns: list of predicted token strings, list of losses per sample.
    """
    pixel_values  = batch['pixel_values'].to(device)
    input_ids     = batch['input_ids'].to(device)
    attention_mask= batch['attention_mask'].to(device)
    labels        = batch['labels'].to(device)
    question_types= batch.get('question_type')
    if question_types is not None:
        question_types = question_types.to(device)

    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        question_types=question_types,
    )

    # Teacher-forced logits: [B, T, V]
    logits = outputs.answer_logits            # [B, T, V]
    B, T, V = logits.shape

    # Greedy decode full answer from teacher-forced logits
    greedy_ids = logits.argmax(dim=-1)        # [B, T]
    valid_mask = (labels != -100)             # [B, T]

    # First-token confidence (for gate alpha correlation analysis)
    first_tok_conf = logits[:, 0, :].softmax(dim=-1).max(dim=-1).values  # [B]

    # Per-sample CE loss (difficulty proxy)
    flat_logits = logits.reshape(B * T, V)
    flat_labels = labels.reshape(B * T)
    per_token_loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction='none')
    per_sample_loss = per_token_loss.reshape(B, T)
    per_sample_loss = (per_sample_loss * valid_mask.float()).sum(1) / valid_mask.float().sum(1).clamp(min=1)

    return greedy_ids.cpu(), valid_mask.cpu(), first_tok_conf.cpu(), per_sample_loss.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Post-batch cross-modal alignment (requires text + vision vectors)
# ─────────────────────────────────────────────────────────────────────────────

def compute_cross_modal_cos(store_data, num_flamingo_layers):
    """
    Cosine similarity between text_meanpool and mean of vision features
    at different stages: pre-Flamingo and post-Flamingo.
    Returns dict of [N] tensors.
    """
    result = {}
    text_vec = store_data.get('text_meanpool_vec')   # [N, D]
    if text_vec is None:
        return result

    # Pre-Flamingo vision mean
    vision_in = store_data.get('vision_proj_input')   # [N, P, D]
    if vision_in is not None:
        vision_mean_pre = vision_in.mean(dim=1)        # [N, D]
        result['cos_text_vision_preflamingo'] = F.cosine_similarity(
            text_vec, vision_mean_pre, dim=-1)

    # Post-last-Flamingo vision mean
    last_idx = num_flamingo_layers - 1
    vision_post = store_data.get(f'flamingo_{last_idx}_vision_out')  # [N, P, D]
    if vision_post is not None:
        vision_mean_post = vision_post.mean(dim=1)     # [N, D]
        result['cos_text_vision_postflamingo'] = F.cosine_similarity(
            text_vec, vision_mean_post, dim=-1)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',    required=True)
    p.add_argument('--csv_path',      required=True)
    p.add_argument('--image_folder',  required=True)
    p.add_argument('--output_dir',    default='probe_output/')
    p.add_argument('--n_samples',     type=int, default=None,
                   help='Limit to first N samples (None = full val set)')
    p.add_argument('--batch_size',    type=int, default=8)
    p.add_argument('--device',        default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--vision_model',  default='google/siglip-base-patch16-224')
    return p.parse_args()


def load_model(checkpoint_path, device):
    """Reuse eval.py logic to auto-detect architecture from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    saved_args   = ckpt.get('args', {})
    state_keys   = list(ckpt['model_state_dict'].keys())

    has_text_lora    = any('encoder.base_model' in k for k in state_keys)
    has_vision_lora  = any('vision_encoder' in k and 'lora_A' in k for k in state_keys)
    has_vision_gate  = any('vision_gating' in k for k in state_keys)
    has_delta_gate   = any('orig_proj' in k for k in state_keys)
    has_type_task    = any('type_head' in k for k in state_keys)
    has_logits_bias  = any('logits_bias' in k for k in state_keys)
    has_type_adapter = any('vision_adapter' in k for k in state_keys)

    # Detect LoRA rank from weight shapes
    text_lora_r = saved_args.get('text_lora_r', 16)
    for k in state_keys:
        if k.startswith('encoder.base_model.model.layers.0.self_attn.q_proj.lora_A'):
            text_lora_r = ckpt['model_state_dict'][k].shape[0]
            break

    # Detect num_fusion_layers
    fusion_indices = set()
    for k in state_keys:
        if 'flamingo_fusion.' in k:
            idx = int(k.split('flamingo_fusion.')[1].split('.')[0])
            fusion_indices.add(idx)
    num_fusion_layers = max(fusion_indices) + 1 if fusion_indices else 2

    use_siglip_pooler  = saved_args.get('use_siglip_pooler', False)
    use_mean_pool_cls  = saved_args.get('use_mean_pool_cls', False)
    gate_max_alpha     = saved_args.get('vision_gate_max_alpha', 1.0)
    text_lora_alpha    = saved_args.get('text_lora_alpha', 32)
    fusion_type        = saved_args.get('fusion_type', 'text2vision')

    print(f"Checkpoint: epoch={ckpt.get('epoch','?')}, val_em={ckpt.get('val_em','?')}")
    print(f"  siglip_pooler={use_siglip_pooler}, mean_pool_cls={use_mean_pool_cls}")
    print(f"  gate_max_alpha={gate_max_alpha}, fusion_layers={num_fusion_layers}")
    print(f"  text_lora: r={text_lora_r}, alpha={text_lora_alpha}")

    model = DeterministicVQA(
        vision_model_name=saved_args.get('vision_model', 'google/siglip-base-patch16-224'),
        bartpho_model_name='vinai/bartpho-syllable',
        num_fusion_layers=num_fusion_layers,
        fusion_type=fusion_type,
        num_heads=8,
        dropout=0.0,                        # eval mode — no dropout
        gradient_checkpointing=False,
        use_vision_lora=has_vision_lora,
        vision_lora_r=saved_args.get('vision_lora_r', 8),
        vision_lora_alpha=saved_args.get('vision_lora_alpha', 16),
        use_text_lora=has_text_lora,
        text_lora_r=text_lora_r,
        text_lora_alpha=text_lora_alpha,
        use_vision_gate=has_vision_gate,
        vision_gate_init=saved_args.get('vision_gate_init', 1.5),
        vision_gate_min_alpha=saved_args.get('vision_gate_min_alpha', 0.35),
        vision_gate_max_alpha=gate_max_alpha,
        use_delta_gate=has_delta_gate,
        use_type_task=has_type_task,
        use_logits_bias=has_logits_bias,
        use_type_adapter=has_type_adapter,
        use_siglip_pooler=use_siglip_pooler,
        use_mean_pool_cls=use_mean_pool_cls,
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    return model, num_fusion_layers


def collate_fn(batch):
    out = {
        'pixel_values':   torch.stack([b['pixel_values'] for b in batch]),
        'input_ids':      torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'labels':         torch.stack([b['labels'] for b in batch]),
        'answer':         [b['raw_answer'] for b in batch],   # dataset key is raw_answer
    }
    if 'question_type' in batch[0]:
        out['question_type'] = torch.stack([b['question_type'] for b in batch])
    return out


def decode_greedy(tokenizer, greedy_ids, valid_mask):
    """Decode full greedy sequence using teacher-forced positions."""
    preds = []
    for ids, mask in zip(greedy_ids.tolist(), valid_mask.tolist()):
        valid_ids = [t for t, m in zip(ids, mask) if m]
        decoded = tokenizer.decode(valid_ids, skip_special_tokens=True).strip()
        preds.append(decoded)
    return preds


def is_correct_em(pred, gt_answer):
    """Exact match after lowercasing + strip."""
    return pred.lower().strip() == gt_answer.lower().strip()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("Loading model...")
    model, num_fusion_layers = load_model(args.checkpoint, device)

    print("Loading dataset...")
    processor = AutoProcessor.from_pretrained(
        model.vision_encoder.config._name_or_path
        if hasattr(model.vision_encoder, 'config') else 'google/siglip-base-patch16-224'
    )
    # Load CSV for question text (dataset doesn't expose raw question by default)
    df_csv = pd.read_csv(args.csv_path).reset_index(drop=True)

    dataset = VQAGenDataset(
        csv_path=args.csv_path,
        image_folder=args.image_folder,
        vision_processor=processor,
        include_question_type=True,
        auto_detect_type=True,
    )
    n_total = len(dataset)
    if args.n_samples:
        n_total = min(args.n_samples, n_total)
        dataset = Subset(dataset, range(n_total))
    df_csv = df_csv.iloc[:n_total].reset_index(drop=True)

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=2, collate_fn=collate_fn)

    # Register hooks
    handles = register_hooks(model)

    # ── Main loop ──────────────────────────────────────────────────────────
    records = []
    sample_offset = 0
    print(f"\nProbing {len(dataset)} samples (batch={args.batch_size})...")

    for batch in tqdm(loader):
        store.reset()

        greedy_ids, valid_mask, confidence, sample_loss = forward_with_probe(model, batch, device)
        pred_tokens = decode_greedy(model.tokenizer, greedy_ids, valid_mask)

        # Cross-modal cosine after accumulating all vectors
        cross_cos = compute_cross_modal_cos(store, num_fusion_layers)

        B = len(batch['answer'])
        for i in range(B):
            rec = {}
            csv_row = df_csv.iloc[sample_offset + i]
            rec['question']  = csv_row.get('question', '')
            rec['gt_answer'] = batch['answer'][i]
            rec['pred_tok']  = pred_tokens[i]
            rec['correct']   = is_correct_em(pred_tokens[i], batch['answer'][i])
            rec['loss']      = sample_loss[i].item()
            rec['confidence']= confidence[i].item()

            # Question type
            if 'question_type' in batch:
                type_id = batch['question_type'][i].item()
            else:
                type_id = -1
            rec['type_id']   = type_id
            rec['type_name'] = TYPE_NAMES.get(type_id, 'unknown')

            # SigLIP raw patch stats
            for key in ['raw_patch_norm_mean', 'raw_patch_norm_std', 'siglip_pooler_norm',
                        'global_proj_norm', 'vision_proj_norm_mean']:
                t = store.get(key)
                rec[key] = t[i].item() if t is not None else None

            # BOS vs mean-pool text stats
            for key in ['text_bos_norm', 'text_meanpool_norm', 'text_bos_meanpool_cos']:
                t = store.get(key)
                rec[key] = t[i].item() if t is not None else None

            # Flamingo vision norm per layer (track how much vision changes)
            for li in range(num_fusion_layers):
                t = store.get(f'flamingo_{li}_vision_norm_mean')
                rec[f'flamingo_{li}_vision_norm'] = t[i].item() if t is not None else None
                t = store.get(f'flamingo_{li}_text_norm_mean')
                rec[f'flamingo_{li}_text_norm'] = t[i].item() if t is not None else None

            # Flamingo Δ: relative change vs pre-Flamingo patches
            vision_in = store.get('vision_proj_input')
            for li in range(num_fusion_layers):
                vision_out = store.get(f'flamingo_{li}_vision_out')
                if vision_in is not None and vision_out is not None:
                    delta = (vision_out[i] - vision_in[i]).norm(dim=-1).mean().item()
                    base  = vision_in[i].norm(dim=-1).mean().item() + 1e-8
                    rec[f'flamingo_{li}_delta_rel'] = delta / base
                else:
                    rec[f'flamingo_{li}_delta_rel'] = None

            # Gate α stats
            for key in ['gate_alpha_mean', 'gate_alpha_std', 'gate_alpha_max', 'gate_alpha_min']:
                t = store.get(key)
                rec[key] = t[i].item() if t is not None else None

            # Cross-modal alignment
            for key, val in cross_cos.items():
                rec[key] = val[i].item() if val is not None and i < len(val) else None

            records.append(rec)

        sample_offset += B

    remove_hooks(handles)

    # ── Save per-sample CSV ────────────────────────────────────────────────
    df = pd.DataFrame(records)
    out_csv = os.path.join(args.output_dir, 'probe_samples.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nSaved per-sample CSV: {out_csv}  ({len(df)} rows)")

    # ── Summary statistics ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY — per type × correct/wrong")
    print("=" * 80)

    # Greedy accuracy proxy per type
    print("\n[Greedy first-token accuracy proxy]")
    print(f"  Overall: {df['correct'].mean():.1%}")
    for tid, tname in TYPE_NAMES.items():
        sub = df[df['type_id'] == tid]
        if len(sub) == 0:
            continue
        print(f"  {tname:10s}: {sub['correct'].mean():.1%}  (n={len(sub)})")

    # Key metrics by type × correct
    metrics = [
        ('gate_alpha_mean',          'Gate α mean'),
        ('gate_alpha_std',           'Gate α std'),
        ('gate_alpha_max',           'Gate α max'),
        ('text_bos_meanpool_cos',    'cos(BOS, mean-pool)'),
        ('text_meanpool_norm',       'Text mean-pool norm'),
        ('cos_text_vision_preflamingo',  'cos(text,vision) pre-Flamingo'),
        ('cos_text_vision_postflamingo', 'cos(text,vision) post-Flamingo'),
        ('flamingo_0_delta_rel',     'Flamingo-0 relative Δ'),
        ('global_proj_norm',         'SigLIP global token norm'),
        ('loss',                     'Per-sample loss'),
        ('confidence',               'First-token confidence'),
    ]

    for col, label in metrics:
        if col not in df.columns or df[col].isna().all():
            continue
        print(f"\n[{label}]")
        header = f"  {'Type':10s}  {'Correct':>10s}  {'Wrong':>10s}  {'Δ (C-W)':>10s}"
        print(header)
        print("  " + "-" * 46)
        for tid, tname in TYPE_NAMES.items():
            sub = df[df['type_id'] == tid]
            if len(sub) == 0:
                continue
            c = sub[sub['correct'] == True][col].dropna().mean()
            w = sub[sub['correct'] == False][col].dropna().mean()
            delta = (c - w) if (not np.isnan(c) and not np.isnan(w)) else float('nan')
            print(f"  {tname:10s}  {c:10.4f}  {w:10.4f}  {delta:+10.4f}")

    # Flamingo Δ progression (does later layer change more?)
    flamingo_cols = [f'flamingo_{i}_delta_rel' for i in range(num_fusion_layers)
                     if f'flamingo_{i}_delta_rel' in df.columns]
    if flamingo_cols:
        print("\n[Flamingo relative Δ by layer (overall)]")
        for col in flamingo_cols:
            val = df[col].dropna().mean()
            print(f"  {col}: {val:.4f}")

    # Gate α full distribution per type
    gate_all = store.get('gate_alpha_all')  # might be None after loop (store was reset)
    # Use df instead for per-type summary
    print("\n[Gate α distribution per type (from per-sample means)]")
    print(f"  {'Type':10s}  {'α_mean':>8s}  {'α_std':>8s}  {'α_max':>8s}")
    print("  " + "-" * 40)
    for tid, tname in TYPE_NAMES.items():
        sub = df[df['type_id'] == tid]
        if len(sub) == 0:
            continue
        am = sub['gate_alpha_mean'].dropna().mean()
        as_ = sub['gate_alpha_std'].dropna().mean()
        ax = sub['gate_alpha_max'].dropna().mean()
        print(f"  {tname:10s}  {am:8.4f}  {as_:8.4f}  {ax:8.4f}")

    # Top-5 worst samples per type (highest loss, wrong)
    print("\n[Top-5 hardest wrong samples per type]")
    for tid, tname in TYPE_NAMES.items():
        sub = df[(df['type_id'] == tid) & (df['correct'] == False)].nlargest(5, 'loss')
        if len(sub) == 0:
            continue
        print(f"\n  {tname}:")
        for _, row in sub.iterrows():
            q = row['question'][:60]
            print(f"    loss={row['loss']:.3f} | gt='{row['gt_answer']}' pred='{row['pred_tok']}' | {q}")

    # Save summary to text file
    summary_path = os.path.join(args.output_dir, 'probe_summary.txt')
    # Re-run print to file
    import io, contextlib
    buf = io.StringIO()
    # Write key numbers
    with open(summary_path, 'w') as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Samples: {len(df)}\n\n")
        for tid, tname in TYPE_NAMES.items():
            sub = df[df['type_id'] == tid]
            if len(sub) == 0:
                continue
            f.write(f"{tname}: acc={sub['correct'].mean():.1%} n={len(sub)}\n")
            for col, label in metrics:
                if col not in df.columns:
                    continue
                c = sub[sub['correct'] == True][col].dropna().mean()
                w = sub[sub['correct'] == False][col].dropna().mean()
                f.write(f"  {label}: correct={c:.4f} wrong={w:.4f} delta={c-w:+.4f}\n")
            f.write("\n")
    print(f"\nSaved summary: {summary_path}")


if __name__ == '__main__':
    main()
