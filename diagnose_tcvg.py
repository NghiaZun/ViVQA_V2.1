"""
TCVG Diagnostic — chạy sau eval để kiểm tra TypeConditioned VisionGating có học không.

3 thứ cần check:
  1. vision_bias: nếu σ(bias) ≈ 1.0 → gate luôn "mở" = không selective
  2. Type embedding similarity: nếu 4 type embeddings gần nhau → không differentiate
  3. Per-type alpha distribution: nếu α mean giống nhau qua 4 types → TCVG là no-op

Usage:
  python diagnose_tcvg.py --checkpoint best_model.pt \
      --csv_path test.csv --image_folder images/
"""
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from dataset import VQAGenDataset, detect_question_type
from model import DeterministicVQA


_INT_TO_TYPE = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}
_TYPE_TO_INT = {v: k for k, v in _INT_TO_TYPE.items()}


def diagnose_weights(model):
    """Static analysis of TCVG weights from checkpoint — no data needed."""
    vg = model.vision_gating

    print("\n" + "="*60)
    print("1. VISION BIAS (gate default)")
    print("="*60)
    bias = vg.vision_bias.item()
    sigma = torch.sigmoid(torch.tensor(bias)).item()
    print(f"  vision_bias = {bias:.4f}  →  σ(bias) = {sigma:.4f}")
    if sigma > 0.95:
        print("  ⚠️  σ ≈ 1.0: gate is ALWAYS OPEN — not selective at all")
    elif sigma > 0.85:
        print("  ⚠️  σ > 0.85: gate leans strongly toward using all vision")
    elif sigma > 0.6:
        print("  ✅  Reasonable range — gate can suppress low-signal patches")
    else:
        print("  ℹ️  Low bias — gate suppresses much of vision features")

    print("\n" + "="*60)
    print("2. TYPE EMBEDDING SIMILARITY (do types differentiate?)")
    print("="*60)
    embs = vg.type_embedding.weight  # [4, D]
    norms = embs.norm(dim=1)
    print(f"  Embedding norms: " + "  ".join(f"{_INT_TO_TYPE[i]}={norms[i]:.3f}" for i in range(4)))
    print()
    print(f"  {'':10} " + "  ".join(f"{_INT_TO_TYPE[j]:10}" for j in range(4)))
    for i in range(4):
        row = []
        for j in range(4):
            if i == j:
                row.append("  1.000    ")
            else:
                sim = F.cosine_similarity(embs[i:i+1], embs[j:j+1]).item()
                row.append(f"  {sim:+.3f}   ")
        print(f"  {_INT_TO_TYPE[i]:10} {''.join(row)}")
    print()

    # Avg off-diagonal similarity
    sims = []
    for i in range(4):
        for j in range(i+1, 4):
            sims.append(F.cosine_similarity(embs[i:i+1], embs[j:j+1]).item())
    avg_sim = sum(sims) / len(sims)
    print(f"  Average off-diagonal cosine similarity: {avg_sim:.4f}")
    if avg_sim > 0.9:
        print("  ⚠️  Embeddings nearly IDENTICAL — types not differentiated")
    elif avg_sim > 0.7:
        print("  ⚠️  Embeddings similar — partial differentiation only")
    elif avg_sim > 0.3:
        print("  ✅  Reasonable separation — types are differentiated")
    else:
        print("  ✅  Strong separation — embeddings are highly type-specific")

    print("\n" + "="*60)
    print("3. GATE NETWORK WEIGHT NORMS (is gate_net doing something?)")
    print("="*60)
    for name, param in vg.gate_net.named_parameters():
        if param.requires_grad:
            print(f"  gate_net.{name}: norm={param.norm().item():.4f}  shape={list(param.shape)}")


def diagnose_runtime(model, dataloader, device):
    """Run inference on test set and collect per-type alpha distributions."""
    model.eval()
    type_alphas = defaultdict(list)  # type_name → list of mean alpha per sample

    # Hook to capture alpha values from VisionGating
    captured = {}

    def hook_fn(module, input, output):
        _, alpha = output  # gated_vision, alpha [B, P]
        captured['alpha'] = alpha.detach().cpu()

    handle = model.vision_gating.register_forward_hook(hook_fn)

    # Also need to know which type each sample was predicted as
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting gate values"):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass to trigger gating (with labels=None is fine for gate capture)
            # We use the model's text encoder + type head to get predicted types
            text_enc = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_cls = text_enc.last_hidden_state[:, 0, :]
            type_logits = model.type_head(text_cls)
            pred_types = torch.argmax(type_logits, dim=-1).cpu().tolist()

            # Run full forward to trigger VisionGating hook
            model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=batch['labels'].to(device),
            )

            # captured['alpha'] shape: [B, P]
            if 'alpha' in captured:
                alpha = captured['alpha']  # [B, P]
                for i, t_int in enumerate(pred_types):
                    t_name = _INT_TO_TYPE[t_int]
                    type_alphas[t_name].append(alpha[i].mean().item())

    handle.remove()

    print("\n" + "="*60)
    print("4. PER-TYPE ALPHA DISTRIBUTION (runtime)")
    print("="*60)
    print(f"  {'Type':<12} {'n':>5}  {'α_mean':>8}  {'α_std':>8}  {'α_min':>8}  {'α_max':>8}")
    print(f"  {'-'*55}")

    all_means = []
    for t in ['OBJECT', 'COUNT', 'COLOR', 'LOCATION']:
        vals = type_alphas[t]
        if not vals:
            continue
        import statistics
        mu = statistics.mean(vals)
        sigma_v = statistics.stdev(vals) if len(vals) > 1 else 0.0
        mn, mx = min(vals), max(vals)
        all_means.append(mu)
        print(f"  {t:<12} {len(vals):>5}  {mu:>8.4f}  {sigma_v:>8.4f}  {mn:>8.4f}  {mx:>8.4f}")

    print()
    spread = max(all_means) - min(all_means) if all_means else 0
    print(f"  Max spread (max_type_mean - min_type_mean): {spread:.4f}")
    if spread < 0.01:
        print("  ⚠️  Nearly IDENTICAL alpha across types — TCVG not type-conditioning")
    elif spread < 0.05:
        print("  ⚠️  Very small spread — weak type-specific gating")
    elif spread < 0.15:
        print("  ✅  Moderate spread — TCVG shows type-specific behavior")
    else:
        print("  ✅  Large spread — strong type-specific gating")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--vision_model', type=str, default='google/siglip-base-patch16-224')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weights_only', action='store_true',
                        help='Only check weights (no inference) — fast')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict']

    has_vision_gate = any('vision_gating' in k for k in state_dict)
    has_type_task = any(k.startswith('type_head') for k in state_dict)
    has_text_lora = any('encoder.base_model.model' in k for k in state_dict)

    if not has_vision_gate:
        print("❌ Checkpoint has no vision_gating — TCVG not present in this model")
        return

    text_lora_r = 16
    if has_text_lora:
        for key in state_dict:
            if 'encoder.base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight' in key:
                text_lora_r = state_dict[key].shape[0]
                break

    fusion_indices = set()
    for k in state_dict:
        if k.startswith('flamingo_fusion.'):
            parts = k.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                fusion_indices.add(int(parts[1]))
    num_fusion = max(fusion_indices) + 1 if fusion_indices else 4

    saved_args = checkpoint.get('args', {})
    fusion_type = saved_args.get('fusion_type', 'text2vision')

    model = DeterministicVQA(
        vision_model_name=args.vision_model,
        bartpho_model_name='vinai/bartpho-syllable',
        num_fusion_layers=num_fusion,
        fusion_type=fusion_type,
        use_text_lora=has_text_lora,
        text_lora_r=text_lora_r,
        text_lora_alpha=32,
        use_vision_gate=True,
        use_type_task=has_type_task,
        gradient_checkpointing=False,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Loaded weights from epoch {checkpoint.get('epoch', 'N/A')}\n")

    # Static weight analysis — always run
    diagnose_weights(model)

    if args.weights_only:
        return

    # Runtime alpha collection
    from transformers import AutoImageProcessor
    vision_processor = AutoImageProcessor.from_pretrained(args.vision_model)
    dataset = VQAGenDataset(
        csv_path=args.csv_path,
        image_folder=args.image_folder,
        vision_processor=vision_processor,
        tokenizer_name='vinai/bartpho-syllable',
        include_question_type=False,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    print(f"Dataset: {len(dataset)} samples")

    diagnose_runtime(model, dataloader, device)

    print("\n" + "="*60)
    print("INTERPRETATION GUIDE")
    print("="*60)
    print("""
  If vision_bias σ > 0.95 AND alpha spread < 0.05:
    → TCVG collapsed: chỉ là linear transform, không type-specific
    → Fix: lower init_bias (0.0 thay vì 1.5), stronger type_loss weight

  If type embeddings avg_cos_sim > 0.7:
    → Type embeddings didn't diverge during training
    → Fix: add explicit type contrastive loss on embeddings

  If alpha spread moderate (0.05-0.15) AND type acc 97%:
    → TCVG working but weakly — type signal is there but gating is gentle
    → This is acceptable; bottleneck is vision encoder quality, not gating

  Current bottlenecks by type (from eval data):
    COUNT:    off-by-1 bias → needs higher resolution or count augmentation
    COLOR:    wrong specific color → SigLIP patch resolution limitation
    LOCATION: 75% completely wrong → needs global scene features
""")


if __name__ == '__main__':
    main()
