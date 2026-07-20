"""
Shared helpers for thesis visualizations (TCVG heatmap + qualitative examples).

Replicates the exact checkpoint-arg auto-detection used in eval.py so the model
is rebuilt with the SAME architecture flags it was trained with (siglip_pooler,
gate min/max alpha, delta gate, type head, text LoRA rank, ...).

Vietnamese diacritics render fine with matplotlib's default DejaVu Sans font.
"""
import torch
from model import DeterministicVQA


TYPE_NAMES = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}
# Nhãn tiếng Việt để hiển thị trên hình khóa luận
TYPE_NAMES_VI = {0: 'ĐỐI TƯỢNG', 1: 'ĐẾM', 2: 'MÀU SẮC', 3: 'VỊ TRÍ'}

# Câu hỏi đại diện mặc định cho từng loại (dùng cho heatmap 4-loại 1-ảnh)
DEFAULT_TYPE_QUESTIONS = {
    0: 'trong hình có con vật gì',
    1: 'trong hình có bao nhiêu con vật',
    2: 'con vật trong hình có màu gì',
    3: 'con vật nằm ở đâu trong hình',
}


def build_model_from_checkpoint(checkpoint_path, vision_model='google/siglip-base-patch16-224',
                                device=None, fusion_type=None, verbose=True):
    """
    Rebuild DeterministicVQA from a checkpoint, auto-detecting architecture flags.

    Returns
    -------
    model : DeterministicVQA (eval mode, on device)
    info  : dict with keys: use_siglip_pooler, num_patches, has_vision_gate,
            has_type_task, epoch, saved_args
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    keys = list(state_dict.keys())

    # ── Feature detection (mirrors eval.py) ────────────────────────────────
    has_vision_lora = any(('lora_A' in k or 'lora_B' in k) for k in keys if 'vision_encoder' in k)
    has_text_lora = any(k.startswith('encoder.base_model.model') for k in keys)
    has_vision_gate = any('vision_gating' in k for k in keys)
    has_delta_gate = any('vision_gating.orig_proj' in k for k in keys)
    has_type_adapter = any('vision_adapter' in k for k in keys)
    has_type_task = any(k.startswith('type_head') for k in keys)
    has_logits_bias = any(k.startswith('logits_bias') for k in keys)

    # Fusion layers
    fusion_idx = set()
    for k in keys:
        if k.startswith('flamingo_fusion.'):
            parts = k.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                fusion_idx.add(int(parts[1]))
    num_fusion_layers = max(fusion_idx) + 1 if fusion_idx else 4

    saved_args = checkpoint.get('args', {})
    fusion_type = fusion_type or saved_args.get('fusion_type', 'text2vision')

    # LoRA ranks/alphas
    text_lora_r = 16
    vision_lora_r = 8
    text_lora_alpha = saved_args.get('text_lora_alpha', 32)
    vision_lora_alpha = saved_args.get('vision_lora_alpha', 16)
    if has_text_lora:
        for k in keys:
            if k.startswith('encoder.base_model.model.layers.0.self_attn.q_proj.lora_A'):
                text_lora_r = state_dict[k].shape[0]
                break
    if has_vision_lora:
        for k in keys:
            if 'vision_encoder' in k and 'lora_A' in k:
                vision_lora_r = state_dict[k].shape[0]
                break
        else:
            vision_lora_r = saved_args.get('vision_lora_r', 8)

    use_siglip_pooler = saved_args.get('use_siglip_pooler', False)
    use_mean_pool_cls = saved_args.get('use_mean_pool_cls', False)
    use_attn_pool_cls = saved_args.get('use_attn_pool_cls', False)
    vision_gate_max_alpha = saved_args.get('vision_gate_max_alpha', 1.0)
    use_type_text_adapter = saved_args.get('use_type_text_adapter', False)
    type_text_adapter_bottleneck = saved_args.get('type_text_adapter_bottleneck', 64)

    if verbose:
        print(f"[viz] Checkpoint: {checkpoint_path} (epoch {checkpoint.get('epoch', 'N/A')})")
        print(f"[viz] fusion={fusion_type}x{num_fusion_layers} | vision_gate={has_vision_gate} "
              f"(delta={has_delta_gate}) | type_head={has_type_task}")
        print(f"[viz] text_lora r={text_lora_r} a={text_lora_alpha} | siglip_pooler={use_siglip_pooler} "
              f"| gate max_alpha={vision_gate_max_alpha}")

    model = DeterministicVQA(
        vision_model_name=vision_model,
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
        vision_gate_min_alpha=saved_args.get('vision_gate_min_alpha', 0.0),
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

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    info = {
        'use_siglip_pooler': use_siglip_pooler,
        'num_patches': model.num_patches,          # 196 for SigLIP-base 224/16
        'has_vision_gate': has_vision_gate,
        'has_type_task': has_type_task,
        'epoch': checkpoint.get('epoch', 'N/A'),
        'saved_args': saved_args,
    }
    return model, info
