"""
🔬 DIAGNOSTIC TOOLS FOR VQA MODEL DEBUGGING
============================================

Evidence-based approach to identify root causes before changing architecture.

Usage:
    python diagnostic_tools.py --checkpoint <path> --val_csv <path> --image_dir <path>

Tests performed:
    1. Gate Statistics: Check if vision is being suppressed
    2. Vision Ablation: Measure actual vision dependency
    3. Feature Quality: Analyze DINOv2 feature distribution
    4. Per-Type Breakdown: Identify which question types fail

Decision tree:
    - Gate < 0.3 → Fix gating mechanism (don't change encoder)
    - Drop < 10% + poor features → Consider CLIP/SigLIP
    - Drop < 10% + good features → Fix dataset bias
    - Drop > 25% → DINOv2 working well, fix reasoning layer
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor  # 🔥 For dataset

from dataset import VQAGenDataset
from model_no_latent import DeterministicVQA


# ============================================================================
# TEST A: GATE BEHAVIOR ANALYSIS
# ============================================================================

def analyze_gate_statistics(model, dataloader, device='cuda'):
    """
    Check if vision gating is suppressing vision features
    
    Critical thresholds:
        - Gate < 0.3: Vision heavily suppressed (BAD!)
        - Gate ~ 0.5: Balanced fusion
        - Gate > 0.7: Strong vision reliance (GOOD for VQA)
    
    Returns:
        dict with mean, std, min, max, median, percentiles
    """
    print("\n" + "="*80)
    print("TEST A: VISION GATE BEHAVIOR ANALYSIS")
    print("="*80)
    
    if not model.use_vision_gate:
        print("⚠️  Vision gating is DISABLED - skipping this test")
        return None
    
    model.eval()
    all_gate_values = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing gates"):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass to get gate values
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            if outputs.gate_stats is not None:
                # Gate stats are already computed in forward pass
                # But we want per-sample distribution, so extract from model
                # This is a simplified version - you may need to modify model
                # to return actual gate values instead of just stats
                pass
    
    # For now, collect from output stats (will need model modification for full analysis)
    print("⚠️  Full gate analysis requires model modification to return gate values")
    print("    Currently only aggregate stats are available in gate_stats dict")
    
    return None


def analyze_gate_statistics_v2(model, dataloader, device='cuda'):
    """
    Enhanced version that extracts gate values by running forward manually
    """
    print("\n" + "="*80)
    print("TEST A: VISION GATE BEHAVIOR ANALYSIS")
    print("="*80)
    
    if not model.use_vision_gate:
        print("⚠️  Vision gating is DISABLED - skipping this test")
        return None
    
    model.eval()
    all_gates = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting gate values"):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Manual forward to extract gates
            # 1. Vision encoding
            vision_outputs = model.vision_encoder(pixel_values=pixel_values)
            patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]
            patch_tokens = patch_tokens + model.vision_pos_embed.expand(pixel_values.size(0), -1, -1)
            vision_features = model.vision_proj(patch_tokens)
            
            # 2. Text encoding
            text_outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_features = text_outputs.last_hidden_state
            
            # 3. Fusion
            fused_vision = vision_features
            for fusion_layer in model.flamingo_fusion:
                fused_vision = fusion_layer(fused_vision, text_features, attention_mask)
            
            # 4. Get gate values (if gating enabled)
            if model.use_vision_gate:
                text_cls = text_features[:, 0, :]
                type_logits = model.type_head(text_cls)
                predicted_types = torch.argmax(type_logits, dim=-1)
                
                _, gate_values = model.vision_gating(
                    fused_vision, 
                    text_features,
                    type_ids=predicted_types
                )
                
                all_gates.append(gate_values.cpu())
    
    # Analyze
    all_gates = torch.cat(all_gates, dim=0)  # [total_samples * num_patches]
    
    stats = {
        'mean': all_gates.mean().item(),
        'std': all_gates.std().item(),
        'min': all_gates.min().item(),
        'max': all_gates.max().item(),
        'median': all_gates.median().item(),
        'p25': all_gates.quantile(0.25).item(),
        'p75': all_gates.quantile(0.75).item(),
        'p90': all_gates.quantile(0.90).item(),
    }
    
    print(f"\n📊 Gate Statistics (across {len(all_gates)} patch-level gates):")
    print(f"   Mean:   {stats['mean']:.3f}")
    print(f"   Median: {stats['median']:.3f}")
    print(f"   Std:    {stats['std']:.3f}")
    print(f"   Min:    {stats['min']:.3f}")
    print(f"   Max:    {stats['max']:.3f}")
    print(f"   P25:    {stats['p25']:.3f}")
    print(f"   P75:    {stats['p75']:.3f}")
    print(f"   P90:    {stats['p90']:.3f}")
    
    # Diagnosis
    print(f"\n🔍 Diagnosis:")
    if stats['mean'] < 0.3:
        print("   ❌ PROBLEM: Vision is heavily suppressed (mean < 0.3)")
        print("   → Model is NOT using vision features effectively")
        print("   → ACTION: Fix gating mechanism or remove gate")
    elif stats['mean'] < 0.5:
        print("   ⚠️  WARNING: Vision usage is low (mean < 0.5)")
        print("   → Model prefers text over vision")
        print("   → Consider: Increase gate init bias or add gate regularization")
    elif stats['mean'] < 0.7:
        print("   ✅ OK: Balanced vision-text fusion (mean 0.5-0.7)")
        print("   → Model is using both modalities")
    else:
        print("   ✅ GOOD: Strong vision reliance (mean > 0.7)")
        print("   → Model heavily uses vision features")
    
    return stats, all_gates


# ============================================================================
# TEST B: VISION ABLATION (DEPENDENCY MEASURE)
# ============================================================================

def test_vision_dependency(model, dataloader, device='cuda'):
    """
    Measure how much model relies on vision by comparing:
        - Accuracy with real images
        - Accuracy with blank/noise images
    
    Critical thresholds:
        - Drop < 10%: Model doesn't need vision (BAD for VQA!)
        - Drop 10-25%: Moderate vision usage
        - Drop > 25%: Strong vision dependency (GOOD!)
    """
    print("\n" + "="*80)
    print("TEST B: VISION DEPENDENCY (ABLATION TEST)")
    print("="*80)
    
    model.eval()
    
    acc_with_vision = []
    acc_without_vision = []
    acc_with_noise = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing vision dependency"):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Test 1: With real images
            outputs_real = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            preds_real = outputs_real.answer_logits.argmax(dim=-1)
            
            # Test 2: With blank images (zeros)
            blank_images = torch.zeros_like(pixel_values)
            outputs_blank = model(
                pixel_values=blank_images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            preds_blank = outputs_blank.answer_logits.argmax(dim=-1)
            
            # Test 3: With noise images (random)
            noise_images = torch.randn_like(pixel_values)
            outputs_noise = model(
                pixel_values=noise_images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            preds_noise = outputs_noise.answer_logits.argmax(dim=-1)
            
            # Compute accuracy (token-level)
            mask = (labels != -100)
            
            acc_real = (preds_real[mask] == labels[mask]).float().mean().item()
            acc_blank = (preds_blank[mask] == labels[mask]).float().mean().item()
            acc_noise = (preds_noise[mask] == labels[mask]).float().mean().item()
            
            acc_with_vision.append(acc_real)
            acc_without_vision.append(acc_blank)
            acc_with_noise.append(acc_noise)
    
    # Aggregate
    acc_real_mean = np.mean(acc_with_vision)
    acc_blank_mean = np.mean(acc_without_vision)
    acc_noise_mean = np.mean(acc_with_noise)
    
    drop_blank = acc_real_mean - acc_blank_mean
    drop_noise = acc_real_mean - acc_noise_mean
    
    print(f"\n📊 Vision Dependency Results:")
    print(f"   Accuracy with REAL images:  {acc_real_mean:.1%}")
    print(f"   Accuracy with BLANK images: {acc_blank_mean:.1%}")
    print(f"   Accuracy with NOISE images: {acc_noise_mean:.1%}")
    print(f"   ")
    print(f"   Drop when removing vision (blank): {drop_blank:.1%}")
    print(f"   Drop when adding noise:            {drop_noise:.1%}")
    
    # Diagnosis
    print(f"\n🔍 Diagnosis:")
    if drop_blank < 0.10:
        print("   ❌ PROBLEM: Model doesn't rely on vision (drop < 10%)")
        print("   → Text-only is sufficient for high accuracy")
        print("   → Possible causes:")
        print("      1. Dataset bias (questions answerable from text)")
        print("      2. Vision features not informative")
        print("      3. Gating suppressing vision")
        print("   → ACTION: Run Test C to check feature quality")
    elif drop_blank < 0.25:
        print("   ⚠️  WARNING: Moderate vision usage (drop 10-25%)")
        print("   → Model uses vision but not heavily")
        print("   → Consider: Harder questions or better fusion")
    else:
        print("   ✅ GOOD: Strong vision dependency (drop > 25%)")
        print("   → Model requires vision to perform well")
        print("   → Vision encoder is being used effectively")
    
    return {
        'acc_real': acc_real_mean,
        'acc_blank': acc_blank_mean,
        'acc_noise': acc_noise_mean,
        'drop_blank': drop_blank,
        'drop_noise': drop_noise
    }


# ============================================================================
# TEST C: VISION FEATURE QUALITY
# ============================================================================

def check_vision_feature_quality(model, dataloader, device='cuda'):
    """
    Analyze DINOv2 feature statistics to detect:
        - Feature collapse (low std)
        - Low diversity (similar features across images)
        - Outliers or NaN/Inf values
    """
    print("\n" + "="*80)
    print("TEST C: VISION FEATURE QUALITY ANALYSIS")
    print("="*80)
    
    model.eval()
    all_vision_features = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting vision features"):
            pixel_values = batch['pixel_values'].to(device)
            
            # Extract DINOv2 features
            vision_outputs = model.vision_encoder(pixel_values=pixel_values)
            patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]  # Remove CLS
            
            all_vision_features.append(patch_tokens.cpu())
    
    # Concatenate
    all_feats = torch.cat(all_vision_features, dim=0)  # [num_samples, num_patches, D]
    
    # Statistics
    stats = {
        'mean': all_feats.mean().item(),
        'std': all_feats.std().item(),
        'min': all_feats.min().item(),
        'max': all_feats.max().item(),
        'has_nan': torch.isnan(all_feats).any().item(),
        'has_inf': torch.isinf(all_feats).any().item(),
    }
    
    # Feature diversity (pairwise distance)
    # Sample 100 random images to avoid memory issues
    sample_size = min(100, len(all_feats))
    indices = torch.randperm(len(all_feats))[:sample_size]
    sampled_feats = all_feats[indices].view(sample_size, -1)  # Flatten patches
    
    pairwise_dist = torch.pdist(sampled_feats).mean().item()
    stats['diversity'] = pairwise_dist
    
    print(f"\n📊 Vision Feature Statistics:")
    print(f"   Shape: {all_feats.shape}")
    print(f"   Mean:  {stats['mean']:.4f}")
    print(f"   Std:   {stats['std']:.4f}")
    print(f"   Min:   {stats['min']:.4f}")
    print(f"   Max:   {stats['max']:.4f}")
    print(f"   Diversity (pairwise L2): {stats['diversity']:.4f}")
    print(f"   Has NaN: {stats['has_nan']}")
    print(f"   Has Inf: {stats['has_inf']}")
    
    # Diagnosis
    print(f"\n🔍 Diagnosis:")
    if stats['has_nan'] or stats['has_inf']:
        print("   ❌ CRITICAL: Features contain NaN/Inf!")
        print("   → ACTION: Check data preprocessing and normalization")
    elif stats['std'] < 0.1:
        print("   ❌ PROBLEM: Low feature variance (std < 0.1)")
        print("   → Features are collapsing or not diverse")
        print("   → Possible causes:")
        print("      1. Vision encoder frozen and not adapted to domain")
        print("      2. Poor image preprocessing")
        print("      3. Dataset images too similar")
        print("   → ACTION: Consider fine-tuning vision encoder or changing encoder")
    elif stats['diversity'] < 1.0:
        print("   ⚠️  WARNING: Low feature diversity")
        print("   → Different images produce similar features")
        print("   → Consider: More diverse dataset or different encoder")
    else:
        print("   ✅ GOOD: Features are diverse and well-distributed")
        print("   → DINOv2 is extracting meaningful features")
    
    return stats


# ============================================================================
# TEST D: PER-TYPE BREAKDOWN
# ============================================================================

def analyze_per_type_performance(model, dataloader, device='cuda'):
    """
    Break down accuracy by question type to identify weak spots
    """
    print("\n" + "="*80)
    print("TEST D: PER-TYPE PERFORMANCE BREAKDOWN")
    print("="*80)
    
    model.eval()
    
    type_names = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}
    
    results_by_type = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing by type"):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            question_types = batch.get('question_types', None)
            
            if question_types is None:
                print("⚠️  Question types not available in dataset - skipping")
                return None
            
            question_types = question_types.to(device)
            
            # Forward
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                question_types=question_types
            )
            
            preds = outputs.answer_logits.argmax(dim=-1)
            
            # Per-sample accuracy
            mask = (labels != -100)
            for i in range(len(question_types)):
                sample_mask = mask[i]
                sample_correct = (preds[i][sample_mask] == labels[i][sample_mask]).all().item()
                
                q_type = question_types[i].item()
                results_by_type[q_type]['correct'] += sample_correct
                results_by_type[q_type]['total'] += 1
    
    # Print results
    print(f"\n📊 Accuracy by Question Type:")
    print(f"   {'Type':<12} {'Accuracy':<10} {'Samples':<10}")
    print(f"   {'-'*35}")
    
    for type_id in sorted(results_by_type.keys()):
        type_name = type_names.get(type_id, f'UNKNOWN_{type_id}')
        correct = results_by_type[type_id]['correct']
        total = results_by_type[type_id]['total']
        acc = correct / total if total > 0 else 0
        
        print(f"   {type_name:<12} {acc:<10.1%} {total:<10}")
    
    # Overall
    total_correct = sum(r['correct'] for r in results_by_type.values())
    total_samples = sum(r['total'] for r in results_by_type.values())
    overall_acc = total_correct / total_samples if total_samples > 0 else 0
    print(f"   {'-'*35}")
    print(f"   {'OVERALL':<12} {overall_acc:<10.1%} {total_samples:<10}")
    
    return results_by_type


# ============================================================================
# MAIN DIAGNOSTIC RUNNER
# ============================================================================

def run_full_diagnostic(
    checkpoint_path,
    eval_csv,  # 🔥 Renamed: có thể là val hoặc test
    image_dir,
    batch_size=16,
    device='cuda'
):
    """
    Run all diagnostic tests and provide recommendation
    
    Args:
        checkpoint_path: Path to model checkpoint
        eval_csv: Path to CSV file (val.csv hoặc test.csv đều OK!)
        image_dir: Path to image directory
        batch_size: Batch size for evaluation
        device: cuda or cpu
    """
    print("\n" + "="*80)
    print("🔬 VQA MODEL DIAGNOSTIC SUITE")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Eval CSV: {eval_csv}")
    print(f"Image Dir: {image_dir}")
    print("="*80)
    
    # Load model
    print("\n📥 Loading model...")
    model = DeterministicVQA(
        use_vision_gate=True,
        vision_gate_init=1.5,
        use_vision_lora=True,
        use_text_lora=True
    )
    
    # Load checkpoint and adapt state_dict keys if necessary
    raw_ckpt = torch.load(checkpoint_path, map_location=device)
    state = raw_ckpt.get('model_state_dict', raw_ckpt)

    # Remap common PEFT/base_model prefixes to current model's key names
    adapted_state = {}
    skipped_adapter_keys = []
    for k, v in state.items():
        new_k = k

        # Remove nested base_model.model wrappers from some HF PEFT-wrapped checkpoints
        # Examples seen in the wild: 'vision_encoder.base_model.model.encoder.layer...' -> 'vision_encoder.encoder.layer...'
        new_k = new_k.replace('vision_encoder.base_model.model.', 'vision_encoder.')
        new_k = new_k.replace('encoder.base_model.model.', 'encoder.')
        new_k = new_k.replace('vision_encoder.base_model.', 'vision_encoder.')
        new_k = new_k.replace('encoder.base_model.', 'encoder.')
        new_k = new_k.replace('.base_model.model.', '.')

        # Skip PEFT/LoRA adapter params if current model wasn't created with PEFT
        # Adapter keys often contain 'lora' or 'lora_A'/'lora_B' suffixes
        if ('lora' in new_k) or ('.lora_' in new_k) or ('lora_A' in new_k) or ('lora_B' in new_k):
            skipped_adapter_keys.append(new_k)
            continue

        adapted_state[new_k] = v

    if skipped_adapter_keys:
        print(f"⚠️  Skipping {len(skipped_adapter_keys)} PEFT/LoRA adapter keys from checkpoint (not applied):")
        print("   ", skipped_adapter_keys[:10])

    # Load adapted state dict with non-strict mode to surface missing/unexpected keys
    load_res = model.load_state_dict(adapted_state, strict=False)
    try:
        missing = load_res.missing_keys
        unexpected = load_res.unexpected_keys
    except Exception:
        # Older/newer torch versions may return different types; fallback to printing the result
        print("Load result:", load_res)
        missing = None
        unexpected = None

    if missing is not None:
        print(f"Loaded checkpoint with {len(missing)} missing keys and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print("  Missing keys (examples):", missing[:10])
        if len(unexpected) > 0:
            print("  Unexpected keys (examples):", unexpected[:10])
    
    model = model.to(device)
    model.eval()
    
    # Load dataset
    print("📥 Loading evaluation dataset...")
    
    # Create vision processor (DINOv2)
    vision_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    
    eval_dataset = VQAGenDataset(
        csv_path=eval_csv,  # 🔥 Đổi thành csv_path
        image_folder=image_dir,  # 🔥 Đổi thành image_folder
        vision_processor=vision_processor,  # 🔥 Thêm vision_processor
        tokenizer_name='vinai/bartpho-syllable',
        max_q_len=32,
        max_a_len=20,
        include_question_type=True,  # 🔥 Enable question type
        auto_detect_type=True  # 🔥 Auto-detect from question text
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✅ Loaded {len(eval_dataset)} evaluation samples")
    
    # Run tests
    gate_stats = analyze_gate_statistics_v2(model, eval_loader, device)
    ablation_results = test_vision_dependency(model, eval_loader, device)
    feature_stats = check_vision_feature_quality(model, eval_loader, device)
    type_results = analyze_per_type_performance(model, eval_loader, device)
    
    # Final recommendation
    print("\n" + "="*80)
    print("🎯 FINAL RECOMMENDATION")
    print("="*80)
    
    # Decision tree
    if gate_stats is not None and gate_stats[0]['mean'] < 0.3:
        print("❌ DON'T change vision encoder yet!")
        print("✅ ACTION: Fix gating mechanism")
        print("   Reason: Gating is suppressing vision features")
        print("   Try:")
        print("      1. Increase vision_gate_init (e.g., 3.0 instead of 1.5)")
        print("      2. Add gate regularization loss")
        print("      3. Remove gating entirely (use_vision_gate=False)")
        
    elif ablation_results['drop_blank'] < 0.10:
        if feature_stats['std'] < 0.1 or feature_stats['diversity'] < 1.0:
            print("✅ CONSIDER: Switching to CLIP or SigLIP")
            print("   Reason: DINOv2 features lack diversity/quality")
            print("   Evidence:")
            print(f"      - Vision dependency drop: {ablation_results['drop_blank']:.1%} (low)")
            print(f"      - Feature std: {feature_stats['std']:.4f} (low)")
            print(f"      - Feature diversity: {feature_stats['diversity']:.4f} (low)")
            print("   Try:")
            print("      1. openai/clip-vit-large-patch14")
            print("      2. google/siglip-so400m-patch14-384")
        else:
            print("❌ DON'T change vision encoder!")
            print("✅ ACTION: Fix dataset bias or add harder questions")
            print("   Reason: Model can answer from text alone")
            print("   Evidence:")
            print(f"      - Vision dependency drop: {ablation_results['drop_blank']:.1%} (low)")
            print(f"      - But features are good (std={feature_stats['std']:.4f})")
            print("   Try:")
            print("      1. Data augmentation (harder visual questions)")
            print("      2. Filter out text-biased questions")
            print("      3. Add visual reasoning tasks")
    
    else:
        print("❌ DON'T change vision encoder!")
        print("✅ DINOv2 is working well!")
        print("   Evidence:")
        print(f"      - Vision dependency drop: {ablation_results['drop_blank']:.1%} (strong)")
        print("   Issue is likely in:")
        print("      1. Fusion mechanism (try more cross-attention layers)")
        print("      2. Decoder reasoning (try larger decoder)")
        print("      3. Training strategy (try longer training, better LR)")
    
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run VQA model diagnostics')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file (val.csv or test.csv)')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    run_full_diagnostic(
        checkpoint_path=args.checkpoint,
        eval_csv=args.csv,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        device=args.device
    )
