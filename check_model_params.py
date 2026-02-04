#!/usr/bin/env python3
"""
Script to analyze model parameters in detail.
Shows total params, trainable params, and breakdown by component.
"""

import torch
import argparse
from model_no_latent import DeterministicVQA


def format_number(num):
    """Format number with commas for readability."""
    return f"{num:,}"


def get_param_count(model, name=""):
    """Get parameter count for a module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def analyze_model_params(
    vision_model='google/siglip-base-patch16-224',
    use_vision_lora=False,
    vision_lora_r=8,
    use_text_lora=True,
    text_lora_r=64,
    num_fusion_layers=6,
    use_vision_gate=True,
    use_type_adapter=False,
    type_adapter_rank=64
):
    """Analyze parameters of DeterministicVQA model."""
    
    print("="*80)
    print("MODEL PARAMETER ANALYSIS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Vision Model: {vision_model}")
    print(f"  Vision LoRA: {use_vision_lora} (r={vision_lora_r})")
    print(f"  Text LoRA: {use_text_lora} (r={text_lora_r})")
    print(f"  Vision Gate: {use_vision_gate}")
    print(f"  Type Adapter: {use_type_adapter} (rank={type_adapter_rank})")
    print(f"  Fusion Layers: {num_fusion_layers}")
    print()
    
    # Create model
    print("Building model...")
    model = DeterministicVQA(
        vision_model_name=vision_model,
        bartpho_model_name='vinai/bartpho-syllable',
        num_fusion_layers=num_fusion_layers,
        num_heads=8,
        dropout=0.2,
        gradient_checkpointing=False,
        use_vision_lora=use_vision_lora,
        vision_lora_r=vision_lora_r,
        vision_lora_alpha=16,
        vision_lora_dropout=0.1,
        use_text_lora=use_text_lora,
        text_lora_r=text_lora_r,
        text_lora_alpha=128,
        text_lora_dropout=0.1,
        use_vision_gate=use_vision_gate,
        use_type_adapter=use_type_adapter,
        type_adapter_rank=type_adapter_rank,
        type_adapter_bias=2.0
    )
    
    print("✅ Model built successfully!\n")
    
    # Overall statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print("="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print(f"Total Parameters:      {format_number(total_params):>15} ({total_params/1e6:.2f}M)")
    print(f"Trainable Parameters:  {format_number(trainable_params):>15} ({trainable_params/1e6:.2f}M)")
    print(f"Frozen Parameters:     {format_number(frozen_params):>15} ({frozen_params/1e6:.2f}M)")
    print(f"Trainable Percentage:  {trainable_params/total_params*100:>14.2f}%")
    print()
    
    # Component breakdown
    print("="*80)
    print("COMPONENT BREAKDOWN")
    print("="*80)
    
    components = {}
    
    # Vision Encoder
    if hasattr(model, 'vision_encoder'):
        vision_total, vision_trainable = get_param_count(model.vision_encoder)
        components['Vision Encoder'] = (vision_total, vision_trainable)
    
    # Vision Projection
    if hasattr(model, 'vision_proj'):
        proj_total, proj_trainable = get_param_count(model.vision_proj)
        components['Vision Projection'] = (proj_total, proj_trainable)
    
    # Vision Gate
    if hasattr(model, 'vision_gate') and model.vision_gate is not None:
        gate_total, gate_trainable = get_param_count(model.vision_gate)
        components['Vision Gate'] = (gate_total, gate_trainable)
    
    # Type Adapter
    if hasattr(model, 'vision_adapter') and model.vision_adapter is not None:
        adapter_total, adapter_trainable = get_param_count(model.vision_adapter)
        components['Type Adapter'] = (adapter_total, adapter_trainable)
    
    # Fusion Layers
    if hasattr(model, 'flamingo_fusion'):
        fusion_total, fusion_trainable = get_param_count(model.flamingo_fusion)
        components['Fusion Layers'] = (fusion_total, fusion_trainable)
    
    # Text Encoder (BARTpho Encoder)
    if hasattr(model, 'encoder'):
        encoder_total, encoder_trainable = get_param_count(model.encoder)
        components['Text Encoder (BARTpho)'] = (encoder_total, encoder_trainable)
    
    # Text Decoder (BARTpho Decoder)
    if hasattr(model, 'decoder'):
        decoder_total, decoder_trainable = get_param_count(model.decoder)
        components['Text Decoder (BARTpho)'] = (decoder_total, decoder_trainable)
    
    # Type Prediction Head
    if hasattr(model, 'type_head'):
        type_head_total, type_head_trainable = get_param_count(model.type_head)
        components['Type Prediction Head'] = (type_head_total, type_head_trainable)
    
    # Type-Aware Logits Bias
    if hasattr(model, 'type_logits_bias'):
        bias_params = model.type_logits_bias.numel()
        components['Type-Aware Logits Bias'] = (bias_params, bias_params)
    
    # LM Head
    if hasattr(model, 'lm_head'):
        lm_head_total, lm_head_trainable = get_param_count(model.lm_head)
        components['LM Head'] = (lm_head_total, lm_head_trainable)
    
    # Print component table
    print(f"{'Component':<30} {'Total Params':>15} {'Trainable':>15} {'Frozen':>15} {'Train %':>10}")
    print("-" * 90)
    
    for name, (total, trainable) in components.items():
        frozen = total - trainable
        train_pct = trainable / total * 100 if total > 0 else 0
        print(f"{name:<30} {format_number(total):>15} {format_number(trainable):>15} {format_number(frozen):>15} {train_pct:>9.2f}%")
    
    # LoRA-specific analysis
    if use_text_lora or use_vision_lora:
        print()
        print("="*80)
        print("LoRA DETAILS")
        print("="*80)
        
        lora_params = 0
        
        if use_text_lora:
            # Count text LoRA params
            text_lora_params = sum(
                p.numel() for name, p in model.named_parameters() 
                if 'lora_A' in name or 'lora_B' in name
            )
            lora_params += text_lora_params
            print(f"Text LoRA Parameters:  {format_number(text_lora_params):>15} ({text_lora_params/1e6:.2f}M)")
            print(f"  Rank: {text_lora_r}, Alpha: {text_lora_r * 2}")
        
        if use_vision_lora:
            # Count vision LoRA params
            vision_lora_params = sum(
                p.numel() for name, p in model.named_parameters() 
                if 'vision_lora' in name
            )
            lora_params += vision_lora_params
            print(f"Vision LoRA Parameters: {format_number(vision_lora_params):>15} ({vision_lora_params/1e6:.2f}M)")
            print(f"  Rank: {vision_lora_r}, Alpha: {vision_lora_r * 2}")
        
        print(f"\nTotal LoRA Parameters: {format_number(lora_params):>15} ({lora_params/1e6:.2f}M)")
        print(f"LoRA % of Trainable:   {lora_params/trainable_params*100:>14.2f}%")
    
    # Memory estimation
    print()
    print("="*80)
    print("MEMORY ESTIMATION (FP32)")
    print("="*80)
    total_memory_mb = total_params * 4 / (1024**2)  # 4 bytes per param in FP32
    trainable_memory_mb = trainable_params * 4 / (1024**2)
    print(f"Total Model Size:      {total_memory_mb:>10.2f} MB ({total_memory_mb/1024:.2f} GB)")
    print(f"Trainable Params Size: {trainable_memory_mb:>10.2f} MB ({trainable_memory_mb/1024:.2f} GB)")
    print()
    
    # Training memory estimate (rough)
    # Gradients + Optimizer states (Adam: 2x params) + Activations
    training_memory_mb = trainable_memory_mb * 4  # Approximate
    print(f"Estimated Training Memory (GPU): ~{training_memory_mb:.2f} MB (~{training_memory_mb/1024:.2f} GB)")
    print("  (Includes gradients, optimizer states, approximate activations)")
    print()
    
    print("="*80)
    print("✅ Analysis Complete!")
    print("="*80)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params,
        'components': components
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze DeterministicVQA model parameters')
    parser.add_argument('--vision_model', type=str, default='google/siglip-base-patch16-224',
                       help='Vision encoder model')
    parser.add_argument('--use_vision_lora', action='store_true',
                       help='Enable vision LoRA')
    parser.add_argument('--vision_lora_r', type=int, default=8,
                       help='Vision LoRA rank')
    parser.add_argument('--use_text_lora', action='store_true', default=True,
                       help='Enable text LoRA (default: True)')
    parser.add_argument('--no_text_lora', action='store_false', dest='use_text_lora',
                       help='Disable text LoRA')
    parser.add_argument('--text_lora_r', type=int, default=64,
                       help='Text LoRA rank')
    parser.add_argument('--num_fusion_layers', type=int, default=6,
                       help='Number of Flamingo fusion layers')
    parser.add_argument('--use_vision_gate', action='store_true', default=True,
                       help='Enable vision gating (default: True)')
    parser.add_argument('--no_vision_gate', action='store_false', dest='use_vision_gate',
                       help='Disable vision gating')
    parser.add_argument('--use_type_adapter', action='store_true',
                       help='Enable type-conditioned vision adapter')
    parser.add_argument('--type_adapter_rank', type=int, default=64,
                       help='Type adapter rank')
    
    args = parser.parse_args()
    
    analyze_model_params(
        vision_model=args.vision_model,
        use_vision_lora=args.use_vision_lora,
        vision_lora_r=args.vision_lora_r,
        use_text_lora=args.use_text_lora,
        text_lora_r=args.text_lora_r,
        num_fusion_layers=args.num_fusion_layers,
        use_vision_gate=args.use_vision_gate,
        use_type_adapter=args.use_type_adapter,
        type_adapter_rank=args.type_adapter_rank
    )


if __name__ == '__main__':
    main()
