"""
Check if checkpoint contains Type Adapter weights
"""
import torch
import sys

checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "best_model.pt"

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("\n" + "="*80)
print("CHECKPOINT ANALYSIS")
print("="*80)

# Check model state dict
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print(f"\n✅ Model state dict found with {len(state_dict)} keys")
    
    # Look for adapter keys
    adapter_keys = [k for k in state_dict.keys() if 'type_adapter' in k.lower()]
    
    if adapter_keys:
        print(f"\n🎯 TYPE ADAPTER FOUND! ({len(adapter_keys)} parameters)")
        print("\nAdapter parameters:")
        for key in adapter_keys[:10]:  # Show first 10
            print(f"  - {key}: {state_dict[key].shape}")
        if len(adapter_keys) > 10:
            print(f"  ... and {len(adapter_keys) - 10} more")
    else:
        print("\n❌ NO TYPE ADAPTER FOUND!")
        print("\nAll keys:")
        for key in sorted(state_dict.keys())[:20]:
            print(f"  - {key}")
        if len(state_dict) > 20:
            print(f"  ... and {len(state_dict) - 20} more")
else:
    print("\n❌ No model_state_dict found!")
    print(f"Available keys: {list(checkpoint.keys())}")

# Check training config
if 'config' in checkpoint:
    config = checkpoint['config']
    print("\n" + "="*80)
    print("TRAINING CONFIG")
    print("="*80)
    if isinstance(config, dict):
        print(f"use_type_adapter: {config.get('use_type_adapter', 'NOT FOUND')}")
        print(f"use_vision_gate: {config.get('use_vision_gate', 'NOT FOUND')}")
        print(f"use_type_loss: {config.get('use_type_loss', 'NOT FOUND')}")

print("\n" + "="*80)
