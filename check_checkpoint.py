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
    
    # Look for adapter and vision-related keys
    adapter_keys = [k for k in state_dict.keys() if 'adapter' in k.lower()]
    vision_keys = [k for k in state_dict.keys() if k.startswith('vision')]
    gate_keys = [k for k in state_dict.keys() if 'gate' in k.lower()]
    fusion_keys = [k for k in state_dict.keys() if 'fusion' in k.lower()]
    
    if adapter_keys:
        print(f"\n🎯 ADAPTER FOUND! ({len(adapter_keys)} parameters)")
        print("\nAdapter parameters:")
        for key in adapter_keys:
            print(f"  - {key}: {state_dict[key].shape}")
    else:
        print("\n❌ NO ADAPTER KEYS FOUND!")
    
    if vision_keys:
        print(f"\n👁️  Vision-related keys ({len(vision_keys)}):")
        for key in vision_keys[:10]:
            print(f"  - {key}: {state_dict[key].shape}")
        if len(vision_keys) > 10:
            print(f"  ... and {len(vision_keys) - 10} more")
    
    if gate_keys:
        print(f"\n🚪 Gate keys ({len(gate_keys)}):")
        for key in gate_keys:
            print(f"  - {key}: {state_dict[key].shape}")
    
    if fusion_keys:
        print(f"\n🔗 Fusion keys ({len(fusion_keys)}):")
        for key in fusion_keys[:5]:
            print(f"  - {key}: {state_dict[key].shape}")
        if len(fusion_keys) > 5:
            print(f"  ... and {len(fusion_keys) - 5} more")
    
    # Component breakdown
    print(f"\n📊 Component breakdown:")
    components = {}
    for k in state_dict.keys():
        prefix = k.split('.')[0]
        components[prefix] = components.get(prefix, 0) + 1
    
    for prefix in sorted(components.keys()):
        print(f"  {prefix}: {components[prefix]} keys")
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

# Check training info
print("\n" + "="*80)
print("TRAINING INFO")
print("="*80)
print(f"Epoch: {checkpoint.get('epoch', 'NOT FOUND')}")
print(f"Stage: {checkpoint.get('stage', 'NOT FOUND')}")
print(f"Train loss: {checkpoint.get('train_loss', 'NOT FOUND')}")
print(f"Val loss: {checkpoint.get('val_loss', 'NOT FOUND')}")
print(f"Best val loss: {checkpoint.get('best_val_loss', 'NOT FOUND')}")

print("\n" + "="*80)
