#!/usr/bin/env python3
"""
Quick test: Verify SigLIP vision_model extraction and forward pass
"""

import torch
from transformers import AutoModel, AutoProcessor

print("="*60)
print("TEST: SigLIP Vision Model Extraction")
print("="*60)

# Load full SigLIP model
model_name = "google/siglip-base-patch16-224"
print(f"\n1. Loading full model: {model_name}")
full_model = AutoModel.from_pretrained(model_name)

# Check structure
print(f"\n2. Model structure:")
print(f"   - Has vision_model: {hasattr(full_model, 'vision_model')}")
print(f"   - Has text_model: {hasattr(full_model, 'text_model')}")

if hasattr(full_model, 'vision_model'):
    vision_model = full_model.vision_model
    print(f"\n3. Extracted vision_model:")
    print(f"   - Type: {type(vision_model).__name__}")
    print(f"   - Config hidden_size: {full_model.config.vision_config.hidden_size}")
    
    # Test forward pass
    print(f"\n4. Testing forward pass...")
    processor = AutoProcessor.from_pretrained(model_name)
    
    from PIL import Image
    import numpy as np
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    inputs = processor(images=dummy_image, return_tensors='pt')
    
    print(f"   - Input shape: {inputs['pixel_values'].shape}")
    
    with torch.no_grad():
        outputs = vision_model(pixel_values=inputs['pixel_values'])
    
    print(f"   - Output shape: {outputs.last_hidden_state.shape}")
    print(f"   - Expected: [1, 197, 768] (1 CLS + 196 patches, 768 hidden)")
    
    if outputs.last_hidden_state.shape[1] == 197:
        print(f"\n✅ SUCCESS! Vision model works correctly")
        print(f"   - Token 0: CLS token")
        print(f"   - Tokens 1-196: Patch embeddings (14x14 grid)")
    else:
        print(f"\n⚠️  Unexpected shape!")
else:
    print(f"\n❌ FAILED: No vision_model found in {model_name}")

print("\n" + "="*60)
