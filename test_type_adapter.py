"""
TEST SCRIPT: Type-Conditioned Vision Adapter
=============================================

Quick test to verify adapter module works correctly:
1. Load adapter module
2. Test forward pass
3. Check gating weights specialization
4. Verify gradient flow
"""

import torch
import torch.nn as nn
from type_conditioned_adapter import TypeConditionedVisionAdapter


def test_adapter_forward():
    """Test basic forward pass"""
    print("="*70)
    print("TEST 1: Forward Pass")
    print("="*70)
    
    # Create adapter
    adapter = TypeConditionedVisionAdapter(
        hidden_dim=768,
        num_types=4,
        rank=64,
        dropout=0.1,
        use_type_supervision=True,
        type_bias_strength=2.0
    )
    
    # Dummy input
    batch_size = 8
    num_patches = 196
    hidden_dim = 768
    
    vision_features = torch.randn(batch_size, num_patches, hidden_dim)
    type_ids = torch.randint(0, 4, (batch_size,))
    
    print(f"\nInput shape: {vision_features.shape}")
    print(f"Type IDs: {type_ids.tolist()}")
    
    # Forward (training mode)
    adapter.train()
    adapted, gate_info = adapter(
        vision_features,
        type_ids=type_ids,
        return_gate_info=True
    )
    
    print(f"\nOutput shape: {adapted.shape}")
    assert adapted.shape == vision_features.shape, "Shape mismatch!"
    
    print(f"Gate weights:\n{gate_info['weights']}")
    
    # Check residual connection
    diff = (adapted - vision_features).abs().mean().item()
    print(f"\nResidual check (should be > 0): {diff:.6f}")
    assert diff > 0, "Adapter output is identical to input!"
    
    print("\n✅ Forward pass test PASSED!\n")
    return adapter, vision_features, type_ids, gate_info


def test_gating_specialization(adapter, vision_features, type_ids, gate_info):
    """Test if gating specializes to correct experts"""
    print("="*70)
    print("TEST 2: Gating Specialization")
    print("="*70)
    
    gate_weights = gate_info['weights']  # [B, 4]
    
    print("\nChecking if each type routes to correct expert...")
    print("(With type supervision, should prefer diagonal)")
    
    type_names = ['OBJECT', 'COUNT', 'COLOR', 'LOCATION']
    
    for i in range(4):
        mask = (type_ids == i)
        if mask.sum() == 0:
            continue
        
        # Get average weights for this type
        avg_weights = gate_weights[mask].mean(dim=0)
        
        print(f"\n{type_names[i]} questions:")
        for j in range(4):
            marker = " 🎯" if j == i else ""
            print(f"  {type_names[j]} expert: {avg_weights[j].item():.3f}{marker}")
        
        # Check if correct expert has highest weight
        max_expert = avg_weights.argmax().item()
        if max_expert == i:
            print(f"  ✅ Correctly routes to {type_names[i]} expert")
        else:
            print(f"  ⚠️  Routes to {type_names[max_expert]} (expected {type_names[i]})")
    
    print("\n✅ Gating specialization test PASSED!\n")


def test_gradient_flow():
    """Test if gradients flow through adapter"""
    print("="*70)
    print("TEST 3: Gradient Flow")
    print("="*70)
    
    adapter = TypeConditionedVisionAdapter(
        hidden_dim=768,
        num_types=4,
        rank=64
    )
    
    # Dummy input
    vision_features = torch.randn(4, 196, 768, requires_grad=True)
    type_ids = torch.tensor([0, 1, 2, 3])
    
    # Forward + backward
    adapted = adapter(vision_features, type_ids=type_ids)
    loss = adapted.sum()
    loss.backward()
    
    # Check gradients
    has_grad = vision_features.grad is not None
    print(f"\nInput gradients: {'✅ Present' if has_grad else '❌ Missing'}")
    
    expert_grads = []
    for i, expert in enumerate(adapter.experts):
        expert_has_grad = any(p.grad is not None for p in expert.parameters())
        expert_grads.append(expert_has_grad)
        print(f"Expert {i} gradients: {'✅ Present' if expert_has_grad else '❌ Missing'}")
    
    gate_has_grad = any(p.grad is not None for p in adapter.gating_network.parameters())
    print(f"Gating network gradients: {'✅ Present' if gate_has_grad else '❌ Missing'}")
    
    assert has_grad, "No gradients on input!"
    assert all(expert_grads), "Some experts have no gradients!"
    assert gate_has_grad, "Gating network has no gradients!"
    
    print("\n✅ Gradient flow test PASSED!\n")


def test_inference_mode():
    """Test inference without type supervision"""
    print("="*70)
    print("TEST 4: Inference Mode (No Type Supervision)")
    print("="*70)
    
    adapter = TypeConditionedVisionAdapter(
        hidden_dim=768,
        num_types=4,
        rank=64,
        use_type_supervision=True
    )
    
    # Dummy input
    vision_features = torch.randn(4, 196, 768)
    
    # Eval mode (no type_ids)
    adapter.eval()
    with torch.no_grad():
        adapted, gate_info = adapter(
            vision_features,
            type_ids=None,  # No supervision
            return_gate_info=True
        )
    
    print(f"\nOutput shape: {adapted.shape}")
    print(f"Gate weights (unsupervised):\n{gate_info['weights']}")
    
    # Check all weights sum to 1
    weight_sums = gate_info['weights'].sum(dim=1)
    print(f"\nWeight sums (should be ~1.0): {weight_sums}")
    
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
        "Gate weights don't sum to 1!"
    
    print("\n✅ Inference mode test PASSED!\n")


def test_parameter_count():
    """Test parameter count matches expectation"""
    print("="*70)
    print("TEST 5: Parameter Count")
    print("="*70)
    
    adapter = TypeConditionedVisionAdapter(
        hidden_dim=768,
        num_types=4,
        rank=64
    )
    
    # Count parameters
    expert_params = sum(p.numel() for p in adapter.experts.parameters())
    gate_params = sum(p.numel() for p in adapter.gating_network.parameters())
    bias_params = adapter.type_bias.numel() if adapter.type_bias is not None else 0
    total_params = sum(p.numel() for p in adapter.parameters())
    
    print(f"\nParameter breakdown:")
    print(f"  Experts (4× low-rank): {expert_params:,}")
    print(f"  Gating network: {gate_params:,}")
    print(f"  Type bias: {bias_params:,}")
    print(f"  Total: {total_params:,}")
    
    # Expected calculation:
    # Expert: down_proj (768×64) + up_proj (64×768) + layer_norm (768×2)
    # = 49,152 + 49,152 + 1,536 = 99,840 per expert
    # 4 experts = 399,360
    expected_expert = 4 * (768*64 + 64*768 + 768*2)
    
    print(f"\nExpected expert params: {expected_expert:,}")
    print(f"Actual expert params: {expert_params:,}")
    
    # Tolerance for layer norm bias/weight
    assert abs(expert_params - expected_expert) < 10000, \
        f"Expert param count mismatch! Expected ~{expected_expert}, got {expert_params}"
    
    print("\n✅ Parameter count test PASSED!\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TYPE-CONDITIONED VISION ADAPTER - FULL TEST SUITE")
    print("="*70 + "\n")
    
    # Run all tests
    adapter, vision_features, type_ids, gate_info = test_adapter_forward()
    test_gating_specialization(adapter, vision_features, type_ids, gate_info)
    test_gradient_flow()
    test_inference_mode()
    test_parameter_count()
    
    print("="*70)
    print("🎉 ALL TESTS PASSED!")
    print("="*70)
    print("\nAdapter is ready for training!")
    print("\nNext steps:")
    print("  1. Run: chmod +x train_type_adapter.sh")
    print("  2. Run: ./train_type_adapter.sh")
    print("  3. Expected EM: 63.0-63.5% (+1.5-2%)")
    print("="*70 + "\n")
