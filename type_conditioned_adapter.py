"""
TYPE-CONDITIONED VISION ADAPTER
================================

Lightweight adapter that transforms vision features based on question type.

Key ideas:
- 4 expert networks (OBJECT, COUNT, COLOR, LOCATION)
- Low-rank bottleneck (768 → 64 → 768)
- Gating network for soft routing
- Type supervision during training
- Residual connection to prevent collapse

Expected improvement: +1.5-2% EM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankExpert(nn.Module):
    """
    Low-rank expert network for efficient feature transformation.
    
    Uses bottleneck: 768 → rank → 768
    Much smaller than full MLP: 768 → 768 → 768
    
    Params: 2 × rank × hidden_dim (e.g., 2 × 64 × 768 = 98K)
    vs Full MLP: ~1.2M params
    """
    
    def __init__(self, hidden_dim=768, rank=64, dropout=0.1):
        super().__init__()
        
        self.down_proj = nn.Linear(hidden_dim, rank, bias=False)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(rank, hidden_dim, bias=False)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize: start close to identity
        nn.init.normal_(self.down_proj.weight, std=0.02)
        nn.init.zeros_(self.up_proj.weight)
    
    def forward(self, x):
        """
        Args:
            x: [B, num_patches, hidden_dim]
        Returns:
            transformed: [B, num_patches, hidden_dim]
        """
        # Down-project to bottleneck
        h = self.down_proj(x)  # [B, num_patches, rank]
        h = self.activation(h)
        h = self.dropout(h)
        
        # Up-project back to original dim
        h = self.up_proj(h)  # [B, num_patches, hidden_dim]
        h = self.layer_norm(h)
        
        return h


class GatingNetwork(nn.Module):
    """
    Learns to route inputs to appropriate experts.
    
    Input: Pooled vision features [B, hidden_dim]
    Output: Gating weights [B, num_types] (probabilities)
    """
    
    def __init__(self, hidden_dim=768, num_types=4, dropout=0.1):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),  # Compress: 768 → 192
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_types),  # 192 → 4
        )
        
        # Initialize to uniform distribution
        # (All experts equally likely at start)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
    
    def forward(self, pooled_features):
        """
        Args:
            pooled_features: [B, hidden_dim]
        Returns:
            gate_weights: [B, num_types] (softmax probabilities)
        """
        logits = self.network(pooled_features)  # [B, num_types]
        gate_weights = F.softmax(logits, dim=-1)
        return gate_weights, logits


class TypeConditionedVisionAdapter(nn.Module):
    """
    Type-conditioned adapter for vision features.
    
    Architecture:
        Vision features → Pool → Gating Network → Weights
                       ↓
        4 Expert Networks → Weighted combination → Output
    
    Training modes:
        - Supervised: Use ground-truth type_ids to bias gating
        - Unsupervised: Let gating network learn from data
    
    Usage:
        adapter = TypeConditionedVisionAdapter()
        
        # Training (with type supervision)
        adapted = adapter(vision_features, type_ids=labels)
        
        # Inference (no type ids)
        adapted = adapter(vision_features)
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_types: int = 4,
        rank: int = 64,
        dropout: float = 0.1,
        use_type_supervision: bool = True,
        type_bias_strength: float = 2.0,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_types = num_types
        self.rank = rank
        self.use_type_supervision = use_type_supervision
        
        print(f"[Type-Conditioned Adapter] Initializing...")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Num types: {num_types} (0=OBJECT, 1=COUNT, 2=COLOR, 3=LOCATION)")
        print(f"  Rank: {rank}")
        print(f"  Type supervision: {use_type_supervision}")
        
        # ================================================================
        # EXPERT NETWORKS (4 low-rank transformations)
        # ================================================================
        self.experts = nn.ModuleList([
            LowRankExpert(hidden_dim, rank, dropout)
            for _ in range(num_types)
        ])
        
        expert_params = sum(p.numel() for p in self.experts.parameters())
        print(f"  Expert networks: {expert_params:,} params")
        
        # ================================================================
        # GATING NETWORK (learns routing)
        # ================================================================
        self.gating_network = GatingNetwork(hidden_dim, num_types, dropout)
        
        gate_params = sum(p.numel() for p in self.gating_network.parameters())
        print(f"  Gating network: {gate_params:,} params")
        
        # ================================================================
        # TYPE SUPERVISION (optional bias matrix)
        # ================================================================
        if use_type_supervision:
            # Bias matrix: [num_types, num_types]
            # Diagonal elements = positive bias for correct expert
            # Off-diagonal = 0 (no bias for wrong experts)
            self.type_bias = nn.Parameter(
                torch.eye(num_types) * type_bias_strength
            )
            print(f"  Type bias: {type_bias_strength} (diagonal)")
        else:
            self.type_bias = None
        
        total_params = expert_params + gate_params
        if self.type_bias is not None:
            total_params += self.type_bias.numel()
        
        print(f"[Type-Conditioned Adapter] ✓ Total: {total_params:,} params")
    
    def forward(
        self,
        vision_features,
        type_ids=None,
        return_gate_info=False
    ):
        """
        Apply type-conditioned transformation to vision features.
        
        Args:
            vision_features: [B, num_patches, hidden_dim] 
                            (e.g., [B, 196, 768] from SigLIP)
            type_ids: [B] - Ground-truth question types (0-3)
                      Used during training for supervised gating
                      Optional during inference
            return_gate_info: bool - Whether to return gating weights/logits
        
        Returns:
            adapted_features: [B, num_patches, hidden_dim]
            (Optional) gate_info: dict with 'weights' and 'logits'
        """
        batch_size, num_patches, hidden_dim = vision_features.shape
        
        # ================================================================
        # STEP 1: Compute gating weights
        # ================================================================
        # Pool vision features (mean over patches)
        pooled = vision_features.mean(dim=1)  # [B, hidden_dim]
        
        # Gating network: pooled → routing weights
        gate_weights, gate_logits = self.gating_network(pooled)  # [B, num_types]
        
        # ================================================================
        # STEP 2: Apply type supervision (training only)
        # ================================================================
        if type_ids is not None and self.type_bias is not None and self.training:
            # Add bias based on ground-truth types
            # This encourages gating to route to correct expert
            type_bias = self.type_bias[type_ids]  # [B, num_types]
            
            # Add bias to logits and re-normalize
            gate_logits = gate_logits + type_bias
            gate_weights = F.softmax(gate_logits, dim=-1)
        
        # ================================================================
        # STEP 3: Apply all expert networks
        # ================================================================
        expert_outputs = []
        for expert in self.experts:
            # Each expert transforms features independently
            expert_out = expert(vision_features)  # [B, num_patches, hidden_dim]
            expert_outputs.append(expert_out)
        
        # Stack: [num_types, B, num_patches, hidden_dim]
        expert_outputs = torch.stack(expert_outputs, dim=0)
        
        # Permute to [B, num_types, num_patches, hidden_dim]
        expert_outputs = expert_outputs.permute(1, 0, 2, 3)
        
        # ================================================================
        # STEP 4: Weighted combination (soft routing)
        # ================================================================
        # Reshape gate_weights for broadcasting: [B, num_types, 1, 1]
        gate_weights_expanded = gate_weights.unsqueeze(2).unsqueeze(3)
        
        # Weighted sum over experts
        # [B, num_types, num_patches, hidden_dim] → [B, num_patches, hidden_dim]
        adapted = (expert_outputs * gate_weights_expanded).sum(dim=1)
        
        # ================================================================
        # STEP 5: Residual connection (CRITICAL for stability!)
        # ================================================================
        # Without residual: experts might collapse to zero
        # With residual: experts learn DELTA from original features
        adapted_features = adapted + vision_features
        
        # ================================================================
        # OPTIONAL: Return gating information for analysis
        # ================================================================
        if return_gate_info:
            gate_info = {
                'weights': gate_weights,  # [B, num_types] - probabilities
                'logits': gate_logits,    # [B, num_types] - raw scores
            }
            return adapted_features, gate_info
        
        return adapted_features
    
    def get_expert_specialization(self, dataloader, device):
        """
        Analyze which expert is used for which question type.
        
        Useful for debugging/validation:
        - Check if OBJECT expert activates for OBJECT questions
        - Check if gating network learns meaningful routing
        
        Args:
            dataloader: DataLoader with type_ids
            device: torch device
        
        Returns:
            specialization_matrix: [num_types, num_types]
                specialization_matrix[i, j] = avg weight of expert j for type i
        """
        self.eval()
        
        # Accumulator: [num_types, num_types]
        total_weights = torch.zeros(self.num_types, self.num_types)
        type_counts = torch.zeros(self.num_types)
        
        with torch.no_grad():
            for batch in dataloader:
                vision_features = batch['vision_features'].to(device)
                type_ids = batch['type_ids'].to(device)
                
                # Get gating weights
                _, gate_info = self.forward(
                    vision_features,
                    type_ids=None,  # Don't use supervision for analysis
                    return_gate_info=True
                )
                gate_weights = gate_info['weights']  # [B, num_types]
                
                # Accumulate
                for i in range(self.num_types):
                    mask = (type_ids == i)
                    if mask.sum() > 0:
                        total_weights[i] += gate_weights[mask].sum(dim=0).cpu()
                        type_counts[i] += mask.sum().item()
        
        # Average
        specialization_matrix = total_weights / type_counts.unsqueeze(1)
        
        return specialization_matrix


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def print_specialization_matrix(matrix, type_names=None):
    """
    Pretty-print specialization matrix.
    
    Args:
        matrix: [num_types, num_types] tensor
        type_names: List of type names (default: OBJECT, COUNT, COLOR, LOCATION)
    """
    if type_names is None:
        type_names = ['OBJECT', 'COUNT', 'COLOR', 'LOCATION']
    
    print("\nExpert Specialization Matrix:")
    print("(Rows = Question Type, Cols = Expert activated)")
    print(f"{'Type':<10}", end='')
    for name in type_names:
        print(f"{name:<10}", end='')
    print()
    print("-" * 50)
    
    for i, type_name in enumerate(type_names):
        print(f"{type_name:<10}", end='')
        for j in range(len(type_names)):
            weight = matrix[i, j].item()
            print(f"{weight:>9.3f}", end=' ')
        print()
    
    print("\nIdeal matrix (perfect specialization):")
    print("  OBJECT expert activates for OBJECT (diagonal = 1.0)")
    print("  Off-diagonal should be < 0.5")


def visualize_gating_distribution(gate_weights, type_ids, save_path=None):
    """
    Visualize gating weight distribution per question type.
    
    Args:
        gate_weights: [N, num_types] - collected from validation set
        type_ids: [N] - ground-truth types
        save_path: Optional path to save plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    type_names = ['OBJECT', 'COUNT', 'COLOR', 'LOCATION']
    num_types = len(type_names)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i in range(num_types):
        mask = (type_ids == i)
        if mask.sum() == 0:
            continue
        
        # Get weights for this type
        weights = gate_weights[mask]  # [N_i, num_types]
        
        # Plot distribution for each expert
        ax = axes[i]
        for j in range(num_types):
            ax.hist(weights[:, j].cpu().numpy(), 
                   bins=20, alpha=0.5, label=f'{type_names[j]} expert')
        
        ax.set_title(f'Gating distribution for {type_names[i]} questions')
        ax.set_xlabel('Expert weight')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved gating visualization to {save_path}")
    else:
        plt.show()


# ================================================================
# EXAMPLE USAGE
# ================================================================

if __name__ == "__main__":
    """
    Test the adapter module.
    """
    
    print("="*60)
    print("TYPE-CONDITIONED VISION ADAPTER - TEST")
    print("="*60)
    
    # Create adapter
    adapter = TypeConditionedVisionAdapter(
        hidden_dim=768,
        num_types=4,
        rank=64,
        dropout=0.1,
        use_type_supervision=True,
        type_bias_strength=2.0
    )
    
    # Dummy input (batch_size=8, num_patches=196, hidden_dim=768)
    batch_size = 8
    num_patches = 196
    hidden_dim = 768
    
    vision_features = torch.randn(batch_size, num_patches, hidden_dim)
    type_ids = torch.randint(0, 4, (batch_size,))  # Random types
    
    print(f"\nInput shape: {vision_features.shape}")
    print(f"Type IDs: {type_ids.tolist()}")
    
    # Forward pass (training mode with supervision)
    adapter.train()
    adapted_features, gate_info = adapter(
        vision_features,
        type_ids=type_ids,
        return_gate_info=True
    )
    
    print(f"\nOutput shape: {adapted_features.shape}")
    print(f"Gate weights shape: {gate_info['weights'].shape}")
    print(f"\nGate weights:")
    print(gate_info['weights'])
    
    # Check residual connection works
    print(f"\nResidual check (should NOT be zero):")
    print(f"  Mean abs diff: {(adapted_features - vision_features).abs().mean():.6f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in adapter.parameters())
    trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    print("\n" + "="*60)
    print("✓ Test passed!")
    print("="*60)
