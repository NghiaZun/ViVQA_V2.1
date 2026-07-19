import torch
import torch.nn as nn
import torch.nn.functional as F


class TypeConditionedVisionAdapter(nn.Module):
    """
    Type-conditioned bottleneck adapter applied to SigLIP patch tokens.

    Learns type-specific scale+bias in the bottleneck space so that each
    question type (COLOR/COUNT/LOCATION/OBJECT) gets a slightly different
    view of the vision features, reducing cross-type gradient interference
    through shared LoRA weights.

    Initialized to near-identity (zero down/up weights) so training starts
    from the same baseline as without the adapter.

    Architecture per sample:
        x → LayerNorm → down(H→R) → GELU → type_scale · h + type_bias
          → dropout → up(R→H) → residual add
    """

    def __init__(self, hidden_dim, num_types=4, rank=64, dropout=0.1,
                 use_type_supervision=True, type_bias_strength=2.0):
        super().__init__()
        self.rank = rank
        self.use_type_supervision = use_type_supervision
        self.type_bias_strength = type_bias_strength

        self.norm = nn.LayerNorm(hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, rank, bias=False)
        self.up_proj = nn.Linear(rank, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Type-specific scale and bias in bottleneck space
        self.type_scale = nn.Embedding(num_types, rank)
        self.type_bias = nn.Embedding(num_types, rank)

        # Near-identity init: zero down/up so adapter starts as a no-op
        nn.init.zeros_(self.down_proj.weight)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.ones_(self.type_scale.weight)
        nn.init.zeros_(self.type_bias.weight)

    def forward(self, x, type_ids=None):
        """
        x:        [B, N, H]  patch tokens (after vision projection)
        type_ids: [B]        question type indices (0=OBJ,1=COUNT,2=COLOR,3=LOC)
        returns:  [B, N, H]  adapted patch tokens
        """
        h = self.down_proj(self.norm(x))   # [B, N, R]
        h = F.gelu(h)

        if type_ids is not None and self.use_type_supervision:
            scale = self.type_scale(type_ids).unsqueeze(1)   # [B, 1, R]
            bias = self.type_bias(type_ids).unsqueeze(1)     # [B, 1, R]
            h = h * scale + bias * self.type_bias_strength

        h = self.dropout(h)
        h = self.up_proj(h)                # [B, N, H]
        return x + h
