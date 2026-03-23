"""
Mixture of Experts (MoE) for EEG Foundation Models.
Enables specialized experts for different EEG patterns, brain regions, or tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class Expert(nn.Module):
    """
    Single expert network (FFN).
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Router(nn.Module):
    """
    Router network that assigns tokens to experts.
    Supports top-k routing and load balancing.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 0.1,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.capacity_factor = capacity_factor

        # Router weights
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Args:
            x: (B, N, D) input tokens

        Returns:
            expert_weights: (B, N, top_k) weights for selected experts
            expert_indices: (B, N, top_k) indices of selected experts
            load_balancing_loss: scalar loss for load balancing
        """
        B, N, D = x.shape

        # Compute router logits
        router_logits = self.gate(x)  # (B, N, num_experts)

        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise

        # Compute routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)  # (B, N, num_experts)

        # Select top-k experts
        expert_weights, expert_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )  # (B, N, top_k)

        # Normalize weights
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        # Load balancing loss
        # Encourage uniform distribution of tokens across experts
        load_balancing_loss = self._compute_load_balancing_loss(router_probs)

        return expert_weights, expert_indices, load_balancing_loss

    def _compute_load_balancing_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss to encourage uniform expert usage.

        Args:
            router_probs: (B, N, num_experts)

        Returns:
            loss: scalar
        """
        # Average probability of routing to each expert
        expert_usage = router_probs.mean(dim=[0, 1])  # (num_experts,)

        # Ideal uniform distribution
        uniform = torch.ones_like(expert_usage) / self.num_experts

        # Coefficient of variation loss
        loss = (expert_usage * torch.log(expert_usage / uniform + 1e-8)).sum()

        return loss


class MoELayer(nn.Module):
    """
    Mixture of Experts layer.
    Replaces standard FFN in Transformer blocks.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 8,
        expert_hidden_dim: Optional[int] = None,
        top_k: int = 2,
        dropout: float = 0.0,
        noise_std: float = 0.1,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight

        expert_hidden_dim = expert_hidden_dim or dim * 4

        # Router
        self.router = Router(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            noise_std=noise_std,
        )

        # Experts
        self.experts = nn.ModuleList([
            Expert(dim, expert_hidden_dim, dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer.

        Args:
            x: (B, N, D) input tokens

        Returns:
            output: (B, N, D) output tokens
            aux_loss: scalar auxiliary loss (load balancing)
        """
        B, N, D = x.shape
        original_shape = x.shape

        # Route tokens to experts
        expert_weights, expert_indices, load_balance_loss = self.router(x)
        # expert_weights: (B, N, top_k)
        # expert_indices: (B, N, top_k)

        # Initialize output
        output = torch.zeros_like(x)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (expert_indices == expert_idx).any(dim=-1)  # (B, N)

            if not expert_mask.any():
                continue

            # Get tokens for this expert
            expert_input = x[expert_mask]  # (num_tokens, D)

            # Process through expert
            expert_output = self.experts[expert_idx](expert_input)  # (num_tokens, D)

            # Get weights for this expert
            # Find positions where this expert is selected
            expert_weight_mask = (expert_indices == expert_idx)  # (B, N, top_k)
            expert_weight = expert_weights * expert_weight_mask.float()  # (B, N, top_k)
            expert_weight = expert_weight.sum(dim=-1)  # (B, N)

            # Accumulate weighted output
            output[expert_mask] += expert_weight[expert_mask].unsqueeze(-1) * expert_output

        # Auxiliary loss
        aux_loss = self.load_balance_weight * load_balance_loss

        return output, aux_loss


class MoETransformerBlock(nn.Module):
    """
    Transformer block with MoE layer instead of standard FFN.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_experts: int = 8,
        top_k: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
        use_moe: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_dropout, batch_first=True
        )

        self.drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        # MoE or standard FFN
        self.use_moe = use_moe
        if use_moe:
            self.ffn = MoELayer(
                dim=dim,
                num_experts=num_experts,
                expert_hidden_dim=int(dim * mlp_ratio),
                top_k=top_k,
                dropout=dropout,
            )
        else:
            hidden_dim = int(dim * mlp_ratio)
            self.ffn = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: (B, N, D)

        Returns:
            x: (B, N, D)
            aux_loss: scalar or None
        """
        # Self-attention
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.drop_path(attn_out)

        # FFN or MoE
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(self.norm2(x))
            x = x + self.drop_path(ffn_out)
            return x, aux_loss
        else:
            ffn_out = self.ffn(self.norm2(x))
            x = x + self.drop_path(ffn_out)
            return x, None


class MoELaBraMBackbone(nn.Module):
    """
    LaBraM backbone with MoE layers.
    Can selectively replace FFN layers with MoE.
    """

    def __init__(
        self,
        base_backbone: nn.Module,
        moe_layers: List[int],  # Which layers to replace with MoE
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()
        self.base_backbone = base_backbone
        self.moe_layers = set(moe_layers)
        self.num_experts = num_experts
        self.top_k = top_k

        # Replace specified layers with MoE
        self._replace_with_moe()

    def _replace_with_moe(self):
        """Replace standard FFN with MoE in specified layers."""
        for layer_idx in self.moe_layers:
            if layer_idx >= len(self.base_backbone.blocks):
                continue

            block = self.base_backbone.blocks[layer_idx]
            dim = block.mlp.fc1.in_features
            hidden_dim = block.mlp.fc1.out_features

            # Create MoE layer
            moe_layer = MoELayer(
                dim=dim,
                num_experts=self.num_experts,
                expert_hidden_dim=hidden_dim,
                top_k=self.top_k,
            )

            # Replace FFN with MoE
            block.mlp = moe_layer

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with MoE.

        Returns:
            output: model output
            total_aux_loss: sum of all MoE auxiliary losses
        """
        # Forward through base backbone
        # We need to manually handle MoE layers to collect aux losses
        B, N, A, T = x.shape

        # Patch embedding
        x_embed = self.base_backbone.patch_embed(x)
        x_embed = self.base_backbone.patch_proj(x_embed)

        cls_tokens = self.base_backbone.cls_token.expand(B, -1, -1)
        x_embed = torch.cat([cls_tokens, x_embed], dim=1)

        # Position encoding
        if not self.base_backbone.use_rope:
            pos = self.base_backbone.spatial_embed(N, A, batch_size=B)
            x_embed = x_embed + pos
            te = self.base_backbone.temporal_embed(x_embed, N, A)
            x_embed[:, 1:, :] = x_embed[:, 1:, :] + te

        x_embed = self.base_backbone.pos_drop(x_embed)

        # Forward through blocks, collecting MoE losses
        total_aux_loss = 0.0
        for layer_idx, block in enumerate(self.base_backbone.blocks):
            if layer_idx in self.moe_layers:
                # MoE block
                # Attention
                x_embed = x_embed + block.drop_path(block.attn(block.norm1(x_embed)))
                # MoE FFN
                ffn_out, aux_loss = block.mlp(block.norm2(x_embed))
                x_embed = x_embed + block.drop_path(ffn_out)
                total_aux_loss = total_aux_loss + aux_loss
            else:
                # Standard block
                x_embed = block(x_embed)

        x_embed = self.base_backbone.norm(x_embed)

        # Pooling
        if self.base_backbone.fc_norm is not None:
            output = self.base_backbone.fc_norm(x_embed[:, 1:].mean(1))
        else:
            output = x_embed[:, 0]

        return output, total_aux_loss


def build_moe_backbone(
    base_backbone: nn.Module,
    moe_layers: Optional[List[int]] = None,
    num_experts: int = 8,
    top_k: int = 2,
    moe_frequency: int = 2,  # Apply MoE every N layers
) -> MoELaBraMBackbone:
    """
    Build MoE-enhanced backbone.

    Args:
        base_backbone: Base LaBraM backbone
        moe_layers: Specific layers to replace with MoE (if None, use moe_frequency)
        num_experts: Number of experts
        top_k: Number of experts to activate per token
        moe_frequency: Apply MoE every N layers (if moe_layers is None)

    Returns:
        moe_backbone: MoE-enhanced backbone
    """
    if moe_layers is None:
        # Apply MoE to every Nth layer
        total_layers = len(base_backbone.blocks)
        moe_layers = list(range(moe_frequency - 1, total_layers, moe_frequency))

    print(f"Applying MoE to layers: {moe_layers}")
    print(f"Number of experts: {num_experts}, Top-k: {top_k}")

    moe_backbone = MoELaBraMBackbone(
        base_backbone=base_backbone,
        moe_layers=moe_layers,
        num_experts=num_experts,
        top_k=top_k,
    )

    return moe_backbone


class ExpertSpecialization:
    """
    Utility class to analyze expert specialization.
    Helps understand what each expert learns.
    """

    def __init__(self, model: MoELaBraMBackbone):
        self.model = model
        self.expert_usage = {}
        self.expert_task_affinity = {}

    def track_expert_usage(
        self,
        layer_idx: int,
        expert_indices: torch.Tensor,
        task_labels: Optional[torch.Tensor] = None,
    ):
        """
        Track which experts are used for which samples/tasks.

        Args:
            layer_idx: Layer index
            expert_indices: (B, N, top_k) expert indices
            task_labels: (B,) task labels (optional)
        """
        if layer_idx not in self.expert_usage:
            self.expert_usage[layer_idx] = torch.zeros(self.model.num_experts)

        # Count expert usage
        for expert_idx in range(self.model.num_experts):
            count = (expert_indices == expert_idx).sum().item()
            self.expert_usage[layer_idx][expert_idx] += count

        # Track task affinity
        if task_labels is not None:
            if layer_idx not in self.expert_task_affinity:
                self.expert_task_affinity[layer_idx] = {}

            for task_id in task_labels.unique():
                task_mask = (task_labels == task_id)
                task_expert_indices = expert_indices[task_mask]

                if task_id.item() not in self.expert_task_affinity[layer_idx]:
                    self.expert_task_affinity[layer_idx][task_id.item()] = \
                        torch.zeros(self.model.num_experts)

                for expert_idx in range(self.model.num_experts):
                    count = (task_expert_indices == expert_idx).sum().item()
                    self.expert_task_affinity[layer_idx][task_id.item()][expert_idx] += count

    def get_expert_statistics(self) -> dict:
        """Get statistics about expert usage."""
        stats = {}

        for layer_idx, usage in self.expert_usage.items():
            total = usage.sum()
            usage_pct = (usage / total * 100).tolist()

            stats[f'layer_{layer_idx}'] = {
                'usage_percentage': usage_pct,
                'most_used_expert': usage.argmax().item(),
                'least_used_expert': usage.argmin().item(),
                'usage_variance': usage.var().item(),
            }

        return stats

    def print_statistics(self):
        """Print expert usage statistics."""
        stats = self.get_expert_statistics()

        print("\n" + "=" * 60)
        print("Expert Usage Statistics")
        print("=" * 60)

        for layer_name, layer_stats in stats.items():
            print(f"\n{layer_name}:")
            print(f"  Most used expert: {layer_stats['most_used_expert']}")
            print(f"  Least used expert: {layer_stats['least_used_expert']}")
            print(f"  Usage variance: {layer_stats['usage_variance']:.4f}")
            print(f"  Usage distribution: {[f'{x:.1f}%' for x in layer_stats['usage_percentage']]}")
