"""
Second-level Temporal Transformer for long-form EEG.
Operates on sequences of CLS embeddings from LaBraM backbone.
Supports In-Context Learning (ICL) inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from models.rope import RoPEAttention, RotaryEmbedding


class TemporalTransformerBlock(nn.Module):
    """
    Transformer block for temporal modeling.
    Similar to LaBraMTransformerBlock but designed for sequence-level processing.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0, use_rope: bool = True,
                 max_seq_len: int = 512):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)

        if use_rope:
            self.attn = RoPEAttention(dim, num_heads=num_heads, qkv_bias=True,
                                      qk_norm=True, attn_drop=attn_drop, proj_drop=drop,
                                      max_seq_len=max_seq_len)
        else:
            from models.labram_backbone import Attention
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True,
                                  qk_norm=True, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TemporalTransformer(nn.Module):
    """
    Second-level Transformer for temporal modeling of EEG epoch sequences.

    Takes a sequence of CLS embeddings from LaBraM backbone and models
    temporal dependencies across epochs.

    Supports two modes:
    1. Standard: Process sequence and output predictions
    2. ICL (In-Context Learning): Given (embedding, label) pairs + query,
       predict query label
    """

    def __init__(
        self,
        embed_dim: int = 200,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_rope: bool = True,
        max_seq_len: int = 512,
        num_classes: int = 2,
    ):
        """
        Args:
            embed_dim: Dimension of input embeddings (should match LaBraM CLS dim)
            depth: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            drop_path_rate: Stochastic depth rate
            use_rope: Whether to use RoPE
            max_seq_len: Maximum sequence length
            num_classes: Number of output classes
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_rope = use_rope

        # Input projection (optional, if needed to adjust dimensions)
        self.input_proj = nn.Linear(embed_dim, embed_dim)

        # Positional encoding (only if not using RoPE)
        if not use_rope:
            self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        self.pos_drop = nn.Dropout(drop_rate)

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TemporalTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                use_rope=use_rope,
                max_seq_len=max_seq_len,
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # ICL-specific components
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.demo_marker = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.query_marker = nn.Parameter(torch.zeros(1, 1, embed_dim))

        nn.init.trunc_normal_(self.demo_marker, std=0.02)
        nn.init.trunc_normal_(self.query_marker, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Standard forward pass.

        Args:
            x: (B, T, D) - sequence of epoch embeddings
            mask: (B, T) - optional attention mask

        Returns:
            logits: (B, num_classes)
        """
        B, T, D = x.shape

        x = self.input_proj(x)

        # Add positional encoding
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, :T, :]

        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Global average pooling
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            x = x.mean(dim=1)

        # Classification
        logits = self.head(x)
        return logits

    def forward_icl(
        self,
        demo_embeddings: torch.Tensor,
        demo_labels: torch.Tensor,
        query_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        In-Context Learning forward pass.

        Given demonstration pairs (embedding, label) and query embeddings,
        predict labels for queries.

        Args:
            demo_embeddings: (B, n_demo, D) - demonstration epoch embeddings
            demo_labels: (B, n_demo) - demonstration labels
            query_embeddings: (B, n_query, D) - query epoch embeddings

        Returns:
            logits: (B, n_query, num_classes)
        """
        B, n_demo, D = demo_embeddings.shape
        n_query = query_embeddings.shape[1]

        # Project inputs
        demo_emb = self.input_proj(demo_embeddings)
        query_emb = self.input_proj(query_embeddings)

        # Embed demonstration labels
        label_emb = self.label_embed(demo_labels)  # (B, n_demo, D)

        # Combine demonstration embeddings with label embeddings
        # Format: [demo_emb_1 + label_emb_1 + demo_marker, ..., query_emb_1 + query_marker, ...]
        demo_combined = demo_emb + label_emb + self.demo_marker.expand(B, n_demo, -1)
        query_combined = query_emb + self.query_marker.expand(B, n_query, -1)

        # Concatenate demonstrations and queries
        x = torch.cat([demo_combined, query_combined], dim=1)  # (B, n_demo + n_query, D)

        # Add positional encoding
        if self.pos_embed is not None:
            seq_len = n_demo + n_query
            x = x + self.pos_embed[:, :seq_len, :]

        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Extract query outputs
        query_outputs = x[:, n_demo:, :]  # (B, n_query, D)

        # Predict labels for queries
        logits = self.head(query_outputs)  # (B, n_query, num_classes)

        return logits

    def forward_icl_single_query(
        self,
        demo_embeddings: torch.Tensor,
        demo_labels: torch.Tensor,
        query_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        ICL inference for a single query.

        Args:
            demo_embeddings: (B, n_demo, D)
            demo_labels: (B, n_demo)
            query_embedding: (B, D)

        Returns:
            logits: (B, num_classes)
        """
        query_emb = query_embedding.unsqueeze(1)  # (B, 1, D)
        logits = self.forward_icl(demo_embeddings, demo_labels, query_emb)
        return logits.squeeze(1)  # (B, num_classes)


def build_temporal_transformer(
    embed_dim: int = 200,
    size: str = 'base',
    num_classes: int = 2,
    use_rope: bool = True,
    max_seq_len: int = 512,
) -> TemporalTransformer:
    """
    Factory function to build TemporalTransformer.

    Args:
        embed_dim: Should match LaBraM backbone embed_dim
        size: 'small', 'base', or 'large'
        num_classes: Number of output classes
        use_rope: Whether to use RoPE
        max_seq_len: Maximum sequence length
    """
    configs = {
        'small': dict(depth=4, num_heads=4, mlp_ratio=4.0),
        'base': dict(depth=6, num_heads=8, mlp_ratio=4.0),
        'large': dict(depth=12, num_heads=12, mlp_ratio=4.0),
    }

    cfg = configs[size]

    return TemporalTransformer(
        embed_dim=embed_dim,
        depth=cfg['depth'],
        num_heads=cfg['num_heads'],
        mlp_ratio=cfg['mlp_ratio'],
        use_rope=use_rope,
        max_seq_len=max_seq_len,
        num_classes=num_classes,
    )
