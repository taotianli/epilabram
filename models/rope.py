"""
Rotary Position Embedding (RoPE) implementation for LaBraM.
Replaces absolute positional encoding to enable context length extension.
"""

import torch
import torch.nn as nn
from typing import Tuple


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Applies rotation to query and key vectors based on their position.
    Supports dynamic sequence lengths and is more efficient for long sequences.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        """
        Args:
            dim: Dimension per head (must be even)
            max_seq_len: Maximum sequence length to precompute
            base: Base for frequency computation
        """
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos/sin for max_seq_len
        self._update_cos_sin_cache(max_seq_len)

    def _update_cos_sin_cache(self, seq_len: int):
        """Precompute cos and sin values for given sequence length"""
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cos, sin: (seq_len, dim)
        """
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len
            self._update_cos_sin_cache(seq_len)

        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                         cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors.

    Args:
        q: (B, H, N, head_dim)
        k: (B, H, N, head_dim)
        cos: (N, head_dim)
        sin: (N, head_dim)

    Returns:
        q_embed, k_embed with rotary position applied
    """
    # Expand cos/sin to match q/k shape
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, N, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class RoPEAttention(nn.Module):
    """
    Multi-head self-attention with RoPE instead of absolute position encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 qk_norm: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0,
                 max_seq_len: int = 2048):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        B, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat([
                self.q_bias,
                torch.zeros_like(self.v_bias, requires_grad=False),
                self.v_bias
            ])

        qkv = torch.nn.functional.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, H, N, head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
            k = self.k_norm(k).type_as(v)

        # Apply RoPE
        cos, sin = self.rope(N)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if return_attention:
            return attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
