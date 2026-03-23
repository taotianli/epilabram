"""
LoRA (Low-Rank Adaptation) implementation for parameter-efficient fine-tuning.
Adds trainable low-rank matrices to attention layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LoRALayer(nn.Module):
    """
    LoRA layer that adds low-rank adaptation to a linear layer.

    W' = W + BA, where B is (out_dim, r) and A is (r, in_dim)
    Only A and B are trainable during LoRA fine-tuning.
    """

    def __init__(self, in_dim: int, out_dim: int, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        """
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            rank: LoRA rank (r)
            alpha: LoRA scaling factor
            dropout: Dropout rate for LoRA path
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize A with kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (*, in_dim)
        Returns:
            LoRA output: (*, out_dim)
        """
        # x @ A^T @ B^T
        result = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result * self.scaling


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    Combines frozen pretrained weights with trainable LoRA weights.
    """

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        """
        Args:
            linear: Pretrained linear layer to adapt
            rank: LoRA rank
            alpha: LoRA scaling factor
            dropout: Dropout rate
        """
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha, dropout)

        # Freeze original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)


class LoRAAttention(nn.Module):
    """
    Attention layer with LoRA applied to Q, K, V projections.
    """

    def __init__(self, attn_module, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0,
                 adapt_qkv: bool = True, adapt_proj: bool = True):
        """
        Args:
            attn_module: Original attention module (Attention or RoPEAttention)
            rank: LoRA rank
            alpha: LoRA scaling factor
            dropout: Dropout rate
            adapt_qkv: Whether to add LoRA to QKV projection
            adapt_proj: Whether to add LoRA to output projection
        """
        super().__init__()
        self.attn = attn_module
        self.adapt_qkv = adapt_qkv
        self.adapt_proj = adapt_proj

        dim = attn_module.qkv.in_features

        # Add LoRA to QKV projection
        if adapt_qkv:
            self.qkv_lora = LoRALayer(dim, dim * 3, rank, alpha, dropout)

        # Add LoRA to output projection
        if adapt_proj:
            self.proj_lora = LoRALayer(dim, dim, rank, alpha, dropout)

        # Freeze original attention weights
        for param in self.attn.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """Forward with LoRA adaptation"""
        B, N, C = x.shape

        # QKV projection with LoRA
        qkv_bias = None
        if self.attn.q_bias is not None:
            qkv_bias = torch.cat([
                self.attn.q_bias,
                torch.zeros_like(self.attn.v_bias, requires_grad=False),
                self.attn.v_bias
            ])

        qkv = F.linear(x, self.attn.qkv.weight, qkv_bias)
        if self.adapt_qkv:
            qkv = qkv + self.qkv_lora(x)

        qkv = qkv.reshape(B, N, 3, self.attn.num_heads, self.attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Apply normalization
        if self.attn.q_norm is not None:
            q = self.attn.q_norm(q).type_as(v)
            k = self.attn.k_norm(k).type_as(v)

        # Apply RoPE if available
        if hasattr(self.attn, 'rope'):
            from models.rope import apply_rotary_pos_emb
            cos, sin = self.attn.rope(N)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Attention computation
        attn = (q * self.attn.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn.attn_drop(attn)

        if return_attention:
            return attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Output projection with LoRA
        x = self.attn.proj(x)
        if self.adapt_proj:
            x = x + self.proj_lora(x)

        x = self.attn.proj_drop(x)
        return x


def add_lora_to_model(model, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0,
                      target_modules: Optional[list] = None):
    """
    Add LoRA adapters to attention layers in the model.

    Args:
        model: Model to add LoRA to (typically LaBraMBackbone)
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout rate
        target_modules: List of module names to adapt (default: all attention layers)

    Returns:
        Modified model with LoRA adapters
    """
    if target_modules is None:
        target_modules = ['attn']

    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if hasattr(module, 'qkv') and hasattr(module, 'proj'):
                # This is an attention module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                parent = model
                for part in parent_name.split('.'):
                    if part:
                        parent = getattr(parent, part)

                # Replace with LoRA attention
                lora_attn = LoRAAttention(module, rank, alpha, dropout)
                setattr(parent, child_name, lora_attn)

    return model


def get_lora_params(model):
    """Get all LoRA parameters for optimization"""
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            lora_params.append(param)
    return lora_params


def merge_lora_weights(model):
    """
    Merge LoRA weights into the base model weights.
    After merging, LoRA adapters can be removed.
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            # Merge: W' = W + BA
            with torch.no_grad():
                lora_weight = module.lora.lora_B @ module.lora.lora_A * module.lora.scaling
                module.linear.weight.data += lora_weight
