"""
任务提示令牌模块：TaskPromptTokens + PromptAdapter
"""

import torch
import torch.nn as nn
from typing import Optional


class TaskPromptTokens(nn.Module):
    """
    为每个任务维护 n_prompt 个可学习 prompt tokens。

    4个任务：TUAB(0), TUSZ(1), TUEV(2), TUEP(3)
    prompt tokens 维度与 backbone hidden size 一致。
    """

    def __init__(self, n_tasks: int = 4, n_prompt: int = 10, embed_dim: int = 200):
        super().__init__()
        self.n_tasks = n_tasks
        self.n_prompt = n_prompt
        # (n_tasks, n_prompt, embed_dim)
        self.prompt_tokens = nn.Parameter(torch.zeros(n_tasks, n_prompt, embed_dim))
        nn.init.trunc_normal_(self.prompt_tokens, std=0.02)

    def forward(self, task_id: torch.Tensor, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """
        将对应任务的 prompt tokens 拼接到 patch_embeddings 序列头部。

        Args:
            task_id: (B,) int，每个样本的任务ID
            patch_embeddings: (B, L, D)，patch token序列（含CLS）

        Returns:
            (B, n_prompt + L, D)
        """
        B = patch_embeddings.shape[0]
        prompts = self.prompt_tokens[task_id]  # (B, n_prompt, D)
        return torch.cat([prompts, patch_embeddings], dim=1)


class PromptAdapter(nn.Module):
    """
    轻量级 adapter，插入每个 Transformer block 之后。
    结构：Linear(d, d//r) → GELU → Linear(d//r, d)
    残差连接：output = x + adapter(x)
    """

    def __init__(self, embed_dim: int = 200, bottleneck_ratio: int = 4):
        super().__init__()
        hidden = embed_dim // bottleneck_ratio
        self.down = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden, embed_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


class AdaptedTransformerBlock(nn.Module):
    """
    将 PromptAdapter 包装在 LaBraMTransformerBlock 之后。
    训练时只更新 adapter 参数，block 参数冻结。
    """

    def __init__(self, block: nn.Module, adapter: PromptAdapter):
        super().__init__()
        self.block = block
        self.adapter = adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = self.adapter(x)
        return x
