"""
完整 EpiLaBraM 模型，整合所有组件。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from models.labram_backbone import LaBraMBackbone
from models.neural_tokenizer import NeuralTokenizer
from models.task_prompt import TaskPromptTokens, PromptAdapter, AdaptedTransformerBlock
from models.prediction_heads import (
    BinaryClassificationHead, MultiClassificationHead, HierarchicalHead,
    ArtifactClassificationHead,
)


class EpiLaBraM(nn.Module):
    """
    完整 EpiLaBraM 模型。

    三阶段训练：
      Stage1: PADM续训（masked EEG modeling）
      Stage2: 多任务微调（冻结backbone，训练prompt+adapter+head）
      Stage3: CPA-DPO偏好对齐
    """

    def __init__(
        self,
        backbone: LaBraMBackbone,
        tokenizer: NeuralTokenizer,
        task_prompts: TaskPromptTokens,
        heads: nn.ModuleDict,
        n_prompt: int = 10,
        adapter_bottleneck_ratio: int = 4,
    ):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.task_prompts = task_prompts
        self.heads = heads

        embed_dim = backbone.embed_dim

        # 为每个 Transformer block 添加 adapter
        self.adapters = nn.ModuleList([
            PromptAdapter(embed_dim, adapter_bottleneck_ratio)
            for _ in backbone.blocks
        ])

        # Stage1 用的 mask token 和 lm_head
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.lm_head = nn.Linear(embed_dim, tokenizer.codebook.n_embed)

    # ------------------------------------------------------------------
    # Stage 1: PADM 续训
    # ------------------------------------------------------------------

    def forward_stage1(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
        sym_mask: torch.Tensor,
        input_chans: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage1：病理感知动态掩码续训，预测 masked tokens。

        Args:
            eeg: (B, N, A, T)
            mask: (B, N*A) bool，True=被掩码
            sym_mask: (B, N*A) bool，对称掩码
            input_chans: 通道索引

        Returns:
            logits: (B, N*A, n_embed)
            sym_logits: (B, N*A, n_embed)
        """
        def _forward_with_mask(m):
            x_out = self.backbone(
                eeg,
                input_chans=input_chans,
                bool_masked_pos=m,
                mask_token=self.mask_token,
                return_patch_tokens=True,
            )
            return self.lm_head(x_out)

        logits = _forward_with_mask(mask)
        sym_logits = _forward_with_mask(sym_mask)
        return logits, sym_logits

    # ------------------------------------------------------------------
    # Stage 2: 多任务微调（带 prompt + adapter）
    # ------------------------------------------------------------------

    def forward_stage2(
        self,
        eeg: torch.Tensor,
        task_id: torch.Tensor,
        input_chans: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Stage2：多任务微调。
        backbone 冻结，只更新 task_prompts + adapters + heads。

        Args:
            eeg: (B, N, A, T)
            task_id: (B,) int

        Returns:
            dict of task logits
        """
        B, N, A, T = eeg.shape

        # 1. patch embedding（不经过完整 backbone forward，手动逐层）
        x = self.backbone.patch_embed(eeg)          # (B, N*A, temporal_out_dim)
        x = self.backbone.patch_proj(x)             # (B, N*A, embed_dim)
        cls = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)              # (B, 1+N*A, D)

        pos = self.backbone.spatial_embed(N, A, batch_size=B, input_chans=input_chans)
        x = x + pos
        te = self.backbone.temporal_embed(x, N, A)
        x[:, 1:] = x[:, 1:] + te
        x = self.backbone.pos_drop(x)

        # 2. 拼接 task prompt tokens
        x = self.task_prompts(task_id, x)           # (B, n_prompt+1+N*A, D)

        # 3. 逐层 block + adapter
        for blk, adapter in zip(self.backbone.blocks, self.adapters):
            x = blk(x)
            x = adapter(x)

        x = self.backbone.norm(x)

        # 4. average pooling（跳过 prompt tokens 和 CLS）
        n_prompt = self.task_prompts.n_prompt
        patch_tokens = x[:, n_prompt + 1:, :]       # (B, N*A, D)
        pooled = patch_tokens.mean(dim=1)            # (B, D)
        if self.backbone.fc_norm is not None:
            pooled = self.backbone.fc_norm(pooled)

        # 5. 按任务分发到对应预测头
        # 同一 batch 内可能有多个任务，逐任务处理
        results = {}
        unique_tasks = task_id.unique()
        for tid in unique_tasks:
            tid_int = tid.item()
            idx = (task_id == tid).nonzero(as_tuple=True)[0]
            feat = pooled[idx]
            task_name = {0: 'TUAB', 1: 'TUSZ', 2: 'TUEV', 3: 'TUEP'}[tid_int]
            results[task_name] = (idx, self.heads[task_name](feat))

        return results

    # ------------------------------------------------------------------
    # Stage 3: DPO reward logits
    # ------------------------------------------------------------------

    def forward_stage3_reward(
        self,
        eeg: torch.Tensor,
        task_id: torch.Tensor,
        input_chans: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Stage3：返回每个类别的 log probability，用于 DPO。

        Args:
            eeg: (B, N, A, T)
            task_id: (B,) int，假设 batch 内同一任务

        Returns:
            log_probs: (B, n_classes)
        """
        results = self.forward_stage2(eeg, task_id, input_chans)
        task_name = {0: 'TUAB', 1: 'TUSZ', 2: 'TUEV', 3: 'TUEP'}[task_id[0].item()]
        _, logits = results[task_name]
        return F.log_softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # 冻结 / 解冻工具
    # ------------------------------------------------------------------

    def freeze_backbone(self):
        """冻结 backbone 参数（Stage2 使用）"""
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)

    def get_stage2_params(self):
        """返回 Stage2 可训练参数"""
        params = (
            list(self.task_prompts.parameters())
            + list(self.adapters.parameters())
            + list(self.heads.parameters())
        )
        return params

    def get_stage3_params(self):
        """返回 Stage3 可训练参数（同 Stage2）"""
        return self.get_stage2_params()


def build_epilabram(
    backbone_size: str = 'base',
    pretrained_path: Optional[str] = None,
    vqnsp_path: Optional[str] = None,
    n_prompt: int = 10,
    adapter_bottleneck_ratio: int = 4,
    n_embed: int = 8192,
    codebook_dim: int = 64,
    task_mode: str = 'default',
    n_classes: int = 5,
) -> EpiLaBraM:
    """工厂函数：构建完整 EpiLaBraM 模型"""
    backbone = LaBraMBackbone(size=backbone_size)
    if pretrained_path:
        backbone.load_pretrained(pretrained_path)

    tokenizer = NeuralTokenizer(backbone, n_embed=n_embed, embed_dim=codebook_dim)
    if vqnsp_path:
        tokenizer.load_pretrained_vqnsp(vqnsp_path)

    embed_dim = backbone.embed_dim
    task_prompts = TaskPromptTokens(n_tasks=4, n_prompt=n_prompt, embed_dim=embed_dim)

    if task_mode == 'artifact':
        tuab_head = ArtifactClassificationHead(embed_dim, n_classes=n_classes)
    else:
        tuab_head = BinaryClassificationHead(embed_dim)

    heads = nn.ModuleDict({
        'TUAB': tuab_head,
        'TUSZ': BinaryClassificationHead(embed_dim),
        'TUEV': MultiClassificationHead(embed_dim, n_classes=6),
        'TUEP': BinaryClassificationHead(embed_dim),
    })

    return EpiLaBraM(backbone, tokenizer, task_prompts, heads,
                     n_prompt=n_prompt,
                     adapter_bottleneck_ratio=adapter_bottleneck_ratio)
