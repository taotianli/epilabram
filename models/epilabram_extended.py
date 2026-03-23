"""
Extended EpiLaBraM model with RoPE, LoRA, and Temporal Transformer support.
Integrates all new features for long-form EEG and ICL inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from models.labram_backbone import LaBraMBackbone
from models.neural_tokenizer import NeuralTokenizer
from models.task_prompt import TaskPromptTokens, PromptAdapter
from models.prediction_heads import (
    BinaryClassificationHead, MultiClassificationHead,
    ArtifactClassificationHead,
)
from models.lora import add_lora_to_model, get_lora_params
from models.temporal_transformer import TemporalTransformer, build_temporal_transformer


class EpiLaBraMExtended(nn.Module):
    """
    Extended EpiLaBraM with:
    1. RoPE support for context length extension
    2. LoRA adapters for parameter-efficient continual pre-training
    3. Second-level Temporal Transformer for long-form EEG
    4. ICL (In-Context Learning) inference capability
    """

    def __init__(
        self,
        backbone: LaBraMBackbone,
        tokenizer: NeuralTokenizer,
        task_prompts: TaskPromptTokens,
        heads: nn.ModuleDict,
        temporal_transformer: Optional[TemporalTransformer] = None,
        n_prompt: int = 10,
        adapter_bottleneck_ratio: int = 4,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.task_prompts = task_prompts
        self.heads = heads
        self.temporal_transformer = temporal_transformer
        self.use_lora = use_lora

        embed_dim = backbone.embed_dim

        # Stage2 adapters
        self.adapters = nn.ModuleList([
            PromptAdapter(embed_dim, adapter_bottleneck_ratio)
            for _ in backbone.blocks
        ])

        # Stage1 components
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.lm_head = nn.Linear(embed_dim, tokenizer.codebook.n_embed)

        # Apply LoRA if requested
        if use_lora:
            self.backbone = add_lora_to_model(
                self.backbone,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=0.0
            )

    # ------------------------------------------------------------------
    # Stage 1: PADM with optional LoRA
    # ------------------------------------------------------------------

    def forward_stage1(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
        sym_mask: torch.Tensor,
        input_chans: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage1: PADM continual pre-training with optional LoRA"""
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
    # Stage 2: Multi-task fine-tuning
    # ------------------------------------------------------------------

    def forward_stage2(
        self,
        eeg: torch.Tensor,
        task_id: torch.Tensor,
        input_chans: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Stage2: Multi-task fine-tuning with prompts and adapters"""
        B, N, A, T = eeg.shape

        # Patch embedding
        x = self.backbone.patch_embed(eeg)
        x = self.backbone.patch_proj(x)
        cls = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Position encoding (only if not using RoPE)
        if not self.backbone.use_rope:
            pos = self.backbone.spatial_embed(N, A, batch_size=B, input_chans=input_chans)
            x = x + pos
            te = self.backbone.temporal_embed(x, N, A)
            x[:, 1:] = x[:, 1:] + te

        x = self.backbone.pos_drop(x)

        # Add task prompts
        x = self.task_prompts(task_id, x)

        # Apply blocks + adapters
        for blk, adapter in zip(self.backbone.blocks, self.adapters):
            x = blk(x)
            x = adapter(x)

        x = self.backbone.norm(x)

        # Average pooling
        n_prompt = self.task_prompts.n_prompt
        patch_tokens = x[:, n_prompt + 1:, :]
        pooled = patch_tokens.mean(dim=1)
        if self.backbone.fc_norm is not None:
            pooled = self.backbone.fc_norm(pooled)

        # Task-specific heads
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
    # Extract CLS embeddings for temporal modeling
    # ------------------------------------------------------------------

    def extract_cls_embeddings(
        self,
        eeg: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
        input_chans: Optional[torch.Tensor] = None,
        use_prompts: bool = False,
    ) -> torch.Tensor:
        """
        Extract CLS token embeddings from backbone for temporal modeling.

        Args:
            eeg: (B, N, A, T) - single epoch or (B*T_seq, N, A, T) for sequence
            task_id: Optional task IDs if using prompts
            input_chans: Optional channel indices
            use_prompts: Whether to use task prompts

        Returns:
            cls_embeddings: (B, D) or (B*T_seq, D)
        """
        B, N, A, T = eeg.shape

        # Patch embedding
        x = self.backbone.patch_embed(eeg)
        x = self.backbone.patch_proj(x)
        cls = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Position encoding
        if not self.backbone.use_rope:
            pos = self.backbone.spatial_embed(N, A, batch_size=B, input_chans=input_chans)
            x = x + pos
            te = self.backbone.temporal_embed(x, N, A)
            x[:, 1:] = x[:, 1:] + te

        x = self.backbone.pos_drop(x)

        # Optionally add task prompts
        if use_prompts and task_id is not None:
            x = self.task_prompts(task_id, x)
            n_prompt = self.task_prompts.n_prompt
        else:
            n_prompt = 0

        # Apply blocks (with or without adapters)
        if use_prompts:
            for blk, adapter in zip(self.backbone.blocks, self.adapters):
                x = blk(x)
                x = adapter(x)
        else:
            for blk in self.backbone.blocks:
                x = blk(x)

        x = self.backbone.norm(x)

        # Extract CLS token
        cls_emb = x[:, n_prompt, :]  # CLS is at position n_prompt (after prompts if any)

        return cls_emb

    # ------------------------------------------------------------------
    # Temporal Transformer forward
    # ------------------------------------------------------------------

    def forward_temporal(
        self,
        eeg_sequence: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
        input_chans: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for long-form EEG using temporal transformer.

        Args:
            eeg_sequence: (B, T_seq, N, A, T) - sequence of EEG epochs
            task_id: (B,) - task IDs
            input_chans: Optional channel indices
            mask: (B, T_seq) - optional sequence mask

        Returns:
            logits: (B, num_classes)
        """
        if self.temporal_transformer is None:
            raise ValueError("Temporal transformer not initialized")

        B, T_seq, N, A, T = eeg_sequence.shape

        # Reshape to process all epochs
        eeg_flat = eeg_sequence.view(B * T_seq, N, A, T)

        # Expand task_id if provided
        if task_id is not None:
            task_id_expanded = task_id.unsqueeze(1).expand(-1, T_seq).reshape(-1)
        else:
            task_id_expanded = None

        # Extract CLS embeddings for each epoch
        cls_embeddings = self.extract_cls_embeddings(
            eeg_flat,
            task_id=task_id_expanded,
            input_chans=input_chans,
            use_prompts=(task_id is not None)
        )

        # Reshape back to sequence
        cls_embeddings = cls_embeddings.view(B, T_seq, -1)

        # Apply temporal transformer
        logits = self.temporal_transformer(cls_embeddings, mask=mask)

        return logits

    # ------------------------------------------------------------------
    # ICL (In-Context Learning) inference
    # ------------------------------------------------------------------

    def forward_icl(
        self,
        demo_eeg: torch.Tensor,
        demo_labels: torch.Tensor,
        query_eeg: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
        input_chans: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        In-Context Learning inference.

        Args:
            demo_eeg: (B, n_demo, N, A, T) - demonstration EEG epochs
            demo_labels: (B, n_demo) - demonstration labels
            query_eeg: (B, n_query, N, A, T) - query EEG epochs
            task_id: (B,) - task IDs
            input_chans: Optional channel indices

        Returns:
            logits: (B, n_query, num_classes)
        """
        if self.temporal_transformer is None:
            raise ValueError("Temporal transformer not initialized")

        B, n_demo, N, A, T = demo_eeg.shape
        n_query = query_eeg.shape[1]

        # Extract embeddings for demonstrations
        demo_flat = demo_eeg.view(B * n_demo, N, A, T)
        if task_id is not None:
            task_id_demo = task_id.unsqueeze(1).expand(-1, n_demo).reshape(-1)
        else:
            task_id_demo = None

        demo_embeddings = self.extract_cls_embeddings(
            demo_flat, task_id=task_id_demo, input_chans=input_chans, use_prompts=(task_id is not None)
        )
        demo_embeddings = demo_embeddings.view(B, n_demo, -1)

        # Extract embeddings for queries
        query_flat = query_eeg.view(B * n_query, N, A, T)
        if task_id is not None:
            task_id_query = task_id.unsqueeze(1).expand(-1, n_query).reshape(-1)
        else:
            task_id_query = None

        query_embeddings = self.extract_cls_embeddings(
            query_flat, task_id=task_id_query, input_chans=input_chans, use_prompts=(task_id is not None)
        )
        query_embeddings = query_embeddings.view(B, n_query, -1)

        # ICL forward
        logits = self.temporal_transformer.forward_icl(
            demo_embeddings, demo_labels, query_embeddings
        )

        return logits

    # ------------------------------------------------------------------
    # Stage 3: DPO reward
    # ------------------------------------------------------------------

    def forward_stage3_reward(
        self,
        eeg: torch.Tensor,
        task_id: torch.Tensor,
        input_chans: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Stage3: DPO reward computation"""
        results = self.forward_stage2(eeg, task_id, input_chans)
        task_name = {0: 'TUAB', 1: 'TUSZ', 2: 'TUEV', 3: 'TUEP'}[task_id[0].item()]
        _, logits = results[task_name]
        return F.log_softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------

    def freeze_backbone(self):
        """Freeze backbone (except LoRA if enabled)"""
        for name, param in self.backbone.named_parameters():
            if 'lora' not in name.lower():
                param.requires_grad_(False)

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad_(True)

    def get_stage1_params(self):
        """Get Stage1 trainable parameters (backbone + LoRA if enabled)"""
        if self.use_lora:
            return get_lora_params(self.backbone) + list(self.lm_head.parameters())
        else:
            return list(self.backbone.parameters()) + list(self.lm_head.parameters())

    def get_stage2_params(self):
        """Get Stage2 trainable parameters"""
        return (
            list(self.task_prompts.parameters())
            + list(self.adapters.parameters())
            + list(self.heads.parameters())
        )

    def get_temporal_params(self):
        """Get temporal transformer parameters"""
        if self.temporal_transformer is None:
            return []
        return list(self.temporal_transformer.parameters())


def build_epilabram_extended(
    backbone_size: str = 'base',
    pretrained_path: Optional[str] = None,
    vqnsp_path: Optional[str] = None,
    n_prompt: int = 10,
    adapter_bottleneck_ratio: int = 4,
    n_embed: int = 8192,
    codebook_dim: int = 64,
    task_mode: str = 'default',
    n_classes: int = 5,
    use_rope: bool = True,
    use_lora: bool = True,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    use_temporal: bool = True,
    temporal_size: str = 'base',
    temporal_num_classes: int = 2,
    max_seq_len: int = 512,
) -> EpiLaBraMExtended:
    """
    Factory function to build extended EpiLaBraM model.

    Args:
        backbone_size: 'base', 'large', or 'huge'
        pretrained_path: Path to pretrained backbone weights
        vqnsp_path: Path to pretrained VQ-NSP weights
        n_prompt: Number of prompt tokens
        adapter_bottleneck_ratio: Adapter bottleneck ratio
        n_embed: Codebook size
        codebook_dim: Codebook dimension
        task_mode: Task mode ('default' or 'artifact')
        n_classes: Number of classes for artifact mode
        use_rope: Whether to use RoPE
        use_lora: Whether to use LoRA
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        use_temporal: Whether to include temporal transformer
        temporal_size: Temporal transformer size
        temporal_num_classes: Number of classes for temporal transformer
        max_seq_len: Maximum sequence length for RoPE
    """
    # Build backbone with RoPE support
    backbone = LaBraMBackbone(
        size=backbone_size,
        use_rope=use_rope,
        max_seq_len=max_seq_len
    )

    if pretrained_path:
        backbone.load_pretrained(pretrained_path)

    # Build tokenizer
    tokenizer = NeuralTokenizer(backbone, n_embed=n_embed, embed_dim=codebook_dim)
    if vqnsp_path:
        tokenizer.load_pretrained_vqnsp(vqnsp_path)

    # Build task prompts
    embed_dim = backbone.embed_dim
    task_prompts = TaskPromptTokens(n_tasks=4, n_prompt=n_prompt, embed_dim=embed_dim)

    # Build prediction heads
    if task_mode == 'artifact':
        tuab_head = ArtifactClassificationHead(embed_dim, n_classes=n_classes)
    else:
        from models.prediction_heads import BinaryClassificationHead
        tuab_head = BinaryClassificationHead(embed_dim)

    heads = nn.ModuleDict({
        'TUAB': tuab_head,
        'TUSZ': BinaryClassificationHead(embed_dim),
        'TUEV': MultiClassificationHead(embed_dim, n_classes=6),
        'TUEP': BinaryClassificationHead(embed_dim),
    })

    # Build temporal transformer if requested
    temporal_transformer = None
    if use_temporal:
        temporal_transformer = build_temporal_transformer(
            embed_dim=embed_dim,
            size=temporal_size,
            num_classes=temporal_num_classes,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
        )

    return EpiLaBraMExtended(
        backbone=backbone,
        tokenizer=tokenizer,
        task_prompts=task_prompts,
        heads=heads,
        temporal_transformer=temporal_transformer,
        n_prompt=n_prompt,
        adapter_bottleneck_ratio=adapter_bottleneck_ratio,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
