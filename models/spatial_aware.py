"""
Spatial-Aware EEG Module

为 TUH EEG（23导联，国际10-20系统）加入空间信息：
  1. ElectrodeCoordEmbedding  — 把3D坐标（近似MNI空间）投影到 embed_dim
  2. EEGGraphConv             — 基于电极距离的轻量图卷积（channel维度空间聚合）
  3. SpatialAwareLaBraM       — 包装 LaBraMBackbone，在 patch embedding 后注入空间信息

设计原则：
  - 不修改任何现有文件
  - 坐标保持在近似MNI空间，为后续 MRI 融合留好接口
  - 可单独使用坐标编码（use_gcn=False），也可叠加图卷积（use_gcn=True）
  - 与原始 backbone 的 forward 签名完全兼容

TUH 23导联顺序（与 data/preprocessing.py STANDARD_CHANNELS 一致）：
  FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 FZ CZ PZ A1 A2 T1 T2
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_

from models.labram_backbone import LaBraMBackbone


# ---------------------------------------------------------------------------
# 10-20 系统 23 导联近似 MNI 坐标（单位 mm，球面投影到头皮）
# 来源：标准 10-20 模板，适合与 MNI 空间 MRI 对齐
# ---------------------------------------------------------------------------
# fmt: off
TUH_23CH_MNI_XYZ = {
    'FP1': (-21,  72,  -4), 'FP2': ( 21,  72,  -4),
    'F7':  (-54,  33,  -3), 'F3':  (-40,  38,  27), 'FZ':  (  0,  46,  38),
    'F4':  ( 40,  38,  27), 'F8':  ( 54,  33,  -3),
    'T1':  (-63,  -1, -18), 'T3':  (-71, -18,  -7), 'C3':  (-53,  -2,  57),
    'CZ':  (  0,  -2,  74), 'C4':  ( 53,  -2,  57), 'T4':  ( 71, -18,  -7),
    'T2':  ( 63,  -1, -18),
    'T5':  (-61, -60,   2), 'P3':  (-40, -57,  52), 'PZ':  (  0, -62,  64),
    'P4':  ( 40, -57,  52), 'T6':  ( 61, -60,   2),
    'O1':  (-25, -93,   5), 'O2':  ( 25, -93,   5),
    'A1':  (-80, -20, -30), 'A2':  ( 80, -20, -30),
}
# fmt: on

# 与 STANDARD_CHANNELS 顺序对齐的坐标矩阵
STANDARD_CHANNELS = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ', 'A1', 'A2', 'T1', 'T2'
]
N_CHANNELS = 23

_coords = torch.tensor(
    [TUH_23CH_MNI_XYZ[ch] for ch in STANDARD_CHANNELS], dtype=torch.float32
)  # (23, 3)

# 归一化到 [-1, 1]
_coords = _coords / _coords.abs().max()


# ---------------------------------------------------------------------------
# 1. 电极坐标嵌入
# ---------------------------------------------------------------------------

class ElectrodeCoordEmbedding(nn.Module):
    """
    把每个电极的 3D MNI 坐标通过 MLP 投影到 embed_dim。

    输出加到对应 channel 的所有时间 patch token 上，
    让模型知道每个 token 来自哪个空间位置。

    这是 MRI 融合的关键接口：坐标在 MNI 空间，
    后续可直接与 MRI ROI 特征做 cross-attention。
    """

    def __init__(self, embed_dim: int, n_channels: int = N_CHANNELS):
        super().__init__()
        self.n_channels = n_channels

        # 注册为 buffer（不参与梯度，但随模型保存/移动设备）
        self.register_buffer('coords', _coords[:n_channels])  # (N, 3)

        # 小 MLP：3 -> embed_dim/2 -> embed_dim
        hidden = max(embed_dim // 2, 16)
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, n_time_patches: int, batch_size: int) -> torch.Tensor:
        """
        返回空间坐标编码，shape (B, N*A, embed_dim)

        Args:
            n_time_patches: 时间 patch 数 A
            batch_size: B
        """
        coord_emb = self.mlp(self.coords)          # (N, D)
        # 扩展到时间维度: (N, D) -> (N, A, D) -> (N*A, D)
        coord_emb = coord_emb.unsqueeze(1).expand(-1, n_time_patches, -1)
        coord_emb = coord_emb.reshape(-1, coord_emb.shape[-1])  # (N*A, D)
        return coord_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N*A, D)

    def get_coords(self) -> torch.Tensor:
        """返回原始 MNI 坐标，供 MRI 融合模块使用。shape (N, 3)"""
        return self.coords


# ---------------------------------------------------------------------------
# 2. 图卷积（channel 维度空间聚合）
# ---------------------------------------------------------------------------

def _build_adjacency(coords: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    基于欧氏距离构建 kNN 邻接矩阵（对称，含自环）。

    Args:
        coords: (N, 3)
        k: 每个节点的邻居数

    Returns:
        adj: (N, N) 归一化邻接矩阵
    """
    N = coords.shape[0]
    dist = torch.cdist(coords, coords)  # (N, N)

    # kNN mask（保留最近 k 个邻居 + 自身）
    _, idx = dist.topk(k + 1, dim=-1, largest=False)
    mask = torch.zeros(N, N, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    mask = mask | mask.t()  # 对称化

    adj = mask.float()
    # 对称归一化: D^{-1/2} A D^{-1/2}
    deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
    adj = adj / deg.sqrt() / deg.sqrt().t()
    return adj


class EEGGraphConv(nn.Module):
    """
    单层图卷积，在 channel 维度做空间聚合。

    对每个时间步，把 N 个 channel token 当作图节点，
    用预计算的 kNN 邻接矩阵做一次消息传递。

    输入/输出 shape 均为 (B, N*A, D)，不改变维度。
    """

    def __init__(self, embed_dim: int, n_channels: int = N_CHANNELS, k: int = 5):
        super().__init__()
        self.n_channels = n_channels
        self.embed_dim = embed_dim

        adj = _build_adjacency(_coords[:n_channels], k=k)
        self.register_buffer('adj', adj)  # (N, N)

        self.linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        trunc_normal_(self.linear.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N*A, D)  — N 个 channel，每个 A 个时间 patch
        """
        B, NA, D = x.shape
        N = self.n_channels
        A = NA // N

        # reshape 到 (B, A, N, D) 方便在 channel 维做图卷积
        x_r = x.reshape(B, N, A, D).permute(0, 2, 1, 3)  # (B, A, N, D)

        # 图卷积: h = adj @ x @ W
        h = torch.einsum('mn, banD -> bamD', self.adj, x_r)  # (B, A, N, D)
        h = self.linear(h)

        # 残差 + LayerNorm
        out = self.norm(x_r + h)  # (B, A, N, D)
        out = out.permute(0, 2, 1, 3).reshape(B, NA, D)  # (B, N*A, D)
        return out


# ---------------------------------------------------------------------------
# 3. SpatialAwareLaBraM — 包装原始 backbone
# ---------------------------------------------------------------------------

class SpatialAwareLaBraM(nn.Module):
    """
    在 LaBraMBackbone 基础上加入空间感知能力。

    流程：
      patch_embed → patch_proj
        → + coord_embedding（坐标编码）
        → gcn（可选，图卷积空间聚合）
        → + spatial_embed + temporal_embed（原始位置编码）
        → Transformer blocks

    Args:
        backbone:   已初始化的 LaBraMBackbone（可带预训练权重）
        use_gcn:    是否使用图卷积（默认 True）
        gcn_k:      图卷积 kNN 邻居数（默认 5）
        n_channels: 导联数（默认 23，TUH 标准）
    """

    def __init__(
        self,
        backbone: LaBraMBackbone,
        use_gcn: bool = True,
        gcn_k: int = 5,
        n_channels: int = N_CHANNELS,
    ):
        super().__init__()
        self.backbone = backbone
        self.use_gcn = use_gcn
        self.n_channels = n_channels
        embed_dim = backbone.embed_dim

        self.coord_embed = ElectrodeCoordEmbedding(embed_dim, n_channels)
        if use_gcn:
            self.gcn = EEGGraphConv(embed_dim, n_channels, k=gcn_k)
        else:
            self.gcn = None

    # ------------------------------------------------------------------
    # 公开属性代理（让外部代码可以直接访问 backbone 属性）
    # ------------------------------------------------------------------

    @property
    def embed_dim(self):
        return self.backbone.embed_dim

    @property
    def blocks(self):
        return self.backbone.blocks

    @property
    def norm(self):
        return self.backbone.norm

    @property
    def fc_norm(self):
        return self.backbone.fc_norm

    @property
    def cls_token(self):
        return self.backbone.cls_token

    @property
    def use_rope(self):
        return self.backbone.use_rope

    def get_electrode_coords(self) -> torch.Tensor:
        """
        返回 MNI 坐标，shape (N, 3)。
        供 MRI 融合模块使用：可以把这些坐标映射到 MRI ROI，
        再做 cross-attention 或特征拼接。
        """
        return self.coord_embed.get_coords()

    # ------------------------------------------------------------------
    # 核心 forward
    # ------------------------------------------------------------------

    def forward_features(
        self,
        x: torch.Tensor,
        input_chans: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
        mask_token: Optional[nn.Parameter] = None,
        return_all_hidden: bool = False,
    ) -> torch.Tensor:
        """
        与 LaBraMBackbone.forward_features 签名完全兼容。

        x: (B, N, A, T)
        """
        B, N, A, T = x.shape
        bb = self.backbone

        # 1. patch embedding（复用 backbone 的 TemporalEncoder + proj）
        x_embed = bb.patch_embed(x)       # (B, N*A, temporal_out_dim)
        x_embed = bb.patch_proj(x_embed)  # (B, N*A, embed_dim)

        # 2. 替换 masked 位置
        if bool_masked_pos is not None and mask_token is not None:
            mt = mask_token.expand(B, x_embed.shape[1], -1)
            w = bool_masked_pos.unsqueeze(-1).type_as(mt)
            x_embed = x_embed * (1 - w) + mt * w

        # 3. 注入坐标编码（在 CLS token 拼接之前，只作用于 patch tokens）
        coord_emb = self.coord_embed(A, B)   # (B, N*A, D)
        x_embed = x_embed + coord_emb

        # 4. 图卷积空间聚合（可选）
        if self.gcn is not None:
            x_embed = self.gcn(x_embed)      # (B, N*A, D)

        # 5. 拼接 CLS token
        cls_tokens = bb.cls_token.expand(B, -1, -1)
        x_embed = torch.cat([cls_tokens, x_embed], dim=1)  # (B, 1+N*A, D)

        # 6. 原始位置编码（spatial + temporal）
        if not bb.use_rope:
            pos = bb.spatial_embed(N, A, batch_size=B, input_chans=input_chans)
            x_embed = x_embed + pos
            te = bb.temporal_embed(x_embed, N, A)
            x_embed[:, 1:, :] = x_embed[:, 1:, :] + te

        x_embed = bb.pos_drop(x_embed)

        # 7. Transformer blocks
        hidden_states = []
        for blk in bb.blocks:
            x_embed = blk(x_embed)
            if return_all_hidden:
                hidden_states.append(x_embed)

        x_embed = bb.norm(x_embed)

        if return_all_hidden:
            return hidden_states, x_embed
        return x_embed

    def forward(
        self,
        x: torch.Tensor,
        input_chans: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
        mask_token: Optional[nn.Parameter] = None,
        return_patch_tokens: bool = False,
        return_all_hidden: bool = False,
    ) -> torch.Tensor:
        """与 LaBraMBackbone.forward 签名完全兼容。"""
        out = self.forward_features(
            x, input_chans, bool_masked_pos, mask_token, return_all_hidden
        )

        if return_all_hidden:
            hidden_states, x_out = out
            return hidden_states, x_out[:, 1:]

        x_out = out
        if return_patch_tokens:
            patch = x_out[:, 1:]
            if self.backbone.fc_norm is not None:
                return self.backbone.fc_norm(patch)
            return patch

        if self.backbone.fc_norm is not None:
            return self.backbone.fc_norm(x_out[:, 1:].mean(1))
        return x_out[:, 0]

    def load_pretrained(self, ckpt_path: str, strict: bool = False):
        """代理到 backbone 的预训练权重加载。"""
        return self.backbone.load_pretrained(ckpt_path, strict=strict)


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def build_spatial_aware_backbone(
    backbone_size: str = 'base',
    pretrained_path: Optional[str] = None,
    use_gcn: bool = True,
    gcn_k: int = 5,
    n_channels: int = N_CHANNELS,
    **backbone_kwargs,
) -> SpatialAwareLaBraM:
    """
    构建带空间感知的 LaBraM backbone。

    用法示例：
        backbone = build_spatial_aware_backbone(
            backbone_size='base',
            pretrained_path='path/to/labram_base.pth',
            use_gcn=True,
        )
        # 替换 build_epilabram 中的 LaBraMBackbone(size=backbone_size)

    MRI 融合接口：
        coords = backbone.get_electrode_coords()  # (23, 3) MNI 坐标
        # 用这些坐标把 EEG token 映射到 MRI ROI，做 cross-attention
    """
    backbone = LaBraMBackbone(size=backbone_size, **backbone_kwargs)
    if pretrained_path:
        backbone.load_pretrained(pretrained_path)

    return SpatialAwareLaBraM(backbone, use_gcn=use_gcn, gcn_k=gcn_k, n_channels=n_channels)
