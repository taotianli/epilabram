"""
LaBraM骨干网络，从原始LaBraM代码适配。
支持 Base / Large / Huge 三种规模，支持加载预训练权重。
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_

from models.rope import RoPEAttention


# ---------------------------------------------------------------------------
# 基础组件
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """Stochastic Depth"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = None,
                 out_features: int = None, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    Multi-head self-attention with QK LayerNorm。
    公式：Attention(Q,K,V) = softmax(LN(Q)LN(K)^T / sqrt(d_head)) * V
    """

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 qk_norm: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
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

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor,
                return_attention: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat([
                self.q_bias,
                torch.zeros_like(self.v_bias, requires_grad=False),
                self.v_bias
            ])
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, H, N, head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
            k = self.k_norm(k).type_as(v)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if return_attention:
            return attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LaBraMTransformerBlock(nn.Module):
    """
    标准ViT block，带QK LayerNorm，省略QKV bias项。
    支持 layer scale (init_values > 0)。
    支持 RoPE 替代绝对位置编码。
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, qk_norm: bool = True,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0, init_values: float = 0.0,
                 use_rope: bool = False, max_seq_len: int = 2048):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)

        if use_rope:
            self.attn = RoPEAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                      qk_norm=qk_norm, attn_drop=attn_drop, proj_drop=drop,
                                      max_seq_len=max_seq_len)
        else:
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                  qk_norm=qk_norm, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1 = self.gamma_2 = None

    def forward(self, x: torch.Tensor,
                return_attention: bool = False) -> torch.Tensor:
        if return_attention:
            return self.attn(self.norm1(x), return_attention=True)

        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Temporal Encoder（3层Conv2d块）
# ---------------------------------------------------------------------------

class TemporalEncoder(nn.Module):
    """
    3层1D卷积块（实现为Conv2d），将EEG patch映射为embedding。

    输入: (B, N*A, 1, T)  其中 T=200（patch长度）
    输出: (B, N*A, out_chans * T')

    Base: out_chans=8, Large: out_chans=16, Huge: out_chans=32
    """

    def __init__(self, in_chans: int = 1, out_chans: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.gelu2 = nn.GELU()
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, A, T)  N=channels, A=time_patches, T=patch_size
        returns: (B, N*A, embed_dim)
        """
        x = rearrange(x, 'B N A T -> B (N A) T')
        B, NA, T = x.shape
        x = x.unsqueeze(1)                          # (B, 1, NA, T)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C NA T -> B NA (T C)')  # (B, NA, T'*C)
        return x


# ---------------------------------------------------------------------------
# Spatial & Temporal Position Embeddings
# ---------------------------------------------------------------------------

class SpatialEmbedding(nn.Module):
    """
    为国际10-20系统所有通道维护可学习embedding（最多128通道）。
    支持通道子集索引。
    """

    def __init__(self, max_channels: int = 128, embed_dim: int = 200):
        super().__init__()
        # +1 for CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, max_channels + 1, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, n_channels: int, time_window: int,
                batch_size: int = 1,
                input_chans: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        返回位置编码，shape (B, 1 + n_channels*time_window, embed_dim)

        Args:
            n_channels:  实际通道数 N
            time_window: 时间 patch 数 A
            batch_size:  B
            input_chans: 通道索引 (N,)，None 则取前 n_channels 个
        """
        if input_chans is not None:
            # input_chans: (N,) indices into [1..max_channels]
            ch_pos = self.pos_embed[:, input_chans + 1, :]  # (1, N, D)
        else:
            ch_pos = self.pos_embed[:, 1:n_channels + 1, :]  # (1, N, D)

        # 扩展时间维度: (1, N, D) -> (B, N*A, D)
        spatial = ch_pos.unsqueeze(2).expand(batch_size, -1, time_window, -1).flatten(1, 2)
        cls_pos = self.pos_embed[:, 0:1, :].expand(batch_size, -1, -1)
        return torch.cat([cls_pos, spatial], dim=1)


class TemporalPositionEmbedding(nn.Module):
    """可学习时间位置编码，最大支持 tmax=16 个时间步"""

    def __init__(self, tmax: int = 16, embed_dim: int = 200):
        super().__init__()
        self.time_embed = nn.Parameter(torch.zeros(1, tmax, embed_dim))
        trunc_normal_(self.time_embed, std=0.02)

    def forward(self, x: torch.Tensor, n_channels: int, time_window: int) -> torch.Tensor:
        """
        返回时间位置编码，shape (B, N*A, embed_dim)
        """
        B = x.shape[0]
        te = self.time_embed[:, :time_window, :].unsqueeze(1).expand(B, n_channels, -1, -1).flatten(1, 2)
        return te


# ---------------------------------------------------------------------------
# 完整骨干网络
# ---------------------------------------------------------------------------

# 三种规模配置
BACKBONE_CONFIGS = {
    'base':  dict(depth=12, hidden=200, heads=10, mlp_ratio=4, out_chans=8),
    'large': dict(depth=24, hidden=400, heads=16, mlp_ratio=4, out_chans=16),
    'huge':  dict(depth=48, hidden=800, heads=16, mlp_ratio=4, out_chans=32),
}


class LaBraMBackbone(nn.Module):
    """
    完整LaBraM骨干网络。

    组合：TemporalEncoder + SpatialEmbedding + TemporalPositionEmbedding
          + 多层 LaBraMTransformerBlock

    输入: x (B, N, A, T)  N=23通道, A=时间patch数, T=200
    输出: hidden states list 或 最终 pooled features
    """

    def __init__(
        self,
        size: str = 'base',
        patch_size: int = 200,
        max_channels: int = 128,
        tmax: int = 16,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        init_values: float = 0.0,
        use_mean_pooling: bool = True,
        use_rope: bool = False,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        cfg = BACKBONE_CONFIGS[size]
        self.embed_dim = cfg['hidden']
        self.depth = cfg['depth']
        self.patch_size = patch_size
        self.use_rope = use_rope

        self.patch_embed = TemporalEncoder(in_chans=1, out_chans=cfg['out_chans'])
        # TemporalEncoder 输出维度: out_chans * ceil(patch_size/8) = out_chans * 25
        temporal_out_dim = cfg['out_chans'] * 25
        self.patch_proj = nn.Linear(temporal_out_dim, self.embed_dim)

        # 如果使用 RoPE，则不需要绝对位置编码
        if not use_rope:
            self.spatial_embed = SpatialEmbedding(max_channels, self.embed_dim)
            self.temporal_embed = TemporalPositionEmbedding(tmax, self.embed_dim)
        else:
            self.spatial_embed = None
            self.temporal_embed = None

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        self.blocks = nn.ModuleList([
            LaBraMTransformerBlock(
                dim=self.embed_dim,
                num_heads=cfg['heads'],
                mlp_ratio=cfg['mlp_ratio'],
                qkv_bias=False,
                qk_norm=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                init_values=init_values,
                use_rope=use_rope,
                max_seq_len=max_seq_len,
            )
            for i in range(self.depth)
        ])

        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.fc_norm = nn.LayerNorm(self.embed_dim, eps=1e-6) if use_mean_pooling else None
        self.use_mean_pooling = use_mean_pooling

        self._init_weights()
        self._fix_init_weight()

    def _init_weights(self):
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self.__init_module)

    def __init_module(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _fix_init_weight(self):
        for layer_id, layer in enumerate(self.blocks):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def forward_features(
        self,
        x: torch.Tensor,
        input_chans: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
        mask_token: Optional[nn.Parameter] = None,
        return_all_hidden: bool = False,
    ) -> torch.Tensor:
        """
        x: (B, N, A, T)
        """
        B, N, A, T = x.shape
        x_embed = self.patch_embed(x)        # (B, N*A, temporal_out_dim)
        x_embed = self.patch_proj(x_embed)   # (B, N*A, embed_dim)

        # 替换masked位置
        if bool_masked_pos is not None and mask_token is not None:
            seq_len = x_embed.shape[1]
            mt = mask_token.expand(B, seq_len, -1)
            w = bool_masked_pos.unsqueeze(-1).type_as(mt)
            x_embed = x_embed * (1 - w) + mt * w

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_embed = torch.cat([cls_tokens, x_embed], dim=1)

        # 位置编码（仅在不使用 RoPE 时）
        if not self.use_rope:
            pos = self.spatial_embed(N, A, batch_size=B, input_chans=input_chans)
            x_embed = x_embed + pos

            te = self.temporal_embed(x_embed, N, A)
            x_embed[:, 1:, :] = x_embed[:, 1:, :] + te

        x_embed = self.pos_drop(x_embed)

        hidden_states = []
        for blk in self.blocks:
            x_embed = blk(x_embed)
            if return_all_hidden:
                hidden_states.append(x_embed)

        x_embed = self.norm(x_embed)

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
        """
        Args:
            x: (B, N, A, T)
            return_patch_tokens: 返回所有patch token（不含CLS）
            return_all_hidden: 返回所有层hidden states

        Returns:
            pooled features (B, D) 或 patch tokens (B, N*A, D)
        """
        out = self.forward_features(x, input_chans, bool_masked_pos, mask_token, return_all_hidden)

        if return_all_hidden:
            hidden_states, x_out = out
            return hidden_states, x_out[:, 1:]

        x_out = out
        if return_patch_tokens:
            patch = x_out[:, 1:]
            if self.fc_norm is not None:
                return self.fc_norm(patch)
            return patch

        if self.fc_norm is not None:
            return self.fc_norm(x_out[:, 1:].mean(1))
        return x_out[:, 0]

    def load_pretrained(self, ckpt_path: str, strict: bool = False):
        """
        加载原始LaBraM预训练权重，处理key名称不匹配问题。
        """
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))

        # 过滤无关key
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith(('loss', 'teacher', 'scaling', 'lm_head', 'student.lm_head'))}

        # key重映射：student.xxx -> xxx
        remapped = {}
        for k, v in state_dict.items():
            new_k = k
            if k.startswith('student.'):
                new_k = k[len('student.'):]
            # patch_embed -> patch_embed (TemporalConv)
            new_k = new_k.replace('patch_embed.', 'patch_embed.')
            # pos_embed -> spatial_embed.pos_embed
            if new_k == 'pos_embed':
                new_k = 'spatial_embed.pos_embed'
            # time_embed -> temporal_embed.time_embed
            if new_k == 'time_embed':
                new_k = 'temporal_embed.time_embed'
            remapped[new_k] = v

        missing, unexpected = self.load_state_dict(remapped, strict=strict)
        print(f"[LaBraMBackbone] Loaded pretrained: missing={len(missing)}, unexpected={len(unexpected)}")
        return missing, unexpected
