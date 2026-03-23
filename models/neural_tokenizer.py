"""
VQ-VAE风格神经tokenizer，与原始LaBraM保持兼容。

关键修复：
- 支持从 vqnsp.pth 加载预训练 codebook（避免 index collapse）
- Stage1 训练时 tokenizer 完全冻结，只作为 target 生成器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange
from timm.models.layers import trunc_normal_

from models.labram_backbone import LaBraMBackbone, LaBraMTransformerBlock


# ---------------------------------------------------------------------------
# LaBraM standard_1020 通道列表（与原始 LaBraM utils.py 完全一致）
# ---------------------------------------------------------------------------

_STANDARD_1020 = [
    'FP1', 'FPZ', 'FP2',
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2',
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8',
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h',
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]

# 我们使用的标准 23 通道（与 data/preprocessing.py 一致）
_OUR_23_CHANNELS = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ', 'A1', 'A2', 'T1', 'T2',
]

# 预计算 input_chans 索引（[0] for CLS + 23 channel indices）
_INPUT_CHANS_23 = [0] + [_STANDARD_1020.index(ch) + 1 for ch in _OUR_23_CHANNELS]
_INPUT_CHANS_TENSOR: Optional[torch.Tensor] = None


def _get_standard_input_chans(device: torch.device) -> torch.Tensor:
    """返回标准 23 通道的 input_chans 索引 tensor（缓存）"""
    global _INPUT_CHANS_TENSOR
    if _INPUT_CHANS_TENSOR is None or _INPUT_CHANS_TENSOR.device != device:
        _INPUT_CHANS_TENSOR = torch.tensor(_INPUT_CHANS_23, dtype=torch.long, device=device)
    return _INPUT_CHANS_TENSOR


# ---------------------------------------------------------------------------
# Codebook
# ---------------------------------------------------------------------------

class NeuralCodebook(nn.Module):
    """
    VQ-VAE codebook，EMA 更新。
    K=8192, D=64，ℓ2 归一化最近邻查找。
    """

    def __init__(self, n_embed: int = 8192, embed_dim: int = 64,
                 beta: float = 1.0, decay: float = 0.99):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.decay = decay

        embedding = torch.randn(n_embed, embed_dim)
        embedding = F.normalize(embedding, dim=-1)
        self.register_buffer('embedding', embedding)
        self.register_buffer('cluster_size', torch.ones(n_embed))
        self.register_buffer('embed_avg', embedding.clone())
        self.register_buffer('initted', torch.zeros(1))   # 0=未初始化, 1=已初始化

    @torch.no_grad()
    def _ema_update(self, flat_input: torch.Tensor, encoding_indices: torch.Tensor):
        one_hot = F.one_hot(encoding_indices, self.n_embed).float()
        self.cluster_size.mul_(self.decay).add_(one_hot.sum(0) * (1 - self.decay))
        embed_sum = one_hot.t() @ flat_input
        self.embed_avg.mul_(self.decay).add_(embed_sum * (1 - self.decay))
        n = self.cluster_size.sum()
        smoothed = (self.cluster_size + 1e-5) / (n + self.n_embed * 1e-5) * n
        self.embedding.copy_(F.normalize(self.embed_avg / smoothed.unsqueeze(1), dim=-1))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z: (B, N, D)
        returns: quantized (B, N, D), indices (B*N,), commitment_loss scalar
        """
        B, N, D = z.shape
        z_flat = z.reshape(-1, D)
        z_norm = F.normalize(z_flat, dim=-1)
        emb_norm = F.normalize(self.embedding, dim=-1)

        dist = (z_norm.pow(2).sum(1, keepdim=True)
                - 2 * z_norm @ emb_norm.t()
                + emb_norm.pow(2).sum(1))
        indices = dist.argmin(dim=-1)

        quantized = F.embedding(indices, self.embedding)
        quantized_norm = F.normalize(quantized, dim=-1)

        if self.training:
            self._ema_update(z_norm.detach(), indices)

        commitment_loss = (
            F.mse_loss(z_norm, quantized_norm.detach())
            + self.beta * F.mse_loss(z_norm.detach(), quantized_norm)
        )

        # straight-through
        quantized_st = z_flat + (quantized_norm - z_flat).detach()
        return quantized_st.reshape(B, N, D), indices, commitment_loss


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class NeuralDecoder(nn.Module):
    def __init__(self, embed_dim: int = 64, hidden_dim: int = 200,
                 num_heads: int = 4, depth: int = 3, out_dim: int = 200):
        super().__init__()
        self.proj_in = nn.Linear(embed_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            LaBraMTransformerBlock(hidden_dim, num_heads, mlp_ratio=4.0)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.amplitude_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.phase_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.proj_in(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.amplitude_head(x), self.phase_head(x)


# ---------------------------------------------------------------------------
# 原始 LaBraM VQNSP encoder（用于加载 vqnsp.pth）
# ---------------------------------------------------------------------------

class VQNSPEncoder(nn.Module):
    """
    与 vqnsp.pth 结构完全对应的轻量 encoder。
    只用于推理（生成 codebook indices），训练时完全冻结。

    vqnsp.pth 结构：
      encoder.cls_token, encoder.pos_embed, encoder.time_embed
      encoder.patch_embed.{conv1,norm1,conv2,norm2,conv3,norm3}
      encoder.blocks.{0..2}.{norm1,attn,norm2,mlp}
      encoder.norm
      quantize.cluster_size
      quantize.embedding.{weight, cluster_size, embed_avg, initted}
    """

    def __init__(self, embed_dim: int = 200, depth: int = 3,
                 num_heads: int = 10, out_chans: int = 8,
                 codebook_dim: int = 64, n_embed: int = 8192,
                 max_channels: int = 128, tmax: int = 16):
        super().__init__()
        from models.labram_backbone import (
            TemporalEncoder, SpatialEmbedding, TemporalPositionEmbedding,
            LaBraMTransformerBlock
        )
        self.embed_dim = embed_dim

        # patch embedding
        self.patch_embed = TemporalEncoder(in_chans=1, out_chans=out_chans)
        temporal_out_dim = out_chans * 25
        self.patch_proj = nn.Linear(temporal_out_dim, embed_dim)

        # position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_channels + 1, embed_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, tmax, embed_dim))

        # transformer blocks
        self.blocks = nn.ModuleList([
            LaBraMTransformerBlock(embed_dim, num_heads, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # projection to codebook dim
        self.encode_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Tanh(),
            nn.Linear(embed_dim, codebook_dim),
        )

        # codebook
        self.codebook = NeuralCodebook(n_embed, codebook_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, A, T)
        returns: indices (B, N*A)
        """
        B, N, A, T = x.shape

        # patch embed
        feat = self.patch_embed(x)       # (B, N*A, temporal_out_dim)
        feat = self.patch_proj(feat)     # (B, N*A, embed_dim)

        # prepend CLS
        cls = self.cls_token.expand(B, -1, -1)
        feat = torch.cat([cls, feat], dim=1)  # (B, 1+N*A, D)

        # spatial pos embed (取前 N 个通道)
        ch_pos = self.pos_embed[:, 1:N+1, :]  # (1, N, D)
        spatial = ch_pos.unsqueeze(2).expand(B, -1, A, -1).flatten(1, 2)  # (B, N*A, D)
        cls_pos = self.pos_embed[:, 0:1, :].expand(B, -1, -1)
        pos = torch.cat([cls_pos, spatial], dim=1)
        feat = feat + pos

        # temporal embed
        te = self.time_embed[:, :A, :].unsqueeze(1).expand(B, N, -1, -1).flatten(1, 2)
        feat[:, 1:] = feat[:, 1:] + te

        # transformer
        for blk in self.blocks:
            feat = blk(feat)
        feat = self.norm(feat)

        # project & quantize
        patch_feat = feat[:, 1:]                          # (B, N*A, D)
        proj = self.encode_proj(patch_feat)               # (B, N*A, codebook_dim)
        _, indices, _ = self.codebook(proj)               # indices: (B*N*A,)
        return indices.reshape(B, N * A)

    def load_vqnsp(self, ckpt_path: str):
        """
        从 vqnsp.pth 加载权重，处理 key 名称映射。
        """
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('model', ckpt)

        mapping = {}
        for k, v in sd.items():
            new_k = k
            # encoder.patch_embed.conv1 -> patch_embed.conv1
            new_k = new_k.replace('encoder.patch_embed.conv1.', 'patch_embed.conv1.')
            new_k = new_k.replace('encoder.patch_embed.norm1.', 'patch_embed.norm1.')
            new_k = new_k.replace('encoder.patch_embed.conv2.', 'patch_embed.conv2.')
            new_k = new_k.replace('encoder.patch_embed.norm2.', 'patch_embed.norm2.')
            new_k = new_k.replace('encoder.patch_embed.conv3.', 'patch_embed.conv3.')
            new_k = new_k.replace('encoder.patch_embed.norm3.', 'patch_embed.norm3.')
            # encoder.blocks -> blocks
            new_k = new_k.replace('encoder.blocks.', 'blocks.')
            # encoder.norm -> norm
            new_k = new_k.replace('encoder.norm.', 'norm.')
            # encoder.cls_token -> cls_token
            new_k = new_k.replace('encoder.cls_token', 'cls_token')
            # encoder.pos_embed -> pos_embed
            new_k = new_k.replace('encoder.pos_embed', 'pos_embed')
            # encoder.time_embed -> time_embed
            new_k = new_k.replace('encoder.time_embed', 'time_embed')
            # quantize.embedding.weight -> codebook.embedding
            if new_k == 'quantize.embedding.weight':
                new_k = 'codebook.embedding'
            elif new_k == 'quantize.embedding.cluster_size':
                new_k = 'codebook.cluster_size'
            elif new_k == 'quantize.embedding.embed_avg':
                new_k = 'codebook.embed_avg'
            elif new_k == 'quantize.embedding.initted':
                new_k = 'codebook.initted'
            elif new_k.startswith('quantize.') or new_k.startswith('decoder.'):
                continue  # 跳过 decoder 和其他 quantize key
            # attn key 映射：qkv.weight, q_bias, v_bias
            mapping[new_k] = v

        missing, unexpected = self.load_state_dict(mapping, strict=False)
        # 过滤掉 encode_proj（随机初始化，不在 vqnsp.pth 里）
        real_missing = [k for k in missing if 'encode_proj' not in k and 'patch_proj' not in k]
        print(f'[VQNSPEncoder] loaded: missing={len(real_missing)}, unexpected={len(unexpected)}')
        if real_missing:
            print(f'  missing keys: {real_missing[:10]}')
        return missing, unexpected


# ---------------------------------------------------------------------------
# NeuralTokenizer（Stage1 用）
# ---------------------------------------------------------------------------

class NeuralTokenizer(nn.Module):
    """
    Stage1 tokenizer：加载预训练 vqnsp.pth，完全冻结，只生成 codebook indices。
    """

    def __init__(
        self,
        backbone: LaBraMBackbone,
        n_embed: int = 8192,
        embed_dim: int = 64,
        decoder_out_dim: int = 200,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        # 保留 backbone 引用（兼容旧接口）
        self.backbone = backbone
        hidden = backbone.embed_dim

        self.encode_proj = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, embed_dim),
        )
        self.codebook = NeuralCodebook(n_embed, embed_dim, decay=ema_decay)
        self.decoder = NeuralDecoder(embed_dim, hidden, out_dim=decoder_out_dim)

        # 预训练 VQNSP encoder（冻结）
        self._vqnsp_encoder: Optional[VQNSPEncoder] = None

    def load_pretrained_vqnsp(self, ckpt_path: str, labram_root: str = None):
        """
        用原始 LaBraM 的 create_model 加载 vqnsp.pth，完全避免结构匹配问题。
        加载后冻结，只用于推理生成 codebook indices。

        Args:
            ckpt_path:   vqnsp.pth 路径
            labram_root: 原始 LaBraM 代码根目录（含 modeling_vqnsp.py）。
                         优先级：参数 > 环境变量 LABRAM_ROOT。
        """
        import sys
        import os
        _labram = labram_root or os.environ.get('LABRAM_ROOT', '')
        if not _labram or not os.path.isdir(_labram):
            raise RuntimeError(
                "找不到原始 LaBraM 代码目录（需要其中的 modeling_vqnsp.py）。\n"
                "请通过以下任一方式指定：\n"
                "  1. 环境变量：export LABRAM_ROOT=/path/to/LaBraM\n"
                "  2. 代码参数：tokenizer.load_pretrained_vqnsp(ckpt_path, labram_root=...)\n"
                f"当前值：labram_root={labram_root!r}, LABRAM_ROOT={os.environ.get('LABRAM_ROOT')!r}"
            )
        if _labram not in sys.path:
            sys.path.insert(0, _labram)

        # monkey-patch torch.load 以兼容 PyTorch 2.6+ 的 weights_only 默认值变化
        import torch as _torch
        _orig_load = _torch.load
        def _patched_load(*args, **kwargs):
            kwargs.setdefault('weights_only', False)
            return _orig_load(*args, **kwargs)
        _torch.load = _patched_load

        try:
            from timm.models import create_model
            import modeling_vqnsp  # noqa: registers the model

            vqnsp = create_model(
                'vqnsp_encoder_base_decoder_3x200x12',
                pretrained=True,
                pretrained_weight=ckpt_path,
                as_tokenzer=True,
                n_code=8192,
                code_dim=64,
            ).eval()
        finally:
            _torch.load = _orig_load
            if _labram in sys.path:
                sys.path.remove(_labram)

        for p in vqnsp.parameters():
            p.requires_grad_(False)

        self._vqnsp_encoder = vqnsp
        print(f'[NeuralTokenizer] VQNSP loaded and frozen from {ckpt_path}')

    @torch.no_grad()
    def get_codebook_indices(self, x: torch.Tensor,
                             input_chans=None) -> torch.Tensor:
        """
        生成 codebook indices，用于 Stage1 masked modeling 的 target。

        x: (B, N, A, T)
        returns: (B, N*A) int64
        """
        if self._vqnsp_encoder is not None:
            device = x.device
            enc = self._vqnsp_encoder.to(device)

            # 构建 input_chans 索引（对应 LaBraM standard_1020 列表）
            if input_chans is None:
                input_chans = _get_standard_input_chans(device)

            _, indices, _ = enc.encode(x, input_chans=input_chans)
            return indices.reshape(x.shape[0], -1)

        # fallback（不推荐，会 collapse）
        features = self.backbone(x, input_chans, return_patch_tokens=True)
        proj = self.encode_proj(features)
        _, indices, _ = self.codebook(proj)
        return indices.reshape(x.shape[0], -1)

    def encode(self, x: torch.Tensor,
               input_chans=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(x, input_chans, return_patch_tokens=True)
        proj = self.encode_proj(features)
        quantized, indices, loss = self.codebook(proj)
        return quantized, indices, loss

    def forward(self, x: torch.Tensor, input_chans=None):
        if x.dim() == 3:
            B, N, AT = x.shape
            T = 200
            A = AT // T
            x = x.reshape(B, N, A, T)

        x_fft = torch.fft.rfft(x, dim=-1)
        amplitude_target = torch.abs(x_fft)
        phase_target = torch.angle(x_fft)

        def std_norm(t):
            mean = t.mean(dim=(1, 2, 3), keepdim=True)
            std = t.std(dim=(1, 2, 3), keepdim=True) + 1e-8
            return (t - mean) / std

        amplitude_target = std_norm(amplitude_target)
        phase_target = std_norm(phase_target)

        quantized, indices, commit_loss = self.encode(x, input_chans)
        amp_pred, phase_pred = self.decoder(quantized)

        B, N, A, Tf = amplitude_target.shape
        amp_target_flat = amplitude_target.reshape(B, N * A, Tf)
        phase_target_flat = phase_target.reshape(B, N * A, Tf)

        rec_loss = F.mse_loss(amp_pred, amp_target_flat)
        phase_loss = F.mse_loss(phase_pred, phase_target_flat)
        total_loss = commit_loss + rec_loss + phase_loss

        return total_loss, {
            'train/commit_loss': commit_loss.detach(),
            'train/rec_loss': rec_loss.detach(),
            'train/phase_loss': phase_loss.detach(),
            'train/total_loss': total_loss.detach(),
        }
