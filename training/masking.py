"""
PathologyAwareDynamicMasking — 病理感知动态掩码策略
全向量化实现，无逐样本 Python 循环，无 numpy AVX-512 路径
"""

import torch
import torch.nn.functional as F
from typing import Tuple


class PathologyAwareDynamicMasking:
    """
    病理感知动态掩码（PADM）。

    对每个 EEG patch 计算病理分数，高病理分数的 patch 被掩码概率更高。
    全程使用 torch，避免 numpy AVX-512 SIGILL。

    掩码策略：加权无放回采样，精确控制掩码比例 = base_mask_ratio。
    """

    def __init__(
        self,
        sample_rate: float = 200.0,
        base_mask_ratio: float = 0.5,
        alpha: float = 0.3,
        beta: float = 5.0,
        spectral_weight: float = 0.4,
        line_length_weight: float = 0.4,
        peak_weight: float = 0.2,
    ):
        self.fs = sample_rate
        self.base_mask_ratio = base_mask_ratio
        self.alpha = alpha
        self.beta = beta
        self.w_spec = spectral_weight
        self.w_ll = line_length_weight
        self.w_peak = peak_weight

    # ------------------------------------------------------------------
    # 批量病理分数（全向量化）
    # ------------------------------------------------------------------

    def compute_batch_scores_vectorized(self, patches: torch.Tensor) -> torch.Tensor:
        """
        批量计算所有 patch 的归一化病理分数。
        全向量化，无 Python 循环。

        Args:
            patches: (N, T)，N 个 patch，每个长度 T

        Returns:
            scores: (N,) float32，归一化到 [0, 1]
        """
        N, T = patches.shape
        x = patches.float()

        # 1. 频谱分数：1-30Hz 能量占比（批量 FFT）
        fft_mag = torch.fft.rfft(x, dim=-1).abs() ** 2          # (N, T//2+1)
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.fs)          # (T//2+1,)
        band_mask = (freqs >= 1.0) & (freqs <= 30.0)             # (T//2+1,)
        band_energy = fft_mag[:, band_mask].sum(dim=-1)          # (N,)
        total_energy = fft_mag.sum(dim=-1) + 1e-8                # (N,)
        spec_scores = band_energy / total_energy                  # (N,)

        # 2. 线长特征（批量 diff）
        ll_scores = x.diff(dim=-1).abs().mean(dim=-1)            # (N,)

        # 3. 峰值数量：局部极大值 + 阈值（批量）
        x_abs = x.abs()
        threshold = x_abs.std(dim=-1, keepdim=True) * 0.5       # (N, 1)
        if T > 2:
            left  = x_abs[:, 1:-1] > x_abs[:, :-2]
            right = x_abs[:, 1:-1] > x_abs[:, 2:]
            above = x_abs[:, 1:-1] > threshold
            pk_scores = (left & right & above).float().sum(dim=-1) / (T / self.fs + 1e-8)
        else:
            pk_scores = torch.zeros(N, device=x.device)

        # min-max 归一化
        def minmax(t: torch.Tensor) -> torch.Tensor:
            mn = t.min()
            mx = t.max()
            return (t - mn) / (mx - mn + 1e-8)

        spec_scores = minmax(spec_scores)
        ll_scores   = minmax(ll_scores)
        pk_scores   = minmax(pk_scores)

        return self.w_spec * spec_scores + self.w_ll * ll_scores + self.w_peak * pk_scores

    # ------------------------------------------------------------------
    # 批量掩码生成（全向量化）
    # ------------------------------------------------------------------

    def generate_mask_batch(
        self,
        patch_scores: torch.Tensor,
        base_mask_ratio: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量生成 binary mask。

        Args:
            patch_scores: (B, N) 归一化病理分数
            base_mask_ratio: 覆盖默认值

        Returns:
            mask:     (B, N) bool
            sym_mask: (B, N) bool，互补
        """
        ratio = base_mask_ratio if base_mask_ratio is not None else self.base_mask_ratio
        B, N = patch_scores.shape
        n_mask = max(1, round(N * ratio))

        # 采样权重：中心化后 sigmoid
        weights = self.alpha + (1 - self.alpha) * torch.sigmoid(
            self.beta * (patch_scores.float() - 0.5)
        )  # (B, N)

        # 批量加权无放回采样
        selected = torch.multinomial(weights, num_samples=n_mask, replacement=False)  # (B, n_mask)
        mask = torch.zeros(B, N, dtype=torch.bool, device=patch_scores.device)
        mask.scatter_(1, selected, True)

        return mask, ~mask

    # ------------------------------------------------------------------
    # 主接口：处理整个 batch
    # ------------------------------------------------------------------

    def __call__(
        self,
        eeg_batch: torch.Tensor,
        patch_size: int = 200,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对整个 batch 生成掩码（全向量化）。

        Args:
            eeg_batch: (B, C, T) 或 (C, T) 单样本
            patch_size: 每个 patch 的时间步数

        Returns:
            mask:     (B, C*n_patches) 或 (C*n_patches,)
            sym_mask: 同上，互补
        """
        single = eeg_batch.dim() == 2
        if single:
            eeg_batch = eeg_batch.unsqueeze(0)  # (1, C, T)

        B, C, T = eeg_batch.shape
        n_patches = T // patch_size
        # reshape 成 (B*C*n_patches, patch_size)
        patches = eeg_batch[:, :, :n_patches * patch_size] \
                      .reshape(B, C * n_patches, patch_size) \
                      .reshape(B * C * n_patches, patch_size)

        scores = self.compute_batch_scores_vectorized(patches)   # (B*C*n_patches,)
        scores = scores.reshape(B, C * n_patches)                # (B, N)

        mask, sym_mask = self.generate_mask_batch(scores)        # (B, N)

        if single:
            return mask[0], sym_mask[0]
        return mask, sym_mask
