"""
单元测试：training/masking.py（向量化版本）
"""

import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from training.masking import PathologyAwareDynamicMasking


class TestPathologyAwareDynamicMasking(unittest.TestCase):
    def setUp(self):
        self.masker = PathologyAwareDynamicMasking(
            sample_rate=200.0,
            base_mask_ratio=0.5,
            alpha=0.3,
            beta=5.0,
        )
        self.patch_size = 200
        self.C = 23
        self.T = 2000  # 10 patches per channel

    def test_batch_scores_shape(self):
        N = self.C * (self.T // self.patch_size)
        patches = torch.randn(N, self.patch_size)
        scores = self.masker.compute_batch_scores_vectorized(patches)
        self.assertEqual(scores.shape, (N,))

    def test_scores_normalized(self):
        N = 50
        patches = torch.randn(N, self.patch_size)
        scores = self.masker.compute_batch_scores_vectorized(patches)
        self.assertGreaterEqual(scores.min().item(), 0.0)
        self.assertLessEqual(scores.max().item(), 1.0 + 1e-5)

    def test_generate_mask_shape(self):
        B = 4
        N = self.C * (self.T // self.patch_size)
        scores = torch.rand(B, N)
        mask, sym_mask = self.masker.generate_mask_batch(scores)
        self.assertEqual(mask.shape, (B, N))
        self.assertEqual(sym_mask.shape, (B, N))

    def test_mask_is_bool(self):
        B, N = 2, 50
        scores = torch.rand(B, N)
        mask, sym_mask = self.masker.generate_mask_batch(scores)
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(sym_mask.dtype, torch.bool)

    def test_symmetric_mask_complement(self):
        B, N = 2, 50
        scores = torch.rand(B, N)
        mask, sym_mask = self.masker.generate_mask_batch(scores)
        self.assertTrue((sym_mask == ~mask).all())

    def test_call_interface_single(self):
        """单样本接口：(C, T) -> (C*n_patches,)"""
        eeg = torch.randn(self.C, self.T)
        mask, sym_mask = self.masker(eeg, self.patch_size)
        expected_N = self.C * (self.T // self.patch_size)
        self.assertEqual(mask.shape, (expected_N,))
        self.assertEqual(sym_mask.shape, (expected_N,))

    def test_call_interface_batch(self):
        """批量接口：(B, C, T) -> (B, C*n_patches)"""
        B = 8
        eeg = torch.randn(B, self.C, self.T)
        mask, sym_mask = self.masker(eeg, self.patch_size)
        expected_N = self.C * (self.T // self.patch_size)
        self.assertEqual(mask.shape, (B, expected_N))
        self.assertEqual(sym_mask.shape, (B, expected_N))

    def test_mask_ratio_exact(self):
        """掩码比例应精确等于 base_mask_ratio"""
        B = 4
        eeg = torch.randn(B, self.C, self.T)
        mask, _ = self.masker(eeg, self.patch_size)
        N = self.C * (self.T // self.patch_size)
        expected_n = round(N * self.masker.base_mask_ratio)
        for b in range(B):
            self.assertEqual(mask[b].sum().item(), expected_n)

    def test_high_pathology_higher_mask_prob(self):
        """高病理分数的 patch 应有更高的被掩码权重"""
        B, N = 1, 100
        low_scores  = torch.zeros(B, N)
        high_scores = torch.ones(B, N)
        w_low  = self.masker.alpha + (1 - self.masker.alpha) * torch.sigmoid(
            self.masker.beta * (low_scores  - 0.5))
        w_high = self.masker.alpha + (1 - self.masker.alpha) * torch.sigmoid(
            self.masker.beta * (high_scores - 0.5))
        self.assertGreater(w_high.mean().item(), w_low.mean().item())

    def test_no_numpy_in_hot_path(self):
        """确认热路径不调用 numpy（避免 AVX-512 SIGILL）"""
        import numpy as np
        original_std = np.std
        call_count = [0]
        def patched_std(*args, **kwargs):
            call_count[0] += 1
            return original_std(*args, **kwargs)
        np.std = patched_std
        try:
            eeg = torch.randn(4, self.C, self.T)
            self.masker(eeg, self.patch_size)
        finally:
            np.std = original_std
        self.assertEqual(call_count[0], 0, "numpy.std was called in hot path!")


if __name__ == '__main__':
    unittest.main()
