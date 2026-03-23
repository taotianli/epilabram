"""
单元测试：training/losses.py
"""

import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from training.losses import MaskedEEGModelingLoss, HierarchicalConsistencyLoss, CPADPOLoss


class TestMaskedEEGModelingLoss(unittest.TestCase):
    def setUp(self):
        self.criterion = MaskedEEGModelingLoss()
        self.B, self.N, self.V = 4, 20, 8192

    def test_output_shape(self):
        logits = torch.randn(self.B, self.N, self.V)
        sym_logits = torch.randn(self.B, self.N, self.V)
        targets = torch.randint(0, self.V, (self.B, self.N))
        mask = torch.rand(self.B, self.N) > 0.5
        sym_mask = ~mask
        loss, log = self.criterion(logits, sym_logits, targets, mask, sym_mask)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertIn('loss/total', log)
        self.assertIn('metric/mask_acc', log)

    def test_loss_positive(self):
        logits = torch.randn(self.B, self.N, self.V)
        sym_logits = torch.randn(self.B, self.N, self.V)
        targets = torch.randint(0, self.V, (self.B, self.N))
        mask = torch.ones(self.B, self.N, dtype=torch.bool)
        sym_mask = torch.ones(self.B, self.N, dtype=torch.bool)
        loss, _ = self.criterion(logits, sym_logits, targets, mask, sym_mask)
        self.assertGreater(loss.item(), 0.0)

    def test_perfect_prediction_low_loss(self):
        # 完美预测时 loss 应该接近 0
        targets = torch.zeros(self.B, self.N, dtype=torch.long)
        logits = torch.full((self.B, self.N, self.V), -1e9)
        logits[:, :, 0] = 1e9  # 全部预测为类别0
        sym_logits = logits.clone()
        mask = torch.ones(self.B, self.N, dtype=torch.bool)
        sym_mask = torch.ones(self.B, self.N, dtype=torch.bool)
        loss, log = self.criterion(logits, sym_logits, targets, mask, sym_mask)
        self.assertLess(loss.item(), 0.01)
        self.assertAlmostEqual(log['metric/mask_acc'].item(), 1.0, places=3)

    def test_empty_mask(self):
        logits = torch.randn(self.B, self.N, self.V)
        sym_logits = torch.randn(self.B, self.N, self.V)
        targets = torch.randint(0, self.V, (self.B, self.N))
        mask = torch.zeros(self.B, self.N, dtype=torch.bool)
        sym_mask = torch.zeros(self.B, self.N, dtype=torch.bool)
        loss, _ = self.criterion(logits, sym_logits, targets, mask, sym_mask)
        self.assertEqual(loss.item(), 0.0)


class TestHierarchicalConsistencyLoss(unittest.TestCase):
    def setUp(self):
        self.criterion = HierarchicalConsistencyLoss(lambda1=1.0, lambda2=1.0, gamma=0.5)
        self.B = 8

    def test_l1_only(self):
        logits_l1 = torch.randn(self.B, 2)
        labels_l1 = torch.randint(0, 2, (self.B,))
        loss, log = self.criterion(logits_l1, None, None, labels_l1)
        self.assertGreater(loss.item(), 0.0)
        self.assertIn('loss/l1', log)
        self.assertNotIn('loss/l2', log)

    def test_all_levels(self):
        logits_l1 = torch.randn(self.B, 2)
        logits_l2 = torch.randn(self.B, 2)
        logits_l3 = torch.randn(self.B, 6)
        labels_l1 = torch.randint(0, 2, (self.B,))
        labels_l2 = torch.randint(0, 2, (self.B,))
        labels_l3 = torch.randint(0, 6, (self.B,))
        loss, log = self.criterion(logits_l1, logits_l2, logits_l3,
                                   labels_l1, labels_l2, labels_l3)
        self.assertIn('loss/l1', log)
        self.assertIn('loss/l2', log)
        self.assertIn('loss/l3', log)
        self.assertIn('loss/consistency', log)

    def test_consistency_nonneg(self):
        logits_l1 = torch.randn(self.B, 2)
        logits_l2 = torch.randn(self.B, 2)
        labels = torch.randint(0, 2, (self.B,))
        _, log = self.criterion(logits_l1, logits_l2, None, labels, labels)
        self.assertGreaterEqual(log['loss/consistency'].item(), 0.0)


class TestCPADPOLoss(unittest.TestCase):
    def setUp(self):
        self.criterion = CPADPOLoss(beta=0.1)
        self.B = 16

    def test_output_shape(self):
        chosen = torch.randn(self.B)
        rejected = torch.randn(self.B)
        ref_chosen = torch.randn(self.B)
        ref_rejected = torch.randn(self.B)
        loss, log = self.criterion(chosen, rejected, ref_chosen, ref_rejected)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertIn('loss/dpo', log)
        self.assertIn('metric/reward_acc', log)

    def test_loss_positive(self):
        chosen = torch.randn(self.B)
        rejected = torch.randn(self.B)
        ref_chosen = torch.zeros(self.B)
        ref_rejected = torch.zeros(self.B)
        loss, _ = self.criterion(chosen, rejected, ref_chosen, ref_rejected)
        self.assertGreater(loss.item(), 0.0)

    def test_reward_acc_range(self):
        chosen = torch.randn(self.B)
        rejected = torch.randn(self.B)
        ref_chosen = torch.zeros(self.B)
        ref_rejected = torch.zeros(self.B)
        _, log = self.criterion(chosen, rejected, ref_chosen, ref_rejected)
        acc = log['metric/reward_acc'].item()
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_perfect_preference(self):
        # chosen 明显优于 rejected 时，reward_acc 应该为 1，loss 应该较小
        # beta=0.1, reward_diff = 0.1*(10-(-10)) = 2.0, loss = -log(sigmoid(2)) ≈ 0.127
        chosen = torch.full((self.B,), 10.0)
        rejected = torch.full((self.B,), -10.0)
        ref_chosen = torch.zeros(self.B)
        ref_rejected = torch.zeros(self.B)
        loss, log = self.criterion(chosen, rejected, ref_chosen, ref_rejected)
        self.assertLess(loss.item(), 0.2)
        self.assertAlmostEqual(log['metric/reward_acc'].item(), 1.0, places=3)


if __name__ == '__main__':
    unittest.main()
