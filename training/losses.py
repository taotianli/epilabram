"""
所有损失函数：MaskedEEGModelingLoss / HierarchicalConsistencyLoss / CPADPOLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class MaskedEEGModelingLoss(nn.Module):
    """
    Masked EEG Modeling 损失。

    L_M = -Σ log p(v_i | e_M)  （交叉熵，仅在 masked 位置计算）
    包含对称掩码损失：L = L_M + L_M_sym
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        sym_logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        sym_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            logits:     (B, N, n_embed)
            sym_logits: (B, N, n_embed)
            targets:    (B, N) int64，codebook indices
            mask:       (B, N) bool，True=被掩码
            sym_mask:   (B, N) bool，对称掩码

        Returns:
            total_loss, log_dict
        """
        def _ce(lgt, tgt, msk):
            B, N, V = lgt.shape
            lgt_flat = lgt.reshape(B * N, V)
            tgt_flat = tgt.reshape(B * N)
            msk_flat = msk.reshape(B * N)
            loss = F.cross_entropy(lgt_flat, tgt_flat, reduction='none')
            return (loss * msk_flat.float()).sum() / (msk_flat.float().sum() + 1e-8)

        loss_m = _ce(logits, targets, mask)
        loss_sym = _ce(sym_logits, targets, sym_mask)
        total = loss_m + loss_sym

        # mask accuracy
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct = (pred == targets) & mask
            acc = correct.float().sum() / (mask.float().sum() + 1e-8)

        return total, {
            'loss/mask': loss_m.detach(),
            'loss/sym_mask': loss_sym.detach(),
            'loss/total': total.detach(),
            'metric/mask_acc': acc.detach(),
        }


class HierarchicalConsistencyLoss(nn.Module):
    """
    分层一致性损失。

    L_hier = L_L1 + λ1*L_L2 + λ2*L_L3 + γ*L_consistency
    L_consistency = mean(max(0, p_seizure_L2 - p_abnormal_L1))

    λ1=1.0, λ2=1.0, γ=0.5
    """

    def __init__(
        self,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        gamma: float = 0.5,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits_l1: torch.Tensor,
        logits_l2: Optional[torch.Tensor],
        logits_l3: Optional[torch.Tensor],
        labels_l1: torch.Tensor,
        labels_l2: Optional[torch.Tensor] = None,
        labels_l3: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            logits_l1: (B, 2)
            logits_l2: (B, 2) or None
            logits_l3: (B, 6) or None
            labels_*:  (B,) int64

        Returns:
            total_loss, log_dict
        """
        loss_l1 = F.cross_entropy(logits_l1, labels_l1,
                                  label_smoothing=self.label_smoothing)
        total = loss_l1
        log = {'loss/l1': loss_l1.detach()}

        if logits_l2 is not None and labels_l2 is not None:
            loss_l2 = F.cross_entropy(logits_l2, labels_l2,
                                      label_smoothing=self.label_smoothing)
            total = total + self.lambda1 * loss_l2
            log['loss/l2'] = loss_l2.detach()

            # 层级一致性惩罚
            p_l1 = torch.softmax(logits_l1, dim=-1)[:, 1]
            p_l2 = torch.softmax(logits_l2, dim=-1)[:, 1]
            consist = F.relu(p_l2 - p_l1).mean()
            total = total + self.gamma * consist
            log['loss/consistency'] = consist.detach()

        if logits_l3 is not None and labels_l3 is not None:
            loss_l3 = F.cross_entropy(logits_l3, labels_l3,
                                      label_smoothing=self.label_smoothing)
            total = total + self.lambda2 * loss_l3
            log['loss/l3'] = loss_l3.detach()

        log['loss/total'] = total.detach()
        return total, log


class CPADPOLoss(nn.Module):
    """
    临床偏好对齐 DPO 损失。

    L_DPO = -E[log σ(β * (
        log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)
    ))]

    β=0.1
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        chosen_logprobs: torch.Tensor,
        rejected_logprobs: torch.Tensor,
        ref_chosen_logprobs: torch.Tensor,
        ref_rejected_logprobs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            chosen_logprobs:    (B,) log π_θ(y_w|x)
            rejected_logprobs:  (B,) log π_θ(y_l|x)
            ref_chosen_logprobs:  (B,) log π_ref(y_w|x)
            ref_rejected_logprobs:(B,) log π_ref(y_l|x)

        Returns:
            loss, log_dict
        """
        pi_ratio_chosen = chosen_logprobs - ref_chosen_logprobs
        pi_ratio_rejected = rejected_logprobs - ref_rejected_logprobs
        reward_diff = self.beta * (pi_ratio_chosen - pi_ratio_rejected)
        loss = -F.logsigmoid(reward_diff).mean()

        with torch.no_grad():
            reward_acc = (reward_diff > 0).float().mean()

        return loss, {
            'loss/dpo': loss.detach(),
            'metric/reward_acc': reward_acc,
            'metric/reward_margin': reward_diff.mean().detach(),
        }
