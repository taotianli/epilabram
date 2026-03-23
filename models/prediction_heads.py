"""
各任务预测头：BinaryClassificationHead / MultiClassificationHead / HierarchicalHead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class BinaryClassificationHead(nn.Module):
    """用于 TUAB, TUSZ, TUEP 的二分类头"""

    def __init__(self, embed_dim: int = 200, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D) → logits (B, 2)"""
        return self.fc(self.drop(x))


class MultiClassificationHead(nn.Module):
    """用于 TUEV 的6类分类头"""

    def __init__(self, embed_dim: int = 200, n_classes: int = 6, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D) → logits (B, n_classes)"""
        return self.fc(self.drop(x))


class ArtifactClassificationHead(nn.Module):
    """5类伪迹分类头 (TUAR)"""
    def __init__(self, embed_dim: int = 200, n_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(x))


class HierarchicalHead(nn.Module):
    """
    三层层级预测头（Stage2专用）：
      Level1: 正常/异常（TUAB，2类）
      Level2: 发作/非发作（TUSZ，2类）
      Level3: 事件细分（TUEV，6类）

    包含层级一致性约束：
      L_consistency = Σ max(0, ŷ_L2 - ŷ_L1)
    """

    def __init__(self, embed_dim: int = 200, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.level1 = nn.Linear(embed_dim, 2)   # normal/abnormal
        self.level2 = nn.Linear(embed_dim, 2)   # background/seizure
        self.level3 = nn.Linear(embed_dim, 6)   # TUEV 6 classes

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (B, D)
        returns: dict with keys 'level1', 'level2', 'level3'
        """
        h = self.drop(x)
        return {
            'level1': self.level1(h),
            'level2': self.level2(h),
            'level3': self.level3(h),
        }

    def consistency_loss(self, logits: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        层级一致性惩罚：若 level2 异常概率 > level1 异常概率则惩罚。
        L_consistency = mean(max(0, p_abnormal_L2 - p_abnormal_L1))
        """
        p_l1 = torch.softmax(logits['level1'], dim=-1)[:, 1]  # P(abnormal)
        p_l2 = torch.softmax(logits['level2'], dim=-1)[:, 1]  # P(seizure)
        return F.relu(p_l2 - p_l1).mean()
