"""
课程学习数据调度器
训练初期以TUAB为主，逐渐引入TUSZ, TUEV, TUEP
"""

from typing import List


class CurriculumScheduler:
    """
    按课程学习策略调度多数据集采样权重。

    epoch 0-10:  weights=[0.6, 0.2, 0.1, 0.1]  (TUAB主导)
    epoch 11-30: weights=[0.3, 0.3, 0.2, 0.2]  (逐渐均衡)
    epoch 31+:   weights=[0.25, 0.25, 0.25, 0.25] (完全均衡)

    数据集顺序: [TUAB, TUSZ, TUEV, TUEP]
    """

    def __init__(
        self,
        initial_weights: List[float] = None,
        mid_weights: List[float] = None,
        final_weights: List[float] = None,
        stage1_end_epoch: int = 10,
        stage2_end_epoch: int = 30,
    ):
        self.initial_weights = initial_weights or [0.6, 0.2, 0.1, 0.1]
        self.mid_weights = mid_weights or [0.3, 0.3, 0.2, 0.2]
        self.final_weights = final_weights or [0.25, 0.25, 0.25, 0.25]
        self.stage1_end_epoch = stage1_end_epoch
        self.stage2_end_epoch = stage2_end_epoch

    def get_sampling_weights(self, epoch: int, total_epochs: int) -> List[float]:
        """
        返回各数据集的采样权重（随epoch线性变化）。

        Args:
            epoch: 当前epoch（0-indexed）
            total_epochs: 总epoch数

        Returns:
            weights: 长度为4的权重列表，对应[TUAB, TUSZ, TUEV, TUEP]
        """
        if epoch <= self.stage1_end_epoch:
            # 在initial和mid之间线性插值
            t = epoch / self.stage1_end_epoch if self.stage1_end_epoch > 0 else 1.0
            weights = [
                self.initial_weights[i] + t * (self.mid_weights[i] - self.initial_weights[i])
                for i in range(4)
            ]
        elif epoch <= self.stage2_end_epoch:
            # 在mid和final之间线性插值
            span = self.stage2_end_epoch - self.stage1_end_epoch
            t = (epoch - self.stage1_end_epoch) / span if span > 0 else 1.0
            weights = [
                self.mid_weights[i] + t * (self.final_weights[i] - self.mid_weights[i])
                for i in range(4)
            ]
        else:
            weights = list(self.final_weights)

        # 归一化确保和为1
        total = sum(weights)
        return [w / total for w in weights]
