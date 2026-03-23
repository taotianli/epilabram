"""
统一评估器
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional

from models.epilabram import EpiLaBraM
from evaluation.metrics import MetricTracker


class Evaluator:
    """
    统一评估器，支持所有四个任务。
    """

    TASK_ID_MAP = {'TUAB': 0, 'TUSZ': 1, 'TUEV': 2, 'TUEP': 3}

    def __init__(self, model: EpiLaBraM, device: torch.device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        dataset,
        task_name: str,
        batch_size: int = 128,
        num_workers: int = 4,
    ) -> Dict[str, float]:
        """
        对单个任务数据集进行评估。

        Args:
            dataset: torch Dataset，返回 (eeg, label) 或 (eeg, task_id, label)
            task_name: 'TUAB' / 'TUSZ' / 'TUEV' / 'TUEP'
            batch_size: 推理 batch size

        Returns:
            metrics dict
        """
        self.model.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

        tracker = MetricTracker()
        tid = self.TASK_ID_MAP[task_name]

        for batch in loader:
            eeg = batch[0].to(self.device)
            label = batch[-1].to(self.device)

            B, C, T = eeg.shape
            A = T // 200
            eeg = eeg[:, :, :A * 200].reshape(B, C, A, 200)
            task_ids = torch.full((B,), tid, dtype=torch.long, device=self.device)

            results = self.model.forward_stage2(eeg, task_ids)
            _, logits = results[task_name]
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
            labels_np = label.cpu().numpy()

            tracker.update(task_name, preds, labels_np, probs)

        return tracker.compute(task_name)

    def evaluate_all(
        self,
        datasets: Dict[str, object],
        batch_size: int = 128,
    ) -> Dict[str, Dict[str, float]]:
        """
        评估所有任务。

        Args:
            datasets: {'TUAB': ds, 'TUSZ': ds, ...}

        Returns:
            {'TUAB': {metric: val, ...}, ...}
        """
        results = {}
        for task_name, ds in datasets.items():
            if ds is not None:
                results[task_name] = self.evaluate(ds, task_name, batch_size)
        return results
