"""
评估指标：balanced_accuracy, auc_pr, auroc, cohens_kappa,
          weighted_f1, pearson_correlation, r2_score, rmse,
          inter_rater_kappa, MetricTracker
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    balanced_accuracy_score,
    average_precision_score,
    roc_auc_score,
    cohen_kappa_score,
    f1_score,
    mean_squared_error,
)
from scipy.stats import pearsonr


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced accuracy = mean of per-class recall"""
    return float(balanced_accuracy_score(y_true, y_pred))


def auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under Precision-Recall curve (average precision)"""
    if y_score.ndim == 2:
        y_score = y_score[:, 1]
    return float(average_precision_score(y_true, y_score))


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under ROC curve"""
    if y_score.ndim == 2:
        if y_score.shape[1] == 2:
            y_score = y_score[:, 1]
        else:
            return float(roc_auc_score(y_true, y_score, multi_class='ovr'))
    return float(roc_auc_score(y_true, y_score))


def cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cohen's kappa coefficient"""
    return float(cohen_kappa_score(y_true, y_pred))


def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted F1 score"""
    return float(f1_score(y_true, y_pred, average='weighted', zero_division=0))


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient"""
    r, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    return float(r)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² coefficient of determination"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def inter_rater_kappa(model_preds: np.ndarray, human_labels: np.ndarray) -> float:
    """
    计算模型预测与临床专家标注的一致性 kappa。
    用于 TBME 临床验证。

    Args:
        model_preds:  (N,) int，模型预测类别
        human_labels: (N,) int，临床专家标注

    Returns:
        kappa: float
    """
    return float(cohen_kappa_score(human_labels, model_preds))


class MetricTracker:
    """
    按任务分别追踪所有评估指标，支持输出格式化表格（用于论文 Table）。
    """

    TASK_METRICS = {
        'TUAB': ['balanced_accuracy', 'auroc', 'auc_pr', 'weighted_f1', 'cohens_kappa'],
        'TUSZ': ['balanced_accuracy', 'auroc', 'auc_pr', 'weighted_f1', 'cohens_kappa'],
        'TUEV': ['balanced_accuracy', 'weighted_f1', 'cohens_kappa'],
        'TUEP': ['balanced_accuracy', 'auroc', 'auc_pr', 'weighted_f1', 'cohens_kappa'],
    }

    def __init__(self):
        self._preds: Dict[str, List] = {}
        self._scores: Dict[str, List] = {}
        self._labels: Dict[str, List] = {}

    def update(self, task: str, preds: np.ndarray,
               labels: np.ndarray, scores: Optional[np.ndarray] = None):
        """
        Args:
            task:   任务名称（TUAB/TUSZ/TUEV/TUEP）
            preds:  (N,) int，预测类别
            labels: (N,) int，真实标签
            scores: (N,) or (N, C)，预测概率（可选）
        """
        if task not in self._preds:
            self._preds[task] = []
            self._scores[task] = []
            self._labels[task] = []
        self._preds[task].append(preds)
        self._labels[task].append(labels)
        if scores is not None:
            self._scores[task].append(scores)

    def compute(self, task: str) -> Dict[str, float]:
        """计算指定任务的所有指标"""
        preds = np.concatenate(self._preds[task])
        labels = np.concatenate(self._labels[task])
        scores = np.concatenate(self._scores[task]) if self._scores[task] else None

        results = {}
        for metric in self.TASK_METRICS.get(task, []):
            try:
                if metric == 'balanced_accuracy':
                    results[metric] = balanced_accuracy(labels, preds)
                elif metric == 'auroc' and scores is not None:
                    results[metric] = auroc(labels, scores)
                elif metric == 'auc_pr' and scores is not None:
                    results[metric] = auc_pr(labels, scores)
                elif metric == 'weighted_f1':
                    results[metric] = weighted_f1(labels, preds)
                elif metric == 'cohens_kappa':
                    results[metric] = cohens_kappa(labels, preds)
            except Exception as e:
                results[metric] = float('nan')
        return results

    def compute_all(self) -> Dict[str, Dict[str, float]]:
        """计算所有任务的指标"""
        return {task: self.compute(task) for task in self._preds}

    def reset(self):
        self._preds.clear()
        self._scores.clear()
        self._labels.clear()

    def format_table(self) -> str:
        """输出 LaTeX 风格的格式化表格"""
        all_metrics = self.compute_all()
        lines = []
        header_tasks = list(all_metrics.keys())
        lines.append("Task\t" + "\t".join(header_tasks))

        # 收集所有指标名
        all_metric_names = set()
        for m in all_metrics.values():
            all_metric_names.update(m.keys())

        for metric_name in sorted(all_metric_names):
            row = [metric_name]
            for task in header_tasks:
                val = all_metrics[task].get(metric_name, float('nan'))
                row.append(f"{val:.4f}" if not np.isnan(val) else "-")
            lines.append("\t".join(row))

        return "\n".join(lines)
