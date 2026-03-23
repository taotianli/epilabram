"""
单元测试：evaluation/metrics.py
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation.metrics import (
    balanced_accuracy, auc_pr, auroc, cohens_kappa,
    weighted_f1, pearson_correlation, r2_score, rmse,
    inter_rater_kappa, MetricTracker
)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        self.y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
        self.y_score = np.array([0.1, 0.6, 0.8, 0.9, 0.2, 0.4, 0.3, 0.7])

    def test_balanced_accuracy(self):
        val = balanced_accuracy(self.y_true, self.y_pred)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)

    def test_balanced_accuracy_perfect(self):
        val = balanced_accuracy(self.y_true, self.y_true)
        self.assertAlmostEqual(val, 1.0)

    def test_auroc(self):
        val = auroc(self.y_true, self.y_score)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)

    def test_auc_pr(self):
        val = auc_pr(self.y_true, self.y_score)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)

    def test_cohens_kappa(self):
        val = cohens_kappa(self.y_true, self.y_pred)
        self.assertGreaterEqual(val, -1.0)
        self.assertLessEqual(val, 1.0)

    def test_cohens_kappa_perfect(self):
        val = cohens_kappa(self.y_true, self.y_true)
        self.assertAlmostEqual(val, 1.0)

    def test_weighted_f1(self):
        val = weighted_f1(self.y_true, self.y_pred)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)

    def test_pearson_correlation(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        val = pearson_correlation(y, y)
        self.assertAlmostEqual(val, 1.0, places=5)

    def test_r2_score_perfect(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        val = r2_score(y, y)
        self.assertAlmostEqual(val, 1.0, places=5)

    def test_rmse_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        val = rmse(y, y)
        self.assertAlmostEqual(val, 0.0, places=5)

    def test_inter_rater_kappa(self):
        val = inter_rater_kappa(self.y_pred, self.y_true)
        self.assertGreaterEqual(val, -1.0)
        self.assertLessEqual(val, 1.0)


class TestMetricTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = MetricTracker()

    def test_update_and_compute(self):
        preds = np.array([0, 1, 1, 0, 1])
        labels = np.array([0, 1, 0, 0, 1])
        scores = np.array([0.1, 0.9, 0.7, 0.2, 0.8])
        self.tracker.update('TUAB', preds, labels, scores)
        results = self.tracker.compute('TUAB')
        self.assertIn('balanced_accuracy', results)
        self.assertIn('auroc', results)
        self.assertIn('weighted_f1', results)

    def test_compute_all(self):
        for task in ['TUAB', 'TUSZ', 'TUEP']:
            preds = np.random.randint(0, 2, 20)
            labels = np.random.randint(0, 2, 20)
            scores = np.random.rand(20)
            self.tracker.update(task, preds, labels, scores)
        results = self.tracker.compute_all()
        self.assertIn('TUAB', results)
        self.assertIn('TUSZ', results)

    def test_reset(self):
        preds = np.array([0, 1])
        labels = np.array([0, 1])
        self.tracker.update('TUAB', preds, labels)
        self.tracker.reset()
        self.assertEqual(len(self.tracker._preds), 0)

    def test_format_table(self):
        preds = np.random.randint(0, 2, 20)
        labels = np.random.randint(0, 2, 20)
        scores = np.random.rand(20)
        self.tracker.update('TUAB', preds, labels, scores)
        table = self.tracker.format_table()
        self.assertIn('TUAB', table)
        self.assertIn('balanced_accuracy', table)

    def test_multiclass_tuev(self):
        preds = np.random.randint(0, 6, 30)
        labels = np.random.randint(0, 6, 30)
        self.tracker.update('TUEV', preds, labels)
        results = self.tracker.compute('TUEV')
        self.assertIn('balanced_accuracy', results)
        self.assertIn('weighted_f1', results)


if __name__ == '__main__':
    unittest.main()
