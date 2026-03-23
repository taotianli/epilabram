"""
单元测试：data/curriculum.py
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.curriculum import CurriculumScheduler


class TestCurriculumScheduler(unittest.TestCase):
    def setUp(self):
        self.scheduler = CurriculumScheduler(
            initial_weights=[0.6, 0.2, 0.1, 0.1],
            mid_weights=[0.3, 0.3, 0.2, 0.2],
            final_weights=[0.25, 0.25, 0.25, 0.25],
            stage1_end_epoch=10,
            stage2_end_epoch=30,
        )

    def test_weights_sum_to_one(self):
        for epoch in [0, 5, 10, 15, 30, 50]:
            weights = self.scheduler.get_sampling_weights(epoch, 100)
            self.assertAlmostEqual(sum(weights), 1.0, places=5)

    def test_initial_phase_tuab_dominant(self):
        weights = self.scheduler.get_sampling_weights(0, 100)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[0], weights[2])
        self.assertGreater(weights[0], weights[3])

    def test_final_phase_uniform(self):
        weights = self.scheduler.get_sampling_weights(50, 100)
        for w in weights:
            self.assertAlmostEqual(w, 0.25, places=3)

    def test_weights_length(self):
        weights = self.scheduler.get_sampling_weights(0, 100)
        self.assertEqual(len(weights), 4)

    def test_monotone_tuab_decrease(self):
        w0 = self.scheduler.get_sampling_weights(0, 100)[0]
        w10 = self.scheduler.get_sampling_weights(10, 100)[0]
        w30 = self.scheduler.get_sampling_weights(30, 100)[0]
        self.assertGreaterEqual(w0, w10)
        self.assertGreaterEqual(w10, w30)

    def test_mid_phase_transition(self):
        # epoch=10 应该等于 mid_weights（线性插值 t=1）
        weights = self.scheduler.get_sampling_weights(10, 100)
        for i, mw in enumerate([0.3, 0.3, 0.2, 0.2]):
            self.assertAlmostEqual(weights[i], mw, places=3)

    def test_all_weights_positive(self):
        for epoch in range(0, 60, 5):
            weights = self.scheduler.get_sampling_weights(epoch, 100)
            for w in weights:
                self.assertGreater(w, 0.0)


if __name__ == '__main__':
    unittest.main()
