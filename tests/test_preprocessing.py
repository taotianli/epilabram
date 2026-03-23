"""
单元测试：data/preprocessing.py
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.preprocessing import (
    bandpass_filter, notch_filter, resample_eeg,
    amplitude_normalize, ChannelAligner, EEGPreprocessor,
    STANDARD_CHANNELS, N_CHANNELS
)


class TestBandpassFilter(unittest.TestCase):
    def setUp(self):
        self.fs = 256.0
        self.T = 2560
        self.C = 4
        self.eeg = np.random.randn(self.C, self.T).astype(np.float32)

    def test_output_shape(self):
        out = bandpass_filter(self.eeg, self.fs)
        self.assertEqual(out.shape, self.eeg.shape)

    def test_attenuates_dc(self):
        # DC component should be attenuated
        dc = np.ones((1, self.T), dtype=np.float32) * 100.0
        out = bandpass_filter(dc, self.fs, low=0.1, high=75.0)
        self.assertLess(np.abs(out).mean(), 10.0)


class TestNotchFilter(unittest.TestCase):
    def test_output_shape(self):
        eeg = np.random.randn(4, 2560).astype(np.float32)
        out = notch_filter(eeg, 256.0)
        self.assertEqual(out.shape, eeg.shape)

    def test_attenuates_50hz(self):
        fs = 256.0
        T = 2560
        t = np.arange(T) / fs
        # Pure 50Hz signal
        sig = np.sin(2 * np.pi * 50 * t).reshape(1, -1).astype(np.float32)
        out = notch_filter(sig, fs, freq=50.0)
        self.assertLess(np.abs(out).mean(), 0.1)


class TestResample(unittest.TestCase):
    def test_upsample(self):
        eeg = np.random.randn(4, 2560).astype(np.float32)
        out = resample_eeg(eeg, orig_fs=256.0, target_fs=200.0)
        expected_T = int(2560 * 200 / 256)
        self.assertAlmostEqual(out.shape[1], expected_T, delta=5)

    def test_same_rate(self):
        eeg = np.random.randn(4, 2000).astype(np.float32)
        out = resample_eeg(eeg, orig_fs=200.0, target_fs=200.0)
        np.testing.assert_array_equal(out, eeg)


class TestAmplitudeNormalize(unittest.TestCase):
    def test_scale(self):
        eeg = np.ones((4, 200), dtype=np.float32) * 100.0
        out = amplitude_normalize(eeg, scale=100.0)
        np.testing.assert_allclose(out, np.ones_like(out))


class TestChannelAligner(unittest.TestCase):
    def setUp(self):
        self.aligner = ChannelAligner()

    def test_full_match(self):
        T = 200
        eeg = np.random.randn(N_CHANNELS, T).astype(np.float32)
        aligned, mask = self.aligner.align(eeg, STANDARD_CHANNELS)
        self.assertEqual(aligned.shape, (N_CHANNELS, T))
        self.assertTrue(mask.all())

    def test_partial_match(self):
        T = 200
        ch_names = ['FP1', 'C3', 'O1']
        eeg = np.random.randn(3, T).astype(np.float32)
        aligned, mask = self.aligner.align(eeg, ch_names)
        self.assertEqual(aligned.shape, (N_CHANNELS, T))
        self.assertEqual(mask.sum(), 3)

    def test_missing_channels_zero(self):
        T = 200
        eeg = np.random.randn(1, T).astype(np.float32)
        aligned, mask = self.aligner.align(eeg, ['FP1'])
        # All channels except FP1 should be zero
        fp1_idx = STANDARD_CHANNELS.index('FP1')
        for i in range(N_CHANNELS):
            if i != fp1_idx:
                np.testing.assert_array_equal(aligned[i], np.zeros(T))

    def test_alias_mapping(self):
        T = 200
        eeg = np.random.randn(2, T).astype(np.float32)
        aligned, mask = self.aligner.align(eeg, ['FP1-REF', 'C3-LE'])
        self.assertEqual(mask.sum(), 2)

    def test_empty_channels(self):
        T = 200
        eeg = np.zeros((0, T), dtype=np.float32)
        aligned, mask = self.aligner.align(eeg, [])
        self.assertEqual(aligned.shape, (N_CHANNELS, T))
        self.assertFalse(mask.any())


class TestEEGPreprocessor(unittest.TestCase):
    def test_pipeline(self):
        preprocessor = EEGPreprocessor()
        eeg = np.random.randn(4, 2560).astype(np.float32) * 50
        ch_names = ['FP1', 'C3', 'O1', 'FZ']
        aligned, mask = preprocessor(eeg, ch_names, orig_fs=256.0)
        self.assertEqual(aligned.shape[0], N_CHANNELS)
        self.assertEqual(mask.sum(), 4)


if __name__ == '__main__':
    unittest.main()
