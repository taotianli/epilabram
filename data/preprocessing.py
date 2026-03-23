"""
EEG预处理流水线
包含带通滤波、陷波滤波、重采样、归一化、通道对齐
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy import signal


# 国际10-20系统标准23通道集合
STANDARD_CHANNELS = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ', 'A1', 'A2', 'T1', 'T2'
]
N_CHANNELS = len(STANDARD_CHANNELS)  # 23


def bandpass_filter(eeg: np.ndarray, fs: float, low: float = 0.1, high: float = 75.0, order: int = 4) -> np.ndarray:
    """
    Butterworth带通滤波器

    Args:
        eeg: shape (C, T)
        fs: 采样率
        low: 低截止频率 Hz
        high: 高截止频率 Hz
        order: 滤波器阶数

    Returns:
        filtered eeg, shape (C, T)
    """
    nyq = fs / 2.0
    b, a = signal.butter(order, [low / nyq, high / nyq], btype='band')
    return signal.filtfilt(b, a, eeg, axis=-1)


def notch_filter(eeg: np.ndarray, fs: float, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """
    陷波滤波器，去除工频干扰

    Args:
        eeg: shape (C, T)
        fs: 采样率
        freq: 陷波频率 Hz
        q: 品质因数

    Returns:
        filtered eeg, shape (C, T)
    """
    b, a = signal.iirnotch(freq / (fs / 2.0), q)
    return signal.filtfilt(b, a, eeg, axis=-1)


def resample_eeg(eeg: np.ndarray, orig_fs: float, target_fs: float = 200.0) -> np.ndarray:
    """
    使用 resample_poly 重采样至目标采样率

    Args:
        eeg: shape (C, T)
        orig_fs: 原始采样率
        target_fs: 目标采样率

    Returns:
        resampled eeg, shape (C, T')
    """
    if orig_fs == target_fs:
        return eeg
    from math import gcd
    orig_fs_int = int(round(orig_fs))
    target_fs_int = int(round(target_fs))
    g = gcd(orig_fs_int, target_fs_int)
    up = target_fs_int // g
    down = orig_fs_int // g
    return signal.resample_poly(eeg, up, down, axis=-1)


def amplitude_normalize(eeg: np.ndarray, scale: float = 100.0) -> np.ndarray:
    """
    幅值归一化：除以scale（单位0.1mV → 值域约[-1, 1]）

    Args:
        eeg: shape (C, T)
        scale: 归一化系数，默认100

    Returns:
        normalized eeg, shape (C, T)
    """
    return eeg / scale


class ChannelAligner:
    """
    将任意电极配置对齐至国际10-20系统23通道集合。
    缺失通道用零向量填充，并记录有效通道mask。

    标准通道集合（23通道）：
    ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2',
     'F7','F8','T3','T4','T5','T6','FZ','CZ','PZ','A1','A2','T1','T2']
    """

    # 常见别名映射（原始名 -> 标准名）
    ALIAS_MAP = {
        'FP1': 'FP1', 'FP2': 'FP2',
        'F3': 'F3', 'F4': 'F4',
        'C3': 'C3', 'C4': 'C4',
        'P3': 'P3', 'P4': 'P4',
        'O1': 'O1', 'O2': 'O2',
        'F7': 'F7', 'F8': 'F8',
        'T3': 'T3', 'T4': 'T4',
        'T5': 'T5', 'T6': 'T6',
        'FZ': 'FZ', 'CZ': 'CZ', 'PZ': 'PZ',
        'A1': 'A1', 'A2': 'A2',
        'T1': 'T1', 'T2': 'T2',
        # TUH常见变体
        'EEG FP1': 'FP1', 'EEG FP2': 'FP2',
        'EEG F3': 'F3', 'EEG F4': 'F4',
        'EEG C3': 'C3', 'EEG C4': 'C4',
        'EEG P3': 'P3', 'EEG P4': 'P4',
        'EEG O1': 'O1', 'EEG O2': 'O2',
        'EEG F7': 'F7', 'EEG F8': 'F8',
        'EEG T3': 'T3', 'EEG T4': 'T4',
        'EEG T5': 'T5', 'EEG T6': 'T6',
        'EEG FZ': 'FZ', 'EEG CZ': 'CZ', 'EEG PZ': 'PZ',
        'EEG A1': 'A1', 'EEG A2': 'A2',
        'EEG T1': 'T1', 'EEG T2': 'T2',
        # 带-REF后缀
        'FP1-REF': 'FP1', 'FP2-REF': 'FP2',
        'F3-REF': 'F3', 'F4-REF': 'F4',
        'C3-REF': 'C3', 'C4-REF': 'C4',
        'P3-REF': 'P3', 'P4-REF': 'P4',
        'O1-REF': 'O1', 'O2-REF': 'O2',
        'F7-REF': 'F7', 'F8-REF': 'F8',
        'T3-REF': 'T3', 'T4-REF': 'T4',
        'T5-REF': 'T5', 'T6-REF': 'T6',
        'FZ-REF': 'FZ', 'CZ-REF': 'CZ', 'PZ-REF': 'PZ',
        'A1-REF': 'A1', 'A2-REF': 'A2',
        'T1-REF': 'T1', 'T2-REF': 'T2',
        # 带-LE后缀
        'FP1-LE': 'FP1', 'FP2-LE': 'FP2',
        'F3-LE': 'F3', 'F4-LE': 'F4',
        'C3-LE': 'C3', 'C4-LE': 'C4',
        'P3-LE': 'P3', 'P4-LE': 'P4',
        'O1-LE': 'O1', 'O2-LE': 'O2',
        'F7-LE': 'F7', 'F8-LE': 'F8',
        'T3-LE': 'T3', 'T4-LE': 'T4',
        'T5-LE': 'T5', 'T6-LE': 'T6',
        'FZ-LE': 'FZ', 'CZ-LE': 'CZ', 'PZ-LE': 'PZ',
        'A1-LE': 'A1', 'A2-LE': 'A2',
        'T1-LE': 'T1', 'T2-LE': 'T2',
    }

    def __init__(self, standard_channels: Optional[List[str]] = None):
        self.standard_channels = standard_channels or STANDARD_CHANNELS
        self.ch_to_idx = {ch: i for i, ch in enumerate(self.standard_channels)}

    def _normalize_name(self, name: str) -> str:
        """统一大写并去除空格，再查别名表"""
        name_upper = name.upper().strip()
        return self.ALIAS_MAP.get(name_upper, name_upper)

    def align(self, eeg: np.ndarray, channel_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        将输入EEG对齐至标准23通道。

        Args:
            eeg: shape (C_in, T)，C_in为输入通道数
            channel_names: 长度为C_in的通道名列表

        Returns:
            aligned_eeg: shape (23, T)，缺失通道填零
            valid_mask: shape (23,) bool，True表示该通道有效
        """
        T = eeg.shape[1]
        aligned = np.zeros((len(self.standard_channels), T), dtype=eeg.dtype)
        valid_mask = np.zeros(len(self.standard_channels), dtype=bool)

        for src_idx, ch_name in enumerate(channel_names):
            std_name = self._normalize_name(ch_name)
            if std_name in self.ch_to_idx:
                dst_idx = self.ch_to_idx[std_name]
                aligned[dst_idx] = eeg[src_idx]
                valid_mask[dst_idx] = True

        return aligned, valid_mask


class EEGPreprocessor:
    """
    完整EEG预处理流水线：
    1. 带通滤波 0.1-75Hz
    2. 陷波滤波 50Hz
    3. 重采样至200Hz
    4. 幅值归一化 /100
    5. 通道对齐至23通道
    """

    def __init__(
        self,
        target_fs: float = 200.0,
        bandpass_low: float = 0.1,
        bandpass_high: float = 75.0,
        notch_freq: float = 50.0,
        amplitude_scale: float = 100.0,
        standard_channels: Optional[List[str]] = None,
    ):
        self.target_fs = target_fs
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.notch_freq = notch_freq
        self.amplitude_scale = amplitude_scale
        self.aligner = ChannelAligner(standard_channels)

    def __call__(
        self,
        eeg: np.ndarray,
        channel_names: List[str],
        orig_fs: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            eeg: shape (C, T)
            channel_names: 通道名列表
            orig_fs: 原始采样率

        Returns:
            aligned_eeg: shape (23, T')
            valid_mask: shape (23,)
        """
        eeg = bandpass_filter(eeg, orig_fs, self.bandpass_low, self.bandpass_high)
        eeg = notch_filter(eeg, orig_fs, self.notch_freq)
        eeg = resample_eeg(eeg, orig_fs, self.target_fs)
        eeg = amplitude_normalize(eeg, self.amplitude_scale)
        aligned, valid_mask = self.aligner.align(eeg, channel_names)
        return aligned, valid_mask
