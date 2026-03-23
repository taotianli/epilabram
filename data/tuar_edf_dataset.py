"""
直接从EDF文件加载TUAR数据的Dataset（无需预先转h5）
支持 TUAR (TUH EEG Artifact Corpus) v3.0.1 格式

任务：5类伪迹分类
  0: bckg  — 背景（无伪迹）
  1: musc  — 肌电伪迹（musc, chew, shiv 及其组合）
  2: eyem  — 眼动伪迹（eyem 及其组合）
  3: elec  — 电极伪迹（elec, elpp 及其组合）
  4: seiz  — 发作（fnsz, gnsz, cpsz, tcsz 等）
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
import mne

mne.set_log_level('WARNING')

# TUAR 数据中的标准EEG通道
TUAR_EEG_CHANNELS = [
    'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF',
    'EEG C3-REF',  'EEG C4-REF',  'EEG P3-REF', 'EEG P4-REF',
    'EEG O1-REF',  'EEG O2-REF',  'EEG F7-REF', 'EEG F8-REF',
    'EEG T3-REF',  'EEG T4-REF',  'EEG T5-REF', 'EEG T6-REF',
    'EEG A1-REF',  'EEG A2-REF',  'EEG FZ-REF', 'EEG CZ-REF',
    'EEG PZ-REF',  'EEG T1-REF',  'EEG T2-REF',
]

TUAR_TO_STANDARD = {
    'EEG FP1-REF': 'FP1', 'EEG FP2-REF': 'FP2',
    'EEG F3-REF':  'F3',  'EEG F4-REF':  'F4',
    'EEG C3-REF':  'C3',  'EEG C4-REF':  'C4',
    'EEG P3-REF':  'P3',  'EEG P4-REF':  'P4',
    'EEG O1-REF':  'O1',  'EEG O2-REF':  'O2',
    'EEG F7-REF':  'F7',  'EEG F8-REF':  'F8',
    'EEG T3-REF':  'T3',  'EEG T4-REF':  'T4',
    'EEG T5-REF':  'T5',  'EEG T6-REF':  'T6',
    'EEG A1-REF':  'A1',  'EEG A2-REF':  'A2',
    'EEG FZ-REF':  'FZ',  'EEG CZ-REF':  'CZ',
    'EEG PZ-REF':  'PZ',  'EEG T1-REF':  'T1',
    'EEG T2-REF':  'T2',
}

STANDARD_23 = [
    'FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2',
    'F7','F8','T3','T4','T5','T6','FZ','CZ','PZ','A1','A2','T1','T2'
]
STD_IDX = {ch: i for i, ch in enumerate(STANDARD_23)}

# ---------------------------------------------------------------------------
# 5类伪迹标签映射
# ---------------------------------------------------------------------------

ARTIFACT_CLASSES = {
    'bckg': 0,
    'null': 0,
    # 肌电
    'musc': 1, 'chew': 1, 'shiv': 1,
    'musc_elec': 1, 'chew_musc': 1, 'chew_elec': 1,
    'shiv_musc': 1, 'shiv_elec': 1, 'eyem_chew': 2,  # eyem_chew 归 eyem
    # 眼动
    'eyem': 2, 'eyem_musc': 2, 'eyem_elec': 2,
    'eyem_shiv': 2,
    # 电极
    'elec': 3, 'elpp': 3, 'elst': 3,
    # 发作
    'fnsz': 4, 'gnsz': 4, 'cpsz': 4, 'tcsz': 4,
    'spsz': 4, 'absz': 4, 'tnsz': 4, 'cnsz': 4,
    'atsz': 4, 'mysz': 4, 'nesz': 4,
    'seiz': 4, 'spsw': 4, 'gped': 4, 'pled': 4,
}

ARTIFACT_CLASS_NAMES = ['bckg', 'musc', 'eyem', 'elec', 'seiz']
N_ARTIFACT_CLASSES = 5


def map_label(raw_label: str) -> Optional[int]:
    """将原始标签映射到 0-4 类，未知标签返回 None"""
    return ARTIFACT_CLASSES.get(raw_label.strip().lower(), None)


# ---------------------------------------------------------------------------
# EDF 加载
# ---------------------------------------------------------------------------

def load_edf(edf_path: str, target_fs: float = 200.0) -> Tuple[np.ndarray, List[str]]:
    """
    读取EDF文件，只保留EEG通道，重采样到target_fs。

    Returns:
        eeg: (23, T) float32，对齐到标准23通道，缺失通道填0
        ch_names: 标准通道名列表
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    orig_fs = raw.info['sfreq']

    available = [ch for ch in TUAR_EEG_CHANNELS if ch in raw.ch_names]
    raw.pick_channels(available)

    if orig_fs != target_fs:
        raw.resample(target_fs, verbose=False)

    data = raw.get_data() * 1e6  # V -> μV
    T = data.shape[1]
    aligned = np.zeros((23, T), dtype=np.float32)
    for i, ch in enumerate(available):
        std_name = TUAR_TO_STANDARD.get(ch)
        if std_name and std_name in STD_IDX:
            aligned[STD_IDX[std_name]] = data[i].astype(np.float32)

    aligned = aligned / 100.0  # 幅值归一化
    return aligned, STANDARD_23


# ---------------------------------------------------------------------------
# 标注解析：将 CSV 标注转为时间轴标签数组
# ---------------------------------------------------------------------------

def parse_annotations(csv_path: str, n_samples: int,
                      fs: float = 200.0) -> Optional[np.ndarray]:
    """
    将 CSV 标注转为逐采样点的类别标签数组。

    Returns:
        labels: (n_samples,) int8，-1 表示未标注区域
    """
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, comment='#')
        df.columns = df.columns.str.strip()
        if not all(c in df.columns for c in ['start_time', 'stop_time', 'label']):
            return None
    except Exception:
        return None

    labels = np.full(n_samples, -1, dtype=np.int8)
    for _, row in df.iterrows():
        cls = map_label(str(row['label']))
        if cls is None:
            continue
        start = int(float(row['start_time']) * fs)
        stop  = min(int(float(row['stop_time']) * fs), n_samples)
        if start < stop:
            labels[start:stop] = cls

    return labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TUAREDFDataset(Dataset):
    """
    TUAR EDF 数据集，支持两种模式：

    mode='pretrain'：无标签，用于 Stage1 续训
        返回: (eeg_tensor [23, T], 0)

    mode='artifact'：5类伪迹分类，用于 Stage2 微调
        返回: (eeg_tensor [23, T], label)
        label: 0=bckg, 1=musc, 2=eyem, 3=elec, 4=seiz
        只保留有明确标注的窗口（排除 label=-1 的区域）
    """

    def __init__(
        self,
        root: str,
        window_sec: float = 10.0,
        stride_sec: float = 10.0,
        target_fs: float = 200.0,
        max_files: Optional[int] = None,
        mode: str = 'pretrain',   # 'pretrain' or 'artifact'
    ):
        self.root = root
        self.window = int(window_sec * target_fs)
        self.stride = int(stride_sec * target_fs)
        self.target_fs = target_fs
        self.max_files = max_files
        self.mode = mode

        self.samples: List[Tuple[np.ndarray, int]] = []
        self._load()

    def _load(self):
        edf_files = sorted(glob.glob(
            os.path.join(self.root, '**', '*.edf'), recursive=True
        ))
        if self.max_files:
            edf_files = edf_files[:self.max_files]

        print(f"Loading {len(edf_files)} EDF files (mode={self.mode})...")
        skipped = 0
        for edf_path in edf_files:
            try:
                eeg, _ = load_edf(edf_path, self.target_fs)
                T = eeg.shape[1]

                if self.mode == 'pretrain':
                    for start in range(0, T - self.window + 1, self.stride):
                        seg = eeg[:, start:start + self.window].copy()
                        self.samples.append((seg, 0))

                else:  # artifact mode
                    csv_path = edf_path.replace('.edf', '.csv')
                    ann = parse_annotations(csv_path, T, self.target_fs)
                    if ann is None:
                        skipped += 1
                        continue

                    for start in range(0, T - self.window + 1, self.stride):
                        win_labels = ann[start:start + self.window]
                        # 取窗口内出现最多的有效标签（排除 -1）
                        valid = win_labels[win_labels >= 0]
                        if len(valid) < self.window * 0.5:
                            continue  # 超过50%未标注，跳过
                        counts = np.bincount(valid.astype(np.int64), minlength=N_ARTIFACT_CLASSES)
                        label = int(counts.argmax())
                        seg = eeg[:, start:start + self.window].copy()
                        self.samples.append((seg, label))

            except Exception as e:
                skipped += 1

        label_dist = {}
        for _, lbl in self.samples:
            label_dist[lbl] = label_dist.get(lbl, 0) + 1

        print(f"Total segments: {len(self.samples)}  (skipped {skipped} files)")
        if self.mode == 'artifact':
            for cls_id, cls_name in enumerate(ARTIFACT_CLASS_NAMES):
                print(f"  {cls_name}: {label_dist.get(cls_id, 0)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        seg, label = self.samples[idx]
        return torch.from_numpy(seg), label
