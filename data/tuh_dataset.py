"""
TUH EEG Corpus 数据集加载器（懒加载版本）
支持 TUAB / TUSZ / TUEV / TUEP / MultiTask

懒加载策略：_load() 只扫描文件建立索引，__getitem__ 按需读取 h5 数据。
避免把数百 GB 数据一次性加载进内存。
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Dict
import h5py

from data.preprocessing import EEGPreprocessor, STANDARD_CHANNELS

# 进程级 h5 文件句柄缓存：每个 DataLoader worker 进程独立持有，
# 避免 __getitem__ 每次都重新打开/关闭文件（最大 IO 瓶颈）。
_h5_cache: Dict[str, h5py.File] = {}


def _get_h5(h5_path: str) -> h5py.File:
    if h5_path not in _h5_cache:
        _h5_cache[h5_path] = h5py.File(h5_path, 'r')
    return _h5_cache[h5_path]

TASK_IDS = {'TUAB': 0, 'TUSZ': 1, 'TUEV': 2, 'TUEP': 3}
TUEV_LABELS = {'SPSW': 0, 'GPED': 1, 'PLED': 2, 'EYEM': 3, 'ARTF': 4, 'BCKG': 5}
TUSZ_SEIZURE_TYPES = ['FNSZ', 'GNSZ', 'ABSZ', 'TNSZ', 'CPSZ', 'TCSZ', 'MYSZ']


def _build_index(h5_path: str, window_size: int, stride: int):
    """
    扫描 h5 文件，返回 [(subj_key, seg_idx, ch_names), ...]
    seg_idx: 预切片格式为片段编号，完整录音格式为起始采样点
    """
    entries = []
    try:
        with h5py.File(h5_path, 'r') as f:
            for subj_key in f.keys():
                shape = f[subj_key]['eeg'].shape
                ch_names = list(f[subj_key]['eeg'].attrs.get('chOrder', STANDARD_CHANNELS))
                if len(shape) == 3:          # (N_segments, C, T)
                    for i in range(shape[0]):
                        entries.append((subj_key, i, ch_names))
                elif len(shape) == 2:        # (C, T)
                    T = shape[1]
                    for start in range(0, T - window_size + 1, stride):
                        entries.append((subj_key, start, ch_names))
    except Exception:
        pass
    return entries


def _read_seg(h5_path: str, subj_key: str, idx: int, window_size: int) -> np.ndarray:
    """从 h5 文件读取单个片段（复用进程级缓存句柄，避免重复开关文件）"""
    f = _get_h5(h5_path)
    eeg = f[subj_key]['eeg']
    if eeg.ndim == 3:
        return np.array(eeg[idx])                        # (C, T)
    else:
        return np.array(eeg[:, idx:idx + window_size])   # (C, T)


# ============================================================
# TUAB
# ============================================================
class TUABDataset(Dataset):
    """TUAB 正常/异常二分类。返回: (eeg [23,T], label, subject_id, recording_id)"""

    def __init__(self, root: str, window_sec: float = 10.0, stride_sec: float = 5.0,
                 sample_rate: float = 200.0, preprocessor: Optional[EEGPreprocessor] = None,
                 split: str = 'train'):
        self.root = root
        self.window_size = int(window_sec * sample_rate)
        self.stride = int(stride_sec * sample_rate)
        self.preprocessor = preprocessor or EEGPreprocessor()
        self.split = split
        # (h5_path, subj_key, seg_idx, label_id, recording_id, ch_names)
        self.samples: List[Tuple] = []
        self._load()

    def _load(self):
        for label_name, label_id in [('normal', 0), ('abnormal', 1)]:
            pattern = os.path.join(self.root, self.split, label_name, '*.h5')
            for h5_path in sorted(glob.glob(pattern)):
                recording_id = os.path.splitext(os.path.basename(h5_path))[0]
                for subj_key, idx, ch_names in _build_index(h5_path, self.window_size, self.stride):
                    self.samples.append((h5_path, subj_key, idx, label_id, recording_id, ch_names))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        h5_path, subj_key, idx, label, recording_id, ch_names = self.samples[i]
        subject_id = recording_id.split('_')[0] if '_' in recording_id else recording_id
        seg = _read_seg(h5_path, subj_key, idx, self.window_size)
        aligned, _ = self.preprocessor.aligner.align(seg, ch_names)
        return torch.from_numpy(aligned).float(), label, subject_id, recording_id


# ============================================================
# TUSZ
# ============================================================
class TUSZDataset(Dataset):
    """TUSZ 发作检测。返回: (eeg [23,T], label, onset, offset)"""

    def __init__(self, root: str, window_sec: float = 10.0, stride_sec: float = 5.0,
                 sample_rate: float = 200.0, preprocessor: Optional[EEGPreprocessor] = None,
                 split: str = 'train', seizure_types: Optional[List[str]] = None):
        self.root = root
        self.window_size = int(window_sec * sample_rate)
        self.stride = int(stride_sec * sample_rate)
        self.sample_rate = sample_rate
        self.preprocessor = preprocessor or EEGPreprocessor()
        self.split = split
        # (h5_path, subj_key, seg_idx, label, onset, offset, ch_names)
        self.samples: List[Tuple] = []
        self._load()

    def _load(self):
        pattern = os.path.join(self.root, self.split, '*.h5')
        for h5_path in sorted(glob.glob(pattern)):
            try:
                with h5py.File(h5_path, 'r') as f:
                    for subj_key in f.keys():
                        shape = f[subj_key]['eeg'].shape
                        ch_names = list(f[subj_key]['eeg'].attrs.get('chOrder', STANDARD_CHANNELS))
                        labels_ds = f[subj_key].get('label', None)
                        if labels_ds is None:
                            continue
                        labels_arr = labels_ds[:]

                        if len(shape) == 3:       # (N_segments, C, T) — 预切片
                            for i in range(shape[0]):
                                lbl = int(labels_arr[i]) if labels_arr.ndim == 1 else 0
                                onset  = i * (self.window_size / self.sample_rate)
                                offset = onset + self.window_size / self.sample_rate
                                self.samples.append((h5_path, subj_key, i, lbl, onset, offset, ch_names))
                        else:                      # (C, T) — 完整录音
                            T = shape[1]
                            for start in range(0, T - self.window_size + 1, self.stride):
                                if labels_arr.ndim == 1:
                                    lbl = int(labels_arr[start:start + self.window_size].mean() >= 0.5)
                                else:
                                    lbl = 0
                                onset  = start / self.sample_rate
                                offset = (start + self.window_size) / self.sample_rate
                                self.samples.append((h5_path, subj_key, start, lbl, onset, offset, ch_names))
            except Exception:
                pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        h5_path, subj_key, idx, label, onset, offset, ch_names = self.samples[i]
        seg = _read_seg(h5_path, subj_key, idx, self.window_size)
        aligned, _ = self.preprocessor.aligner.align(seg, ch_names)
        return torch.from_numpy(aligned).float(), label, onset, offset


# ============================================================
# TUEV
# ============================================================
class TUEVDataset(Dataset):
    """TUEV 6类事件分类。返回: (eeg [23,T], label)"""

    def __init__(self, root: str, window_sec: float = 10.0, stride_sec: float = 5.0,
                 sample_rate: float = 200.0, preprocessor: Optional[EEGPreprocessor] = None,
                 split: str = 'train'):
        self.root = root
        self.window_size = int(window_sec * sample_rate)
        self.stride = int(stride_sec * sample_rate)
        self.preprocessor = preprocessor or EEGPreprocessor()
        self.split = split
        self.samples: List[Tuple] = []
        self._load()

    def _load(self):
        for label_name, label_id in TUEV_LABELS.items():
            pattern = os.path.join(self.root, self.split, label_name.lower(), '*.h5')
            for h5_path in sorted(glob.glob(pattern)):
                for subj_key, idx, ch_names in _build_index(h5_path, self.window_size, self.stride):
                    self.samples.append((h5_path, subj_key, idx, label_id, ch_names))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        h5_path, subj_key, idx, label, ch_names = self.samples[i]
        seg = _read_seg(h5_path, subj_key, idx, self.window_size)
        aligned, _ = self.preprocessor.aligner.align(seg, ch_names)
        return torch.from_numpy(aligned).float(), label


# ============================================================
# TUEP
# ============================================================
class TUEPDataset(Dataset):
    """TUEP 癫痫诊断。返回: (eeg [23,T], label)"""

    def __init__(self, root: str, window_sec: float = 10.0, stride_sec: float = 5.0,
                 sample_rate: float = 200.0, preprocessor: Optional[EEGPreprocessor] = None,
                 split: str = 'train'):
        self.root = root
        self.window_size = int(window_sec * sample_rate)
        self.stride = int(stride_sec * sample_rate)
        self.preprocessor = preprocessor or EEGPreprocessor()
        self.split = split
        self.samples: List[Tuple] = []
        self._load()

    def _load(self):
        for label_name, label_id in [('no_epilepsy', 0), ('epilepsy', 1)]:
            pattern = os.path.join(self.root, self.split, label_name, '*.h5')
            for h5_path in sorted(glob.glob(pattern)):
                for subj_key, idx, ch_names in _build_index(h5_path, self.window_size, self.stride):
                    self.samples.append((h5_path, subj_key, idx, label_id, ch_names))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        h5_path, subj_key, idx, label, ch_names = self.samples[i]
        seg = _read_seg(h5_path, subj_key, idx, self.window_size)
        aligned, _ = self.preprocessor.aligner.align(seg, ch_names)
        return torch.from_numpy(aligned).float(), label


# ============================================================
# MultiTask
# ============================================================
class MultiTaskTUHDataset(Dataset):
    """组合四个数据集。返回: (eeg [23,T], task_id, label)"""

    def __init__(self, tuab_dataset=None, tusz_dataset=None,
                 tuev_dataset=None, tuep_dataset=None):
        self.datasets: List[Dataset] = []
        self.task_ids: List[int] = []
        self.offsets: List[int] = [0]

        for ds, tid in [(tuab_dataset, 0), (tusz_dataset, 1),
                        (tuev_dataset, 2), (tuep_dataset, 3)]:
            if ds is not None:
                self.datasets.append(ds)
                self.task_ids.append(tid)
                self.offsets.append(self.offsets[-1] + len(ds))

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, idx):
        ds_idx = 0
        for i in range(len(self.offsets) - 1):
            if self.offsets[i] <= idx < self.offsets[i + 1]:
                ds_idx = i
                break
        local_idx = idx - self.offsets[ds_idx]
        item = self.datasets[ds_idx][local_idx]
        return item[0], self.task_ids[ds_idx], item[1]
