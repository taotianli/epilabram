"""
TUH EEG Corpus 数据集加载器
支持 TUAB / TUSZ / TUEV / TUEP / MultiTask
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Dict
import h5py

from data.preprocessing import EEGPreprocessor, STANDARD_CHANNELS

# 任务ID映射
TASK_IDS = {'TUAB': 0, 'TUSZ': 1, 'TUEV': 2, 'TUEP': 3}

# TUEV 6类标签
TUEV_LABELS = {'SPSW': 0, 'GPED': 1, 'PLED': 2, 'EYEM': 3, 'ARTF': 4, 'BCKG': 5}

# TUSZ 发作类型
TUSZ_SEIZURE_TYPES = ['FNSZ', 'GNSZ', 'ABSZ', 'TNSZ', 'CPSZ', 'TCSZ', 'MYSZ']


def _load_h5_eeg(h5_path: str, window_size: int, stride: int) -> List[np.ndarray]:
    """从h5文件中按滑窗切分EEG片段，返回list of (C, window_size)"""
    segments = []
    with h5py.File(h5_path, 'r') as f:
        for subj in f.keys():
            eeg = f[subj]['eeg'][:]  # (C, T)
            T = eeg.shape[1]
            for start in range(0, T - window_size + 1, stride):
                segments.append(eeg[:, start:start + window_size])
    return segments


class TUABDataset(Dataset):
    """
    TUAB 正常/异常二分类数据集。

    目录结构（h5格式）：
        tuab_path/
            normal/   *.h5
            abnormal/ *.h5

    返回: (eeg_tensor [23, T], label, subject_id, recording_id)
        label: 0=normal, 1=abnormal
    """

    def __init__(
        self,
        root: str,
        window_sec: float = 10.0,
        stride_sec: float = 5.0,
        sample_rate: float = 200.0,
        preprocessor: Optional[EEGPreprocessor] = None,
        split: str = 'train',
    ):
        self.root = root
        self.window_size = int(window_sec * sample_rate)
        self.stride = int(stride_sec * sample_rate)
        self.sample_rate = sample_rate
        self.preprocessor = preprocessor or EEGPreprocessor()
        self.split = split

        self.samples: List[Tuple[np.ndarray, int, str, str]] = []
        self._load()

    def _load(self):
        for label_name, label_id in [('normal', 0), ('abnormal', 1)]:
            pattern = os.path.join(self.root, self.split, label_name, '*.h5')
            for h5_path in sorted(glob.glob(pattern)):
                recording_id = os.path.splitext(os.path.basename(h5_path))[0]
                subject_id = recording_id.split('_')[0] if '_' in recording_id else recording_id
                try:
                    with h5py.File(h5_path, 'r') as f:
                        for subj_key in f.keys():
                            eeg = f[subj_key]['eeg'][:]  # (C, T)
                            ch_names = list(f[subj_key]['eeg'].attrs.get('chOrder', STANDARD_CHANNELS))
                            T = eeg.shape[1]
                            for start in range(0, T - self.window_size + 1, self.stride):
                                seg = eeg[:, start:start + self.window_size]
                                self.samples.append((seg, label_id, subject_id, recording_id, ch_names))
                except Exception:
                    pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        seg, label, subject_id, recording_id, ch_names = self.samples[idx]
        aligned, _ = self.preprocessor.aligner.align(seg, ch_names)
        eeg_tensor = torch.from_numpy(aligned).float()
        return eeg_tensor, label, subject_id, recording_id


class TUSZDataset(Dataset):
    """
    TUSZ 发作检测数据集。

    目录结构（h5格式）：
        tusz_path/
            {split}/
                *.h5  (含 label 字段: 0=background, 1=seizure)

    返回: (eeg_tensor [23, T], label, onset, offset)
        label: 0=background, 1=seizure
    """

    def __init__(
        self,
        root: str,
        window_sec: float = 10.0,
        stride_sec: float = 5.0,
        sample_rate: float = 200.0,
        preprocessor: Optional[EEGPreprocessor] = None,
        split: str = 'train',
        seizure_types: Optional[List[str]] = None,
    ):
        self.root = root
        self.window_size = int(window_sec * sample_rate)
        self.stride = int(stride_sec * sample_rate)
        self.sample_rate = sample_rate
        self.preprocessor = preprocessor or EEGPreprocessor()
        self.split = split
        self.seizure_types = seizure_types  # None表示不过滤

        self.samples: List[Tuple] = []
        self._load()

    def _load(self):
        pattern = os.path.join(self.root, self.split, '*.h5')
        for h5_path in sorted(glob.glob(pattern)):
            try:
                with h5py.File(h5_path, 'r') as f:
                    for subj_key in f.keys():
                        eeg = f[subj_key]['eeg'][:]  # (C, T)
                        ch_names = list(f[subj_key]['eeg'].attrs.get('chOrder', STANDARD_CHANNELS))
                        labels_arr = f[subj_key].get('label', None)
                        if labels_arr is None:
                            continue
                        labels_arr = labels_arr[:]  # (T,) or (N_events, 3)
                        T = eeg.shape[1]
                        for start in range(0, T - self.window_size + 1, self.stride):
                            seg = eeg[:, start:start + self.window_size]
                            # 取窗口内多数标签
                            if labels_arr.ndim == 1:
                                win_labels = labels_arr[start:start + self.window_size]
                                label = int(win_labels.mean() >= 0.5)
                                onset = start / self.sample_rate
                                offset = (start + self.window_size) / self.sample_rate
                            else:
                                label = 0
                                onset = start / self.sample_rate
                                offset = (start + self.window_size) / self.sample_rate
                            self.samples.append((seg, label, onset, offset, ch_names))
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float, float]:
        seg, label, onset, offset, ch_names = self.samples[idx]
        aligned, _ = self.preprocessor.aligner.align(seg, ch_names)
        eeg_tensor = torch.from_numpy(aligned).float()
        return eeg_tensor, label, onset, offset


class TUEVDataset(Dataset):
    """
    TUEV 6类事件分类数据集。

    标签: SPSW=0, GPED=1, PLED=2, EYEM=3, ARTF=4, BCKG=5

    返回: (eeg_tensor [23, T], label)
    """

    def __init__(
        self,
        root: str,
        window_sec: float = 10.0,
        stride_sec: float = 5.0,
        sample_rate: float = 200.0,
        preprocessor: Optional[EEGPreprocessor] = None,
        split: str = 'train',
    ):
        self.root = root
        self.window_size = int(window_sec * sample_rate)
        self.stride = int(stride_sec * sample_rate)
        self.sample_rate = sample_rate
        self.preprocessor = preprocessor or EEGPreprocessor()
        self.split = split

        self.samples: List[Tuple] = []
        self._load()

    def _load(self):
        for label_name, label_id in TUEV_LABELS.items():
            pattern = os.path.join(self.root, self.split, label_name.lower(), '*.h5')
            for h5_path in sorted(glob.glob(pattern)):
                try:
                    with h5py.File(h5_path, 'r') as f:
                        for subj_key in f.keys():
                            eeg = f[subj_key]['eeg'][:]
                            ch_names = list(f[subj_key]['eeg'].attrs.get('chOrder', STANDARD_CHANNELS))
                            T = eeg.shape[1]
                            for start in range(0, T - self.window_size + 1, self.stride):
                                seg = eeg[:, start:start + self.window_size]
                                self.samples.append((seg, label_id, ch_names))
                except Exception:
                    pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        seg, label, ch_names = self.samples[idx]
        aligned, _ = self.preprocessor.aligner.align(seg, ch_names)
        eeg_tensor = torch.from_numpy(aligned).float()
        return eeg_tensor, label


class TUEPDataset(Dataset):
    """
    TUEP 癫痫诊断数据集。

    返回: (eeg_tensor [23, T], label)
        label: 0=no_epilepsy, 1=epilepsy
    """

    def __init__(
        self,
        root: str,
        window_sec: float = 10.0,
        stride_sec: float = 5.0,
        sample_rate: float = 200.0,
        preprocessor: Optional[EEGPreprocessor] = None,
        split: str = 'train',
    ):
        self.root = root
        self.window_size = int(window_sec * sample_rate)
        self.stride = int(stride_sec * sample_rate)
        self.sample_rate = sample_rate
        self.preprocessor = preprocessor or EEGPreprocessor()
        self.split = split

        self.samples: List[Tuple] = []
        self._load()

    def _load(self):
        for label_name, label_id in [('no_epilepsy', 0), ('epilepsy', 1)]:
            pattern = os.path.join(self.root, self.split, label_name, '*.h5')
            for h5_path in sorted(glob.glob(pattern)):
                try:
                    with h5py.File(h5_path, 'r') as f:
                        for subj_key in f.keys():
                            eeg = f[subj_key]['eeg'][:]
                            ch_names = list(f[subj_key]['eeg'].attrs.get('chOrder', STANDARD_CHANNELS))
                            T = eeg.shape[1]
                            for start in range(0, T - self.window_size + 1, self.stride):
                                seg = eeg[:, start:start + self.window_size]
                                self.samples.append((seg, label_id, ch_names))
                except Exception:
                    pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        seg, label, ch_names = self.samples[idx]
        aligned, _ = self.preprocessor.aligner.align(seg, ch_names)
        eeg_tensor = torch.from_numpy(aligned).float()
        return eeg_tensor, label


class MultiTaskTUHDataset(Dataset):
    """
    组合 TUAB / TUSZ / TUEV / TUEP 四个数据集的多任务数据集。

    返回: (eeg_tensor [23, T], task_id, label)
        task_id: 0=TUAB, 1=TUSZ, 2=TUEV, 3=TUEP
    """

    def __init__(
        self,
        tuab_dataset: Optional[TUABDataset] = None,
        tusz_dataset: Optional[TUSZDataset] = None,
        tuev_dataset: Optional[TUEVDataset] = None,
        tuep_dataset: Optional[TUEPDataset] = None,
    ):
        self.datasets: List[Dataset] = []
        self.task_ids: List[int] = []
        self.offsets: List[int] = [0]

        for ds, tid in [
            (tuab_dataset, 0),
            (tusz_dataset, 1),
            (tuev_dataset, 2),
            (tuep_dataset, 3),
        ]:
            if ds is not None:
                self.datasets.append(ds)
                self.task_ids.append(tid)
                self.offsets.append(self.offsets[-1] + len(ds))

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        # 找到对应子数据集
        ds_idx = 0
        for i in range(len(self.offsets) - 1):
            if self.offsets[i] <= idx < self.offsets[i + 1]:
                ds_idx = i
                break
        local_idx = idx - self.offsets[ds_idx]
        item = self.datasets[ds_idx][local_idx]
        eeg_tensor = item[0]
        label = item[1]
        task_id = self.task_ids[ds_idx]
        return eeg_tensor, task_id, label
