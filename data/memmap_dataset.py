"""
MemmapEEGDataset：从预构建的 numpy memmap 文件读取 EEG 数据。
适用于线性探针、全量微调、RoPE 实验等任意训练场景。

memmap 文件由 build_memmap.py 生成：
  memmap/{TASK}/{split}_eeg.npy    float16  (N, C, T)
  memmap/{TASK}/{split}_labels.npy int32    (N,)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapEEGDataset(Dataset):
    def __init__(self, task: str, split: str, memmap_dir: str = 'memmap'):
        eeg_path = os.path.join(memmap_dir, task, f'{split}_eeg.npy')
        lbl_path = os.path.join(memmap_dir, task, f'{split}_labels.npy')

        if not os.path.exists(eeg_path):
            raise FileNotFoundError(
                f'{eeg_path} 不存在，请先运行 build_memmap.py')

        # 只读 memmap：OS 页缓存自动管理冷热数据，不占额外 RAM
        self.eeg    = np.load(eeg_path,  mmap_mode='r')
        self.labels = np.load(lbl_path,  mmap_mode='r')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        # np.array() 触发实际读取，避免把 memmap 切片传给 DataLoader
        eeg   = torch.from_numpy(np.array(self.eeg[i], dtype='float32'))
        label = int(self.labels[i])
        return eeg, label

    @classmethod
    def exists(cls, task: str, split: str, memmap_dir: str = 'memmap') -> bool:
        return os.path.exists(os.path.join(memmap_dir, task, f'{split}_eeg.npy'))
