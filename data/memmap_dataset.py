"""
MemmapEEGDataset：从预构建的 numpy memmap 文件读取 EEG 数据。
适用于线性探针、全量微调、RoPE 实验等任意训练场景。

memmap 文件由 build_memmap.py 生成：
  memmap/{TASK}/{split}_eeg.npy      float16 原始二进制 (N, C, T)
  memmap/{TASK}/{split}_labels.npy   int32   原始二进制 (N,)
  memmap/{TASK}/{split}_meta.json    shape 元信息
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapEEGDataset(Dataset):
    def __init__(self, task: str, split: str, memmap_dir: str = 'memmap'):
        task_dir   = os.path.join(memmap_dir, task)
        eeg_path   = os.path.join(task_dir, f'{split}_eeg.npy')
        lbl_path   = os.path.join(task_dir, f'{split}_labels.npy')
        meta_path  = os.path.join(task_dir, f'{split}_meta.json')

        if not os.path.exists(eeg_path):
            raise FileNotFoundError(
                f'{eeg_path} 不存在，请先运行 build_memmap.py')

        with open(meta_path) as f:
            meta = json.load(f)
        N, C, T = meta['N'], meta['C'], meta['T']

        # np.memmap 直接读原始二进制，OS 页缓存自动管理冷热数据
        self.eeg    = np.memmap(eeg_path, dtype='float16', mode='r', shape=(N, C, T))
        self.labels = np.memmap(lbl_path, dtype='int32',   mode='r', shape=(N,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        # np.array() 触发实际读取，避免把 memmap 切片传给 DataLoader
        eeg   = torch.from_numpy(np.array(self.eeg[i], dtype='float32'))
        label = int(self.labels[i])
        return eeg, label

    @classmethod
    def exists(cls, task: str, split: str, memmap_dir: str = 'memmap') -> bool:
        d = os.path.join(memmap_dir, task)
        return (os.path.exists(os.path.join(d, f'{split}_eeg.npy')) and
                os.path.exists(os.path.join(d, f'{split}_meta.json')))
