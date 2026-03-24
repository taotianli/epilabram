"""
ShardedMemmapDataset：为 TUEG 等超大预训练数据集设计的分片 memmap。

目录结构（由 build_sharded_memmap.py 生成）：
  memmap/TUEG/
    meta.json                  ← 分片信息（总样本数、每片大小等）
    shard_0000_eeg.npy         ← float16 (shard_size, 23, T)
    shard_0001_eeg.npy
    ...

用法：
  ds = ShardedMemmapDataset('TUEG', memmap_dir='/projects/u6da/tuh_processed/memmap')
  loader = DataLoader(ds, batch_size=512, shuffle=True, num_workers=8)
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class ShardedMemmapDataset(Dataset):
    def __init__(self, task: str, memmap_dir: str = 'memmap'):
        task_dir  = os.path.join(memmap_dir, task)
        meta_path = os.path.join(task_dir, 'meta.json')

        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f'{meta_path} 不存在，请先运行 build_sharded_memmap.py --tasks {task}')

        with open(meta_path) as f:
            self.meta = json.load(f)

        self.task_dir   = task_dir
        self.n_shards   = self.meta['n_shards']
        self.shard_size = self.meta['shard_size']   # 最后一片可能不足
        self.total      = self.meta['total']
        self.shape      = tuple(self.meta['shape'])  # (C, T)

        # 懒加载：首次访问分片时才打开 memmap（避免一次性占用所有文件描述符）
        self._shards: dict = {}

    def _get_shard(self, shard_id: int) -> np.memmap:
        if shard_id not in self._shards:
            path  = os.path.join(self.task_dir, f'shard_{shard_id:04d}_eeg.npy')
            start = shard_id * self.shard_size
            n     = min(self.shard_size, self.total - start)
            self._shards[shard_id] = np.load(path, mmap_mode='r')
        return self._shards[shard_id]

    def __len__(self):
        return self.total

    def __getitem__(self, i):
        shard_id  = i // self.shard_size
        local_idx = i %  self.shard_size
        shard     = self._get_shard(shard_id)
        eeg = torch.from_numpy(np.array(shard[local_idx], dtype='float32'))
        return eeg

    @classmethod
    def exists(cls, task: str, memmap_dir: str = 'memmap') -> bool:
        return os.path.exists(os.path.join(memmap_dir, task, 'meta.json'))
