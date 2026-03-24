#!/usr/bin/env python3
"""
为超大数据集（TUEG / TUAR / TUSL 等）构建分片 memmap。
每片约 shard_gb GB，避免单文件过大。

用法：
  python build_sharded_memmap.py --tasks TUEG --shard_gb 50 --num_workers 32 \\
      --out_dir /projects/u6da/tuh_processed/memmap

输出结构：
  memmap/TUEG/
    meta.json
    shard_0000_eeg.npy   ← ~50GB，float16 (shard_size, 23, 2000)
    shard_0001_eeg.npy
    ...
"""

import os
import sys
import json
import argparse
import time
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- 各数据集路径和 Dataset 类 ----
DEFAULT_DATA_PATHS = {
    'TUAB': '/projects/u6da/tuh_processed/tuab',
    'TUSZ': '/projects/u6da/tuh_processed/tusz',
    'TUEV': '/projects/u6da/tuh_processed/tuev',
    'TUEP': '/projects/u6da/tuh_processed/tuep',
    'TUAR': '/projects/u6da/tuh_processed/tuar',
    'TUEG': '/projects/u6da/tuh_processed/tueg',
    'TUSL': '/projects/u6da/tuh_processed/tusl',
}

def get_ds_cls(task: str):
    from data.tuh_dataset import TUABDataset, TUSZDataset, TUEVDataset, TUEPDataset
    mapping = {
        'TUAB': TUABDataset,
        'TUSZ': TUSZDataset,
        'TUEV': TUEVDataset,
        'TUEP': TUEPDataset,
    }
    if task in mapping:
        return mapping[task]
    # TUAR / TUEG / TUSL：结构与 TUAB 相同（normal/abnormal 或单目录）
    # 若有专用 Dataset 类可在此替换
    return TUABDataset


def build_sharded(ds_cls, data_path, split, out_dir, shard_gb, num_workers):
    ds = ds_cls(data_path, split=split, window_sec=10.0, stride_sec=10.0)
    N  = len(ds)
    sample     = ds[0][0]
    C, T       = sample.shape
    bytes_each = C * T * 2                          # float16
    shard_size = max(1, int(shard_gb * 1e9 / bytes_each))
    n_shards   = (N + shard_size - 1) // shard_size
    total_gb   = N * bytes_each / 1e9

    print(f'  [{split}] 样本={N:,}  每片≈{shard_gb}GB  共{n_shards}片  总计≈{total_gb:.1f}GB')

    meta_path = os.path.join(out_dir, 'meta.json')
    if os.path.exists(meta_path):
        existing = json.load(open(meta_path))
        if existing.get('total') == N and existing.get('n_shards') == n_shards:
            print(f'  [{split}] 已存在，跳过')
            return

    loader = DataLoader(ds, batch_size=1024, shuffle=False,
                        num_workers=num_workers, pin_memory=False,
                        persistent_workers=(num_workers > 0),
                        prefetch_factor=4 if num_workers > 0 else None)

    t0      = time.time()
    global_i = 0
    shard_id = 0
    cur_mm   = None
    cur_pos  = 0

    def open_shard(sid, size):
        path = os.path.join(out_dir, f'shard_{sid:04d}_eeg.npy')
        return np.memmap(path, dtype='float16', mode='w+', shape=(size, C, T))

    cur_size = min(shard_size, N)
    cur_mm   = open_shard(0, cur_size)

    for batch in tqdm(loader, desc=f'  [{split}]', unit='batch', dynamic_ncols=True):
        eeg = batch[0].numpy().astype('float16')   # (B, C, T)
        b   = eeg.shape[0]
        written = 0

        while written < b:
            space   = cur_size - cur_pos
            to_write = min(b - written, space)
            cur_mm[cur_pos:cur_pos + to_write] = eeg[written:written + to_write]
            cur_pos  += to_write
            written  += to_write
            global_i += to_write

            if cur_pos >= cur_size:
                cur_mm.flush()
                shard_id += 1
                if global_i < N:
                    cur_size = min(shard_size, N - global_i)
                    cur_mm   = open_shard(shard_id, cur_size)
                    cur_pos  = 0

    if cur_mm is not None:
        cur_mm.flush()

    meta = {
        'split':      split,
        'total':      N,
        'n_shards':   n_shards,
        'shard_size': shard_size,
        'shape':      [C, T],
        'dtype':      'float16',
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f'  [{split}] 完成  耗时={elapsed/60:.1f}min  → {out_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks',       nargs='+', default=['TUEG'])
    parser.add_argument('--splits',      nargs='+', default=['train', 'eval'])
    parser.add_argument('--shard_gb',    type=float, default=50,
                        help='每个分片的目标大小（GB），默认 50')
    parser.add_argument('--out_dir',     type=str, default='memmap')
    parser.add_argument('--num_workers', type=int, default=32)
    for t in ['tuab','tusz','tuev','tuep','tuar','tueg','tusl']:
        parser.add_argument(f'--{t}_path', type=str, default=None)
    args = parser.parse_args()

    data_paths = {k.upper(): getattr(args, f'{k}_path') or DEFAULT_DATA_PATHS[k.upper()]
                  for k in ['tuab','tusz','tuev','tuep','tuar','tueg','tusl']}

    for task in args.tasks:
        print(f'\n=== {task} ===')
        out_dir = os.path.join(args.out_dir, task)
        os.makedirs(out_dir, exist_ok=True)
        for split in args.splits:
            build_sharded(get_ds_cls(task), data_paths[task],
                          split, out_dir, args.shard_gb, args.num_workers)


if __name__ == '__main__':
    main()
