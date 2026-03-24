#!/usr/bin/env python3
"""
一次性将 h5 数据集转为 numpy memmap，加速后续所有训练（包括全量微调/RoPE 实验）。

输出结构：
  memmap/{TASK}/{split}_eeg.npy    float16  (N, 23, T)
  memmap/{TASK}/{split}_labels.npy int32    (N,)

用法：
  python build_memmap.py --tasks TUAB TUSZ TUEV TUEP
"""

import os
import argparse
import time
import numpy as np
from tqdm.auto import tqdm

from data.tuh_dataset import TUABDataset, TUSZDataset, TUEVDataset, TUEPDataset

DEFAULT_DATA_PATHS = {
    'TUAB': '/projects/u6da/tuh_processed/tuab',
    'TUSZ': '/projects/u6da/tuh_processed/tusz',
    'TUEV': '/projects/u6da/tuh_processed/tuev',
    'TUEP': '/projects/u6da/tuh_processed/tuep',
}

TASK_DS_CLS = {
    'TUAB': TUABDataset,
    'TUSZ': TUSZDataset,
    'TUEV': TUEVDataset,
    'TUEP': TUEPDataset,
}


def build_split(ds_cls, data_path, split, out_dir, num_workers=16):
    from torch.utils.data import DataLoader
    ds = ds_cls(data_path, split=split, window_sec=10.0, stride_sec=10.0)
    N = len(ds)
    T = ds[0][0].shape[-1]   # 采样点数，通常 2000
    C = ds[0][0].shape[0]    # 通道数，通常 23

    eeg_path = os.path.join(out_dir, f'{split}_eeg.npy')
    lbl_path = os.path.join(out_dir, f'{split}_labels.npy')

    if os.path.exists(eeg_path) and os.path.exists(lbl_path):
        size_gb = os.path.getsize(eeg_path) / 1e9
        print(f'  [{split}] 已存在 ({size_gb:.1f} GB)，跳过')
        return

    size_gb = N * C * T * 2 / 1e9
    print(f'  [{split}] 样本数={N:,}  通道={C}  采样点={T}  预计大小={size_gb:.1f} GB (fp16)')

    eeg_mm  = np.memmap(eeg_path, dtype='float16', mode='w+', shape=(N, C, T))
    lbl_mm  = np.memmap(lbl_path, dtype='int32',   mode='w+', shape=(N,))

    loader = DataLoader(ds, batch_size=512, shuffle=False,
                        num_workers=num_workers, pin_memory=False,
                        persistent_workers=(num_workers > 0),
                        prefetch_factor=4 if num_workers > 0 else None)

    t0  = time.time()
    idx = 0
    for batch in tqdm(loader, desc=f'  [{split}]', unit='batch',
                      total=len(loader), dynamic_ncols=True):
        eeg   = batch[0].numpy().astype('float16')
        label = batch[1].numpy().astype('int32')
        b = eeg.shape[0]
        eeg_mm[idx:idx+b] = eeg
        lbl_mm[idx:idx+b] = label
        idx += b

    eeg_mm.flush()
    lbl_mm.flush()
    elapsed = time.time() - t0
    print(f'  [{split}] 完成  耗时={elapsed/60:.1f}min  '
          f'实际大小={os.path.getsize(eeg_path)/1e9:.1f}GB  → {out_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks',       nargs='+', default=['TUAB','TUSZ','TUEV','TUEP'])
    parser.add_argument('--splits',      nargs='+', default=['train','eval'])
    parser.add_argument('--out_dir',     type=str,  default='memmap')
    parser.add_argument('--num_workers', type=int,  default=16)
    parser.add_argument('--tuab_path', type=str, default=None)
    parser.add_argument('--tusz_path', type=str, default=None)
    parser.add_argument('--tuev_path', type=str, default=None)
    parser.add_argument('--tuep_path', type=str, default=None)
    args = parser.parse_args()

    data_paths = {
        'TUAB': args.tuab_path or DEFAULT_DATA_PATHS['TUAB'],
        'TUSZ': args.tusz_path or DEFAULT_DATA_PATHS['TUSZ'],
        'TUEV': args.tuev_path or DEFAULT_DATA_PATHS['TUEV'],
        'TUEP': args.tuep_path or DEFAULT_DATA_PATHS['TUEP'],
    }

    for task in args.tasks:
        print(f'\n=== {task} ===')
        out_dir = os.path.join(args.out_dir, task)
        os.makedirs(out_dir, exist_ok=True)
        for split in args.splits:
            build_split(TASK_DS_CLS[task], data_paths[task],
                        split, out_dir, args.num_workers)


if __name__ == '__main__':
    main()
