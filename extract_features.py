#!/usr/bin/env python3
"""
离线预提取 LaBraM backbone 特征。
只需运行一次，之后线性探针训练直接读 embedding，IO 不再是瓶颈。

用法：
  python extract_features.py --ckpt checkpoints/labram-base.pth --tasks TUAB TUSZ TUEV TUEP
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.labram_backbone import LaBraMBackbone
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


@torch.no_grad()
def extract_split(backbone, ds_cls, data_path, split, batch_size, num_workers,
                  out_dir, device):
    ds = ds_cls(data_path, split=split, window_sec=10.0, stride_sec=10.0)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    all_feats, all_labels = [], []
    for batch in tqdm(loader, desc=f'  {split}', unit='batch'):
        eeg = batch[0]
        label = batch[1]

        if eeg.ndim == 3:
            B, N, T = eeg.shape
            A = T // 200
            eeg = eeg.reshape(B, N, A, 200)

        eeg = eeg.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            feats = backbone(eeg)   # (B, embed_dim)

        all_feats.append(feats.float().cpu().numpy())
        all_labels.append(label.numpy() if hasattr(label, 'numpy') else np.array(label))

    feats_arr  = np.concatenate(all_feats,  axis=0).astype(np.float32)
    labels_arr = np.concatenate(all_labels, axis=0)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f'{split}_feats.npy'),  feats_arr)
    np.save(os.path.join(out_dir, f'{split}_labels.npy'), labels_arr)
    print(f'  saved {split}: feats={feats_arr.shape}  labels={labels_arr.shape}'
          f'  → {out_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',         type=str, required=True)
    parser.add_argument('--backbone_size',type=str, default='base',
                        choices=['base', 'large', 'huge'])
    parser.add_argument('--tasks',        nargs='+', default=['TUAB', 'TUSZ', 'TUEV', 'TUEP'])
    parser.add_argument('--splits',       nargs='+', default=['train', 'eval'])
    parser.add_argument('--batch_size',   type=int, default=2048)
    parser.add_argument('--num_workers',  type=int, default=32)
    parser.add_argument('--out_dir',      type=str, default='features')
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    backbone = LaBraMBackbone(size=args.backbone_size)
    backbone.load_pretrained(args.ckpt)
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    for task in args.tasks:
        print(f'\n=== {task} ===')
        out_dir = os.path.join(args.out_dir, task)
        for split in args.splits:
            extract_split(
                backbone, TASK_DS_CLS[task], data_paths[task],
                split, args.batch_size, args.num_workers,
                out_dir, device,
            )


if __name__ == '__main__':
    main()
