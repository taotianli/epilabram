"""
Rebuild TUEV memmap with two modes:

  --mode stratified (default):
      Merge official train+eval, stratified 80/20 per class.
      Ensures all 6 classes appear in both splits.

  --mode official:
      Follow BIOT protocol: official eval/ stays as test,
      official train/ is split 80/20 by recording into train/val.

Usage:
    python rebuild_tuev_memmap.py --mode official \
        --src /projects/u6da/tuh_processed/tuev \
        --out /projects/u6da/tuh_processed/memmap/TUEV
"""
import argparse
import glob
import json
import os
import random

import h5py
import numpy as np
from tqdm.auto import tqdm

TUEV_LABELS = {'spsw': 0, 'gped': 1, 'pled': 2, 'eyem': 3, 'artf': 4, 'bckg': 5}


def collect_files(src, splits):
    files = []
    for split in splits:
        for cls, label_id in TUEV_LABELS.items():
            pattern = os.path.join(src, split, cls, '*.h5')
            for p in sorted(glob.glob(pattern)):
                files.append((p, label_id))
    return files


def count_segments(files):
    total = 0
    for path, _ in files:
        with h5py.File(path, 'r') as f:
            k = list(f.keys())[0]
            total += f[k]['eeg'].shape[0]
    return total


def write_split(files, out_dir, split_name, C=23, T=2000):
    N = count_segments(files)
    print(f'  [{split_name}] {len(files)} files → {N:,} segments')

    eeg_path = os.path.join(out_dir, f'{split_name}_eeg.npy')
    lbl_path = os.path.join(out_dir, f'{split_name}_labels.npy')

    eeg_mm = np.memmap(eeg_path, dtype='float16', mode='w+', shape=(N, C, T))
    lbl_mm = np.memmap(lbl_path, dtype='int32',   mode='w+', shape=(N,))

    idx = 0
    for path, label_id in tqdm(files, desc=f'  [{split_name}]'):
        with h5py.File(path, 'r') as f:
            k = list(f.keys())[0]
            eeg = f[k]['eeg'][:]
        n = eeg.shape[0]
        eeg_mm[idx:idx+n] = eeg.astype('float16')
        lbl_mm[idx:idx+n] = label_id
        idx += n

    eeg_mm.flush()
    lbl_mm.flush()

    meta = {'N': N, 'C': C, 'T': T, 'eeg_dtype': 'float16', 'lbl_dtype': 'int32'}
    with open(os.path.join(out_dir, f'{split_name}_meta.json'), 'w') as f:
        json.dump(meta, f)

    print(f'  [{split_name}] done → {eeg_path}')


def mode_stratified(src, out, seed, eval_ratio):
    """Merge train+eval, stratified split per class."""
    rng = random.Random(seed)
    all_files = collect_files(src, ['train', 'eval'])
    print(f'Total files: {len(all_files)}')

    train_files, eval_files = [], []
    by_class = {i: [] for i in range(6)}
    for path, label_id in all_files:
        by_class[label_id].append((path, label_id))

    for label_id, cls_files in by_class.items():
        rng.shuffle(cls_files)
        n_eval = max(1, int(len(cls_files) * eval_ratio))
        eval_files  += cls_files[:n_eval]
        train_files += cls_files[n_eval:]
        cls_name = [k for k, v in TUEV_LABELS.items() if v == label_id][0]
        print(f'  {cls_name}: total={len(cls_files)}  train={len(cls_files)-n_eval}  eval={n_eval}')

    rng.shuffle(train_files)
    rng.shuffle(eval_files)
    return train_files, eval_files


def mode_official(src, out, seed, val_ratio):
    """BIOT protocol: official eval/ = test, official train/ split 80/20 by recording (stratified per class)."""
    rng = random.Random(seed)

    # official eval stays as test
    eval_files = collect_files(src, ['eval'])
    print(f'Official eval files: {len(eval_files)}')

    # stratified split of official train by class
    train_all = collect_files(src, ['train'])
    by_class = {i: [] for i in range(6)}
    for path, label_id in train_all:
        by_class[label_id].append((path, label_id))

    train_files, val_files = [], []
    for label_id, cls_files in by_class.items():
        rng.shuffle(cls_files)
        n_val = max(1, int(len(cls_files) * val_ratio)) if cls_files else 0
        val_files   += cls_files[:n_val]
        train_files += cls_files[n_val:]
        cls_name = [k for k, v in TUEV_LABELS.items() if v == label_id][0]
        print(f'  {cls_name}: total={len(cls_files)}  train={len(cls_files)-n_val}  val={n_val}')

    rng.shuffle(train_files)
    rng.shuffle(val_files)

    write_split(train_files, out, 'train')
    write_split(val_files,   out, 'eval')   # used for early stopping
    write_split(eval_files,  out, 'test')   # final test set
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['stratified', 'official'], default='stratified')
    parser.add_argument('--src',  default='/projects/u6da/tuh_processed/tuev')
    parser.add_argument('--out',  default='/projects/u6da/tuh_processed/memmap/TUEV')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_ratio', type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.mode == 'official':
        print('Mode: official (BIOT protocol)')
        mode_official(args.src, args.out, args.seed, args.eval_ratio)
    else:
        print('Mode: stratified (custom split)')
        train_files, eval_files = mode_stratified(args.src, args.out, args.seed, args.eval_ratio)
        write_split(train_files, args.out, 'train')
        write_split(eval_files,  args.out, 'eval')

    print('\nDone. Run explore_data.py to verify class distribution.')


if __name__ == '__main__':
    main()

