#!/usr/bin/env python3
"""
TUH EEG 数据预处理脚本
用法：python preprocess_tuh.py [--tuh_root ...] [--output_dir ...] [--workers N] [--datasets ...]
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import h5py
import mne
from tqdm.auto import tqdm

# ============================================================
# 常量
# ============================================================
TARGET_FS       = 200.0
WINDOW_SEC      = 10.0
STRIDE_SEC      = 5.0
EVAL_STRIDE_SEC = 10.0

STANDARD_23 = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ', 'A1', 'A2', 'T1', 'T2',
]
STD_IDX: Dict[str, int] = {ch: i for i, ch in enumerate(STANDARD_23)}

_ALIAS_MAP: Dict[str, str] = {}
for _ch in STANDARD_23:
    for _v in [_ch, f'EEG {_ch}-REF', f'EEG {_ch}-LE', f'{_ch}-REF', f'{_ch}-LE', f'EEG {_ch}']:
        _ALIAS_MAP[_v.upper()] = _ch

TUEV_LABELS      = {'spsw': 0, 'gped': 1, 'pled': 2, 'eyem': 3, 'artf': 4, 'bckg': 5}
TUEV_CLASS_NAMES = ['spsw', 'gped', 'pled', 'eyem', 'artf', 'bckg']

TUSZ_SEIZURE_LABELS = {
    'fnsz', 'gnsz', 'absz', 'tnsz', 'cpsz', 'tcsz', 'mysz',
    'spsz', 'cnsz', 'atsz', 'nesz', 'seiz',
}

# ============================================================
# 核心 EDF 读取函数
# ============================================================
def load_and_preprocess_edf(
    edf_path: str,
    target_fs: float = TARGET_FS,
) -> Tuple[Optional[np.ndarray], List[str], float]:
    """读取 EDF → 选通道 → 重采样 → μV/100 归一化 → 对齐 23 通道"""
    mne.set_log_level('ERROR')
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        return None, [], 0.0

    orig_fs = raw.info['sfreq']
    available = [ch for ch in raw.ch_names if _ALIAS_MAP.get(ch.upper().strip())]
    if not available:
        return None, [], orig_fs

    raw.pick_channels(available, ordered=False)
    if abs(orig_fs - target_fs) > 1.0:
        raw.resample(target_fs, verbose=False)

    data = raw.get_data() * 1e6
    T = data.shape[1]
    aligned = np.zeros((23, T), dtype=np.float32)
    for i, ch in enumerate(raw.ch_names):
        std = _ALIAS_MAP.get(ch.upper().strip())
        if std and std in STD_IDX:
            aligned[STD_IDX[std]] = data[i].astype(np.float32)
    aligned /= 100.0
    return aligned, STANDARD_23, orig_fs

# ============================================================
# Worker 函数（每个进程处理一个 EDF）
# ============================================================
def _worker_simple(task):
    """TUAB / TUEV / TUEP 通用 worker：固定标签"""
    edf_path, out_path, label_id, win, stp = task
    if os.path.exists(out_path):
        return 'skip', 0

    eeg, ch_names, _ = load_and_preprocess_edf(edf_path)
    if eeg is None:
        return 'error', 0

    T = eeg.shape[1]
    segs = [eeg[:, s:s + win] for s in range(0, T - win + 1, stp)]
    if not segs:
        return 'error', 0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rec_id = Path(edf_path).stem
    try:
        with h5py.File(out_path, 'w') as hf:
            grp = hf.create_group(rec_id)
            ds = grp.create_dataset('eeg', data=np.stack(segs),
                                    compression='gzip', compression_opts=4)
            ds.attrs['chOrder'] = ch_names
            ds.attrs['label']   = label_id
            grp.attrs['n_segments'] = len(segs)
    except Exception:
        return 'error', 0
    return 'ok', len(segs)


def _worker_tusz(task):
    """TUSZ worker：每个 segment 有独立标签"""
    edf_path, out_path, tse_path, win, stp = task
    if os.path.exists(out_path):
        return 'skip', 0, 0

    events = _parse_tusz_tse(tse_path)
    eeg, ch_names, _ = load_and_preprocess_edf(edf_path)
    if eeg is None:
        return 'error', 0, 0

    T = eeg.shape[1]
    label_arr = np.zeros(T, dtype=np.int8)
    if events:
        for s_t, e_t, lbl in events:
            s, e = int(s_t * TARGET_FS), min(int(e_t * TARGET_FS), T)
            if s < e:
                label_arr[s:e] = lbl

    segs, labels = [], []
    for start in range(0, T - win + 1, stp):
        seg_lbl = label_arr[start:start + win]
        lbl = 1 if seg_lbl.sum() > win * 0.3 else 0
        segs.append(eeg[:, start:start + win])
        labels.append(lbl)

    if not segs:
        return 'error', 0, 0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rec_id = Path(edf_path).stem
    try:
        with h5py.File(out_path, 'w') as hf:
            grp = hf.create_group(rec_id)
            ds = grp.create_dataset('eeg', data=np.stack(segs),
                                    compression='gzip', compression_opts=4)
            ds.attrs['chOrder'] = ch_names
            grp.create_dataset('label', data=np.array(labels, dtype=np.int8))
            grp.attrs['n_segments'] = len(segs)
    except Exception:
        return 'error', 0, 0

    n_seiz = sum(labels)
    return 'ok', len(segs) - n_seiz, n_seiz

# ============================================================
# 辅助函数
# ============================================================
def _parse_tusz_tse(tse_path: str):
    if not os.path.exists(tse_path):
        return None
    events = []
    try:
        with open(tse_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('version'):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                start, stop, label = float(parts[0]), float(parts[1]), parts[2].lower()
                events.append((start, stop, int(label in TUSZ_SEIZURE_LABELS)))
    except Exception:
        return None
    return events or None


def _tuev_label_from_filename(edf_path: str) -> Optional[int]:
    stem = Path(edf_path).stem.lower()
    for cls_name in TUEV_LABELS:
        if stem.startswith(cls_name + '_') or stem == cls_name:
            return TUEV_LABELS[cls_name]
    return None


def _tuev_label_from_lab(edf_path: str) -> Optional[int]:
    lab_path = edf_path.replace('.edf', '_ch000.lab')
    if not os.path.exists(lab_path):
        return None
    counts = defaultdict(float)
    try:
        with open(lab_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                label = parts[2].lower()
                try:
                    duration = int(parts[1]) - int(parts[0])
                except ValueError:
                    continue
                if label in TUEV_LABELS:
                    counts[label] += duration
        if not counts:
            return TUEV_LABELS['bckg']
        non_bckg = {k: v for k, v in counts.items() if k != 'bckg'}
        if non_bckg:
            return TUEV_LABELS[max(non_bckg, key=lambda k: non_bckg[k])]
        return TUEV_LABELS['bckg']
    except Exception:
        return None


def _run_parallel(tasks, worker_fn, desc, num_workers):
    """并行执行 tasks，返回结果列表"""
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(worker_fn, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, unit='file'):
            results.append(fut.result())
    return results

# ============================================================
# 各数据集处理函数
# ============================================================
def process_tuab(tuh_root, output_dir, max_files, num_workers,
                 win, stride_stp, eval_stp):
    print("=" * 60)
    print("TUAB — 异常检测  |  0=normal, 1=abnormal")
    print("=" * 60)

    tuab_root = os.path.join(tuh_root, 'tuab')
    found_any = False

    for split in ('train', 'eval'):
        stp = stride_stp if split == 'train' else eval_stp
        counts = defaultdict(int)
        skipped = 0

        for label_name, label_id in [('normal', 0), ('abnormal', 1)]:
            files = []
            for base in (
                os.path.join(tuab_root, 'edf', split, label_name),
                os.path.join(tuab_root, split, label_name),
            ):
                files = sorted(glob.glob(os.path.join(base, '**', '*.edf'), recursive=True))
                if files:
                    break
            if not files:
                continue
            found_any = True
            if max_files:
                files = files[:max_files]

            out_dir = os.path.join(output_dir, 'tuab', split, label_name)
            os.makedirs(out_dir, exist_ok=True)
            print(f"  [{split}/{label_name}]  {len(files)} files")

            tasks = [
                (f, os.path.join(out_dir, f'{Path(f).stem}.h5'), label_id, win, stp)
                for f in files
            ]
            results = _run_parallel(tasks, _worker_simple,
                                     f'TUAB {split}/{label_name}', num_workers)
            for status, n in results:
                if status == 'error':
                    skipped += 1
                else:
                    counts[label_name] += n

        if found_any:
            print(f"    normal: {counts['normal']:>8}  abnormal: {counts['abnormal']:>8}  skipped: {skipped}")

    if not found_any:
        print(f"  [WARN] 未找到 EDF 文件: {tuab_root}")


def process_tusz(tuh_root, output_dir, max_files, num_workers,
                 win, stride_stp, eval_stp):
    print("=" * 60)
    print("TUSZ — 发作检测  |  0=background, 1=seizure")
    print("=" * 60)

    tusz_root = os.path.join(tuh_root, 'tusz')
    found_any = False

    for split in ('train', 'dev', 'eval'):
        stp = stride_stp if split == 'train' else eval_stp
        files = []
        for base in (
            os.path.join(tusz_root, 'edf', split),
            os.path.join(tusz_root, split),
        ):
            files = sorted(glob.glob(os.path.join(base, '**', '*.edf'), recursive=True))
            if files:
                break
        if not files:
            continue
        found_any = True
        if max_files:
            files = files[:max_files]

        out_dir = os.path.join(output_dir, 'tusz', split)
        os.makedirs(out_dir, exist_ok=True)
        print(f"  [{split}]  {len(files)} files")

        tasks = [
            (f, os.path.join(out_dir, f'{Path(f).stem}.h5'),
             f.replace('.edf', '.tse'), win, stp)
            for f in files
        ]
        results = _run_parallel(tasks, _worker_tusz, f'TUSZ {split}', num_workers)
        n_bckg = n_seiz = skipped = 0
        for status, nb, ns in results:
            if status == 'error':
                skipped += 1
            else:
                n_bckg += nb
                n_seiz += ns
        print(f"    background: {n_bckg:>8}  seizure: {n_seiz:>8}  skipped: {skipped}")

    if not found_any:
        print(f"  [WARN] 未找到 EDF 文件: {tusz_root}")


def process_tuev(tuh_root, output_dir, max_files, num_workers,
                 win, stride_stp, eval_stp):
    print("=" * 60)
    print("TUEV — 事件分类  |  0=spsw 1=gped 2=pled 3=eyem 4=artf 5=bckg")
    print("=" * 60)
    print("  标签来源: train→ch000.lab  eval→文件名前缀")

    tuev_root = os.path.join(tuh_root, 'tuev')
    found_any = False

    for split in ('train', 'eval'):
        stp = stride_stp if split == 'train' else eval_stp
        base = os.path.join(tuev_root, 'edf', split)
        if not os.path.exists(base):
            continue

        edf_files = sorted(glob.glob(os.path.join(base, '**', '*.edf'), recursive=True))
        if not edf_files:
            continue
        found_any = True
        if max_files:
            edf_files = edf_files[:max_files]

        entries = []
        for f in edf_files:
            cls_id = (_tuev_label_from_filename(f) if split == 'eval'
                      else _tuev_label_from_lab(f))
            if cls_id is None:
                cls_id = TUEV_LABELS['bckg']
            entries.append((f, cls_id))

        for cls_name in TUEV_CLASS_NAMES:
            os.makedirs(os.path.join(output_dir, 'tuev', split, cls_name), exist_ok=True)

        print(f"  [{split}]  {len(entries)} files")
        tasks = [
            (f, os.path.join(output_dir, 'tuev', split,
                             TUEV_CLASS_NAMES[cls_id], f'{Path(f).stem}.h5'),
             cls_id, win, stp)
            for f, cls_id in entries
        ]
        results = _run_parallel(tasks, _worker_simple, f'TUEV {split}', num_workers)
        cls_counts = defaultdict(int)
        skipped = 0
        for (_, cls_id), (status, n) in zip(entries, results):
            if status == 'error':
                skipped += 1
            else:
                cls_counts[TUEV_CLASS_NAMES[cls_id]] += n

        for cls in TUEV_CLASS_NAMES:
            print(f"    {cls:>6}: {cls_counts[cls]:>8} segments")
        if skipped:
            print(f"    skipped: {skipped}")

    if not found_any:
        print(f"  [WARN] 未找到 EDF 文件: {tuev_root}")


def process_tuep(tuh_root, output_dir, max_files, num_workers,
                 win, stride_stp, eval_stp,
                 train_ratio=0.8, seed=42):
    print("=" * 60)
    print("TUEP — 癫痫诊断  |  0=no_epilepsy, 1=epilepsy")
    print("=" * 60)

    tuep_root = os.path.join(tuh_root, 'tuep')
    label_map = [
        ('00_epilepsy',    1, 'epilepsy'),
        ('01_no_epilepsy', 0, 'no_epilepsy'),
    ]
    found_any = False
    rng = np.random.default_rng(seed)

    for dir_name, label_id, label_name in label_map:
        base = os.path.join(tuep_root, dir_name)
        if not os.path.exists(base):
            print(f"  [WARN] 目录不存在: {base}")
            continue

        patient_dirs = sorted([
            d for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))
        ])
        if not patient_dirs:
            continue

        pd_arr = np.array(patient_dirs)
        rng.shuffle(pd_arr)
        n_train = int(len(pd_arr) * train_ratio)
        split_map = {p: 'train' for p in pd_arr[:n_train]}
        split_map.update({p: 'eval' for p in pd_arr[n_train:]})

        all_files = sorted(glob.glob(os.path.join(base, '**', '*.edf'), recursive=True))
        if not all_files:
            continue
        found_any = True
        if max_files:
            all_files = all_files[:max_files]

        tasks = []
        for f in all_files:
            rel = os.path.relpath(f, base)
            patient_id = rel.split(os.sep)[0]
            split = split_map.get(patient_id, 'train')
            stp = stride_stp if split == 'train' else eval_stp
            out_dir = os.path.join(output_dir, 'tuep', split, label_name)
            os.makedirs(out_dir, exist_ok=True)
            tasks.append((f, os.path.join(out_dir, f'{Path(f).stem}.h5'),
                          label_id, win, stp))

        results = _run_parallel(tasks, _worker_simple, f'TUEP {label_name}', num_workers)
        counts = defaultdict(int)
        skipped = 0
        for (f, out_path, *_), (status, n) in zip(tasks, results):
            split = 'train' if os.path.join(output_dir, 'tuep', 'train') in out_path else 'eval'
            if status == 'error':
                skipped += 1
            else:
                counts[split] += n
        print(f"  {label_name}: train={counts['train']}  eval={counts['eval']}  skipped={skipped}")

    if not found_any:
        print(f"  [WARN] 未找到 EDF 文件: {tuep_root}")


def process_edf_filelist(ds_name, tuh_root, output_dir, max_files):
    print("=" * 60)
    print(f"{ds_name.upper()} — EDF 文件列表（直接读取，不转 h5）")
    print("=" * 60)

    ds_root = os.path.join(tuh_root, ds_name)
    out_dir = os.path.join(output_dir, ds_name)
    os.makedirs(out_dir, exist_ok=True)
    found_any = False

    for split in ('train', 'eval', 'dev'):
        files = []
        for base in (
            os.path.join(ds_root, 'edf', split),
            os.path.join(ds_root, split),
            ds_root,
        ):
            files = sorted(glob.glob(os.path.join(base, '**', '*.edf'), recursive=True))
            if files:
                break
        if not files:
            continue
        found_any = True
        if max_files:
            files = files[:max_files]

        list_path = os.path.join(out_dir, f'{split}_files.txt')
        with open(list_path, 'w') as f:
            for fp in files:
                f.write(fp + '\n')
        print(f"  [{split}]  {len(files)} files  →  {list_path}")

    if not found_any:
        print(f"  [WARN] 未找到 EDF 文件: {ds_root}")

# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='TUH EEG 数据预处理')
    parser.add_argument('--tuh_root',   default='/projects/u6da/tuh_eeg',
                        help='TUH EEG 原始数据根目录')
    parser.add_argument('--output_dir', default='/projects/u6da/tuh_processed',
                        help='输出目录')
    parser.add_argument('--max_files',  type=int, default=None,
                        help='每个 split 最多处理多少文件（调试用，默认全部）')
    parser.add_argument('--workers',    type=int, default=8,
                        help='并行进程数（默认 8）')
    parser.add_argument('--datasets',   nargs='+',
                        default=['tuab', 'tusz', 'tuev', 'tuep', 'tuar', 'tueg'],
                        help='要处理的数据集')
    args = parser.parse_args()

    win        = int(WINDOW_SEC * TARGET_FS)
    stride_stp = int(STRIDE_SEC * TARGET_FS)
    eval_stp   = int(EVAL_STRIDE_SEC * TARGET_FS)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"TUH_ROOT   = {args.tuh_root}")
    print(f"OUTPUT_DIR = {args.output_dir}")
    print(f"MAX_FILES  = {args.max_files}")
    print(f"WORKERS    = {args.workers}")
    print(f"DATASETS   = {args.datasets}")
    print()

    if 'tuab' in args.datasets:
        process_tuab(args.tuh_root, args.output_dir, args.max_files, args.workers,
                     win, stride_stp, eval_stp)

    if 'tusz' in args.datasets:
        process_tusz(args.tuh_root, args.output_dir, args.max_files, args.workers,
                     win, stride_stp, eval_stp)

    if 'tuev' in args.datasets:
        process_tuev(args.tuh_root, args.output_dir, args.max_files, args.workers,
                     win, stride_stp, eval_stp)

    if 'tuep' in args.datasets:
        process_tuep(args.tuh_root, args.output_dir, args.max_files, args.workers,
                     win, stride_stp, eval_stp)

    if 'tuar' in args.datasets:
        process_edf_filelist('tuar', args.tuh_root, args.output_dir, args.max_files)

    if 'tueg' in args.datasets:
        process_edf_filelist('tueg', args.tuh_root, args.output_dir, args.max_files)

    print("\n全部处理完成！")


if __name__ == '__main__':
    main()
