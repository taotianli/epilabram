"""
TUH EEG 数据预处理脚本
将原始 EDF 文件处理成 HDF5 格式，供训练使用。

数据集用途：
  TUAR  → Stage1 无监督续训（EDF 直读，无需预处理成 h5）
  TUAB  → Stage2 异常检测（binary: normal=0 / abnormal=1）
  TUSZ  → Stage2 发作检测（binary: bckg=0 / seizure=1）
  TUEV  → Stage2 事件分类（6类: SPSW/GPED/PLED/EYEM/ARTF/BCKG）
  TUEP  → Stage2 癫痫诊断（binary: no_epilepsy=0 / epilepsy=1）
  TUEG  → Stage1 无监督续训（背景EEG，与TUAR合并使用）

集群数据根目录：/projects/u6da/tuh_eeg/
  tueg/  tuab/  tuar/  tuep/  tuev/  tusz/

用法：
  # 处理所有数据集
  python preprocess_tuh.py --tuh_root /projects/u6da/tuh_eeg --output_dir /projects/u6da/tuh_processed

  # 只处理某个数据集
  python preprocess_tuh.py --tuh_root /projects/u6da/tuh_eeg --output_dir /projects/u6da/tuh_processed --datasets tuab tusz

  # 调试（每个 split 只处理前N个文件）
  python preprocess_tuh.py --tuh_root /projects/u6da/tuh_eeg --output_dir ./data_test --max_files 20

输出目录结构：
  output_dir/
    tuab/
      train/normal/*.h5
      train/abnormal/*.h5
      eval/normal/*.h5
      eval/abnormal/*.h5
    tusz/
      train/*.h5
      dev/*.h5
      eval/*.h5
    tuev/
      train/spsw/*.h5  (等6个子目录)
      eval/spsw/*.h5
    tuep/
      train/epilepsy/*.h5
      train/no_epilepsy/*.h5
      eval/epilepsy/*.h5
      eval/no_epilepsy/*.h5
    tuar/           (EDF直接使用，不做h5转换，只生成文件列表)
      train_files.txt
      eval_files.txt
    tueg/           (同上)
      train_files.txt
      eval_files.txt
"""

import os
import sys
import glob
import argparse
import json
import traceback
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

import numpy as np
import h5py
import mne

mne.set_log_level('WARNING')

# ============================================================
# 常量
# ============================================================

TARGET_FS = 200.0        # 统一重采样到200Hz
WINDOW_SEC = 10.0        # 10秒窗口
STRIDE_SEC = 5.0         # 5秒步进（训练用）
EVAL_STRIDE_SEC = 10.0   # 评估用不重叠窗口

STANDARD_23 = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ', 'A1', 'A2', 'T1', 'T2',
]
STD_IDX = {ch: i for i, ch in enumerate(STANDARD_23)}

# TUH EDF 文件中的通道名 -> 标准名映射
_ALIAS_MAP: Dict[str, str] = {}
for _ch in STANDARD_23:
    _ALIAS_MAP[_ch] = _ch
    _ALIAS_MAP[f'EEG {_ch}-REF'] = _ch
    _ALIAS_MAP[f'EEG {_ch}-LE']  = _ch
    _ALIAS_MAP[f'{_ch}-REF']     = _ch
    _ALIAS_MAP[f'{_ch}-LE']      = _ch
    _ALIAS_MAP[f'EEG {_ch}']     = _ch

# TUEV 6类标签
TUEV_LABELS = {
    'spsw': 0, 'gped': 1, 'pled': 2, 'eyem': 3, 'artf': 4, 'bckg': 5,
}
TUEV_CLASS_NAMES = ['spsw', 'gped', 'pled', 'eyem', 'artf', 'bckg']

# TUSZ 标注中的发作标签（含多种发作类型）
TUSZ_SEIZURE_LABELS = {
    'fnsz', 'gnsz', 'absz', 'tnsz', 'cpsz', 'tcsz', 'mysz',
    'spsz', 'cnsz', 'atsz', 'nesz', 'seiz',
}


# ============================================================
# 通用 EDF 读取与预处理
# ============================================================

def load_and_preprocess_edf(
    edf_path: str,
    target_fs: float = TARGET_FS,
) -> Tuple[Optional[np.ndarray], List[str], float]:
    """
    读取 EDF，对齐到标准23通道，重采样，幅值归一化。

    Returns:
        eeg: (23, T) float32 或 None（读取失败）
        ch_names: STANDARD_23
        orig_fs: 原始采样率
    """
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        print(f"  [ERROR] Cannot read {edf_path}: {e}")
        return None, [], 0.0

    orig_fs = raw.info['sfreq']

    # 选出能映射到标准通道的通道
    available = []
    for ch in raw.ch_names:
        std = _ALIAS_MAP.get(ch.upper().strip(), _ALIAS_MAP.get(ch.strip()))
        if std:
            available.append(ch)
    if not available:
        return None, [], orig_fs

    raw.pick_channels(available, ordered=False)

    # 重采样
    if abs(orig_fs - target_fs) > 1.0:
        raw.resample(target_fs, verbose=False)

    data = raw.get_data() * 1e6   # V -> μV

    # 对齐到 23 通道
    T = data.shape[1]
    aligned = np.zeros((23, T), dtype=np.float32)
    for i, ch in enumerate(raw.ch_names):
        std = _ALIAS_MAP.get(ch.upper().strip(), _ALIAS_MAP.get(ch.strip()))
        if std and std in STD_IDX:
            aligned[STD_IDX[std]] = data[i].astype(np.float32)

    # 幅值归一化：除以100（μV量级 → 约[-1,1]）
    aligned /= 100.0

    return aligned, STANDARD_23, orig_fs


# ============================================================
# TUAB 预处理
# ============================================================

def _find_tuab_edfs(tuab_root: str) -> Dict[str, List[Tuple[str, int]]]:
    """
    扫描 TUAB 目录，返回 {split: [(edf_path, label), ...]}

    TUAB 原始结构（v3.x）：
      tuab/
        edf/
          train/
            normal/    *.edf
            abnormal/  *.edf
          eval/
            normal/    *.edf
            abnormal/  *.edf
    """
    splits = {}
    for split in ('train', 'eval'):
        entries = []
        for label_name, label_id in [('normal', 0), ('abnormal', 1)]:
            pattern = os.path.join(tuab_root, 'edf', split, label_name, '**', '*.edf')
            for f in sorted(glob.glob(pattern, recursive=True)):
                entries.append((f, label_id))
        if entries:
            splits[split] = entries
    return splits


def process_tuab(tuab_root: str, output_dir: str, max_files: Optional[int],
                 window_sec: float, stride_sec: float, eval_stride_sec: float):
    print("\n" + "="*60)
    print("Processing TUAB (Abnormal EEG Corpus)")
    print("="*60)
    print("Labels: 0=normal, 1=abnormal")

    splits_data = _find_tuab_edfs(tuab_root)
    if not splits_data:
        print(f"  [WARN] No EDF files found under {tuab_root}/edf/")
        return

    stats = defaultdict(lambda: defaultdict(int))

    for split, entries in splits_data.items():
        stride = stride_sec if split == 'train' else eval_stride_sec
        win = int(window_sec * TARGET_FS)
        stp = int(stride * TARGET_FS)

        if max_files:
            entries = entries[:max_files]

        for label_name, label_id in [('normal', 0), ('abnormal', 1)]:
            out_subdir = os.path.join(output_dir, 'tuab', split, label_name)
            os.makedirs(out_subdir, exist_ok=True)

        print(f"\n  Split: {split} ({len(entries)} files, stride={stride}s)")
        skipped = 0

        for edf_path, label_id in entries:
            label_name = 'normal' if label_id == 0 else 'abnormal'
            rec_id = Path(edf_path).stem
            out_path = os.path.join(output_dir, 'tuab', split, label_name, f'{rec_id}.h5')

            if os.path.exists(out_path):
                stats[split][label_name] += 1
                continue

            eeg, ch_names, _ = load_and_preprocess_edf(edf_path)
            if eeg is None:
                skipped += 1
                continue

            T = eeg.shape[1]
            segments = []
            for start in range(0, T - win + 1, stp):
                segments.append(eeg[:, start:start + win])

            if not segments:
                skipped += 1
                continue

            with h5py.File(out_path, 'w') as hf:
                grp = hf.create_group(rec_id)
                data_arr = np.stack(segments, axis=0)   # (N, 23, win)
                ds = grp.create_dataset('eeg', data=data_arr, compression='gzip', compression_opts=4)
                ds.attrs['chOrder'] = ch_names
                ds.attrs['label'] = label_id
                ds.attrs['n_segments'] = len(segments)
                ds.attrs['source'] = edf_path

            stats[split][label_name] += len(segments)

        print(f"    normal:   {stats[split]['normal']:>8} segments")
        print(f"    abnormal: {stats[split]['abnormal']:>8} segments")
        if skipped:
            print(f"    skipped:  {skipped} files")


# ============================================================
# TUSZ 预处理
# ============================================================

def _parse_tusz_annotations(tse_path: str) -> Optional[np.ndarray]:
    """
    解析 TUSZ .tse（tab-separated events）标注文件。

    格式：
      version = tse_v1.0.0
      start_time  stop_time  label  confidence
      0.0000  10.0000  bckg  1.0000
      ...
    """
    if not os.path.exists(tse_path):
        return None
    try:
        events = []
        with open(tse_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('version'):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                start, stop, label = float(parts[0]), float(parts[1]), parts[2].lower()
                is_seizure = label in TUSZ_SEIZURE_LABELS
                events.append((start, stop, int(is_seizure)))
        return events if events else None
    except Exception:
        return None


def _find_tusz_edfs(tusz_root: str) -> Dict[str, List[str]]:
    """
    TUSZ 原始结构：
      tusz/
        edf/
          train/  **/*.edf  (配套 *.tse 标注)
          dev/    **/*.edf
          eval/   **/*.edf
    """
    splits = {}
    for split in ('train', 'dev', 'eval'):
        pattern = os.path.join(tusz_root, 'edf', split, '**', '*.edf')
        files = sorted(glob.glob(pattern, recursive=True))
        if files:
            splits[split] = files
    return splits


def process_tusz(tusz_root: str, output_dir: str, max_files: Optional[int],
                 window_sec: float, stride_sec: float, eval_stride_sec: float):
    print("\n" + "="*60)
    print("Processing TUSZ (Seizure Corpus)")
    print("="*60)
    print("Labels: 0=background, 1=seizure")

    splits_data = _find_tusz_edfs(tusz_root)
    if not splits_data:
        print(f"  [WARN] No EDF files found under {tusz_root}/edf/")
        return

    for split, files in splits_data.items():
        stride = stride_sec if split == 'train' else eval_stride_sec
        win = int(window_sec * TARGET_FS)
        stp = int(stride * TARGET_FS)

        if max_files:
            files = files[:max_files]

        out_dir = os.path.join(output_dir, 'tusz', split)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n  Split: {split} ({len(files)} files, stride={stride}s)")
        n_bckg = n_seiz = skipped = 0

        for edf_path in files:
            rec_id = Path(edf_path).stem
            out_path = os.path.join(out_dir, f'{rec_id}.h5')

            if os.path.exists(out_path):
                continue

            # 找对应 .tse 标注
            tse_path = edf_path.replace('.edf', '.tse')
            events = _parse_tusz_annotations(tse_path)

            eeg, ch_names, _ = load_and_preprocess_edf(edf_path)
            if eeg is None:
                skipped += 1
                continue

            T = eeg.shape[1]

            # 构建逐采样点标签（-1=未标注，0=背景，1=发作）
            label_arr = np.zeros(T, dtype=np.int8)
            if events:
                for start_t, stop_t, lbl in events:
                    s = int(start_t * TARGET_FS)
                    e = min(int(stop_t * TARGET_FS), T)
                    if s < e:
                        label_arr[s:e] = lbl

            # 按窗口切分
            segments, labels = [], []
            for start in range(0, T - win + 1, stp):
                seg_labels = label_arr[start:start + win]
                # 取多数标签
                n_seiz_pts = int(seg_labels.sum())
                lbl = 1 if n_seiz_pts > win * 0.3 else 0  # >30%发作点 → 发作窗口
                segments.append(eeg[:, start:start + win])
                labels.append(lbl)
                if lbl == 1:
                    n_seiz += 1
                else:
                    n_bckg += 1

            if not segments:
                skipped += 1
                continue

            with h5py.File(out_path, 'w') as hf:
                grp = hf.create_group(rec_id)
                data_arr = np.stack(segments, axis=0)       # (N, 23, win)
                labels_arr = np.array(labels, dtype=np.int8)
                ds = grp.create_dataset('eeg', data=data_arr, compression='gzip', compression_opts=4)
                ds.attrs['chOrder'] = ch_names
                grp.create_dataset('label', data=labels_arr)
                grp.attrs['n_segments'] = len(segments)
                grp.attrs['source'] = edf_path

        print(f"    background: {n_bckg:>8} segments")
        print(f"    seizure:    {n_seiz:>8} segments")
        if skipped:
            print(f"    skipped:    {skipped} files")


# ============================================================
# TUEV 预处理
# ============================================================

def _find_tuev_edfs(tuev_root: str) -> Dict[str, List[Tuple[str, int]]]:
    """
    TUEV 原始结构：
      tuev/
        edf/
          train/
            spsw/ *.edf  (配套 *.rec 或 *.tse_bi 标注)
            gped/ *.edf
            pled/ *.edf
            eyem/ *.edf
            artf/ *.edf
            bckg/ *.edf
          eval/
            ...

    有些版本直接按类别分目录，有些版本是混合目录+标注文件。
    这里按目录名判断类别（适配常见版本）。
    """
    splits = {}
    for split in ('train', 'eval'):
        entries = []
        for cls_name, cls_id in TUEV_LABELS.items():
            pattern = os.path.join(tuev_root, 'edf', split, cls_name, '**', '*.edf')
            for f in sorted(glob.glob(pattern, recursive=True)):
                entries.append((f, cls_id))
        if entries:
            splits[split] = entries
    return splits


def process_tuev(tuev_root: str, output_dir: str, max_files: Optional[int],
                 window_sec: float, stride_sec: float, eval_stride_sec: float):
    print("\n" + "="*60)
    print("Processing TUEV (EEG Events Corpus)")
    print("="*60)
    print("Labels: 0=spsw, 1=gped, 2=pled, 3=eyem, 4=artf, 5=bckg")

    splits_data = _find_tuev_edfs(tuev_root)
    if not splits_data:
        print(f"  [WARN] No EDF files found under {tuev_root}/edf/")
        print("  Trying flat structure...")
        splits_data = _find_tuev_edfs_flat(tuev_root)
        if not splits_data:
            return

    for split, entries in splits_data.items():
        stride = stride_sec if split == 'train' else eval_stride_sec
        win = int(window_sec * TARGET_FS)
        stp = int(stride * TARGET_FS)

        if max_files:
            entries = entries[:max_files]

        for cls_name in TUEV_LABELS:
            os.makedirs(os.path.join(output_dir, 'tuev', split, cls_name), exist_ok=True)

        print(f"\n  Split: {split} ({len(entries)} files, stride={stride}s)")
        cls_counts = defaultdict(int)
        skipped = 0

        for edf_path, cls_id in entries:
            cls_name = TUEV_CLASS_NAMES[cls_id]
            rec_id = Path(edf_path).stem
            out_path = os.path.join(output_dir, 'tuev', split, cls_name, f'{rec_id}.h5')

            if os.path.exists(out_path):
                cls_counts[cls_name] += 1
                continue

            eeg, ch_names, _ = load_and_preprocess_edf(edf_path)
            if eeg is None:
                skipped += 1
                continue

            T = eeg.shape[1]
            segments = []
            for start in range(0, T - win + 1, stp):
                segments.append(eeg[:, start:start + win])

            if not segments:
                skipped += 1
                continue

            with h5py.File(out_path, 'w') as hf:
                grp = hf.create_group(rec_id)
                data_arr = np.stack(segments, axis=0)
                ds = grp.create_dataset('eeg', data=data_arr, compression='gzip', compression_opts=4)
                ds.attrs['chOrder'] = ch_names
                ds.attrs['label'] = cls_id
                grp.attrs['n_segments'] = len(segments)
                grp.attrs['source'] = edf_path

            cls_counts[cls_name] += len(segments)

        for cls_name in TUEV_CLASS_NAMES:
            print(f"    {cls_name:>6}: {cls_counts[cls_name]:>8} segments")
        if skipped:
            print(f"    skipped: {skipped} files")


def _find_tuev_edfs_flat(tuev_root: str) -> Dict[str, List[Tuple[str, int]]]:
    """备选：扁平目录，从标注文件读取类别"""
    splits = {}
    for split in ('train', 'eval'):
        entries = []
        pattern = os.path.join(tuev_root, 'edf', split, '**', '*.edf')
        for edf_path in sorted(glob.glob(pattern, recursive=True)):
            # 尝试从同名 .tse_bi 文件读取类别
            tse_path = edf_path.replace('.edf', '.tse_bi')
            if not os.path.exists(tse_path):
                tse_path = edf_path.replace('.edf', '.tse')
            cls_id = _infer_tuev_label(tse_path)
            if cls_id is not None:
                entries.append((edf_path, cls_id))
        if entries:
            splits[split] = entries
    return splits


def _infer_tuev_label(tse_path: str) -> Optional[int]:
    """从 .tse_bi 或 .tse 文件推断 TUEV 类别（取最多出现的非背景类）"""
    if not os.path.exists(tse_path):
        return None
    counts = defaultdict(float)
    try:
        with open(tse_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                try:
                    start, stop, label = float(parts[0]), float(parts[1]), parts[2].lower()
                except ValueError:
                    continue
                if label in TUEV_LABELS:
                    counts[label] += stop - start
        if not counts:
            return TUEV_LABELS['bckg']
        # 取持续时间最长的类别，背景作为兜底
        best = max(counts, key=lambda k: counts[k])
        return TUEV_LABELS[best]
    except Exception:
        return None


# ============================================================
# TUEP 预处理
# ============================================================

def _find_tuep_edfs(tuep_root: str) -> Dict[str, List[Tuple[str, int]]]:
    """
    TUEP 原始结构：
      tuep/
        edf/
          train/
            epilepsy/    *.edf
            no_epilepsy/ *.edf
          eval/
            ...
    """
    splits = {}
    for split in ('train', 'eval'):
        entries = []
        for label_name, label_id in [('epilepsy', 1), ('no_epilepsy', 0)]:
            pattern = os.path.join(tuep_root, 'edf', split, label_name, '**', '*.edf')
            for f in sorted(glob.glob(pattern, recursive=True)):
                entries.append((f, label_id))
        if entries:
            splits[split] = entries
    return splits


def process_tuep(tuep_root: str, output_dir: str, max_files: Optional[int],
                 window_sec: float, stride_sec: float, eval_stride_sec: float):
    print("\n" + "="*60)
    print("Processing TUEP (Epilepsy Corpus)")
    print("="*60)
    print("Labels: 0=no_epilepsy, 1=epilepsy")

    splits_data = _find_tuep_edfs(tuep_root)
    if not splits_data:
        print(f"  [WARN] No EDF files found under {tuep_root}/edf/")
        return

    for split, entries in splits_data.items():
        stride = stride_sec if split == 'train' else eval_stride_sec
        win = int(window_sec * TARGET_FS)
        stp = int(stride * TARGET_FS)

        if max_files:
            entries = entries[:max_files]

        for label_name in ('epilepsy', 'no_epilepsy'):
            os.makedirs(os.path.join(output_dir, 'tuep', split, label_name), exist_ok=True)

        print(f"\n  Split: {split} ({len(entries)} files, stride={stride}s)")
        n_epi = n_no = skipped = 0

        for edf_path, label_id in entries:
            label_name = 'epilepsy' if label_id == 1 else 'no_epilepsy'
            rec_id = Path(edf_path).stem
            out_path = os.path.join(output_dir, 'tuep', split, label_name, f'{rec_id}.h5')

            if os.path.exists(out_path):
                continue

            eeg, ch_names, _ = load_and_preprocess_edf(edf_path)
            if eeg is None:
                skipped += 1
                continue

            T = eeg.shape[1]
            segments = []
            for start in range(0, T - win + 1, stp):
                segments.append(eeg[:, start:start + win])

            if not segments:
                skipped += 1
                continue

            with h5py.File(out_path, 'w') as hf:
                grp = hf.create_group(rec_id)
                data_arr = np.stack(segments, axis=0)
                ds = grp.create_dataset('eeg', data=data_arr, compression='gzip', compression_opts=4)
                ds.attrs['chOrder'] = ch_names
                ds.attrs['label'] = label_id
                grp.attrs['n_segments'] = len(segments)
                grp.attrs['source'] = edf_path

            if label_id == 1:
                n_epi += len(segments)
            else:
                n_no += len(segments)

        print(f"    epilepsy:    {n_epi:>8} segments")
        print(f"    no_epilepsy: {n_no:>8} segments")
        if skipped:
            print(f"    skipped:     {skipped} files")


# ============================================================
# TUAR / TUEG：生成文件列表（EDF 直读，无需转 h5）
# ============================================================

def process_edf_filelist(
    ds_name: str,
    ds_root: str,
    output_dir: str,
    max_files: Optional[int],
):
    """
    TUAR 和 TUEG 直接在训练时用 TUAREDFDataset 读取 EDF，
    这里只生成 train_files.txt / eval_files.txt 列表文件。
    """
    print("\n" + "="*60)
    print(f"Processing {ds_name.upper()} (EDF file list only)")
    print("="*60)

    out_dir = os.path.join(output_dir, ds_name)
    os.makedirs(out_dir, exist_ok=True)

    for split in ('train', 'eval', 'dev'):
        # 尝试 edf/{split} 和直接 {split}
        for base in (
            os.path.join(ds_root, 'edf', split),
            os.path.join(ds_root, split),
        ):
            pattern = os.path.join(base, '**', '*.edf')
            files = sorted(glob.glob(pattern, recursive=True))
            if files:
                break

        if not files:
            continue

        if max_files:
            files = files[:max_files]

        list_path = os.path.join(out_dir, f'{split}_files.txt')
        with open(list_path, 'w') as f:
            for fp in files:
                f.write(fp + '\n')

        print(f"  {split}: {len(files)} files → {list_path}")


# ============================================================
# 统计信息汇总
# ============================================================

def print_summary(output_dir: str):
    print("\n" + "="*60)
    print("Dataset Summary")
    print("="*60)

    summary = {}

    for ds in ('tuab', 'tusz', 'tuev', 'tuep'):
        ds_dir = os.path.join(output_dir, ds)
        if not os.path.exists(ds_dir):
            continue
        h5_files = glob.glob(os.path.join(ds_dir, '**', '*.h5'), recursive=True)
        total_seg = 0
        for h5p in h5_files:
            try:
                with h5py.File(h5p, 'r') as hf:
                    for grp in hf.values():
                        n = grp.attrs.get('n_segments', 0)
                        total_seg += int(n) if n else 0
            except Exception:
                pass
        summary[ds] = {'h5_files': len(h5_files), 'segments': total_seg}
        print(f"  {ds.upper():>6}: {len(h5_files):>6} h5 files, {total_seg:>10} segments")

    for ds in ('tuar', 'tueg'):
        ds_dir = os.path.join(output_dir, ds)
        if not os.path.exists(ds_dir):
            continue
        txt_files = glob.glob(os.path.join(ds_dir, '*.txt'))
        total_edfs = 0
        for tp in txt_files:
            with open(tp) as f:
                total_edfs += sum(1 for _ in f)
        summary[ds] = {'edf_files': total_edfs}
        print(f"  {ds.upper():>6}: {total_edfs:>6} EDF files (direct read)")

    # 保存 summary.json
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


# ============================================================
# 主函数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess TUH EEG datasets to HDF5 format'
    )
    parser.add_argument(
        '--tuh_root', type=str, required=True,
        help='TUH EEG 根目录，包含 tuab/tusz/tuev/tuep/tuar/tueg 子目录',
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='处理结果输出目录',
    )
    parser.add_argument(
        '--datasets', nargs='+',
        choices=['tuab', 'tusz', 'tuev', 'tuep', 'tuar', 'tueg', 'all'],
        default=['all'],
        help='要处理的数据集（默认全部）',
    )
    parser.add_argument(
        '--window_sec', type=float, default=WINDOW_SEC,
        help=f'时间窗口长度（秒，默认{WINDOW_SEC}）',
    )
    parser.add_argument(
        '--stride_sec', type=float, default=STRIDE_SEC,
        help=f'训练集滑窗步长（秒，默认{STRIDE_SEC}）',
    )
    parser.add_argument(
        '--eval_stride_sec', type=float, default=EVAL_STRIDE_SEC,
        help=f'验证/测试集步长（秒，默认{EVAL_STRIDE_SEC}，不重叠）',
    )
    parser.add_argument(
        '--max_files', type=int, default=None,
        help='每个split最多处理的文件数（调试用）',
    )
    parser.add_argument(
        '--n_workers', type=int, default=1,
        help='并行处理工作进程数（暂未使用，预留参数）',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"TUH Root:   {args.tuh_root}")
    print(f"Output Dir: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    datasets = args.datasets
    if 'all' in datasets:
        datasets = ['tuab', 'tusz', 'tuev', 'tuep', 'tuar', 'tueg']

    processors = {
        'tuab': lambda: process_tuab(
            os.path.join(args.tuh_root, 'tuab'),
            args.output_dir, args.max_files,
            args.window_sec, args.stride_sec, args.eval_stride_sec,
        ),
        'tusz': lambda: process_tusz(
            os.path.join(args.tuh_root, 'tusz'),
            args.output_dir, args.max_files,
            args.window_sec, args.stride_sec, args.eval_stride_sec,
        ),
        'tuev': lambda: process_tuev(
            os.path.join(args.tuh_root, 'tuev'),
            args.output_dir, args.max_files,
            args.window_sec, args.stride_sec, args.eval_stride_sec,
        ),
        'tuep': lambda: process_tuep(
            os.path.join(args.tuh_root, 'tuep'),
            args.output_dir, args.max_files,
            args.window_sec, args.stride_sec, args.eval_stride_sec,
        ),
        'tuar': lambda: process_edf_filelist(
            'tuar', os.path.join(args.tuh_root, 'tuar'),
            args.output_dir, args.max_files,
        ),
        'tueg': lambda: process_edf_filelist(
            'tueg', os.path.join(args.tuh_root, 'tueg'),
            args.output_dir, args.max_files,
        ),
    }

    for ds in datasets:
        try:
            processors[ds]()
        except Exception as e:
            print(f"\n[ERROR] Failed to process {ds}: {e}")
            traceback.print_exc()

    print_summary(args.output_dir)
    print("\nDone.")


if __name__ == '__main__':
    main()
