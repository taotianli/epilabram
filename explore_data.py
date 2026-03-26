"""
Quick exploration of TUH memmap datasets.
Usage: python explore_data.py --memmap_dir /projects/u6da/tuh_processed/memmap
"""
import argparse
import numpy as np
import os

TASKS = ['TUAB', 'TUSZ', 'TUEV', 'TUEP']
TUEV_CLASS_NAMES = {0: 'spsw', 1: 'gped', 2: 'pled', 3: 'eyem', 4: 'artf', 5: 'bckg'}


def explore_task(task, base):
    path = os.path.join(base, task)
    if not os.path.exists(path):
        print(f'  [skip] {path} not found')
        return

    print(f'\n{"="*50}')
    print(f'  {task}')
    print(f'{"="*50}')

    for split in ['train', 'eval']:
        label_file = os.path.join(path, f'{split}_labels.npy')
        if not os.path.exists(label_file):
            print(f'  [{split}] label file not found')
            continue

        meta_file = os.path.join(path, f'{split}_meta.json')
        if os.path.exists(meta_file):
            import json
            meta = json.load(open(meta_file))
            labels = np.memmap(label_file, dtype=meta['lbl_dtype'], mode='r', shape=(meta['N'],))
        else:
            labels = np.load(label_file, allow_pickle=True)
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        print(f'\n  {split} — {total:,} samples')
        for cls, cnt in zip(unique, counts):
            name = TUEV_CLASS_NAMES.get(cls, '') if task == 'TUEV' else ''
            tag = f' ({name})' if name else ''
            print(f'    class {cls}{tag}: {cnt:7,}  ({cnt/total*100:.1f}%)')

    # check data file
    for split in ['train', 'eval']:
        for ext in ['.npy', '.dat']:
            data_file = os.path.join(path, f'{split}_data{ext}')
            if os.path.exists(data_file):
                size_gb = os.path.getsize(data_file) / 1e9
                print(f'\n  {split}_data{ext}: {size_gb:.2f} GB')
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--memmap_dir', default='/projects/u6da/tuh_processed/memmap')
    parser.add_argument('--tasks', nargs='+', default=TASKS)
    args = parser.parse_args()

    print(f'Memmap dir: {args.memmap_dir}')
    print(f'Tasks: {args.tasks}')

    for task in args.tasks:
        explore_task(task, args.memmap_dir)


if __name__ == '__main__':
    main()
