"""
Inspect h5 file structure for TUEV train vs eval.
Usage: python explore_h5.py
"""
import h5py
import os

FILES = [
    '/projects/u6da/tuh_processed/tuev/train/eyem/aaaaaadm_00000001.h5',
    '/projects/u6da/tuh_processed/tuev/eval/spsw/spsw_001_a_1.h5',
]

for f in FILES:
    if not os.path.exists(f):
        print(f'[skip] {f}')
        continue
    print(f)
    with h5py.File(f, 'r') as h:
        keys = list(h.keys())
        print(f'  top-level keys ({len(keys)}): {keys[:5]}')
        k = keys[0]
        print(f'  h["{k}"] keys: {list(h[k].keys())}')
        for sk in h[k].keys():
            item = h[k][sk]
            if hasattr(item, 'shape'):
                print(f'    {sk}: shape={item.shape} dtype={item.dtype}')
            else:
                print(f'    {sk}: (group)')
    print()
