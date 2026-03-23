"""
Stage3 训练入口
支持 EDF 格式（--data_format edf）和 H5 格式（--data_format h5，默认）
"""

import argparse
import os
import sys
import yaml
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if 'LaBraM' not in p]
sys.path.insert(0, _HERE)

from models.epilabram import build_epilabram
from data.tuh_dataset import TUABDataset, TUSZDataset, TUEVDataset, TUEPDataset
from data.tuar_edf_dataset import TUAREDFDataset
from data.preprocessing import EEGPreprocessor
from training.stage3_trainer import Stage3Trainer, PreferenceDataset
from utils.seed import set_seed
from utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='EpiLaBraM Stage3 Training (CPA-DPO)')
    parser.add_argument('--config', type=str, default='configs/stage3_cpa_dpo.yaml')
    parser.add_argument('--output_dir', type=str, default='experiments/stage3')
    parser.add_argument('--stage2_ckpt', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--task', type=str, default='TUAB',
                        choices=['TUAB', 'TUSZ', 'TUEV', 'TUEP'])
    parser.add_argument('--data_format', type=str, default='h5', choices=['h5', 'edf'])
    parser.add_argument('--edf_path', type=str, default=None)
    parser.add_argument('--max_files', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = {
        **cfg.get('training', {}),
        'beta_dpo': cfg.get('dpo', {}).get('beta', 0.1),
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_epilabram(backbone_size='base').to(device)
    load_checkpoint(args.stage2_ckpt, model, strict=False)

    data_cfg = cfg.get('data', {})
    window_sec = data_cfg.get('window_sec', 10.0)
    stride_sec = data_cfg.get('stride_sec', 5.0)
    pref_cfg = cfg.get('preference_data', {})

    if args.data_format == 'edf':
        edf_path = args.edf_path or data_cfg.get('edf_path')
        assert edf_path and os.path.exists(edf_path), f"EDF路径不存在: {edf_path}"
        base_ds = TUAREDFDataset(edf_path, window_sec=window_sec, stride_sec=stride_sec,
                                 max_files=args.max_files)
    else:
        preprocessor = EEGPreprocessor(target_fs=data_cfg.get('sample_rate', 200))
        task_ds_map = {
            'TUAB': (TUABDataset, 'tuab_path'),
            'TUSZ': (TUSZDataset, 'tusz_path'),
            'TUEV': (TUEVDataset, 'tuev_path'),
            'TUEP': (TUEPDataset, 'tuep_path'),
        }
        cls, path_key = task_ds_map[args.task]
        path = data_cfg.get(path_key)
        assert path and os.path.exists(path), f"Dataset path not found: {path}"
        base_ds = cls(path, window_sec=window_sec, stride_sec=stride_sec,
                      preprocessor=preprocessor, split='train')

    print(f"Building preference dataset from {args.task} ({len(base_ds)} segments)...")
    pref_dataset = PreferenceDataset.build_from_base_dataset(
        base_ds, model, device,
        confidence_threshold=pref_cfg.get('model_confidence_threshold', 0.9),
    )
    print(f"Preference pairs: {len(pref_dataset)}")

    os.makedirs(args.output_dir, exist_ok=True)

    trainer = Stage3Trainer(
        model=model,
        preference_dataset=pref_dataset,
        config=train_cfg,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
    )
    trainer.train(resume_ckpt=args.resume)


if __name__ == '__main__':
    main()
