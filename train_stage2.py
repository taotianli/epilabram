"""
Stage2 训练入口
支持 EDF 格式（--data_format edf）和 H5 格式（--data_format h5，默认）
"""

import argparse
import os
import sys
import yaml
import torch
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if 'LaBraM' not in p]
sys.path.insert(0, _HERE)

from models.epilabram import build_epilabram
from data.tuh_dataset import TUABDataset, TUSZDataset, TUEVDataset, TUEPDataset
from data.tuar_edf_dataset import TUAREDFDataset
from data.preprocessing import EEGPreprocessor
from training.stage2_trainer import Stage2Trainer
from utils.seed import set_seed
from utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='EpiLaBraM Stage2 Training (MTPCT)')
    parser.add_argument('--config', type=str, default='configs/stage2_mtpct.yaml')
    parser.add_argument('--output_dir', type=str, default='experiments/stage2')
    parser.add_argument('--stage1_ckpt', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--data_format', type=str, default='h5', choices=['h5', 'edf'])
    parser.add_argument('--edf_path', type=str, default=None)
    parser.add_argument('--max_files', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.batch_size:
        cfg['training']['batch_size'] = args.batch_size
    if args.epochs:
        cfg['training']['total_epochs'] = args.epochs

    data_cfg = cfg.get('data', {})
    train_cfg = {
        **cfg.get('training', {}),
        **cfg.get('hierarchy', {}),
        'num_workers': data_cfg.get('num_workers', 4),
        'patch_size': 200,
    }

    is_distributed = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    if is_distributed:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    model = build_epilabram(
        backbone_size=cfg.get('model', {}).get('backbone_size', 'base'),
        n_prompt=cfg.get('model', {}).get('n_prompt_tokens', 10),
        adapter_bottleneck_ratio=cfg.get('model', {}).get('adapter_bottleneck_ratio', 4),
    ).to(device)

    load_checkpoint(args.stage1_ckpt, model, strict=False)

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    window_sec = data_cfg.get('window_sec', 10.0)
    stride_sec = data_cfg.get('stride_sec', 5.0)

    if args.data_format == 'edf':
        edf_path = args.edf_path or data_cfg.get('edf_path')
        assert edf_path and os.path.exists(edf_path), f"EDF路径不存在: {edf_path}"
        ds = TUAREDFDataset(edf_path, window_sec=window_sec, stride_sec=stride_sec,
                            max_files=args.max_files)
        # Stage2 EDF 模式：TUAB 任务（二分类），其余为 None
        train_datasets = [ds, None, None, None]
        val_datasets   = [None, None, None, None]
    else:
        preprocessor = EEGPreprocessor(target_fs=data_cfg.get('sample_rate', 200))

        def _make(cls, key, split='train'):
            path = data_cfg.get(key)
            if path and os.path.exists(path):
                return cls(path, window_sec=window_sec, stride_sec=stride_sec,
                           preprocessor=preprocessor, split=split)
            return None

        train_datasets = [_make(TUABDataset, 'tuab_path'),
                          _make(TUSZDataset, 'tusz_path'),
                          _make(TUEVDataset, 'tuev_path'),
                          _make(TUEPDataset, 'tuep_path')]
        val_datasets   = [_make(TUABDataset, 'tuab_path', 'val'),
                          _make(TUSZDataset, 'tusz_path', 'val'),
                          _make(TUEVDataset, 'tuev_path', 'val'),
                          _make(TUEPDataset, 'tuep_path', 'val')]

    os.makedirs(args.output_dir, exist_ok=True)

    trainer = Stage2Trainer(
        model=model,
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        config=train_cfg,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
        local_rank=local_rank,
    )
    trainer.train(resume_ckpt=args.resume)


if __name__ == '__main__':
    main()
