"""
Stage1 训练入口
支持 EDF 格式（--data_format edf）和 H5 格式（--data_format h5，默认）
"""

import argparse
import os
import sys
import yaml
import torch
import torch.distributed as dist

# 确保 epilabram 包优先，避免误导入同名模块
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if 'LaBraM' not in p]
sys.path.insert(0, _HERE)

from models.epilabram import build_epilabram
from data.tuh_dataset import TUABDataset, TUSZDataset, TUEVDataset, TUEPDataset
from data.tuar_edf_dataset import TUAREDFDataset
from data.preprocessing import EEGPreprocessor
from training.stage1_trainer import Stage1Trainer
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='EpiLaBraM Stage1 Training (PADM)')
    parser.add_argument('--config', type=str, default='configs/stage1_dacp.yaml')
    parser.add_argument('--output_dir', type=str, default='experiments/stage1')
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--labram_root', type=str, default=None,
                        help='原始LaBraM代码根目录（含modeling_vqnsp.py），也可用环境变量LABRAM_ROOT')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    # EDF 专用参数
    parser.add_argument('--data_format', type=str, default='h5', choices=['h5', 'edf'],
                        help='数据格式：h5（原始LaBraM格式）或 edf（直接读EDF）')
    parser.add_argument('--edf_path', type=str, default=None,
                        help='EDF格式时的数据根目录（如 /path/to/TUAR/edf）')
    parser.add_argument('--max_files', type=int, default=None,
                        help='EDF格式时最多加载的文件数（调试用）')
    return parser.parse_args()


def _build_edf_datasets(edf_path, window_sec, stride_sec, max_files):
    """构建 EDF 格式数据集列表（Stage1 只需要一个无标签数据集）"""
    ds = TUAREDFDataset(
        edf_path,
        window_sec=window_sec,
        stride_sec=stride_sec,
        max_files=max_files,
    )
    # Stage1 是无监督续训，四个槽位都用同一个数据集，curriculum 权重会均匀分配
    return [ds, None, None, None]


def _build_h5_datasets(data_cfg, window_sec, stride_sec, preprocessor):
    """构建 H5 格式数据集列表"""
    def _make(cls, path_key):
        path = data_cfg.get(path_key)
        if path and os.path.exists(path):
            return cls(path, window_sec=window_sec, stride_sec=stride_sec,
                       preprocessor=preprocessor)
        return None

    return [
        _make(TUABDataset, 'tuab_path'),
        _make(TUSZDataset, 'tusz_path'),
        _make(TUEVDataset, 'tuev_path'),
        _make(TUEPDataset, 'tuep_path'),
    ]


def main():
    args = parse_args()
    set_seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.batch_size:
        cfg['training']['batch_size'] = args.batch_size
    if args.epochs:
        cfg['training']['total_epochs'] = args.epochs
    if args.lr:
        cfg['training']['peak_lr'] = args.lr
    if args.pretrained_path:
        cfg['model']['pretrained_path'] = args.pretrained_path

    data_cfg = cfg.get('data', {})
    train_cfg = {
        **cfg.get('training', {}),
        **cfg.get('masking', {}),
        **cfg.get('curriculum', {}),
        'num_workers': data_cfg.get('num_workers', 4),
        'sample_rate': data_cfg.get('sample_rate', 200),
        'patch_size': cfg['model'].get('patch_window', 200),
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
        backbone_size=cfg['model'].get('backbone_size', 'base'),
        pretrained_path=cfg['model'].get('pretrained_path'),
        vqnsp_path=cfg['model'].get('vqnsp_path'),
        labram_root=args.labram_root or cfg['model'].get('labram_root'),
        n_prompt=cfg.get('model', {}).get('n_prompt_tokens', 10),
    ).to(device)

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    window_sec = data_cfg.get('window_sec', 10.0)
    stride_sec = data_cfg.get('stride_sec', 5.0)

    if args.data_format == 'edf':
        edf_path = args.edf_path or data_cfg.get('edf_path')
        assert edf_path and os.path.exists(edf_path), \
            f"EDF路径不存在: {edf_path}，请用 --edf_path 指定"
        train_datasets = _build_edf_datasets(edf_path, window_sec, stride_sec, args.max_files)
    else:
        preprocessor = EEGPreprocessor(target_fs=data_cfg.get('sample_rate', 200))
        train_datasets = _build_h5_datasets(data_cfg, window_sec, stride_sec, preprocessor)

    os.makedirs(args.output_dir, exist_ok=True)

    trainer = Stage1Trainer(
        model=model,
        train_datasets=train_datasets,
        val_dataset=None,
        config=train_cfg,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
        local_rank=local_rank,
    )
    trainer.train(resume_ckpt=args.resume)


if __name__ == '__main__':
    main()
