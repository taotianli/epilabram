#!/usr/bin/env python3
"""
Stage1 训练启动器（带 faulthandler + 完整日志）
"""
import sys, os, traceback, faulthandler

faulthandler.enable(file=open('/tmp/stage1_fault.log', 'w'), all_threads=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if 'LaBraM' not in p]
sys.path.insert(0, _HERE)

import yaml, torch
from models.epilabram import build_epilabram
from data.tuar_edf_dataset import TUAREDFDataset
from training.stage1_trainer import Stage1Trainer
from utils.seed import set_seed

LOG = '/tmp/stage1_run.log'
def log(msg):
    print(msg, flush=True)
    with open(LOG, 'a') as f:
        f.write(msg + '\n')

try:
    set_seed(42)

    log('[1] config')
    with open('configs/stage1_dacp.yaml') as f:
        cfg = yaml.safe_load(f)
    train_cfg = {
        **cfg.get('training', {}),
        **cfg.get('masking', {}),
        **cfg.get('curriculum', {}),
        'num_workers': 0, 'sample_rate': 200, 'patch_size': 200,
    }
    for k in ('peak_lr', 'min_lr', 'weight_decay', 'ema_decay', 'gradient_clip'):
        if k in train_cfg: train_cfg[k] = float(train_cfg[k])
    for k in ('warmup_epochs', 'total_epochs', 'batch_size', 'accumulation_steps'):
        if k in train_cfg: train_cfg[k] = int(train_cfg[k])
    log(f'    bs={train_cfg["batch_size"]}  accum={train_cfg["accumulation_steps"]}  epochs={train_cfg["total_epochs"]}')

    log('[2] model + vqnsp tokenizer')
    device = torch.device('cuda')
    model = build_epilabram(
        backbone_size='base',
        pretrained_path=cfg['model'].get('pretrained_path', 'checkpoints/labram-base.pth'),
        vqnsp_path=cfg['model'].get('vqnsp_path', 'checkpoints/vqnsp.pth'),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log(f'    params={n_params:.1f}M')

    # 验证 codebook 不 collapse
    log('[3] codebook sanity check')
    test_eeg = torch.randn(4, 23, 10, 200, device=device)
    with torch.no_grad():
        targets = model.tokenizer.get_codebook_indices(test_eeg)
    unique = targets.unique().numel()
    log(f'    unique targets (4 samples): {unique}  (should be >> 1)')
    assert unique > 10, f'Codebook collapse detected! unique={unique}'

    log('[4] dataset')
    edf_path = cfg.get('data', {}).get('edf_path', 'data/tuar/edf')
    ds = TUAREDFDataset(
        edf_path,
        window_sec=10.0, stride_sec=5.0,
    )
    log(f'    {len(ds)} segments')

    log('[5] trainer')
    os.makedirs('experiments/stage1_full', exist_ok=True)
    trainer = Stage1Trainer(
        model=model,
        train_datasets=[ds, None, None, None],
        val_dataset=None,
        config=train_cfg,
        output_dir='experiments/stage1_full',
        local_rank=0,
    )

    log('[6] training start')
    trainer.train()
    log('DONE')

except Exception as e:
    log(f'ERROR: {e}')
    log(traceback.format_exc())
    sys.exit(1)
