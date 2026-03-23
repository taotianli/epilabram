#!/usr/bin/env python3
"""
Stage2 训练启动器（多任务微调 MTPCT）
冻结 backbone，只训练 TaskPromptTokens + PromptAdapters + 预测头
"""
import sys, os, traceback, faulthandler

faulthandler.enable(file=open('/tmp/stage2_fault.log', 'w'), all_threads=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if 'LaBraM' not in p]
sys.path.insert(0, _HERE)

import yaml, torch
from models.epilabram import build_epilabram
from data.tuar_edf_dataset import TUAREDFDataset
from training.stage2_trainer import Stage2Trainer
from utils.seed import set_seed
from utils.checkpoint import load_checkpoint

LOG = '/tmp/stage2_run.log'
def log(msg):
    print(msg, flush=True)
    with open(LOG, 'a') as f:
        f.write(msg + '\n')

try:
    set_seed(42)

    log('[1] config')
    with open('configs/stage2_mtpct.yaml') as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get('data', {})
    train_cfg = {
        **cfg.get('training', {}),
        **cfg.get('hierarchy', {}),
        'num_workers': 0,
        'patch_size': 200,
    }
    for k in ('peak_lr', 'min_lr', 'weight_decay'):
        if k in train_cfg: train_cfg[k] = float(train_cfg[k])
    for k in ('warmup_epochs', 'total_epochs', 'batch_size'):
        if k in train_cfg: train_cfg[k] = int(train_cfg[k])
    log(f'    bs={train_cfg["batch_size"]}  epochs={train_cfg["total_epochs"]}  lr={train_cfg["peak_lr"]}')

    log('[2] model')
    device = torch.device('cuda')
    model = build_epilabram(
        backbone_size='base',
        vqnsp_path=cfg['model'].get('vqnsp_path', 'checkpoints/vqnsp.pth'),
        task_mode='artifact',
        n_classes=5,
    ).to(device)

    log('[3] load Stage1 checkpoint (backbone only)')
    stage1_ckpt = 'experiments/stage1_full/stage1_epoch049.pth'
    ckpt = torch.load(stage1_ckpt, map_location='cpu', weights_only=False)
    sd = ckpt.get('model', ckpt)
    # 只加载 backbone 参数，跳过 heads/adapters/task_prompts（尺寸不匹配）
    sd_backbone = {k: v for k, v in sd.items()
                   if not any(k.startswith(p) for p in ('heads.', 'adapters.', 'task_prompts.'))}
    result = model.load_state_dict(sd_backbone, strict=False)
    log(f'    loaded {len(sd_backbone)} keys  missing={len(result.missing_keys)}')

    log('[4] dataset (TUAR EDF)')
    window_sec = float(data_cfg.get('window_sec', 10.0))
    stride_sec = 5.0
    ds = TUAREDFDataset(
        data_cfg.get('edf_path', '/home/taotl/Desktop/TUAR/v3.0.1/edf'),
        window_sec=window_sec,
        stride_sec=stride_sec,
        mode='artifact',
    )
    log(f'    {len(ds)} segments')

    log('[5] trainer')
    os.makedirs('experiments/stage2_full', exist_ok=True)
    trainer = Stage2Trainer(
        model=model,
        train_datasets=[ds, None, None, None],   # TUAB slot
        val_datasets=[None, None, None, None],
        config=train_cfg,
        output_dir='experiments/stage2_full',
        local_rank=0,
    )

    # 检查是否有可以 resume 的 checkpoint
    resume_ckpt = None
    import glob as _glob
    # 只 resume 本次训练（5类）的 checkpoint，跳过旧的 2 类 checkpoint
    ckpts = sorted(_glob.glob('experiments/stage2_full/stage2_epoch*.pth'))
    for ckpt_path in reversed(ckpts):
        try:
            c = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            sd = c.get('model', c)
            # 检查 head 尺寸是否匹配（5类）
            if sd.get('heads.TUAB.fc.weight', torch.zeros(2,1)).shape[0] == 5:
                resume_ckpt = ckpt_path
                break
        except Exception:
            pass

    if resume_ckpt:
        log(f'[6] resuming from {resume_ckpt}')
    else:
        log('[6] training start (from scratch)')

    trainer.train(resume_ckpt=resume_ckpt)
    log('DONE')

except Exception as e:
    log(f'ERROR: {e}')
    log(traceback.format_exc())
    sys.exit(1)
