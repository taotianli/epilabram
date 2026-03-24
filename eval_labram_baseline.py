#!/usr/bin/env python3
"""
LaBraM 基准评测：在 TUAB / TUSZ / TUEV / TUEP 上做线性探针评估

用法：
  # 评测所有任务
  python eval_labram_baseline.py --ckpt checkpoints/labram-base.pth

  # 只评测某个任务
  python eval_labram_baseline.py --ckpt checkpoints/labram-base.pth --tasks TUAB TUSZ

  # 全量微调（解冻 backbone）
  python eval_labram_baseline.py --ckpt checkpoints/labram-base.pth --finetune
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.labram_backbone import LaBraMBackbone
from data.tuh_dataset import TUABDataset, TUSZDataset, TUEVDataset, TUEPDataset

# ============================================================
# 配置
# ============================================================
DEFAULT_DATA_PATHS = {
    'TUAB': '/projects/u6da/tuh_processed/tuab',
    'TUSZ': '/projects/u6da/tuh_processed/tusz',
    'TUEV': '/projects/u6da/tuh_processed/tuev',
    'TUEP': '/projects/u6da/tuh_processed/tuep',
}

TASK_CONFIGS = {
    'TUAB': {'n_classes': 2,  'ds_cls': TUABDataset,  'type': 'binary'},
    'TUSZ': {'n_classes': 2,  'ds_cls': TUSZDataset,  'type': 'binary'},
    'TUEV': {'n_classes': 6,  'ds_cls': TUEVDataset,  'type': 'multiclass'},
    'TUEP': {'n_classes': 2,  'ds_cls': TUEPDataset,  'type': 'binary'},
}

TUEV_CLASS_NAMES = ['spsw', 'gped', 'pled', 'eyem', 'artf', 'bckg']

# ============================================================
# 模型：LaBraM backbone + 线性分类头
# ============================================================
class LaBraMClassifier(nn.Module):
    def __init__(self, backbone: LaBraMBackbone, n_classes: int, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, n_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 23, T) → 自动 reshape 成 (B, 23, A, 200)
        """
        if x.ndim == 3:
            B, N, T = x.shape
            A = T // 200
            x = x.reshape(B, N, A, 200)
        feats = self.backbone(x)   # (B, embed_dim)
        return self.head(feats)    # (B, n_classes)


# ============================================================
# 数据加载工具
# ============================================================
def get_dataloader(task: str, split: str, batch_size: int, num_workers: int,
                   data_path: str) -> DataLoader:
    cfg = TASK_CONFIGS[task]
    ds = cfg['ds_cls'](data_path, split=split, window_sec=10.0, stride_sec=10.0)
    shuffle = (split == 'train')
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),  # worker 进程跨 epoch 复用，避免重启开销
        prefetch_factor=4 if num_workers > 0 else None,  # 提前预取 batch，掩盖 IO 延迟
    )


def extract_eeg_label(batch, task: str):
    """统一不同数据集的返回格式，只取 eeg 和 label"""
    eeg = batch[0]
    label = batch[1]
    return eeg, label


# ============================================================
# 训练一个 epoch
# ============================================================
def train_epoch(model, loader, optimizer, criterion, device, task, use_bf16):
    model.train()
    total_loss = 0.0
    n = 0
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    for batch in tqdm(loader, desc='  train', leave=False, unit='batch'):
        eeg, label = extract_eeg_label(batch, task)
        eeg   = eeg.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=dtype, enabled=device.type == 'cuda'):
            logits = model(eeg)
            loss   = criterion(logits, label)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * eeg.size(0)
        n += eeg.size(0)

    return total_loss / max(n, 1)


# ============================================================
# 推理 + 指标计算
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, task, n_classes, use_bf16=True):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    for batch in tqdm(loader, desc='  eval', leave=False, unit='batch'):
        eeg, label = extract_eeg_label(batch, task)
        eeg = eeg.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=dtype, enabled=device.type == 'cuda'):
            logits = model(eeg)

        probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
        preds = probs.argmax(axis=-1)

        all_preds.append(preds)
        all_labels.append(label.numpy())
        all_probs.append(probs)

    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    probs  = np.concatenate(all_probs)

    metrics = {}
    metrics['acc']      = float((preds == labels).mean())
    metrics['bal_acc']  = float(balanced_accuracy_score(labels, preds))
    metrics['macro_f1'] = float(f1_score(labels, preds, average='macro',    zero_division=0))
    metrics['w_f1']     = float(f1_score(labels, preds, average='weighted', zero_division=0))

    if n_classes == 2:
        try:
            metrics['auroc'] = float(roc_auc_score(labels, probs[:, 1]))
        except Exception:
            metrics['auroc'] = float('nan')
    else:
        try:
            metrics['auroc'] = float(roc_auc_score(
                labels, probs, multi_class='ovr', average='macro'))
        except Exception:
            metrics['auroc'] = float('nan')

    return metrics, preds, labels


def print_task_results(task, metrics, preds, labels, n_classes):
    print(f"\n{'='*60}")
    print(f"  {task} Results")
    print(f"{'='*60}")
    print(f"  Accuracy:          {metrics['acc']:.4f}")
    print(f"  Balanced Accuracy: {metrics['bal_acc']:.4f}")
    print(f"  Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:       {metrics['w_f1']:.4f}")
    print(f"  AUROC:             {metrics['auroc']:.4f}")

    names = TUEV_CLASS_NAMES if task == 'TUEV' else None
    print(f"\n  Per-class Report:")
    print(classification_report(labels, preds, target_names=names,
                                zero_division=0, digits=4))

    cm = confusion_matrix(labels, preds)
    print(f"  Confusion Matrix:")
    print(cm)


# ============================================================
# 单任务评测流程
# ============================================================
def run_task(task: str, args, device: torch.device) -> dict:
    cfg     = TASK_CONFIGS[task]
    n_cls   = cfg['n_classes']
    path    = args.data_paths.get(task, DEFAULT_DATA_PATHS[task])

    print(f"\n{'#'*60}")
    print(f"  Task: {task}  |  classes={n_cls}  |  mode={'finetune' if args.finetune else 'linear_probe'}")
    print(f"{'#'*60}")

    # ---- 加载 backbone ----
    backbone = LaBraMBackbone(size=args.backbone_size)
    backbone.load_pretrained(args.ckpt)

    model = LaBraMClassifier(backbone, n_cls, freeze_backbone=not args.finetune).to(device)

    # torch.compile 加速（PyTorch 2.x + H200）
    if args.compile:
        print("  [compile] torch.compile 编译模型中...")
        model = torch.compile(model, mode='max-autotune')

    # ---- 数据 ----
    train_loader = get_dataloader(task, 'train', args.batch_size, args.num_workers, path)
    eval_loader  = get_dataloader(task, 'eval',  args.batch_size, args.num_workers, path)
    print(f"  train={len(train_loader.dataset)}  eval={len(eval_loader.dataset)}")

    # ---- 训练 ----
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_bal_acc = 0.0
    best_metrics = {}
    best_preds = best_labels = None

    for ep in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device, task, args.bf16)
        metrics, preds, labels = evaluate(model, eval_loader, device, task, n_cls, args.bf16)
        scheduler.step()

        print(f"  Epoch {ep:2d}/{args.epochs}  loss={loss:.4f}  "
              f"bal_acc={metrics['bal_acc']:.4f}  auroc={metrics['auroc']:.4f}")

        if metrics['bal_acc'] > best_bal_acc:
            best_bal_acc  = metrics['bal_acc']
            best_metrics  = metrics
            best_preds    = preds
            best_labels   = labels

    print_task_results(task, best_metrics, best_preds, best_labels, n_cls)
    return best_metrics


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='LaBraM 基准线性探针评测')
    parser.add_argument('--ckpt',          type=str, required=True,
                        help='LaBraM checkpoint 路径（如 checkpoints/labram-base.pth）')
    parser.add_argument('--backbone_size', type=str, default='base',
                        choices=['base', 'large', 'huge'])
    parser.add_argument('--tasks',         nargs='+', default=['TUAB', 'TUSZ', 'TUEV', 'TUEP'])
    parser.add_argument('--epochs',        type=int, default=10,
                        help='线性探针训练轮数（默认10）')
    parser.add_argument('--batch_size',    type=int, default=256)
    parser.add_argument('--lr',            type=float, default=1e-3)
    parser.add_argument('--num_workers',   type=int, default=8)
    parser.add_argument('--finetune',      action='store_true',
                        help='全量微调（默认为 freeze backbone 线性探针）')
    parser.add_argument('--bf16',          action='store_true', default=True,
                        help='使用 BF16 精度（H200 推荐，默认开启）')
    parser.add_argument('--no_bf16',       dest='bf16', action='store_false')
    parser.add_argument('--compile',       action='store_true', default=False,
                        help='torch.compile 加速（首次运行较慢，后续更快）')
    parser.add_argument('--output_dir',    type=str, default='experiments/labram_baseline')
    # 可逐个覆盖数据路径
    parser.add_argument('--tuab_path', type=str, default=None)
    parser.add_argument('--tusz_path', type=str, default=None)
    parser.add_argument('--tuev_path', type=str, default=None)
    parser.add_argument('--tuep_path', type=str, default=None)
    args = parser.parse_args()

    # 整理数据路径
    args.data_paths = {
        'TUAB': args.tuab_path or DEFAULT_DATA_PATHS['TUAB'],
        'TUSZ': args.tusz_path or DEFAULT_DATA_PATHS['TUSZ'],
        'TUEV': args.tuev_path or DEFAULT_DATA_PATHS['TUEV'],
        'TUEP': args.tuep_path or DEFAULT_DATA_PATHS['TUEP'],
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Device:  {device}")
    print(f"Ckpt:    {args.ckpt}")
    print(f"Mode:    {'finetune' if args.finetune else 'linear_probe'}")
    print(f"Tasks:   {args.tasks}")
    print(f"Epochs:  {args.epochs}")

    all_results = {}
    for task in args.tasks:
        if task not in TASK_CONFIGS:
            print(f"[WARN] Unknown task: {task}, skipping")
            continue
        results = run_task(task, args, device)
        all_results[task] = results

    # ---- 汇总 ----
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    header = f"{'Task':>6}  {'Acc':>7}  {'BalAcc':>7}  {'MacroF1':>8}  {'AUROC':>7}"
    print(header)
    print('-' * len(header))
    for task, m in all_results.items():
        print(f"{task:>6}  {m['acc']:7.4f}  {m['bal_acc']:7.4f}  "
              f"{m['macro_f1']:8.4f}  {m['auroc']:7.4f}")

    # 保存结果
    out_path = os.path.join(args.output_dir, 'baseline_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == '__main__':
    main()
