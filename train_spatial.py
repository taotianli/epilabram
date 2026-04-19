"""
train_spatial.py — Spatial-Aware EEG 训练脚本

在原有 EpiLaBraM Stage2 基础上，用 SpatialAwareLaBraM 替换 backbone，
加入电极坐标编码 + 图卷积空间聚合，其余训练逻辑保持不变。

用法：
    python train_spatial.py \
        --config configs/stage2_mtpct.yaml \
        --stage1_ckpt experiments/stage1/best.pth \
        --output_dir experiments/spatial \
        --use_gcn \
        --gcn_k 5

Ablation：
    --no_gcn   只用坐标编码，不用图卷积
"""

import argparse
import os
import sys
import math
import yaml
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if 'LaBraM' not in p]
sys.path.insert(0, _HERE)

from models.spatial_aware import build_spatial_aware_backbone, SpatialAwareLaBraM
from models.neural_tokenizer import NeuralTokenizer
from models.task_prompt import TaskPromptTokens, PromptAdapter
from models.prediction_heads import (
    BinaryClassificationHead, MultiClassificationHead,
)
from models.epilabram import EpiLaBraM
from data.tuh_dataset import TUABDataset, TUSZDataset, TUEVDataset, TUEPDataset
from data.preprocessing import EEGPreprocessor
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import get_logger
from evaluation.metrics import MetricTracker


# ---------------------------------------------------------------------------
# 构建 spatial-aware EpiLaBraM
# ---------------------------------------------------------------------------

def build_spatial_epilabram(
    backbone_size: str = 'base',
    pretrained_path: Optional[str] = None,
    use_gcn: bool = True,
    gcn_k: int = 5,
    n_prompt: int = 10,
    adapter_bottleneck_ratio: int = 4,
    n_embed: int = 8192,
    codebook_dim: int = 64,
) -> EpiLaBraM:
    """
    与 build_epilabram 等价，但 backbone 换成 SpatialAwareLaBraM。
    EpiLaBraM 其余组件（tokenizer / task_prompts / heads / adapters）完全不变。
    """
    spatial_backbone = build_spatial_aware_backbone(
        backbone_size=backbone_size,
        pretrained_path=pretrained_path,
        use_gcn=use_gcn,
        gcn_k=gcn_k,
    )

    # EpiLaBraM 需要一个真正的 LaBraMBackbone 来初始化 NeuralTokenizer，
    # 这里直接用 spatial_backbone 内部的 backbone 即可
    inner_backbone = spatial_backbone.backbone
    tokenizer = NeuralTokenizer(inner_backbone, n_embed=n_embed, embed_dim=codebook_dim)

    embed_dim = spatial_backbone.embed_dim
    task_prompts = TaskPromptTokens(n_tasks=4, n_prompt=n_prompt, embed_dim=embed_dim)

    heads = nn.ModuleDict({
        'TUAB': BinaryClassificationHead(embed_dim),
        'TUSZ': BinaryClassificationHead(embed_dim),
        'TUEV': MultiClassificationHead(embed_dim, n_classes=6),
        'TUEP': BinaryClassificationHead(embed_dim),
    })

    # 把 spatial_backbone 作为 backbone 传入 EpiLaBraM
    # EpiLaBraM 会访问 backbone.embed_dim / backbone.blocks / backbone.use_rope 等属性
    # SpatialAwareLaBraM 已通过 @property 代理了这些属性
    model = EpiLaBraM(
        backbone=spatial_backbone,
        tokenizer=tokenizer,
        task_prompts=task_prompts,
        heads=heads,
        n_prompt=n_prompt,
        adapter_bottleneck_ratio=adapter_bottleneck_ratio,
    )
    return model


# ---------------------------------------------------------------------------
# 验证函数
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: EpiLaBraM, val_datasets: list, device: torch.device,
             patch_size: int = 200, batch_size: int = 64, num_workers: int = 4) -> Dict:
    model.eval()
    task_names = ['TUAB', 'TUSZ', 'TUEV', 'TUEP']
    task_id_map = {'TUAB': 0, 'TUSZ': 1, 'TUEV': 2, 'TUEP': 3}
    results = {}

    for name, ds in zip(task_names, val_datasets):
        if ds is None:
            continue
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        all_logits, all_labels = [], []
        tid = task_id_map[name]

        for batch in loader:
            eeg = batch[0].to(device, non_blocking=True)
            label = batch[1]
            B, C, T = eeg.shape
            A = T // patch_size
            eeg = eeg[:, :, :A * patch_size].reshape(B, C, A, patch_size)
            task_ids = torch.full((B,), tid, dtype=torch.long, device=device)

            with autocast('cuda'):
                res = model.forward_stage2(eeg, task_ids)
            _, logits = res[name]
            all_logits.append(logits.float().cpu())
            all_labels.append(label)

        if len(all_logits) == 0:
            continue
        logits_cat = torch.cat(all_logits)
        labels_cat = torch.cat(all_labels)
        probs = torch.softmax(logits_cat, dim=-1).numpy()
        preds = logits_cat.argmax(dim=-1).numpy()
        labels_np = labels_cat.numpy()

        tracker = MetricTracker()
        tracker.update(name, preds, labels_np, probs)
        results[name] = tracker.compute(name)

    return results


# ---------------------------------------------------------------------------
# 训练器
# ---------------------------------------------------------------------------

class SpatialTrainer:
    def __init__(self, model: EpiLaBraM, train_datasets: list, val_datasets: list,
                 config: dict, output_dir: str, use_wandb: bool = False,
                 finetune: bool = False):
        self.model = model
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.cfg = config
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.finetune = finetune
        self.logger = get_logger('spatial', output_dir)

        self.label_smoothing = config.get('label_smoothing', 0.1)
        self.scaler = GradScaler('cuda')

        if finetune:
            # 全参数微调：backbone + spatial + prompts + adapters + heads 全部训练
            self.model.unfreeze_backbone()
        else:
            # 参数高效微调：冻结 backbone，只训练新增模块
            self.model.freeze_backbone()
        # 空间模块始终可训练（它们在 wrapper 层，不在 inner backbone 里）
        spatial_bb = model.backbone  # SpatialAwareLaBraM
        for p in spatial_bb.coord_embed.parameters():
            p.requires_grad_(True)
        if spatial_bb.gcn is not None:
            for p in spatial_bb.gcn.parameters():
                p.requires_grad_(True)

        self._setup_optimizer()

    def _get_trainable_params(self):
        if self.finetune:
            return [p for p in self.model.parameters() if p.requires_grad]
        spatial_bb = self.model.backbone
        params = (
            list(spatial_bb.coord_embed.parameters())
            + (list(spatial_bb.gcn.parameters()) if spatial_bb.gcn is not None else [])
            + self.model.get_stage2_params()
        )
        return params

    def _setup_optimizer(self):
        cfg = self.cfg
        self.peak_lr = float(cfg.get('peak_lr', 1e-3))
        self.min_lr = float(cfg.get('min_lr', 1e-6))
        self.warmup_epochs = int(cfg.get('warmup_epochs', 3))
        self.total_epochs = int(cfg.get('total_epochs', 30))
        self.optimizer = torch.optim.AdamW(
            self._get_trainable_params(),
            lr=self.peak_lr,
            weight_decay=float(cfg.get('weight_decay', 0.05)),
        )

    def _get_lr(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.peak_lr * (epoch + 1) / self.warmup_epochs
        progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
        return self.min_lr + (self.peak_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    def _build_loaders(self) -> Dict[str, DataLoader]:
        task_names = ['TUAB', 'TUSZ', 'TUEV', 'TUEP']
        bs = max(self.cfg.get('batch_size', 256) // 4, 1)
        loaders = {}
        for name, ds in zip(task_names, self.train_datasets):
            if ds is not None:
                loaders[name] = DataLoader(
                    ds, batch_size=bs, shuffle=True,
                    num_workers=self.cfg.get('num_workers', 4),
                    pin_memory=True, drop_last=True,
                )
        return loaders

    def _train_epoch(self, loaders: Dict[str, DataLoader], epoch: int) -> Dict[str, float]:
        self.model.train()
        device = next(self.model.parameters()).device
        task_id_map = {'TUAB': 0, 'TUSZ': 1, 'TUEV': 2, 'TUEP': 3}
        patch_size = self.cfg.get('patch_size', 200)

        iters = {name: iter(loader) for name, loader in loaders.items()}
        n_steps = max(len(l) for l in loaders.values())
        metrics: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        self.optimizer.zero_grad()

        for step in range(n_steps):
            total_loss = torch.tensor(0.0, device=device)

            for task_name, it in iters.items():
                try:
                    batch = next(it)
                except StopIteration:
                    iters[task_name] = iter(loaders[task_name])
                    batch = next(iters[task_name])

                eeg = batch[0].to(device, non_blocking=True)
                label = batch[1].to(device, non_blocking=True)
                B, C, T = eeg.shape
                A = T // patch_size
                eeg = eeg[:, :, :A * patch_size].reshape(B, C, A, patch_size)
                task_ids = torch.full((B,), task_id_map[task_name], dtype=torch.long, device=device)

                with autocast('cuda'):
                    res = self.model.forward_stage2(eeg, task_ids)
                    _, logits = res[task_name]
                    loss = F.cross_entropy(logits, label,
                                           label_smoothing=self.label_smoothing)

                total_loss = total_loss + loss
                key = f'{task_name}/loss'
                metrics[key] = metrics.get(key, 0.0) + loss.item()
                counts[key] = counts.get(key, 0) + 1

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self._get_trainable_params(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return {k: v / max(counts[k], 1) for k, v in metrics.items()}

    def train(self, resume_ckpt: Optional[str] = None):
        device = next(self.model.parameters()).device
        start_epoch = 0
        if resume_ckpt:
            start_epoch = load_checkpoint(resume_ckpt, self.model, self.optimizer, self.scaler)

        if self.use_wandb:
            import wandb
            wandb.init(project='epilabram', name='spatial', config=self.cfg)

        loaders = self._build_loaders()
        best_metric = 0.0

        for epoch in range(start_epoch, self.total_epochs):
            lr = self._get_lr(epoch)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr

            train_metrics = self._train_epoch(loaders, epoch)
            train_metrics['lr'] = lr
            self.logger.info(
                f"Epoch {epoch:03d} | " +
                " | ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
            )

            # 每 5 epoch 验证一次
            if (epoch + 1) % 5 == 0 or epoch == self.total_epochs - 1:
                val_results = evaluate(
                    self.model, self.val_datasets, device,
                    patch_size=self.cfg.get('patch_size', 200),
                    batch_size=self.cfg.get('batch_size', 256),
                    num_workers=self.cfg.get('num_workers', 4),
                )
                for task, m in val_results.items():
                    self.logger.info(
                        f"  Val {task}: " + " | ".join(f"{k}={v:.4f}" for k, v in m.items())
                    )

                # 用所有任务 bal_acc 均值作为 best 指标
                avg_bal_acc = sum(
                    m.get('bal_acc', 0.0) for m in val_results.values()
                ) / max(len(val_results), 1)

                save_checkpoint(
                    os.path.join(self.output_dir, f'spatial_epoch{epoch:03d}.pth'),
                    self.model, self.optimizer, self.scaler, epoch,
                )
                if avg_bal_acc > best_metric:
                    best_metric = avg_bal_acc
                    save_checkpoint(
                        os.path.join(self.output_dir, 'spatial_best.pth'),
                        self.model, self.optimizer, self.scaler, epoch,
                    )
                    self.logger.info(f"  New best avg_bal_acc={best_metric:.4f}")

                if self.use_wandb:
                    import wandb
                    flat = {f'val/{t}/{k}': v for t, m in val_results.items() for k, v in m.items()}
                    wandb.log({'epoch': epoch, **train_metrics, **flat})

        if self.use_wandb:
            import wandb
            wandb.finish()


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Spatial-Aware EEG Training')
    parser.add_argument('--config', type=str, default='configs/stage2_mtpct.yaml')
    parser.add_argument('--output_dir', type=str, default='experiments/spatial')
    parser.add_argument('--stage1_ckpt', type=str, required=True,
                        help='Stage1 预训练权重路径（LaBraM backbone）')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--peak_lr', type=float, default=None)
    # 空间模块选项
    parser.add_argument('--use_gcn', action='store_true', default=True,
                        help='使用图卷积空间聚合（默认开启）')
    parser.add_argument('--no_gcn', dest='use_gcn', action='store_false',
                        help='只用坐标编码，不用图卷积（ablation）')
    parser.add_argument('--gcn_k', type=int, default=5,
                        help='图卷积 kNN 邻居数')
    parser.add_argument('--finetune', action='store_true', default=False,
                        help='解冻 backbone 全参数微调（默认冻结 backbone）')
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
    if args.peak_lr:
        cfg['training']['peak_lr'] = args.peak_lr

    data_cfg = cfg.get('data', {})
    train_cfg = {
        **cfg.get('training', {}),
        **cfg.get('hierarchy', {}),
        'num_workers': data_cfg.get('num_workers', 4),
        'patch_size': 200,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_spatial_epilabram(
        backbone_size=cfg.get('model', {}).get('backbone_size', 'base'),
        pretrained_path=args.stage1_ckpt,
        use_gcn=args.use_gcn,
        gcn_k=args.gcn_k,
        n_prompt=cfg.get('model', {}).get('n_prompt_tokens', 10),
        adapter_bottleneck_ratio=cfg.get('model', {}).get('adapter_bottleneck_ratio', 4),
    ).to(device)

    n_spatial = sum(p.numel() for p in model.backbone.coord_embed.parameters())
    n_gcn = sum(p.numel() for p in model.backbone.gcn.parameters()) if model.backbone.gcn else 0
    print(f"[spatial] coord_embed params: {n_spatial:,}  gcn params: {n_gcn:,}")
    print(f"[spatial] use_gcn={args.use_gcn}  gcn_k={args.gcn_k}  finetune={args.finetune}")

    preprocessor = EEGPreprocessor(target_fs=data_cfg.get('sample_rate', 200))
    window_sec = data_cfg.get('window_sec', 10.0)
    stride_sec = data_cfg.get('stride_sec', 5.0)

    def _make(cls, key, split='train'):
        path = data_cfg.get(key)
        if path and os.path.exists(path):
            return cls(path, window_sec=window_sec, stride_sec=stride_sec,
                       preprocessor=preprocessor, split=split)
        return None

    train_datasets = [
        _make(TUABDataset, 'tuab_path'),
        _make(TUSZDataset, 'tusz_path'),
        _make(TUEVDataset, 'tuev_path'),
        _make(TUEPDataset, 'tuep_path'),
    ]
    val_datasets = [
        _make(TUABDataset, 'tuab_path', 'eval'),
        _make(TUSZDataset, 'tusz_path', 'dev'),
        _make(TUEVDataset, 'tuev_path', 'eval'),
        _make(TUEPDataset, 'tuep_path', 'eval'),
    ]

    os.makedirs(args.output_dir, exist_ok=True)

    trainer = SpatialTrainer(
        model=model,
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        config=train_cfg,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
        finetune=args.finetune,
    )
    trainer.train(resume_ckpt=args.resume)


if __name__ == '__main__':
    main()
