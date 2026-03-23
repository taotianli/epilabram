"""
Stage2 训练器：多任务提示微调（MTPCT）
冻结 backbone，只训练 TaskPromptTokens + PromptAdapters + 预测头
"""

import os
import math
from typing import Optional, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.amp import GradScaler, autocast

from models.epilabram import EpiLaBraM
from training.losses import HierarchicalConsistencyLoss
from utils.logger import get_logger
from utils.checkpoint import save_checkpoint, load_checkpoint


class Stage2Trainer:
    """
    Stage2：多任务微调。

    - 冻结 backbone，只训练 TaskPromptTokens + PromptAdapters + 预测头
    - 每个 batch 从多个数据集均匀采样
    - HierarchicalConsistencyLoss
    - AdamW + cosine LR + warmup
    """

    def __init__(
        self,
        model: EpiLaBraM,
        train_datasets: list,   # [tuab, tusz, tuev, tuep]，None 表示不使用
        val_datasets: list,
        config: dict,
        output_dir: str,
        use_wandb: bool = False,
        local_rank: int = 0,
    ):
        self.model = model
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.cfg = config
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.local_rank = local_rank
        self.is_main = (local_rank == 0)
        self.logger = get_logger('stage2', output_dir) if self.is_main else None

        self.criterion = HierarchicalConsistencyLoss(
            lambda1=config.get('lambda1', 1.0),
            lambda2=config.get('lambda2', 1.0),
            gamma=config.get('gamma', 0.5),
            label_smoothing=config.get('label_smoothing', 0.1),
        )
        self.scaler = GradScaler('cuda')

        # 冻结 backbone
        self.model.freeze_backbone()
        self._setup_optimizer()

    def _setup_optimizer(self):
        cfg = self.cfg
        self.peak_lr = float(cfg.get('peak_lr', 1e-3))
        self.min_lr = float(cfg.get('min_lr', 1e-6))
        self.warmup_epochs = int(cfg.get('warmup_epochs', 3))
        self.total_epochs = int(cfg.get('total_epochs', 30))
        params = self.model.get_stage2_params()
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.peak_lr,
            weight_decay=float(cfg.get('weight_decay', 0.05)),
        )

    def _get_lr(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.peak_lr * (epoch + 1) / self.warmup_epochs
        progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
        return self.min_lr + (self.peak_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def _build_loaders(self) -> Dict[str, DataLoader]:
        """为每个任务单独建 DataLoader，训练时轮流采样"""
        task_names = ['TUAB', 'TUSZ', 'TUEV', 'TUEP']
        loaders = {}
        bs = self.cfg.get('batch_size', 256) // 4  # 每任务 batch size
        for name, ds in zip(task_names, self.train_datasets):
            if ds is not None:
                loaders[name] = DataLoader(
                    ds,
                    batch_size=max(bs, 1),
                    shuffle=True,
                    num_workers=self.cfg.get('num_workers', 8),
                    pin_memory=True,
                    drop_last=True,
                )
        return loaders

    def _train_epoch(self, loaders: Dict[str, DataLoader], epoch: int) -> Dict[str, float]:
        self.model.train()
        device = next(self.model.parameters()).device
        task_id_map = {'TUAB': 0, 'TUSZ': 1, 'TUEV': 2, 'TUEP': 3}

        # 将所有 loader 转为 iterator
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
                tid = task_id_map[task_name]

                B, C, T = eeg.shape
                patch_size = self.cfg.get('patch_size', 200)
                A = T // patch_size
                eeg = eeg[:, :, :A * patch_size].reshape(B, C, A, patch_size)
                task_ids = torch.full((B,), tid, dtype=torch.long, device=device)

                with autocast('cuda'):
                    results = self.model.forward_stage2(eeg, task_ids)
                    _, logits = results[task_name]

                    if task_name == 'TUEV':
                        loss, log = self.criterion(logits, None, None, label)
                    else:
                        loss, log = self.criterion(logits, None, None, label)

                total_loss = total_loss + loss

                for k, v in log.items():
                    key = f'{task_name}/{k}'
                    metrics[key] = metrics.get(key, 0.0) + v.item()
                    counts[key] = counts.get(key, 0) + 1

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.get_stage2_params(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return {k: v / max(counts[k], 1) for k, v in metrics.items()}

    def train(self, resume_ckpt: Optional[str] = None):
        start_epoch = 0
        if resume_ckpt:
            start_epoch = load_checkpoint(resume_ckpt, self.model, self.optimizer, self.scaler)

        if self.use_wandb and self.is_main:
            import wandb
            wandb.init(project='epilabram', name='stage2', config=self.cfg)

        loaders = self._build_loaders()

        for epoch in range(start_epoch, self.total_epochs):
            lr = self._get_lr(epoch)
            self._set_lr(lr)

            metrics = self._train_epoch(loaders, epoch)
            metrics['lr'] = lr

            if self.is_main:
                self.logger.info(f"Epoch {epoch:03d} | " +
                                 " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
                if self.use_wandb:
                    import wandb
                    wandb.log({'epoch': epoch, **metrics})

                if (epoch + 1) % 5 == 0 or epoch == self.total_epochs - 1:
                    save_checkpoint(
                        os.path.join(self.output_dir, f'stage2_epoch{epoch:03d}.pth'),
                        self.model, self.optimizer, self.scaler, epoch,
                    )
                    save_checkpoint(
                        os.path.join(self.output_dir, 'stage2_best.pth'),
                        self.model, self.optimizer, self.scaler, epoch,
                    )

        if self.use_wandb and self.is_main:
            import wandb
            wandb.finish()
