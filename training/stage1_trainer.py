"""
Stage1 训练器：基于 PathologyAwareDynamicMasking 的续训
"""

import os
import math
import copy
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast

from models.epilabram import EpiLaBraM
from training.masking import PathologyAwareDynamicMasking
from training.losses import MaskedEEGModelingLoss
from data.curriculum import CurriculumScheduler
from utils.logger import get_logger
from utils.checkpoint import save_checkpoint, load_checkpoint


class Stage1Trainer:
    """
    Stage1：PADM 续训。

    - PathologyAwareDynamicMasking 生成 mask
    - CurriculumScheduler 调度多数据集采样权重
    - 梯度累积（accumulation_steps=4）
    - AdamW + cosine LR schedule + warmup
    - EMA 权重更新（decay=0.996）
    - 支持 DDP 多 GPU
    """

    def __init__(
        self,
        model: EpiLaBraM,
        train_datasets: list,
        val_dataset,
        config: dict,
        output_dir: str,
        use_wandb: bool = False,
        local_rank: int = 0,
    ):
        self.model = model
        self.train_datasets = train_datasets   # [tuab, tusz, tuev, tuep]
        self.val_dataset = val_dataset
        self.cfg = config
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.local_rank = local_rank
        self.is_main = (local_rank == 0)
        self.logger = get_logger('stage1', output_dir) if self.is_main else None

        self.masker = PathologyAwareDynamicMasking(
            sample_rate=config.get('sample_rate', 200.0),
            base_mask_ratio=config.get('base_mask_ratio', 0.5),
            alpha=config.get('alpha', 0.3),
            beta=config.get('beta', 5.0),
        )
        self.criterion = MaskedEEGModelingLoss()
        self.curriculum = CurriculumScheduler(
            initial_weights=config.get('initial_weights', [0.6, 0.2, 0.1, 0.1]),
            mid_weights=config.get('mid_weights', [0.3, 0.3, 0.2, 0.2]),
            final_weights=config.get('final_weights', [0.25, 0.25, 0.25, 0.25]),
            stage1_end_epoch=config.get('stage1_end_epoch', 10),
            stage2_end_epoch=config.get('stage2_end_epoch', 30),
        )

        self.scaler = GradScaler('cuda')
        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        self._setup_optimizer()

    # ------------------------------------------------------------------
    # 优化器 & LR schedule
    # ------------------------------------------------------------------

    def _setup_optimizer(self):
        cfg = self.cfg
        self.peak_lr = float(cfg.get('peak_lr', 5e-4))
        self.min_lr = float(cfg.get('min_lr', 1e-5))
        self.warmup_epochs = int(cfg.get('warmup_epochs', 5))
        self.total_epochs = int(cfg.get('total_epochs', 50))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.peak_lr,
            weight_decay=float(cfg.get('weight_decay', 0.05)),
            betas=(0.9, 0.95),
        )

    def _get_lr(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.peak_lr * (epoch + 1) / self.warmup_epochs
        progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine

    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    # ------------------------------------------------------------------
    # EMA 更新
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_ema(self, decay: float = 0.996):
        for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)

    # ------------------------------------------------------------------
    # DataLoader（按课程权重采样）
    # ------------------------------------------------------------------

    def _build_loader(self, epoch: int) -> DataLoader:
        weights = self.curriculum.get_sampling_weights(epoch, self.total_epochs)
        sample_weights = []
        for ds, w in zip(self.train_datasets, weights):
            if ds is not None:
                sample_weights.extend([w / len(ds)] * len(ds))

        from torch.utils.data import ConcatDataset
        active_ds = [ds for ds in self.train_datasets if ds is not None]
        combined = ConcatDataset(active_ds)

        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float),
            num_samples=len(combined),
            replacement=True,
        )
        return DataLoader(
            combined,
            batch_size=self.cfg.get('batch_size', 512),
            sampler=sampler,
            num_workers=self.cfg.get('num_workers', 4),
            pin_memory=True,
            drop_last=True,
        )

    # ------------------------------------------------------------------
    # 单 epoch 训练
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        accum_steps = self.cfg.get('accumulation_steps', 4)
        grad_clip = self.cfg.get('gradient_clip', 3.0)
        device = next(self.model.parameters()).device

        total_loss = 0.0
        total_acc = 0.0
        n_steps = 0

        self.optimizer.zero_grad()

        for step, batch in enumerate(loader):
            eeg = batch[0].to(device, non_blocking=True)  # (B, C, T)

            B, C, T = eeg.shape
            patch_size = self.cfg.get('patch_size', 200)
            A = T // patch_size
            eeg = eeg[:, :, :A * patch_size].reshape(B, C, A, patch_size)

            # 批量生成 mask（向量化，无逐样本循环）
            eeg_3d = eeg.reshape(B, C, A * patch_size).cpu()
            mask, sym_mask = self.masker(eeg_3d, patch_size)   # (B, C*A)
            mask     = mask.to(device)
            sym_mask = sym_mask.to(device)

            # 获取 tokenizer 目标
            with torch.no_grad():
                targets = self.model.tokenizer.get_codebook_indices(eeg)  # (B, C*A)

            with autocast('cuda'):
                logits, sym_logits = self.model.forward_stage1(eeg, mask, sym_mask)
                loss, log = self.criterion(logits, sym_logits, targets, mask, sym_mask)
                loss = loss / accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self._update_ema(self.cfg.get('ema_decay', 0.996))

            total_loss += log['loss/total'].item()
            total_acc += log['metric/mask_acc'].item()
            n_steps += 1

        return {
            'train/loss': total_loss / max(n_steps, 1),
            'train/mask_acc': total_acc / max(n_steps, 1),
        }

    # ------------------------------------------------------------------
    # 主训练循环
    # ------------------------------------------------------------------

    def train(self, resume_ckpt: Optional[str] = None):
        start_epoch = 0
        if resume_ckpt:
            start_epoch = load_checkpoint(resume_ckpt, self.model, self.optimizer, self.scaler)

        if self.use_wandb and self.is_main:
            import wandb
            wandb.init(project='epilabram', name='stage1', config=self.cfg)

        for epoch in range(start_epoch, self.total_epochs):
            lr = self._get_lr(epoch)
            self._set_lr(lr)

            loader = self._build_loader(epoch)
            metrics = self._train_epoch(loader, epoch)
            metrics['lr'] = lr

            if self.is_main:
                self.logger.info(f"Epoch {epoch:03d} | " +
                                 " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
                if self.use_wandb:
                    import wandb
                    wandb.log({'epoch': epoch, **metrics})

                if (epoch + 1) % 10 == 0 or epoch == self.total_epochs - 1:
                    save_checkpoint(
                        os.path.join(self.output_dir, f'stage1_epoch{epoch:03d}.pth'),
                        self.model, self.optimizer, self.scaler, epoch,
                    )
                    save_checkpoint(
                        os.path.join(self.output_dir, 'stage1_ema_latest.pth'),
                        self.ema_model, None, None, epoch,
                    )

        if self.use_wandb and self.is_main:
            import wandb
            wandb.finish()
