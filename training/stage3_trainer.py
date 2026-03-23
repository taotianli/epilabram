"""
Stage3 训练器：临床偏好对齐 DPO（CPA-DPO）
"""

import os
import copy
import math
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

from models.epilabram import EpiLaBraM
from training.losses import CPADPOLoss
from utils.logger import get_logger
from utils.checkpoint import save_checkpoint, load_checkpoint


class PreferenceDataset(Dataset):
    """
    偏好数据集：
    - chosen:   高一致性样本（kappa > threshold）
    - rejected: 模型高置信度错误预测样本

    返回: (eeg_chosen, eeg_rejected, task_id, label)
    """

    def __init__(
        self,
        chosen_samples: list,    # list of (eeg_tensor, task_id, label)
        rejected_samples: list,  # list of (eeg_tensor, task_id, label)
    ):
        assert len(chosen_samples) == len(rejected_samples), \
            "chosen 和 rejected 样本数量必须相同"
        self.chosen = chosen_samples
        self.rejected = rejected_samples

    def __len__(self) -> int:
        return len(self.chosen)

    def __getitem__(self, idx: int) -> Tuple:
        eeg_c, tid_c, lbl_c = self.chosen[idx]
        eeg_r, tid_r, lbl_r = self.rejected[idx]
        return eeg_c, eeg_r, tid_c, lbl_c

    @staticmethod
    def build_from_base_dataset(
        base_dataset,
        model: EpiLaBraM,
        device: torch.device,
        high_kappa_threshold: float = 0.8,
        confidence_threshold: float = 0.9,
        batch_size: int = 64,
    ) -> 'PreferenceDataset':
        """
        从基础数据集自动构建偏好对。
        若高置信度样本不足，自动降低阈值至 0.5 保证能构建出偏好对。
        """
        model.eval()
        all_samples = []  # (eeg, task_id, label, conf, correct)

        loader = DataLoader(base_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        with torch.no_grad():
            for batch in loader:
                eeg = batch[0].to(device)
                task_ids = (batch[1].to(device) if len(batch) > 2
                            else torch.zeros(eeg.shape[0], dtype=torch.long, device=device))
                labels = batch[-1].to(device)

                B, C, T = eeg.shape
                A = T // 200
                eeg_r = eeg[:, :, :A * 200].reshape(B, C, A, 200)

                log_probs = model.forward_stage3_reward(eeg_r, task_ids)
                probs = log_probs.exp()
                conf, pred = probs.max(dim=-1)
                correct = (pred == labels)

                for i in range(B):
                    all_samples.append((
                        eeg[i].cpu(),
                        task_ids[i].item(),
                        labels[i].item(),
                        conf[i].item(),
                        correct[i].item(),
                    ))

        # 从高到低尝试阈值，直到能构建出至少1对
        for thresh in [confidence_threshold, 0.7, 0.6, 0.5]:
            chosen   = [(s[0], s[1], s[2]) for s in all_samples if s[4] and s[3] >= thresh]
            rejected = [(s[0], s[1], s[2]) for s in all_samples if not s[4] and s[3] >= thresh]
            n = min(len(chosen), len(rejected))
            if n > 0:
                if thresh < confidence_threshold:
                    print(f"  [PreferenceDataset] 降低置信度阈值至 {thresh:.1f}，得到 {n} 对")
                return PreferenceDataset(chosen[:n], rejected[:n])

        # 最后兜底：随机分配 chosen/rejected
        print("  [PreferenceDataset] 警告：无法按置信度区分，随机构建偏好对（仅用于调试）")
        half = len(all_samples) // 2
        chosen   = [(s[0], s[1], s[2]) for s in all_samples[:half]]
        rejected = [(s[0], s[1], s[2]) for s in all_samples[half:half*2]]
        n = min(len(chosen), len(rejected))
        return PreferenceDataset(chosen[:n], rejected[:n])


class Stage3Trainer:
    """
    Stage3：CPA-DPO 偏好对齐训练。

    - 冻结参考模型 π_ref（Stage2 训练后的模型副本）
    - 使用 CPADPOLoss 优化策略模型 π_θ
    - 超参数：peak_lr=1e-5, β_dpo=0.1, total_epochs=10
    """

    def __init__(
        self,
        model: EpiLaBraM,
        preference_dataset: PreferenceDataset,
        config: dict,
        output_dir: str,
        use_wandb: bool = False,
        local_rank: int = 0,
    ):
        self.model = model
        self.pref_dataset = preference_dataset
        self.cfg = config
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.local_rank = local_rank
        self.is_main = (local_rank == 0)
        self.logger = get_logger('stage3', output_dir) if self.is_main else None

        # 冻结参考模型
        self.ref_model = copy.deepcopy(model)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

        self.criterion = CPADPOLoss(beta=config.get('beta_dpo', 0.1))
        self.scaler = GradScaler('cuda')
        self._setup_optimizer()

    def _setup_optimizer(self):
        cfg = self.cfg
        self.peak_lr = float(cfg.get('peak_lr', 1e-5))
        self.min_lr = float(cfg.get('min_lr', 1e-7))
        self.warmup_epochs = int(cfg.get('warmup_epochs', 1))
        self.total_epochs = int(cfg.get('total_epochs', 10))
        params = self.model.get_stage3_params()
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.peak_lr,
            weight_decay=0.0,
        )
        self.total_epochs = cfg.get('total_epochs', 10)

    def _get_lr(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.peak_lr * (epoch + 1) / self.warmup_epochs
        progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
        return self.min_lr + (self.peak_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def _get_logprob(self, model: EpiLaBraM, eeg: torch.Tensor,
                     task_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """提取指定标签的 log probability"""
        log_probs = model.forward_stage3_reward(eeg, task_ids)  # (B, n_classes)
        return log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # (B,)

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        device = next(self.model.parameters()).device

        total_metrics: Dict[str, float] = {}
        n_steps = 0

        for batch in loader:
            eeg_c, eeg_r, task_ids, labels = batch
            eeg_c = eeg_c.to(device)
            eeg_r = eeg_r.to(device)
            task_ids = task_ids.to(device)
            labels = labels.to(device)

            # 重塑
            def reshape(eeg):
                B, C, T = eeg.shape
                A = T // 200
                return eeg[:, :, :A * 200].reshape(B, C, A, 200)

            eeg_c = reshape(eeg_c)
            eeg_r = reshape(eeg_r)

            with autocast('cuda'):
                chosen_lp = self._get_logprob(self.model, eeg_c, task_ids, labels)
                rejected_lp = self._get_logprob(self.model, eeg_r, task_ids, labels)

                with torch.no_grad():
                    ref_chosen_lp = self._get_logprob(self.ref_model, eeg_c, task_ids, labels)
                    ref_rejected_lp = self._get_logprob(self.ref_model, eeg_r, task_ids, labels)

                loss, log = self.criterion(chosen_lp, rejected_lp, ref_chosen_lp, ref_rejected_lp)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.get_stage3_params(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            for k, v in log.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v.item()
            n_steps += 1

        return {k: v / max(n_steps, 1) for k, v in total_metrics.items()}

    def train(self, resume_ckpt: Optional[str] = None):
        start_epoch = 0
        if resume_ckpt:
            start_epoch = load_checkpoint(resume_ckpt, self.model, self.optimizer, self.scaler)

        if self.use_wandb and self.is_main:
            import wandb
            wandb.init(project='epilabram', name='stage3', config=self.cfg)

        loader = DataLoader(
            self.pref_dataset,
            batch_size=self.cfg.get('batch_size', 64),
            shuffle=True,
            num_workers=self.cfg.get('num_workers', 4),
            pin_memory=True,
            drop_last=True,
        )

        for epoch in range(start_epoch, self.total_epochs):
            lr = self._get_lr(epoch)
            self._set_lr(lr)

            metrics = self._train_epoch(loader)
            metrics['lr'] = lr

            if self.is_main:
                self.logger.info(f"Epoch {epoch:03d} | " +
                                 " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
                if self.use_wandb:
                    import wandb
                    wandb.log({'epoch': epoch, **metrics})

                save_checkpoint(
                    os.path.join(self.output_dir, f'stage3_epoch{epoch:03d}.pth'),
                    self.model, self.optimizer, self.scaler, epoch,
                )

        if self.use_wandb and self.is_main:
            import wandb
            wandb.finish()
