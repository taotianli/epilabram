"""
Training script for Stage 1 with LoRA support.
Continual pre-training using PADM with parameter-efficient LoRA adapters.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.epilabram_extended import build_epilabram_extended
from training.losses import MaskedEEGModelingLoss
from training.masking import PathologyAwareDynamicMasking
from data.tuar_edf_dataset import TUAREDFDataset
from utils.logger import setup_logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Stage 1: PADM with LoRA')

    # Model
    parser.add_argument('--backbone_size', type=str, default='base', choices=['base', 'large', 'huge'])
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--vqnsp_path', type=str, required=True)
    parser.add_argument('--use_rope', action='store_true', help='Use RoPE instead of APE')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA for efficient training')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=float, default=16.0)
    parser.add_argument('--max_seq_len', type=int, default=2048)

    # Data
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--n_channels', type=int, default=23)
    parser.add_argument('--time_patches', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=200)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # Masking
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--pathology_boost', type=float, default=1.5)

    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='./checkpoints/stage1_lora')
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None)

    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()


def build_optimizer(model, args, use_lora=False):
    """Build optimizer for Stage 1"""
    if use_lora:
        # Only optimize LoRA parameters + lm_head
        params = model.get_stage1_params()
        print(f"Training with LoRA: {sum(p.numel() for p in params):,} parameters")
    else:
        # Optimize all backbone + lm_head parameters
        params = list(model.backbone.parameters()) + list(model.lm_head.parameters())
        print(f"Training full model: {sum(p.numel() for p in params):,} parameters")

    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )

    return optimizer


def build_scheduler(optimizer, args, steps_per_epoch):
    """Build learning rate scheduler with warmup"""
    warmup_steps = args.warmup_epochs * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def _prepare_eeg(batch, device, patch_size):
    """将 dataset 返回的 (eeg, label) 整理为 (B, C, A, patch_size) 格式"""
    eeg = batch[0].to(device)          # (B, C, T)
    B, C, T = eeg.shape
    A = T // patch_size
    eeg = eeg[:, :, :A * patch_size].reshape(B, C, A, patch_size)
    return eeg


def train_epoch(model, dataloader, optimizer, scheduler, criterion, masking, device, epoch, args):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    total_acc = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch in pbar:
        eeg = _prepare_eeg(batch, device, args.patch_size)   # (B, C, A, T)
        B, C, A, T = eeg.shape

        # Generate masks via PADM (expects (B, C, A*T))
        eeg_3d = eeg.reshape(B, C, A * T).cpu()
        mask, sym_mask = masking(eeg_3d, patch_size=T)       # (B, C*A)
        mask = mask.to(device)
        sym_mask = sym_mask.to(device)

        # Forward
        logits, sym_logits = model.forward_stage1(eeg, mask, sym_mask)

        # Get targets from tokenizer
        with torch.no_grad():
            targets = model.tokenizer.get_codebook_indices(eeg)  # (B, C*A)

        # Compute loss
        loss, metrics = criterion(logits, sym_logits, targets, mask, sym_mask)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_acc += metrics['metric/mask_acc'].item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{metrics["metric/mask_acc"].item():.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })

    return total_loss / max(num_batches, 1), total_acc / max(num_batches, 1)


def validate(model, dataloader, criterion, masking, device, patch_size):
    """Validate the model"""
    model.eval()

    total_loss = 0
    total_acc = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            eeg = _prepare_eeg(batch, device, patch_size)    # (B, C, A, T)
            B, C, A, T = eeg.shape

            eeg_3d = eeg.reshape(B, C, A * T).cpu()
            mask, sym_mask = masking(eeg_3d, patch_size=T)
            mask = mask.to(device)
            sym_mask = sym_mask.to(device)

            logits, sym_logits = model.forward_stage1(eeg, mask, sym_mask)
            targets = model.tokenizer.get_codebook_indices(eeg)

            loss, metrics = criterion(logits, sym_logits, targets, mask, sym_mask)

            total_loss += loss.item()
            total_acc += metrics['metric/mask_acc'].item()
            num_batches += 1

    return total_loss / max(num_batches, 1), total_acc / max(num_batches, 1)


def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.output_dir, 'train.log'))
    logger.info(f"Arguments: {args}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Build model
    logger.info("Building model...")
    model = build_epilabram_extended(
        backbone_size=args.backbone_size,
        pretrained_path=args.pretrained_path,
        vqnsp_path=args.vqnsp_path,
        use_rope=args.use_rope,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        max_seq_len=args.max_seq_len,
        use_temporal=False,  # Not needed for Stage 1
    )
    model = model.to(device)

    # Freeze backbone if using LoRA
    if args.use_lora:
        model.freeze_backbone()
        logger.info("Backbone frozen (LoRA mode)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Efficiency: {trainable_params / total_params * 100:.2f}%")

    # Build datasets
    logger.info("Loading datasets...")
    window_sec = args.time_patches * args.patch_size / 200.0  # patches * patch_size / sample_rate
    train_dataset = TUAREDFDataset(
        edf_root=args.data_dir,
        window_sec=window_sec,
        stride_sec=window_sec / 2,
        split='train',
    )

    val_dataset = TUAREDFDataset(
        edf_root=args.data_dir,
        window_sec=window_sec,
        stride_sec=window_sec / 2,
        split='eval',
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Build optimizer and scheduler
    optimizer = build_optimizer(model, args, use_lora=args.use_lora)
    scheduler = build_scheduler(optimizer, args, len(train_loader))

    # Build loss and masking
    criterion = MaskedEEGModelingLoss()

    masking = PathologyAwareDynamicMasking(
        sample_rate=200.0,
        base_mask_ratio=args.mask_ratio,
    )

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        start_epoch = load_checkpoint(args.resume, model, optimizer)

    # Training loop
    logger.info("Starting training...")

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, masking, device, epoch, args
        )

        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, masking, device, args.patch_size)
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if (epoch + 1) % args.save_freq == 0 or is_best:
            save_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(save_path, model, optimizer, None, epoch)
            logger.info(f"Saved checkpoint to {save_path}")

            if is_best:
                best_path = os.path.join(args.output_dir, 'best_model.pth')
                save_checkpoint(best_path, model, optimizer, None, epoch)
                logger.info(f"Saved best model to {best_path}")

    logger.info("Training completed!")


if __name__ == '__main__':
    main()
