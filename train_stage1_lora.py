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
from training.losses import PADMLoss
from training.masking import PathologyAwareMasking
from data.tuar_edf_dataset import TUARDataset
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


def train_epoch(model, dataloader, optimizer, scheduler, criterion, masking, device, epoch, args):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    total_acc = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch in pbar:
        eeg = batch['eeg'].to(device)  # (B, N, A, T)
        pathology_mask = batch.get('pathology_mask', None)
        if pathology_mask is not None:
            pathology_mask = pathology_mask.to(device)

        B, N, A, T = eeg.shape

        # Generate masks
        mask, sym_mask = masking.generate_masks(
            B, N, A,
            pathology_mask=pathology_mask,
            device=device
        )

        # Forward
        logits, sym_logits = model.forward_stage1(eeg, mask, sym_mask)

        # Get targets from tokenizer
        with torch.no_grad():
            targets = model.tokenizer.encode(eeg)  # (B, N*A)

        # Compute loss
        loss, metrics = criterion(logits, sym_logits, targets, mask, sym_mask)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        scheduler.step()

        # Metrics
        total_loss += loss.item()
        total_acc += metrics['accuracy']
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{metrics["accuracy"]:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches

    return avg_loss, avg_acc


def validate(model, dataloader, criterion, masking, device):
    """Validate the model"""
    model.eval()

    total_loss = 0
    total_acc = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            eeg = batch['eeg'].to(device)
            pathology_mask = batch.get('pathology_mask', None)
            if pathology_mask is not None:
                pathology_mask = pathology_mask.to(device)

            B, N, A, T = eeg.shape

            # Generate masks
            mask, sym_mask = masking.generate_masks(
                B, N, A,
                pathology_mask=pathology_mask,
                device=device
            )

            # Forward
            logits, sym_logits = model.forward_stage1(eeg, mask, sym_mask)

            # Get targets
            targets = model.tokenizer.encode(eeg)

            # Compute loss
            loss, metrics = criterion(logits, sym_logits, targets, mask, sym_mask)

            total_loss += loss.item()
            total_acc += metrics['accuracy']
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches

    return avg_loss, avg_acc


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
    train_dataset = TUARDataset(
        data_dir=args.data_dir,
        split='train',
        n_channels=args.n_channels,
        time_patches=args.time_patches,
        patch_size=args.patch_size,
    )

    val_dataset = TUARDataset(
        data_dir=args.data_dir,
        split='val',
        n_channels=args.n_channels,
        time_patches=args.time_patches,
        patch_size=args.patch_size,
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
    criterion = PADMLoss(
        n_embed=model.tokenizer.codebook.n_embed,
        symmetric_weight=0.5,
    )

    masking = PathologyAwareMasking(
        mask_ratio=args.mask_ratio,
        pathology_boost=args.pathology_boost,
    )

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

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
        val_loss, val_acc = validate(model, val_loader, criterion, masking, device)
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if (epoch + 1) % args.save_freq == 0 or is_best:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'args': args,
            }

            save_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(checkpoint, save_path)
            logger.info(f"Saved checkpoint to {save_path}")

            if is_best:
                best_path = os.path.join(args.output_dir, 'best_model.pth')
                save_checkpoint(checkpoint, best_path)
                logger.info(f"Saved best model to {best_path}")

    logger.info("Training completed!")


if __name__ == '__main__':
    main()
