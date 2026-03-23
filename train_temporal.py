"""
Training script for Temporal Transformer.
Trains the second-level transformer for long-form EEG modeling.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.epilabram_extended import build_epilabram_extended
from utils.logger import setup_logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train Temporal Transformer')

    # Model
    parser.add_argument('--backbone_size', type=str, default='base', choices=['base', 'large', 'huge'])
    parser.add_argument('--backbone_ckpt', type=str, required=True, help='Pretrained backbone checkpoint')
    parser.add_argument('--vqnsp_path', type=str, required=True)
    parser.add_argument('--use_rope', action='store_true')
    parser.add_argument('--temporal_size', type=str, default='base', choices=['small', 'base', 'large'])
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--max_seq_len', type=int, default=512)

    # Data
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seq_length', type=int, default=16, help='Number of epochs in sequence')
    parser.add_argument('--n_channels', type=int, default=23)
    parser.add_argument('--time_patches', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=200)

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='./checkpoints/temporal')
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)

    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()


class SequenceEEGDataset(torch.utils.data.Dataset):
    """
    Dataset that returns sequences of EEG epochs.
    Each sample is a sequence of consecutive epochs with a single label.
    """

    def __init__(self, data_dir, split, seq_length, n_channels, time_patches, patch_size):
        self.data_dir = data_dir
        self.split = split
        self.seq_length = seq_length
        self.n_channels = n_channels
        self.time_patches = time_patches
        self.patch_size = patch_size

        # Load metadata (implement based on your data structure)
        # This is a placeholder - adapt to your actual data format
        self.samples = self._load_samples()

    def _load_samples(self):
        # Placeholder: Load list of (sequence_path, label) tuples
        # Each sequence should contain seq_length consecutive epochs
        samples = []
        # TODO: Implement based on your data structure
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence_path, label = self.samples[idx]

        # Load EEG sequence (T_seq, N, A, T)
        # TODO: Implement based on your data format
        eeg_sequence = torch.randn(self.seq_length, self.n_channels,
                                   self.time_patches, self.patch_size)

        return {
            'eeg_sequence': eeg_sequence,
            'label': label,
            'mask': torch.ones(self.seq_length).bool(),  # All valid
        }


def build_optimizer(model, args):
    """Build optimizer for temporal transformer"""
    # Only train temporal transformer, freeze backbone
    params = model.get_temporal_params()

    print(f"Training temporal transformer: {sum(p.numel() for p in params):,} parameters")

    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
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


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, args):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch in pbar:
        eeg_sequence = batch['eeg_sequence'].to(device)  # (B, T_seq, N, A, T)
        labels = batch['label'].to(device)  # (B,)
        mask = batch['mask'].to(device)  # (B, T_seq)

        # Forward
        logits = model.forward_temporal(eeg_sequence, mask=mask)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.get_temporal_params(), args.grad_clip)

        optimizer.step()
        scheduler.step()

        # Metrics
        preds = logits.argmax(dim=-1)
        correct = (preds == labels).sum().item()

        total_loss += loss.item() * labels.size(0)
        total_correct += correct
        total_samples += labels.size(0)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct / labels.size(0):.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc


def validate(model, dataloader, device, args):
    """Validate the model"""
    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            eeg_sequence = batch['eeg_sequence'].to(device)
            labels = batch['label'].to(device)
            mask = batch['mask'].to(device)

            # Forward
            logits = model.forward_temporal(eeg_sequence, mask=mask)

            # Compute loss
            loss = criterion(logits, labels)

            # Metrics
            preds = logits.argmax(dim=-1)
            correct = (preds == labels).sum().item()

            total_loss += loss.item() * labels.size(0)
            total_correct += correct
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

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
        vqnsp_path=args.vqnsp_path,
        use_rope=args.use_rope,
        use_lora=False,  # Not needed for temporal training
        use_temporal=True,
        temporal_size=args.temporal_size,
        temporal_num_classes=args.num_classes,
        max_seq_len=args.max_seq_len,
    )

    # Load pretrained backbone
    logger.info(f"Loading backbone from {args.backbone_ckpt}")
    checkpoint = torch.load(args.backbone_ckpt, map_location='cpu')
    model.backbone.load_state_dict(checkpoint['model'], strict=False)

    # Freeze backbone
    model.freeze_backbone()
    logger.info("Backbone frozen")

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Build datasets
    logger.info("Loading datasets...")
    train_dataset = SequenceEEGDataset(
        data_dir=args.data_dir,
        split='train',
        seq_length=args.seq_length,
        n_channels=args.n_channels,
        time_patches=args.time_patches,
        patch_size=args.patch_size,
    )

    val_dataset = SequenceEEGDataset(
        data_dir=args.data_dir,
        split='val',
        seq_length=args.seq_length,
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
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, len(train_loader))

    # Resume from checkpoint
    start_epoch = 0
    best_val_acc = 0.0

    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = load_checkpoint(args.resume)
        model.temporal_transformer.load_state_dict(checkpoint['temporal_transformer'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)

    # Training loop
    logger.info("Starting training...")

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, args
        )

        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validate
        val_loss, val_acc = validate(model, val_loader, device, args)
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        if (epoch + 1) % args.save_freq == 0 or is_best:
            checkpoint = {
                'epoch': epoch,
                'temporal_transformer': model.temporal_transformer.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc,
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
