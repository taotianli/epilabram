"""
Training script for Instruction Tuning.
Enables the model to understand task descriptions and perform zero-shot inference.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.labram_backbone import LaBraMBackbone
from models.instruction_tuning import (
    build_instruction_tuned_model,
    InstructionDataset,
    collate_instruction_batch,
    get_instruction,
)
from utils.logger import setup_logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train Instruction-Tuned EEG Model')

    # Model
    parser.add_argument('--backbone_size', type=str, default='base')
    parser.add_argument('--backbone_ckpt', type=str, required=True)
    parser.add_argument('--text_encoder', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--freeze_text_encoder', action='store_true')
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--fusion_dim', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_fusion_layers', type=int, default=2)

    # Data
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # Multi-task
    parser.add_argument('--tasks', nargs='+', default=['seizure_detection', 'sleep_staging', 'artifact_detection'])

    # Output
    parser.add_argument('--output_dir', type=str, default='./checkpoints/instruction_tuning')
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)

    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()


def build_multi_task_dataset(data_dir, split, tasks):
    """
    Build dataset with multiple tasks.
    Each sample has EEG data, task name, and label.
    """
    # Placeholder: Load your actual data
    # This should return lists of (eeg, label, task_name) tuples

    eeg_data = []
    labels = []
    task_names = []

    # TODO: Implement actual data loading based on your data structure
    # Example structure:
    # for task in tasks:
    #     task_data = load_task_data(data_dir, split, task)
    #     eeg_data.extend(task_data['eeg'])
    #     labels.extend(task_data['labels'])
    #     task_names.extend([task] * len(task_data['eeg']))

    return InstructionDataset(eeg_data, labels, task_names)


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, args):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch in pbar:
        eeg = batch['eeg'].to(device)
        labels = batch['labels'].to(device)
        instructions = batch['instructions']
        task_names = batch['task_names']

        # Determine task type for each sample
        # For simplicity, assume binary classification for all tasks
        # You can extend this to handle different task types
        task_type = 'binary'

        # Forward
        logits = model(eeg, instructions, task_type=task_type)

        # Compute loss
        loss = nn.functional.cross_entropy(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

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


def validate(model, dataloader, device):
    """Validate the model"""
    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    # Per-task metrics
    task_metrics = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            eeg = batch['eeg'].to(device)
            labels = batch['labels'].to(device)
            instructions = batch['instructions']
            task_names = batch['task_names']

            # Forward
            logits = model(eeg, instructions, task_type='binary')

            # Compute loss
            loss = nn.functional.cross_entropy(logits, labels)

            # Metrics
            preds = logits.argmax(dim=-1)
            correct = (preds == labels).sum().item()

            total_loss += loss.item() * labels.size(0)
            total_correct += correct
            total_samples += labels.size(0)

            # Per-task metrics
            for i, task_name in enumerate(task_names):
                if task_name not in task_metrics:
                    task_metrics[task_name] = {'correct': 0, 'total': 0}

                task_metrics[task_name]['correct'] += (preds[i] == labels[i]).item()
                task_metrics[task_name]['total'] += 1

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    # Compute per-task accuracy
    for task_name in task_metrics:
        task_metrics[task_name]['accuracy'] = \
            task_metrics[task_name]['correct'] / task_metrics[task_name]['total']

    return avg_loss, avg_acc, task_metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.output_dir, 'train.log'))
    logger.info(f"Arguments: {args}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Build backbone
    logger.info("Building backbone...")
    backbone = LaBraMBackbone(size=args.backbone_size)

    # Load pretrained weights
    logger.info(f"Loading backbone from {args.backbone_ckpt}")
    checkpoint = torch.load(args.backbone_ckpt, map_location='cpu')
    backbone.load_state_dict(checkpoint['model'], strict=False)

    # Freeze backbone if requested
    if args.freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")

    # Build instruction-tuned model
    logger.info("Building instruction-tuned model...")
    model = build_instruction_tuned_model(
        backbone=backbone,
        fusion_dim=args.fusion_dim,
        num_heads=args.num_heads,
        num_fusion_layers=args.num_fusion_layers,
        text_encoder_name=args.text_encoder,
        freeze_text_encoder=args.freeze_text_encoder,
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Build datasets
    logger.info("Loading datasets...")
    train_dataset = build_multi_task_dataset(args.data_dir, 'train', args.tasks)
    val_dataset = build_multi_task_dataset(args.data_dir, 'val', args.tasks)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_instruction_batch,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_instruction_batch,
        pin_memory=True,
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Tasks: {args.tasks}")

    # Build optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Build scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint
    start_epoch = 0
    best_val_acc = 0.0

    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model'])
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
        val_loss, val_acc, task_metrics = validate(model, val_loader, device)
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Log per-task metrics
        logger.info("Per-task accuracy:")
        for task_name, metrics in task_metrics.items():
            logger.info(f"  {task_name}: {metrics['accuracy']:.4f}")

        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        if (epoch + 1) % args.save_freq == 0 or is_best:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc,
                'task_metrics': task_metrics,
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
