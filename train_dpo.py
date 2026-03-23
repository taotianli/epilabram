"""
Training script for DPO (Direct Preference Optimization).
Aligns model predictions with clinical preferences.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

from models.labram_backbone import LaBraMBackbone
from models.dpo import (
    DPOEEGModel,
    PreferenceDataGenerator,
    PreferenceDataset,
    collate_preference_batch,
    ClinicalPreferenceOptimizer,
    evaluate_clinical_metrics,
    print_clinical_metrics,
)
from utils.logger import setup_logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train EEG Model with DPO')

    # Model
    parser.add_argument('--backbone_size', type=str, default='base')
    parser.add_argument('--policy_ckpt', type=str, required=True, help='Initial policy model checkpoint')
    parser.add_argument('--reference_ckpt', type=str, default=None, help='Reference model (if None, use reference-free DPO)')
    parser.add_argument('--beta', type=float, default=0.1, help='DPO temperature parameter')

    # Data
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    # Training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # Preference generation
    parser.add_argument('--preference_strategy', type=str, default='confidence',
                       choices=['confidence', 'clinical', 'consistency'])
    parser.add_argument('--clinical_criterion', type=str, default='high_sensitivity',
                       choices=['high_sensitivity', 'high_specificity'])
    parser.add_argument('--confidence_threshold', type=float, default=0.8)

    # Output
    parser.add_argument('--output_dir', type=str, default='./checkpoints/dpo')
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)

    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()


def generate_preference_dataset(
    model,
    dataloader,
    generator,
    device,
    strategy,
    clinical_criterion=None,
):
    """
    Generate preference pairs from a dataset.
    """
    model.eval()

    all_eeg = []
    all_chosen = []
    all_rejected = []

    print(f"Generating preference pairs using strategy: {strategy}")

    for batch in tqdm(dataloader, desc='Generating preferences'):
        eeg = batch['eeg'].to(device)
        labels = batch['labels'].to(device)

        if strategy == 'confidence':
            eeg_pref, chosen, rejected = generator.generate_confidence_pairs(eeg, labels)
        elif strategy == 'clinical':
            eeg_pref, chosen, rejected = generator.generate_clinical_pairs(
                eeg, labels, clinical_criterion=clinical_criterion
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if len(eeg_pref) > 0:
            all_eeg.append(eeg_pref.cpu())
            all_chosen.append(chosen.cpu())
            all_rejected.append(rejected.cpu())

    # Concatenate
    all_eeg = torch.cat(all_eeg, dim=0)
    all_chosen = torch.cat(all_chosen, dim=0)
    all_rejected = torch.cat(all_rejected, dim=0)

    print(f"Generated {len(all_eeg)} preference pairs")

    # Create dataset
    eeg_list = [all_eeg[i] for i in range(len(all_eeg))]
    chosen_list = all_chosen.tolist()
    rejected_list = all_rejected.tolist()

    dataset = PreferenceDataset(eeg_list, chosen_list, rejected_list)

    return dataset


def train_epoch(dpo_model, dataloader, optimizer, device, epoch):
    """Train for one epoch with DPO"""
    dpo_model.policy_model.train()

    total_loss = 0
    total_reward_margin = 0
    total_reward_acc = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch in pbar:
        eeg = batch['eeg'].to(device)
        chosen_labels = batch['chosen_labels'].to(device)
        rejected_labels = batch['rejected_labels'].to(device)

        # DPO forward
        loss, metrics = dpo_model(eeg, chosen_labels, rejected_labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dpo_model.policy_model.parameters(), 1.0)
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        total_reward_margin += metrics['reward_margin'].item()
        total_reward_acc += metrics['reward_accuracy'].item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'margin': f'{metrics["reward_margin"].item():.4f}',
            'acc': f'{metrics["reward_accuracy"].item():.4f}',
        })

    avg_loss = total_loss / num_batches
    avg_margin = total_reward_margin / num_batches
    avg_acc = total_reward_acc / num_batches

    return avg_loss, avg_margin, avg_acc


def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.output_dir, 'train.log'))
    logger.info(f"Arguments: {args}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Build policy model
    logger.info("Building policy model...")
    policy_model = LaBraMBackbone(size=args.backbone_size)

    # Load policy checkpoint
    logger.info(f"Loading policy model from {args.policy_ckpt}")
    checkpoint = torch.load(args.policy_ckpt, map_location='cpu')
    policy_model.load_state_dict(checkpoint['model'], strict=False)
    policy_model = policy_model.to(device)

    # Build reference model
    reference_model = None
    if args.reference_ckpt:
        logger.info(f"Loading reference model from {args.reference_ckpt}")
        reference_model = LaBraMBackbone(size=args.backbone_size)
        ref_checkpoint = torch.load(args.reference_ckpt, map_location='cpu')
        reference_model.load_state_dict(ref_checkpoint['model'], strict=False)
        reference_model = reference_model.to(device)
        reference_model.eval()
    else:
        logger.info("Using reference-free DPO")

    # Build DPO model
    dpo_model = DPOEEGModel(
        policy_model=policy_model,
        reference_model=reference_model,
        beta=args.beta,
        reference_free=(reference_model is None),
    )

    # Count parameters
    total_params = sum(p.numel() for p in policy_model.parameters())
    trainable_params = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Load data for preference generation
    logger.info("Loading data for preference generation...")
    # TODO: Implement actual data loading
    # train_data_loader = ...
    # val_data_loader = ...

    # Generate preference dataset
    logger.info("Generating preference dataset...")
    generator = PreferenceDataGenerator(
        model=policy_model,
        strategy=args.preference_strategy,
        confidence_threshold=args.confidence_threshold,
    )

    # preference_train_dataset = generate_preference_dataset(
    #     policy_model,
    #     train_data_loader,
    #     generator,
    #     device,
    #     args.preference_strategy,
    #     args.clinical_criterion,
    # )

    # Build dataloader
    # train_loader = DataLoader(
    #     preference_train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     collate_fn=collate_preference_batch,
    #     pin_memory=True,
    # )

    # Build optimizer
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Training loop
    logger.info("Starting DPO training...")

    # for epoch in range(args.epochs):
    #     logger.info(f"\nEpoch {epoch}/{args.epochs}")
    #
    #     # Train
    #     train_loss, train_margin, train_acc = train_epoch(
    #         dpo_model, train_loader, optimizer, device, epoch
    #     )
    #
    #     logger.info(f"Train Loss: {train_loss:.4f}, "
    #                f"Reward Margin: {train_margin:.4f}, "
    #                f"Reward Acc: {train_acc:.4f}")
    #
    #     # Evaluate clinical metrics
    #     if (epoch + 1) % 5 == 0:
    #         metrics = evaluate_clinical_metrics(policy_model, val_data_loader, device)
    #         print_clinical_metrics(metrics)
    #
    #     # Save checkpoint
    #     if (epoch + 1) % args.save_freq == 0:
    #         checkpoint = {
    #             'epoch': epoch,
    #             'model': policy_model.state_dict(),
    #             'optimizer': optimizer.state_dict(),
    #             'train_loss': train_loss,
    #             'args': args,
    #         }
    #
    #         save_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
    #         save_checkpoint(checkpoint, save_path)
    #         logger.info(f"Saved checkpoint to {save_path}")

    logger.info("DPO training completed!")

    # Example: Use ClinicalPreferenceOptimizer for high-level optimization
    logger.info("\nExample: Using ClinicalPreferenceOptimizer")

    # optimizer_clinical = ClinicalPreferenceOptimizer(
    #     model=policy_model,
    #     reference_model=reference_model,
    #     beta=args.beta,
    # )

    # # Optimize for high sensitivity
    # optimizer_clinical.optimize_for_sensitivity(
    #     train_data_loader,
    #     optimizer,
    #     num_epochs=10,
    #     device=device,
    # )


if __name__ == '__main__':
    main()
