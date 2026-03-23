"""
ICL (In-Context Learning) inference script.
Demonstrates few-shot learning with demonstration examples.
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.epilabram_extended import build_epilabram_extended
from utils.logger import setup_logger
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='ICL Inference')

    # Model
    parser.add_argument('--backbone_size', type=str, default='base')
    parser.add_argument('--backbone_ckpt', type=str, required=True)
    parser.add_argument('--temporal_ckpt', type=str, required=True)
    parser.add_argument('--vqnsp_path', type=str, required=True)
    parser.add_argument('--use_rope', action='store_true')
    parser.add_argument('--temporal_size', type=str, default='base')
    parser.add_argument('--num_classes', type=int, default=2)

    # Data
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)

    # ICL settings
    parser.add_argument('--n_demo', type=int, default=4, help='Number of demonstration examples')
    parser.add_argument('--n_query', type=int, default=10, help='Number of query examples per batch')
    parser.add_argument('--demo_strategy', type=str, default='balanced',
                       choices=['balanced', 'random', 'nearest'])

    # Output
    parser.add_argument('--output_dir', type=str, default='./results/icl')
    parser.add_argument('--save_predictions', action='store_true')

    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()


class ICLDataset(torch.utils.data.Dataset):
    """
    Dataset for ICL inference.
    Returns demonstration examples and query examples.
    """

    def __init__(self, data_dir, split, n_demo, n_query, demo_strategy='balanced'):
        self.data_dir = data_dir
        self.split = split
        self.n_demo = n_demo
        self.n_query = n_query
        self.demo_strategy = demo_strategy

        # Load all samples
        self.samples = self._load_samples()

        # Separate by class for balanced sampling
        self.samples_by_class = {}
        for sample in self.samples:
            label = sample['label']
            if label not in self.samples_by_class:
                self.samples_by_class[label] = []
            self.samples_by_class[label].append(sample)

    def _load_samples(self):
        # Placeholder: Load samples
        # TODO: Implement based on your data structure
        samples = []
        return samples

    def _select_demonstrations(self, query_idx):
        """Select demonstration examples based on strategy"""
        if self.demo_strategy == 'balanced':
            # Select equal number from each class
            demos = []
            n_per_class = self.n_demo // len(self.samples_by_class)

            for label, class_samples in self.samples_by_class.items():
                # Exclude query sample
                available = [s for i, s in enumerate(class_samples) if i != query_idx]
                selected = np.random.choice(len(available), n_per_class, replace=False)
                demos.extend([available[i] for i in selected])

            return demos[:self.n_demo]

        elif self.demo_strategy == 'random':
            # Random selection
            available = [s for i, s in enumerate(self.samples) if i != query_idx]
            selected = np.random.choice(len(available), self.n_demo, replace=False)
            return [available[i] for i in selected]

        else:
            raise NotImplementedError(f"Strategy {self.demo_strategy} not implemented")

    def __len__(self):
        return len(self.samples) // self.n_query

    def __getitem__(self, idx):
        # Select query samples
        query_start = idx * self.n_query
        query_samples = self.samples[query_start:query_start + self.n_query]

        # Select demonstrations (avoid overlap with queries)
        demo_samples = self._select_demonstrations(query_start)

        # Load EEG data
        demo_eeg = torch.stack([self._load_eeg(s) for s in demo_samples])
        demo_labels = torch.tensor([s['label'] for s in demo_samples])

        query_eeg = torch.stack([self._load_eeg(s) for s in query_samples])
        query_labels = torch.tensor([s['label'] for s in query_samples])

        return {
            'demo_eeg': demo_eeg,  # (n_demo, N, A, T)
            'demo_labels': demo_labels,  # (n_demo,)
            'query_eeg': query_eeg,  # (n_query, N, A, T)
            'query_labels': query_labels,  # (n_query,)
        }

    def _load_eeg(self, sample):
        # Placeholder: Load EEG data
        # TODO: Implement based on your data format
        return torch.randn(23, 4, 200)


def evaluate_icl(model, dataloader, device, args):
    """Evaluate ICL performance"""
    model.eval()

    all_predictions = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='ICL Inference'):
            demo_eeg = batch['demo_eeg'].to(device)  # (B, n_demo, N, A, T)
            demo_labels = batch['demo_labels'].to(device)  # (B, n_demo)
            query_eeg = batch['query_eeg'].to(device)  # (B, n_query, N, A, T)
            query_labels = batch['query_labels'].to(device)  # (B, n_query)

            # ICL forward
            logits = model.forward_icl(demo_eeg, demo_labels, query_eeg)  # (B, n_query, num_classes)

            # Get predictions
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)  # (B, n_query)
            confidence = probs.max(dim=-1)[0]  # (B, n_query)

            # Collect results
            all_predictions.append(preds.cpu())
            all_labels.append(query_labels.cpu())
            all_confidences.append(confidence.cpu())

    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0).flatten()
    all_labels = torch.cat(all_labels, dim=0).flatten()
    all_confidences = torch.cat(all_confidences, dim=0).flatten()

    # Compute metrics
    accuracy = (all_predictions == all_labels).float().mean().item()

    # Per-class accuracy
    per_class_acc = {}
    for c in range(args.num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class_acc[c] = (all_predictions[mask] == all_labels[mask]).float().mean().item()

    # Confidence statistics
    avg_confidence = all_confidences.mean().item()
    correct_confidence = all_confidences[all_predictions == all_labels].mean().item()
    incorrect_confidence = all_confidences[all_predictions != all_labels].mean().item()

    results = {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc,
        'avg_confidence': avg_confidence,
        'correct_confidence': correct_confidence,
        'incorrect_confidence': incorrect_confidence,
        'predictions': all_predictions.numpy(),
        'labels': all_labels.numpy(),
        'confidences': all_confidences.numpy(),
    }

    return results


def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.output_dir, 'icl_inference.log'))
    logger.info(f"Arguments: {args}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Build model
    logger.info("Building model...")
    model = build_epilabram_extended(
        backbone_size=args.backbone_size,
        vqnsp_path=args.vqnsp_path,
        use_rope=args.use_rope,
        use_lora=False,
        use_temporal=True,
        temporal_size=args.temporal_size,
        temporal_num_classes=args.num_classes,
    )

    # Load checkpoints
    logger.info(f"Loading backbone from {args.backbone_ckpt}")
    backbone_ckpt = torch.load(args.backbone_ckpt, map_location='cpu')
    model.backbone.load_state_dict(backbone_ckpt['model'], strict=False)

    logger.info(f"Loading temporal transformer from {args.temporal_ckpt}")
    temporal_ckpt = torch.load(args.temporal_ckpt, map_location='cpu')
    model.temporal_transformer.load_state_dict(temporal_ckpt['temporal_transformer'])

    model = model.to(device)
    model.eval()

    # Build dataset
    logger.info("Loading dataset...")
    dataset = ICLDataset(
        data_dir=args.data_dir,
        split='test',
        n_demo=args.n_demo,
        n_query=args.n_query,
        demo_strategy=args.demo_strategy,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info(f"Test samples: {len(dataset) * args.n_query}")
    logger.info(f"Demonstrations per query: {args.n_demo}")
    logger.info(f"Demo strategy: {args.demo_strategy}")

    # Run ICL inference
    logger.info("\nRunning ICL inference...")
    results = evaluate_icl(model, dataloader, device, args)

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("ICL Inference Results")
    logger.info("=" * 60)
    logger.info(f"Overall Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Average Confidence: {results['avg_confidence']:.4f}")
    logger.info(f"Correct Confidence: {results['correct_confidence']:.4f}")
    logger.info(f"Incorrect Confidence: {results['incorrect_confidence']:.4f}")

    logger.info("\nPer-class Accuracy:")
    for c, acc in results['per_class_accuracy'].items():
        logger.info(f"  Class {c}: {acc:.4f}")

    # Save predictions
    if args.save_predictions:
        save_path = os.path.join(args.output_dir, 'predictions.npz')
        np.savez(
            save_path,
            predictions=results['predictions'],
            labels=results['labels'],
            confidences=results['confidences'],
        )
        logger.info(f"\nSaved predictions to {save_path}")

    logger.info("\nICL inference completed!")


if __name__ == '__main__':
    main()
