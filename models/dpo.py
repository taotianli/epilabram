"""
Direct Preference Optimization (DPO) for EEG Foundation Models.
Aligns model predictions with clinical preferences without reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class PreferenceExample:
    """
    A single preference example for DPO training.
    """
    eeg: torch.Tensor  # (N, A, T) EEG data
    chosen_output: Union[int, torch.Tensor]  # Preferred prediction
    rejected_output: Union[int, torch.Tensor]  # Rejected prediction
    metadata: Optional[Dict] = None  # Additional info (confidence, reasoning, etc.)


class DPOLoss(nn.Module):
    """
    Direct Preference Optimization loss.

    Based on: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
    https://arxiv.org/abs/2305.18290

    Loss = -E[log σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)))]

    where:
    - y_w: chosen (preferred) output
    - y_l: rejected output
    - π_θ: policy model (current model)
    - π_ref: reference model (frozen)
    - β: temperature parameter
    """

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        reference_free: bool = False,
    ):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.reference_free = reference_free

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: Optional[torch.Tensor] = None,
        reference_rejected_logps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute DPO loss.

        Args:
            policy_chosen_logps: log P(y_chosen | x) from policy model (B,)
            policy_rejected_logps: log P(y_rejected | x) from policy model (B,)
            reference_chosen_logps: log P(y_chosen | x) from reference model (B,)
            reference_rejected_logps: log P(y_rejected | x) from reference model (B,)

        Returns:
            loss: scalar loss
            metrics: dict of metrics for logging
        """
        if self.reference_free:
            # Reference-free variant (simpler, no need for reference model)
            logits = self.beta * (policy_chosen_logps - policy_rejected_logps)
        else:
            # Standard DPO with reference model
            if reference_chosen_logps is None or reference_rejected_logps is None:
                raise ValueError("Reference logps required when reference_free=False")

            policy_logratios = policy_chosen_logps - policy_rejected_logps
            reference_logratios = reference_chosen_logps - reference_rejected_logps
            logits = self.beta * (policy_logratios - reference_logratios)

        # DPO loss: -log sigmoid(logits)
        losses = -F.logsigmoid(logits)

        # Label smoothing
        if self.label_smoothing > 0:
            losses = (1 - self.label_smoothing) * losses + \
                     self.label_smoothing * -F.logsigmoid(-logits)

        loss = losses.mean()

        # Compute metrics
        with torch.no_grad():
            # Implicit reward
            if self.reference_free:
                chosen_rewards = self.beta * policy_chosen_logps
                rejected_rewards = self.beta * policy_rejected_logps
            else:
                chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
                rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)

            reward_margin = chosen_rewards - rejected_rewards
            reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()

        metrics = {
            'loss': loss.detach(),
            'reward_margin': reward_margin.mean().detach(),
            'reward_accuracy': reward_accuracy,
            'chosen_rewards': chosen_rewards.mean().detach(),
            'rejected_rewards': rejected_rewards.mean().detach(),
        }

        return loss, metrics


class DPOEEGModel(nn.Module):
    """
    EEG model with DPO training support.

    Supports preference optimization for:
    - High-confidence correct predictions (chosen) vs low-confidence/wrong (rejected)
    - Clinically preferred predictions (e.g., high sensitivity) vs others
    - Interpretable predictions vs black-box predictions
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: Optional[nn.Module] = None,
        beta: float = 0.1,
        reference_free: bool = False,
    ):
        super().__init__()
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.beta = beta
        self.reference_free = reference_free

        # Freeze reference model
        if self.reference_model is not None:
            for param in self.reference_model.parameters():
                param.requires_grad = False

        # DPO loss
        self.dpo_loss = DPOLoss(
            beta=beta,
            reference_free=reference_free,
        )

    def get_log_probs(
        self,
        model: nn.Module,
        eeg: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get log probabilities for given labels.

        Args:
            model: Model to use
            eeg: (B, N, A, T) EEG data
            labels: (B,) labels

        Returns:
            log_probs: (B,) log probabilities
        """
        logits = model(eeg)  # (B, num_classes)
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probs for given labels
        log_probs = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)

        return log_probs

    def forward(
        self,
        eeg: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for DPO training.

        Args:
            eeg: (B, N, A, T) EEG data
            chosen_labels: (B,) preferred labels
            rejected_labels: (B,) rejected labels

        Returns:
            loss: scalar loss
            metrics: dict of metrics
        """
        # Get log probs from policy model
        policy_chosen_logps = self.get_log_probs(self.policy_model, eeg, chosen_labels)
        policy_rejected_logps = self.get_log_probs(self.policy_model, eeg, rejected_labels)

        # Get log probs from reference model
        if not self.reference_free:
            with torch.no_grad():
                reference_chosen_logps = self.get_log_probs(
                    self.reference_model, eeg, chosen_labels
                )
                reference_rejected_logps = self.get_log_probs(
                    self.reference_model, eeg, rejected_labels
                )
        else:
            reference_chosen_logps = None
            reference_rejected_logps = None

        # Compute DPO loss
        loss, metrics = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )

        return loss, metrics


class PreferenceDataGenerator:
    """
    Generates preference pairs from model predictions and ground truth.

    Strategies:
    1. Confidence-based: High-confidence correct (chosen) vs low-confidence/wrong (rejected)
    2. Clinical-based: Predictions meeting clinical criteria (chosen) vs others (rejected)
    3. Consistency-based: Consistent predictions across augmentations (chosen) vs inconsistent (rejected)
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: str = 'confidence',
        confidence_threshold: float = 0.8,
    ):
        self.model = model
        self.strategy = strategy
        self.confidence_threshold = confidence_threshold

    def generate_confidence_pairs(
        self,
        eeg: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate preference pairs based on confidence.

        Chosen: High-confidence correct predictions
        Rejected: Low-confidence or incorrect predictions
        """
        self.model.eval()

        with torch.no_grad():
            logits = self.model(eeg)
            probs = F.softmax(logits, dim=-1)
            confidences, predictions = probs.max(dim=-1)

        # Identify chosen and rejected samples
        correct = (predictions == labels)
        high_confidence = (confidences > self.confidence_threshold)

        # Chosen: correct AND high confidence
        chosen_mask = correct & high_confidence

        # Rejected: incorrect OR low confidence
        rejected_mask = ~chosen_mask

        # Filter to get valid pairs
        valid_mask = chosen_mask | rejected_mask

        chosen_labels = labels.clone()
        rejected_labels = predictions.clone()

        # For rejected samples, if they're correct but low confidence,
        # use a different (wrong) prediction as rejected
        low_conf_correct = correct & ~high_confidence
        if low_conf_correct.any():
            # Use second-best prediction as rejected
            second_best = probs.topk(2, dim=-1)[1][:, 1]
            rejected_labels[low_conf_correct] = second_best[low_conf_correct]

        return eeg[valid_mask], chosen_labels[valid_mask], rejected_labels[valid_mask]

    def generate_clinical_pairs(
        self,
        eeg: torch.Tensor,
        labels: torch.Tensor,
        clinical_criterion: str = 'high_sensitivity',
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate preference pairs based on clinical criteria.

        For binary classification (e.g., seizure detection):
        - High sensitivity: Prefer predicting positive when uncertain
        - High specificity: Prefer predicting negative when uncertain
        """
        self.model.eval()

        with torch.no_grad():
            logits = self.model(eeg)
            probs = F.softmax(logits, dim=-1)

        if clinical_criterion == 'high_sensitivity':
            # For positive class (label=1), prefer predicting positive
            # Chosen: predict positive when true positive or uncertain
            # Rejected: predict negative when true positive

            positive_samples = (labels == 1)
            chosen_labels = torch.ones_like(labels)
            rejected_labels = torch.zeros_like(labels)

            # Only keep samples where model is uncertain or wrong
            uncertain = (probs[:, 1] > 0.3) & (probs[:, 1] < 0.7)
            valid_mask = positive_samples & uncertain

        elif clinical_criterion == 'high_specificity':
            # For negative class (label=0), prefer predicting negative
            negative_samples = (labels == 0)
            chosen_labels = torch.zeros_like(labels)
            rejected_labels = torch.ones_like(labels)

            uncertain = (probs[:, 0] > 0.3) & (probs[:, 0] < 0.7)
            valid_mask = negative_samples & uncertain

        else:
            raise ValueError(f"Unknown clinical criterion: {clinical_criterion}")

        return eeg[valid_mask], chosen_labels[valid_mask], rejected_labels[valid_mask]

    def generate_consistency_pairs(
        self,
        eeg: torch.Tensor,
        labels: torch.Tensor,
        augment_fn,
        num_augmentations: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate preference pairs based on prediction consistency.

        Chosen: Predictions consistent across augmentations
        Rejected: Inconsistent predictions
        """
        self.model.eval()

        all_predictions = []

        with torch.no_grad():
            # Original prediction
            logits = self.model(eeg)
            predictions = logits.argmax(dim=-1)
            all_predictions.append(predictions)

            # Augmented predictions
            for _ in range(num_augmentations):
                eeg_aug = augment_fn(eeg)
                logits_aug = self.model(eeg_aug)
                predictions_aug = logits_aug.argmax(dim=-1)
                all_predictions.append(predictions_aug)

        # Stack predictions
        all_predictions = torch.stack(all_predictions, dim=1)  # (B, num_aug+1)

        # Compute consistency (mode prediction)
        mode_predictions = torch.mode(all_predictions, dim=1)[0]

        # Consistency score: fraction of predictions matching mode
        consistency = (all_predictions == mode_predictions.unsqueeze(1)).float().mean(dim=1)

        # Chosen: high consistency (> 0.8)
        # Rejected: low consistency (< 0.5)
        high_consistency = consistency > 0.8
        low_consistency = consistency < 0.5

        valid_mask = high_consistency | low_consistency

        chosen_labels = mode_predictions.clone()
        rejected_labels = predictions.clone()

        # For low consistency samples, use the original prediction as rejected
        rejected_labels[low_consistency] = predictions[low_consistency]

        return eeg[valid_mask], chosen_labels[valid_mask], rejected_labels[valid_mask]


class PreferenceDataset(torch.utils.data.Dataset):
    """
    Dataset for DPO training with preference pairs.
    """

    def __init__(
        self,
        eeg_data: List[torch.Tensor],
        chosen_labels: List[int],
        rejected_labels: List[int],
        metadata: Optional[List[Dict]] = None,
    ):
        self.eeg_data = eeg_data
        self.chosen_labels = chosen_labels
        self.rejected_labels = rejected_labels
        self.metadata = metadata or [{}] * len(eeg_data)

        assert len(eeg_data) == len(chosen_labels) == len(rejected_labels)

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        return {
            'eeg': self.eeg_data[idx],
            'chosen_label': self.chosen_labels[idx],
            'rejected_label': self.rejected_labels[idx],
            'metadata': self.metadata[idx],
        }


def collate_preference_batch(batch):
    """Collate function for preference dataset."""
    eeg = torch.stack([item['eeg'] for item in batch])
    chosen_labels = torch.tensor([item['chosen_label'] for item in batch])
    rejected_labels = torch.tensor([item['rejected_label'] for item in batch])

    return {
        'eeg': eeg,
        'chosen_labels': chosen_labels,
        'rejected_labels': rejected_labels,
    }


class ClinicalPreferenceOptimizer:
    """
    High-level interface for clinical preference optimization.

    Supports common clinical scenarios:
    - Optimizing for high sensitivity (minimize false negatives)
    - Optimizing for high specificity (minimize false positives)
    - Balancing sensitivity and specificity
    - Optimizing for clinical confidence
    """

    def __init__(
        self,
        model: nn.Module,
        reference_model: Optional[nn.Module] = None,
        beta: float = 0.1,
    ):
        self.model = model
        self.reference_model = reference_model
        self.beta = beta

        # Create DPO model
        self.dpo_model = DPOEEGModel(
            policy_model=model,
            reference_model=reference_model,
            beta=beta,
            reference_free=(reference_model is None),
        )

    def optimize_for_sensitivity(
        self,
        train_loader,
        optimizer,
        num_epochs: int = 10,
        device: str = 'cuda',
    ):
        """
        Optimize model for high sensitivity (recall).
        Useful for critical detection tasks (e.g., seizure detection).
        """
        print("Optimizing for HIGH SENSITIVITY (minimize false negatives)")

        generator = PreferenceDataGenerator(
            model=self.model,
            strategy='clinical',
        )

        for epoch in range(num_epochs):
            total_loss = 0
            total_reward_margin = 0
            num_batches = 0

            for batch in train_loader:
                eeg = batch['eeg'].to(device)
                labels = batch['labels'].to(device)

                # Generate preference pairs favoring sensitivity
                eeg_pref, chosen, rejected = generator.generate_clinical_pairs(
                    eeg, labels, clinical_criterion='high_sensitivity'
                )

                if len(eeg_pref) == 0:
                    continue

                # DPO training step
                loss, metrics = self.dpo_model(eeg_pref, chosen, rejected)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_reward_margin += metrics['reward_margin'].item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            avg_margin = total_reward_margin / num_batches

            print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, "
                  f"Reward Margin={avg_margin:.4f}")

    def optimize_for_specificity(
        self,
        train_loader,
        optimizer,
        num_epochs: int = 10,
        device: str = 'cuda',
    ):
        """
        Optimize model for high specificity (precision).
        Useful when false positives are costly.
        """
        print("Optimizing for HIGH SPECIFICITY (minimize false positives)")

        generator = PreferenceDataGenerator(
            model=self.model,
            strategy='clinical',
        )

        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            for batch in train_loader:
                eeg = batch['eeg'].to(device)
                labels = batch['labels'].to(device)

                # Generate preference pairs favoring specificity
                eeg_pref, chosen, rejected = generator.generate_clinical_pairs(
                    eeg, labels, clinical_criterion='high_specificity'
                )

                if len(eeg_pref) == 0:
                    continue

                # DPO training step
                loss, metrics = self.dpo_model(eeg_pref, chosen, rejected)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}")

    def optimize_for_confidence(
        self,
        train_loader,
        optimizer,
        num_epochs: int = 10,
        device: str = 'cuda',
        confidence_threshold: float = 0.8,
    ):
        """
        Optimize model to be more confident in correct predictions.
        """
        print(f"Optimizing for CONFIDENCE (threshold={confidence_threshold})")

        generator = PreferenceDataGenerator(
            model=self.model,
            strategy='confidence',
            confidence_threshold=confidence_threshold,
        )

        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            for batch in train_loader:
                eeg = batch['eeg'].to(device)
                labels = batch['labels'].to(device)

                # Generate confidence-based preference pairs
                eeg_pref, chosen, rejected = generator.generate_confidence_pairs(
                    eeg, labels
                )

                if len(eeg_pref) == 0:
                    continue

                # DPO training step
                loss, metrics = self.dpo_model(eeg_pref, chosen, rejected)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}")


def evaluate_clinical_metrics(
    model: nn.Module,
    test_loader,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Evaluate model on clinical metrics.

    Returns:
        metrics: Dict with sensitivity, specificity, PPV, NPV, F1, etc.
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            eeg = batch['eeg'].to(device)
            labels = batch['labels'].to(device)

            logits = model(eeg)
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    # Compute metrics (assuming binary classification)
    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    tn = ((all_preds == 0) & (all_labels == 0)).sum().item()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().item()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    # Average confidence
    avg_confidence = all_probs.max(dim=-1)[0].mean().item()

    metrics = {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'accuracy': accuracy,
        'f1': f1,
        'avg_confidence': avg_confidence,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }

    return metrics


def print_clinical_metrics(metrics: Dict[str, float]):
    """Pretty print clinical metrics."""
    print("\n" + "=" * 60)
    print("Clinical Evaluation Metrics")
    print("=" * 60)
    print(f"Sensitivity (Recall):  {metrics['sensitivity']:.4f}")
    print(f"Specificity:           {metrics['specificity']:.4f}")
    print(f"PPV (Precision):       {metrics['ppv']:.4f}")
    print(f"NPV:                   {metrics['npv']:.4f}")
    print(f"Accuracy:              {metrics['accuracy']:.4f}")
    print(f"F1 Score:              {metrics['f1']:.4f}")
    print(f"Average Confidence:    {metrics['avg_confidence']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['tp']:5d}  FN: {metrics['fn']:5d}")
    print(f"  FP: {metrics['fp']:5d}  TN: {metrics['tn']:5d}")
    print("=" * 60)
