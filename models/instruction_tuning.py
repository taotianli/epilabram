"""
Instruction Tuning for EEG Foundation Models.
Enables the model to understand task descriptions and perform zero-shot inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModel


class InstructionEncoder(nn.Module):
    """
    Encodes task instructions using a pretrained language model.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        freeze: bool = True,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)

        if freeze:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        self.hidden_size = self.text_encoder.config.hidden_size

    def forward(self, instructions: List[str], device: str = 'cuda') -> torch.Tensor:
        """
        Encode instruction strings.

        Args:
            instructions: List of instruction strings
            device: Device to use

        Returns:
            instruction_embeddings: (B, hidden_size)
        """
        # Tokenize
        encoded = self.tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(device)

        # Encode
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.text_encoder(**encoded)
            # Use [CLS] token or mean pooling
            instruction_emb = outputs.last_hidden_state[:, 0, :]  # (B, hidden_size)

        return instruction_emb


class InstructionFusion(nn.Module):
    """
    Fuses EEG features with instruction embeddings using cross-attention.
    """

    def __init__(
        self,
        eeg_dim: int,
        text_dim: int,
        fusion_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.eeg_dim = eeg_dim
        self.text_dim = text_dim
        self.fusion_dim = fusion_dim

        # Project to fusion space
        self.eeg_proj = nn.Linear(eeg_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(fusion_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(fusion_dim) for _ in range(num_layers)
        ])

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(fusion_dim)

    def forward(
        self,
        eeg_features: torch.Tensor,
        instruction_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse EEG features with instruction embeddings.

        Args:
            eeg_features: (B, N, eeg_dim) or (B, eeg_dim)
            instruction_emb: (B, text_dim)

        Returns:
            fused_features: (B, fusion_dim)
        """
        # Handle both sequence and single vector inputs
        if eeg_features.dim() == 2:
            eeg_features = eeg_features.unsqueeze(1)  # (B, 1, eeg_dim)

        B = eeg_features.size(0)

        # Project
        eeg_proj = self.eeg_proj(eeg_features)  # (B, N, fusion_dim)
        text_proj = self.text_proj(instruction_emb).unsqueeze(1)  # (B, 1, fusion_dim)

        # Cross-attention: EEG attends to instruction
        x = eeg_proj
        for attn, norm in zip(self.cross_attn_layers, self.norms):
            # Query: EEG, Key/Value: instruction
            attn_out, _ = attn(x, text_proj, text_proj)
            x = norm(x + attn_out)

        # FFN
        x = self.ffn_norm(x + self.ffn(x))

        # Pool to single vector
        fused = x.mean(dim=1)  # (B, fusion_dim)

        return fused


class InstructionTunedEEGModel(nn.Module):
    """
    Complete instruction-tuned EEG model.

    Supports:
    - Task-specific instructions
    - Zero-shot task generalization
    - Multi-task learning with natural language task descriptions
    """

    def __init__(
        self,
        backbone: nn.Module,
        instruction_encoder: Optional[InstructionEncoder] = None,
        fusion_dim: int = 512,
        num_heads: int = 8,
        num_fusion_layers: int = 2,
        max_num_classes: int = 10,
    ):
        super().__init__()
        self.backbone = backbone
        self.eeg_dim = backbone.embed_dim

        # Instruction encoder
        if instruction_encoder is None:
            instruction_encoder = InstructionEncoder()
        self.instruction_encoder = instruction_encoder
        self.text_dim = instruction_encoder.hidden_size

        # Fusion module
        self.fusion = InstructionFusion(
            eeg_dim=self.eeg_dim,
            text_dim=self.text_dim,
            fusion_dim=fusion_dim,
            num_heads=num_heads,
            num_layers=num_fusion_layers,
        )

        # Task-agnostic output heads
        self.binary_head = nn.Linear(fusion_dim, 2)
        self.multiclass_head = nn.Linear(fusion_dim, max_num_classes)
        self.regression_head = nn.Linear(fusion_dim, 1)

    def forward(
        self,
        eeg: torch.Tensor,
        instructions: List[str],
        task_type: str = 'binary',
        num_classes: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass with instruction conditioning.

        Args:
            eeg: (B, N, A, T) EEG data
            instructions: List of instruction strings (length B)
            task_type: 'binary', 'multiclass', or 'regression'
            num_classes: Number of classes for multiclass tasks

        Returns:
            logits: (B, num_classes) or (B, 1) for regression
        """
        # Extract EEG features
        eeg_features = self.backbone(eeg)  # (B, eeg_dim)

        # Encode instructions
        instruction_emb = self.instruction_encoder(instructions, device=eeg.device)

        # Fuse
        fused_features = self.fusion(eeg_features, instruction_emb)

        # Task-specific output
        if task_type == 'binary':
            logits = self.binary_head(fused_features)
        elif task_type == 'multiclass':
            logits = self.multiclass_head(fused_features)
            if num_classes is not None:
                logits = logits[:, :num_classes]
        elif task_type == 'regression':
            logits = self.regression_head(fused_features)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        return logits

    def extract_features(
        self,
        eeg: torch.Tensor,
        instructions: List[str],
    ) -> torch.Tensor:
        """Extract instruction-conditioned features without classification."""
        eeg_features = self.backbone(eeg)
        instruction_emb = self.instruction_encoder(instructions, device=eeg.device)
        fused_features = self.fusion(eeg_features, instruction_emb)
        return fused_features


# Predefined instruction templates
INSTRUCTION_TEMPLATES = {
    # Seizure detection
    'seizure_detection': "Analyze this EEG segment and determine if it contains epileptic seizure activity. Look for high-amplitude spikes, sharp waves, or rhythmic discharges.",

    'seizure_type': "Classify the type of seizure in this EEG recording. Consider patterns such as spike-wave complexes, focal discharges, or generalized rhythmic activity.",

    # Sleep staging
    'sleep_staging': "Classify the sleep stage of this 30-second EEG epoch. Consider alpha waves (wake), sleep spindles and K-complexes (N2), delta waves (N3), and rapid eye movements (REM).",

    'sleep_quality': "Assess the sleep quality based on this EEG recording. Look for sleep fragmentation, abnormal sleep architecture, or sleep disorders.",

    # Artifact detection
    'artifact_detection': "Identify if this EEG segment contains artifacts. Check for eye blinks, muscle activity, electrode noise, or movement artifacts.",

    'artifact_type': "Classify the type of artifact in this EEG: eye movement, muscle artifact, electrode artifact, or physiological artifact.",

    # Cognitive states
    'attention_level': "Determine the attention level from this EEG recording. High attention shows increased beta activity and reduced alpha in posterior regions.",

    'cognitive_load': "Assess the cognitive workload based on this EEG. High cognitive load is associated with increased theta and decreased alpha power.",

    'emotion_recognition': "Identify the emotional state from this EEG recording. Consider frontal asymmetry, theta/alpha ratios, and beta activity patterns.",

    # Clinical conditions
    'abnormality_detection': "Detect any abnormal patterns in this EEG that may indicate neurological conditions. Look for asymmetries, focal slowing, or epileptiform discharges.",

    'brain_state': "Classify the overall brain state from this EEG: normal, drowsy, pathological, or altered consciousness.",

    # Research tasks
    'motor_imagery': "Detect motor imagery from this EEG. Look for event-related desynchronization in sensorimotor cortex (mu and beta bands).",

    'p300_detection': "Identify P300 event-related potential in this EEG segment. Look for positive deflection around 300ms post-stimulus in parietal regions.",

    # Custom template
    'custom': "{custom_instruction}",
}


def get_instruction(task_name: str, custom_text: Optional[str] = None) -> str:
    """
    Get instruction text for a task.

    Args:
        task_name: Name of the task
        custom_text: Custom instruction text (for 'custom' task)

    Returns:
        instruction: Instruction string
    """
    if task_name not in INSTRUCTION_TEMPLATES:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(INSTRUCTION_TEMPLATES.keys())}")

    instruction = INSTRUCTION_TEMPLATES[task_name]

    if task_name == 'custom' and custom_text is not None:
        instruction = instruction.format(custom_instruction=custom_text)

    return instruction


def build_instruction_tuned_model(
    backbone: nn.Module,
    fusion_dim: int = 512,
    num_heads: int = 8,
    num_fusion_layers: int = 2,
    text_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    freeze_text_encoder: bool = True,
) -> InstructionTunedEEGModel:
    """
    Factory function to build instruction-tuned model.

    Args:
        backbone: Pretrained EEG backbone (e.g., LaBraM)
        fusion_dim: Dimension of fusion space
        num_heads: Number of attention heads
        num_fusion_layers: Number of cross-attention layers
        text_encoder_name: Pretrained text encoder name
        freeze_text_encoder: Whether to freeze text encoder

    Returns:
        model: InstructionTunedEEGModel
    """
    instruction_encoder = InstructionEncoder(
        model_name=text_encoder_name,
        freeze=freeze_text_encoder,
    )

    model = InstructionTunedEEGModel(
        backbone=backbone,
        instruction_encoder=instruction_encoder,
        fusion_dim=fusion_dim,
        num_heads=num_heads,
        num_fusion_layers=num_fusion_layers,
    )

    return model


class InstructionDataset(torch.utils.data.Dataset):
    """
    Dataset for instruction tuning.
    Each sample includes EEG data, instruction text, and label.
    """

    def __init__(
        self,
        eeg_data: List[torch.Tensor],
        labels: List[int],
        task_names: List[str],
        custom_instructions: Optional[List[str]] = None,
    ):
        self.eeg_data = eeg_data
        self.labels = labels
        self.task_names = task_names
        self.custom_instructions = custom_instructions or [None] * len(eeg_data)

        assert len(eeg_data) == len(labels) == len(task_names)

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]
        label = self.labels[idx]
        task_name = self.task_names[idx]
        custom_text = self.custom_instructions[idx]

        # Get instruction
        instruction = get_instruction(task_name, custom_text)

        return {
            'eeg': eeg,
            'label': label,
            'instruction': instruction,
            'task_name': task_name,
        }


def collate_instruction_batch(batch):
    """Collate function for instruction tuning dataloader."""
    eeg = torch.stack([item['eeg'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    instructions = [item['instruction'] for item in batch]
    task_names = [item['task_name'] for item in batch]

    return {
        'eeg': eeg,
        'labels': labels,
        'instructions': instructions,
        'task_names': task_names,
    }
