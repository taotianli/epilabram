"""
Utility functions for working with extended EpiLaBraM features.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count model parameters.

    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_parameter_summary(model: nn.Module, detailed: bool = False):
    """
    Print parameter summary for the model.

    Args:
        model: EpiLaBraMExtended model
        detailed: If True, print per-component breakdown
    """
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)

    print("=" * 60)
    print("Parameter Summary")
    print("=" * 60)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters: {total - trainable:,}")
    print(f"Trainable ratio: {trainable / total * 100:.2f}%")

    if detailed:
        print("\nComponent Breakdown:")
        print("-" * 60)

        # Backbone
        backbone_total = count_parameters(model.backbone, trainable_only=False)
        backbone_trainable = count_parameters(model.backbone, trainable_only=True)
        print(f"Backbone: {backbone_total:,} total, {backbone_trainable:,} trainable")

        # LoRA (if present)
        lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora' in n.lower())
        if lora_params > 0:
            print(f"  LoRA adapters: {lora_params:,}")

        # Tokenizer
        tokenizer_params = count_parameters(model.tokenizer, trainable_only=False)
        print(f"Tokenizer: {tokenizer_params:,}")

        # Task prompts
        prompt_params = count_parameters(model.task_prompts, trainable_only=False)
        print(f"Task prompts: {prompt_params:,}")

        # Adapters
        adapter_params = sum(count_parameters(a, trainable_only=False) for a in model.adapters)
        print(f"Adapters: {adapter_params:,}")

        # Heads
        head_params = sum(count_parameters(h, trainable_only=False) for h in model.heads.values())
        print(f"Prediction heads: {head_params:,}")

        # Temporal transformer (if present)
        if model.temporal_transformer is not None:
            temporal_params = count_parameters(model.temporal_transformer, trainable_only=False)
            print(f"Temporal transformer: {temporal_params:,}")

    print("=" * 60)


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract only LoRA parameters from model state dict.

    Args:
        model: Model with LoRA adapters

    Returns:
        State dict containing only LoRA parameters
    """
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            lora_state[name] = param.data.clone()
    return lora_state


def load_lora_state_dict(model: nn.Module, lora_state: Dict[str, torch.Tensor]):
    """
    Load LoRA parameters into model.

    Args:
        model: Model with LoRA adapters
        lora_state: State dict containing LoRA parameters
    """
    model_state = model.state_dict()
    for name, param in lora_state.items():
        if name in model_state:
            model_state[name].copy_(param)
        else:
            print(f"Warning: LoRA parameter {name} not found in model")


def merge_lora_to_backbone(model, save_path: Optional[str] = None):
    """
    Merge LoRA weights into backbone and optionally save.

    Args:
        model: EpiLaBraMExtended model with LoRA
        save_path: Optional path to save merged backbone

    Returns:
        Merged backbone state dict
    """
    from models.lora import merge_lora_weights

    print("Merging LoRA weights into backbone...")
    merge_lora_weights(model.backbone)

    backbone_state = model.backbone.state_dict()

    if save_path:
        torch.save({'model': backbone_state}, save_path)
        print(f"Saved merged backbone to {save_path}")

    return backbone_state


def extract_cls_embeddings_batch(
    model,
    eeg_batch: torch.Tensor,
    task_id: Optional[torch.Tensor] = None,
    use_prompts: bool = False,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Extract CLS embeddings for a batch of EEG data.

    Args:
        model: EpiLaBraMExtended model
        eeg_batch: (B, N, A, T) EEG data
        task_id: (B,) task IDs
        use_prompts: Whether to use task prompts
        device: Device to use

    Returns:
        cls_embeddings: (B, D)
    """
    model.eval()
    eeg_batch = eeg_batch.to(device)
    if task_id is not None:
        task_id = task_id.to(device)

    with torch.no_grad():
        cls_emb = model.extract_cls_embeddings(
            eeg_batch,
            task_id=task_id,
            use_prompts=use_prompts
        )

    return cls_emb


def compute_sequence_embeddings(
    model,
    eeg_sequence: torch.Tensor,
    task_id: Optional[torch.Tensor] = None,
    use_prompts: bool = False,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Compute embeddings for a sequence of EEG epochs.

    Args:
        model: EpiLaBraMExtended model
        eeg_sequence: (B, T_seq, N, A, T) EEG sequence
        task_id: (B,) task IDs
        use_prompts: Whether to use task prompts
        device: Device to use

    Returns:
        sequence_embeddings: (B, T_seq, D)
    """
    model.eval()
    B, T_seq, N, A, T = eeg_sequence.shape

    # Flatten sequence
    eeg_flat = eeg_sequence.view(B * T_seq, N, A, T).to(device)

    # Expand task_id if provided
    if task_id is not None:
        task_id = task_id.to(device)
        task_id_expanded = task_id.unsqueeze(1).expand(-1, T_seq).reshape(-1)
    else:
        task_id_expanded = None

    # Extract embeddings
    with torch.no_grad():
        embeddings = model.extract_cls_embeddings(
            eeg_flat,
            task_id=task_id_expanded,
            use_prompts=use_prompts
        )

    # Reshape back to sequence
    embeddings = embeddings.view(B, T_seq, -1)

    return embeddings


def prepare_icl_batch(
    demo_eeg: List[torch.Tensor],
    demo_labels: List[int],
    query_eeg: List[torch.Tensor],
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare a batch for ICL inference.

    Args:
        demo_eeg: List of demonstration EEG tensors (N, A, T)
        demo_labels: List of demonstration labels
        query_eeg: List of query EEG tensors (N, A, T)
        device: Device to use

    Returns:
        demo_eeg_batch: (1, n_demo, N, A, T)
        demo_labels_batch: (1, n_demo)
        query_eeg_batch: (1, n_query, N, A, T)
    """
    demo_eeg_batch = torch.stack(demo_eeg).unsqueeze(0).to(device)
    demo_labels_batch = torch.tensor(demo_labels).unsqueeze(0).to(device)
    query_eeg_batch = torch.stack(query_eeg).unsqueeze(0).to(device)

    return demo_eeg_batch, demo_labels_batch, query_eeg_batch


def icl_predict(
    model,
    demo_eeg: List[torch.Tensor],
    demo_labels: List[int],
    query_eeg: List[torch.Tensor],
    device: str = 'cuda',
    return_probs: bool = False
) -> torch.Tensor:
    """
    Perform ICL prediction for query examples.

    Args:
        model: EpiLaBraMExtended model
        demo_eeg: List of demonstration EEG tensors
        demo_labels: List of demonstration labels
        query_eeg: List of query EEG tensors
        device: Device to use
        return_probs: If True, return probabilities instead of predictions

    Returns:
        predictions: (n_query,) or (n_query, num_classes) if return_probs
    """
    model.eval()

    # Prepare batch
    demo_batch, labels_batch, query_batch = prepare_icl_batch(
        demo_eeg, demo_labels, query_eeg, device
    )

    # ICL forward
    with torch.no_grad():
        logits = model.forward_icl(demo_batch, labels_batch, query_batch)

    # Get predictions or probabilities
    logits = logits.squeeze(0)  # (n_query, num_classes)

    if return_probs:
        return torch.softmax(logits, dim=-1)
    else:
        return logits.argmax(dim=-1)


def estimate_memory_usage(
    model,
    batch_size: int,
    seq_length: int = 1,
    n_channels: int = 23,
    time_patches: int = 4,
    patch_size: int = 200,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Estimate memory usage for model inference.

    Args:
        model: EpiLaBraMExtended model
        batch_size: Batch size
        seq_length: Sequence length (for temporal modeling)
        n_channels: Number of EEG channels
        time_patches: Number of time patches
        patch_size: Patch size
        device: Device to use

    Returns:
        Dictionary with memory estimates in MB
    """
    model = model.to(device)
    model.eval()

    # Model parameters
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

    # Input data
    if seq_length > 1:
        input_shape = (batch_size, seq_length, n_channels, time_patches, patch_size)
    else:
        input_shape = (batch_size, n_channels, time_patches, patch_size)

    input_memory = (batch_size * seq_length * n_channels * time_patches * patch_size * 4) / 1024**2

    # Estimate activation memory (rough approximation)
    # Typically 2-3x model parameters for activations during forward pass
    activation_memory = param_memory * 2.5

    total_memory = param_memory + input_memory + activation_memory

    return {
        'parameters_mb': param_memory,
        'input_mb': input_memory,
        'activations_mb': activation_memory,
        'total_mb': total_memory,
        'total_gb': total_memory / 1024,
    }


def benchmark_inference_speed(
    model,
    batch_size: int,
    seq_length: int = 1,
    n_channels: int = 23,
    time_patches: int = 4,
    patch_size: int = 200,
    num_iterations: int = 100,
    device: str = 'cuda',
    use_temporal: bool = False
) -> Dict[str, float]:
    """
    Benchmark model inference speed.

    Args:
        model: EpiLaBraMExtended model
        batch_size: Batch size
        seq_length: Sequence length
        n_channels: Number of channels
        time_patches: Number of time patches
        patch_size: Patch size
        num_iterations: Number of iterations for benchmarking
        device: Device to use
        use_temporal: Whether to use temporal transformer

    Returns:
        Dictionary with timing statistics
    """
    import time

    model = model.to(device)
    model.eval()

    # Prepare dummy input
    if use_temporal:
        dummy_input = torch.randn(
            batch_size, seq_length, n_channels, time_patches, patch_size
        ).to(device)
    else:
        dummy_input = torch.randn(
            batch_size, n_channels, time_patches, patch_size
        ).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            if use_temporal:
                _ = model.forward_temporal(dummy_input)
            else:
                task_id = torch.zeros(batch_size, dtype=torch.long).to(device)
                _ = model.forward_stage2(dummy_input, task_id)

    # Benchmark
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            if use_temporal:
                _ = model.forward_temporal(dummy_input)
            else:
                task_id = torch.zeros(batch_size, dtype=torch.long).to(device)
                _ = model.forward_stage2(dummy_input, task_id)

    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    throughput = batch_size / avg_time

    return {
        'total_time_s': total_time,
        'avg_time_ms': avg_time * 1000,
        'throughput_samples_per_s': throughput,
    }


def visualize_attention_maps(
    model,
    eeg: torch.Tensor,
    layer_idx: int = -1,
    head_idx: int = 0,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Extract attention maps from a specific layer and head.

    Args:
        model: EpiLaBraMExtended model
        eeg: (1, N, A, T) EEG data
        layer_idx: Layer index (-1 for last layer)
        head_idx: Attention head index
        device: Device to use

    Returns:
        attention_map: (N*A+1, N*A+1) attention weights
    """
    model = model.to(device)
    model.eval()
    eeg = eeg.to(device)

    # Get the target block
    block = model.backbone.blocks[layer_idx]

    # Forward through backbone up to target layer
    B, N, A, T = eeg.shape
    x = model.backbone.patch_embed(eeg)
    x = model.backbone.patch_proj(x)
    cls = model.backbone.cls_token.expand(B, -1, -1)
    x = torch.cat([cls, x], dim=1)

    if not model.backbone.use_rope:
        pos = model.backbone.spatial_embed(N, A, batch_size=B)
        x = x + pos
        te = model.backbone.temporal_embed(x, N, A)
        x[:, 1:] = x[:, 1:] + te

    x = model.backbone.pos_drop(x)

    # Forward through blocks up to target
    for i, blk in enumerate(model.backbone.blocks):
        if i < layer_idx:
            x = blk(x)
        elif i == layer_idx:
            # Get attention from this block
            with torch.no_grad():
                attn = blk(model.backbone.norm(x), return_attention=True)
            break

    # Extract specific head
    attn_map = attn[0, head_idx].cpu()  # (N*A+1, N*A+1)

    return attn_map
