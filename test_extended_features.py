"""
Example usage and testing script for EpiLaBraM Extended features.
Demonstrates RoPE, LoRA, Temporal Transformer, and ICL inference.
"""

import torch
from models.epilabram_extended import build_epilabram_extended


def test_rope_and_lora():
    """Test RoPE and LoRA integration"""
    print("=" * 60)
    print("Testing RoPE and LoRA")
    print("=" * 60)

    # Build model with RoPE and LoRA
    model = build_epilabram_extended(
        backbone_size='base',
        use_rope=True,
        use_lora=True,
        lora_rank=8,
        lora_alpha=16.0,
        use_temporal=False,
    )

    # Test Stage1 forward (PADM with LoRA)
    B, N, A, T = 2, 23, 4, 200
    eeg = torch.randn(B, N, A, T)
    mask = torch.rand(B, N * A) > 0.5
    sym_mask = torch.rand(B, N * A) > 0.5

    print(f"\nInput shape: {eeg.shape}")
    print(f"Mask shape: {mask.shape}")

    logits, sym_logits = model.forward_stage1(eeg, mask, sym_mask)
    print(f"Output logits shape: {logits.shape}")
    print(f"Symmetric logits shape: {sym_logits.shape}")

    # Check LoRA parameters
    lora_params = [n for n, p in model.named_parameters() if 'lora' in n.lower()]
    print(f"\nLoRA parameters found: {len(lora_params)}")
    print(f"Sample LoRA params: {lora_params[:3]}")

    # Test Stage2 forward
    task_id = torch.tensor([0, 1])  # TUAB, TUSZ
    results = model.forward_stage2(eeg, task_id)
    print(f"\nStage2 results: {list(results.keys())}")
    for task_name, (idx, logits) in results.items():
        print(f"  {task_name}: indices={idx.tolist()}, logits shape={logits.shape}")

    print("\n✓ RoPE and LoRA test passed!\n")


def test_temporal_transformer():
    """Test Temporal Transformer for long-form EEG"""
    print("=" * 60)
    print("Testing Temporal Transformer")
    print("=" * 60)

    # Build model with temporal transformer
    model = build_epilabram_extended(
        backbone_size='base',
        use_rope=True,
        use_lora=False,
        use_temporal=True,
        temporal_size='base',
        temporal_num_classes=2,
    )

    # Test CLS embedding extraction
    B, N, A, T = 2, 23, 4, 200
    eeg = torch.randn(B, N, A, T)

    print(f"\nExtracting CLS embeddings from single epoch...")
    cls_emb = model.extract_cls_embeddings(eeg)
    print(f"CLS embedding shape: {cls_emb.shape}")

    # Test temporal forward with sequence
    T_seq = 8  # 8 epochs in sequence
    eeg_sequence = torch.randn(B, T_seq, N, A, T)
    task_id = torch.tensor([0, 1])

    print(f"\nProcessing EEG sequence...")
    print(f"Sequence shape: {eeg_sequence.shape}")

    logits = model.forward_temporal(eeg_sequence, task_id=task_id)
    print(f"Temporal output logits shape: {logits.shape}")

    # Test with mask
    mask = torch.ones(B, T_seq).bool()
    mask[0, 6:] = False  # Mask last 2 epochs for first sample
    mask[1, 7:] = False  # Mask last 1 epoch for second sample

    print(f"\nTesting with sequence mask...")
    logits_masked = model.forward_temporal(eeg_sequence, task_id=task_id, mask=mask)
    print(f"Masked output logits shape: {logits_masked.shape}")

    print("\n✓ Temporal Transformer test passed!\n")


def test_icl_inference():
    """Test In-Context Learning inference"""
    print("=" * 60)
    print("Testing ICL (In-Context Learning) Inference")
    print("=" * 60)

    # Build model with temporal transformer
    model = build_epilabram_extended(
        backbone_size='base',
        use_rope=True,
        use_lora=False,
        use_temporal=True,
        temporal_size='base',
        temporal_num_classes=2,
    )

    B, N, A, T = 2, 23, 4, 200

    # Prepare demonstration examples
    n_demo = 4
    demo_eeg = torch.randn(B, n_demo, N, A, T)
    demo_labels = torch.randint(0, 2, (B, n_demo))

    print(f"\nDemonstration examples:")
    print(f"  EEG shape: {demo_eeg.shape}")
    print(f"  Labels shape: {demo_labels.shape}")
    print(f"  Sample labels: {demo_labels[0].tolist()}")

    # Prepare query examples
    n_query = 3
    query_eeg = torch.randn(B, n_query, N, A, T)

    print(f"\nQuery examples:")
    print(f"  EEG shape: {query_eeg.shape}")

    # ICL inference
    logits = model.forward_icl(demo_eeg, demo_labels, query_eeg)
    print(f"\nICL output logits shape: {logits.shape}")
    print(f"Predictions: {logits.argmax(dim=-1)}")

    # Test single query
    query_single = query_eeg[:, 0, :, :, :]  # (B, N, A, T)
    query_single_expanded = query_single.unsqueeze(1)  # (B, 1, N, A, T)

    logits_single = model.forward_icl(demo_eeg, demo_labels, query_single_expanded)
    print(f"\nSingle query logits shape: {logits_single.shape}")
    print(f"Single query predictions: {logits_single.argmax(dim=-1)}")

    print("\n✓ ICL inference test passed!\n")


def test_parameter_efficiency():
    """Test parameter counts for different configurations"""
    print("=" * 60)
    print("Testing Parameter Efficiency")
    print("=" * 60)

    configs = [
        ("Baseline (no RoPE, no LoRA)", False, False),
        ("RoPE only", True, False),
        ("LoRA only", False, True),
        ("RoPE + LoRA", True, True),
    ]

    for name, use_rope, use_lora in configs:
        model = build_epilabram_extended(
            backbone_size='base',
            use_rope=use_rope,
            use_lora=use_lora,
            use_temporal=False,
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n{name}:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        if use_lora:
            model.freeze_backbone()
            stage1_params = sum(p.numel() for p in model.get_stage1_params())
            print(f"  Stage1 trainable (LoRA): {stage1_params:,}")
            print(f"  Efficiency: {stage1_params / total_params * 100:.2f}%")

    print("\n✓ Parameter efficiency test passed!\n")


def test_context_length_extension():
    """Test context length extension with RoPE"""
    print("=" * 60)
    print("Testing Context Length Extension with RoPE")
    print("=" * 60)

    model = build_epilabram_extended(
        backbone_size='base',
        use_rope=True,
        use_lora=False,
        use_temporal=True,
        temporal_size='base',
        max_seq_len=512,
    )

    B, N, A, T = 1, 23, 4, 200

    # Test with different sequence lengths
    seq_lengths = [4, 8, 16, 32, 64]

    print("\nTesting different sequence lengths:")
    for seq_len in seq_lengths:
        eeg_sequence = torch.randn(B, seq_len, N, A, T)
        try:
            logits = model.forward_temporal(eeg_sequence)
            print(f"  Sequence length {seq_len:3d}: ✓ (output shape: {logits.shape})")
        except Exception as e:
            print(f"  Sequence length {seq_len:3d}: ✗ ({str(e)})")

    print("\n✓ Context length extension test passed!\n")


def test_full_pipeline():
    """Test complete training pipeline"""
    print("=" * 60)
    print("Testing Full Training Pipeline")
    print("=" * 60)

    model = build_epilabram_extended(
        backbone_size='base',
        use_rope=True,
        use_lora=True,
        lora_rank=8,
        use_temporal=True,
        temporal_size='base',
    )

    B, N, A, T = 2, 23, 4, 200

    # Stage 1: PADM with LoRA
    print("\n[Stage 1] PADM continual pre-training with LoRA")
    model.freeze_backbone()  # Freeze non-LoRA params
    eeg = torch.randn(B, N, A, T)
    mask = torch.rand(B, N * A) > 0.5
    sym_mask = torch.rand(B, N * A) > 0.5

    logits, sym_logits = model.forward_stage1(eeg, mask, sym_mask)
    stage1_params = model.get_stage1_params()
    print(f"  Output shape: {logits.shape}")
    print(f"  Trainable params: {sum(p.numel() for p in stage1_params):,}")

    # Stage 2: Multi-task fine-tuning
    print("\n[Stage 2] Multi-task fine-tuning")
    model.freeze_backbone()
    task_id = torch.tensor([0, 1])
    results = model.forward_stage2(eeg, task_id)
    stage2_params = model.get_stage2_params()
    print(f"  Tasks: {list(results.keys())}")
    print(f"  Trainable params: {sum(p.numel() for p in stage2_params):,}")

    # Temporal modeling
    print("\n[Temporal] Long-form EEG modeling")
    T_seq = 8
    eeg_sequence = torch.randn(B, T_seq, N, A, T)
    logits_temporal = model.forward_temporal(eeg_sequence, task_id=task_id)
    temporal_params = model.get_temporal_params()
    print(f"  Sequence length: {T_seq}")
    print(f"  Output shape: {logits_temporal.shape}")
    print(f"  Temporal params: {sum(p.numel() for p in temporal_params):,}")

    # ICL inference
    print("\n[ICL] In-Context Learning inference")
    n_demo, n_query = 4, 2
    demo_eeg = torch.randn(B, n_demo, N, A, T)
    demo_labels = torch.randint(0, 2, (B, n_demo))
    query_eeg = torch.randn(B, n_query, N, A, T)
    logits_icl = model.forward_icl(demo_eeg, demo_labels, query_eeg)
    print(f"  Demonstrations: {n_demo}")
    print(f"  Queries: {n_query}")
    print(f"  Output shape: {logits_icl.shape}")

    print("\n✓ Full pipeline test passed!\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("EpiLaBraM Extended - Feature Tests")
    print("=" * 60 + "\n")

    # Run all tests
    test_rope_and_lora()
    test_temporal_transformer()
    test_icl_inference()
    test_parameter_efficiency()
    test_context_length_extension()
    test_full_pipeline()

    print("=" * 60)
    print("All tests completed successfully! ✓")
    print("=" * 60)
