# EpiLaBraM Extended Features

This document describes the new features added to EpiLaBraM for enhanced long-form EEG modeling and parameter-efficient training.

## New Features

### 1. RoPE (Rotary Position Embedding)

**Location**: `models/rope.py`

Replaces absolute positional encoding (APE) with Rotary Position Embedding for better context length extension.

**Benefits**:
- Enables processing of longer EEG sequences without retraining
- More efficient for variable-length inputs
- Better extrapolation to unseen sequence lengths

**Usage**:
```python
from models.epilabram_extended import build_epilabram_extended

model = build_epilabram_extended(
    backbone_size='base',
    use_rope=True,  # Enable RoPE
    max_seq_len=2048  # Maximum sequence length
)
```

### 2. LoRA (Low-Rank Adaptation)

**Location**: `models/lora.py`

Adds trainable low-rank matrices to attention layers for parameter-efficient continual pre-training.

**Benefits**:
- Reduces trainable parameters by ~90% during Stage 1
- Faster training and lower memory usage
- Preserves pretrained knowledge while adapting to new data

**Usage**:
```python
model = build_epilabram_extended(
    backbone_size='base',
    use_lora=True,  # Enable LoRA
    lora_rank=8,    # LoRA rank (lower = fewer parameters)
    lora_alpha=16.0  # LoRA scaling factor
)

# For Stage 1 training with LoRA
model.freeze_backbone()  # Freeze non-LoRA parameters
stage1_params = model.get_stage1_params()  # Get LoRA + lm_head params
optimizer = torch.optim.AdamW(stage1_params, lr=1e-4)
```

**LoRA Configuration**:
- `rank=8`: ~1-2% of original parameters (recommended for most cases)
- `rank=16`: ~2-4% of original parameters (for more capacity)
- `alpha=16.0`: Standard scaling factor (typically 2×rank)

### 3. Temporal Transformer

**Location**: `models/temporal_transformer.py`

Second-level Transformer that operates on sequences of CLS embeddings from LaBraM backbone to handle long-form EEG.

**Benefits**:
- Models temporal dependencies across multiple EEG epochs
- Handles variable-length sequences
- Supports attention masking for padded sequences

**Architecture**:
- Input: Sequence of CLS embeddings from backbone
- Processing: Multi-layer Transformer with RoPE
- Output: Sequence-level predictions

**Usage**:
```python
model = build_epilabram_extended(
    backbone_size='base',
    use_temporal=True,  # Enable temporal transformer
    temporal_size='base',  # 'small', 'base', or 'large'
    temporal_num_classes=2
)

# Process long-form EEG sequence
B, T_seq, N, A, T = 2, 16, 23, 4, 200
eeg_sequence = torch.randn(B, T_seq, N, A, T)  # 16 epochs
task_id = torch.tensor([0, 1])

logits = model.forward_temporal(eeg_sequence, task_id=task_id)
# Output: (B, num_classes)

# With sequence masking (for variable lengths)
mask = torch.ones(B, T_seq).bool()
mask[0, 12:] = False  # First sample has 12 valid epochs
mask[1, 14:] = False  # Second sample has 14 valid epochs

logits = model.forward_temporal(eeg_sequence, task_id=task_id, mask=mask)
```

### 4. ICL (In-Context Learning) Inference

**Location**: `models/temporal_transformer.py` (integrated in `EpiLaBraMExtended`)

Enables few-shot learning by providing demonstration (EEG, label) pairs followed by query EEG.

**Benefits**:
- Few-shot adaptation without gradient updates
- Rapid adaptation to new tasks or domains
- Useful for personalized medicine scenarios

**Usage**:
```python
# Prepare demonstrations (labeled examples)
n_demo = 4
demo_eeg = torch.randn(B, n_demo, N, A, T)
demo_labels = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])  # (B, n_demo)

# Prepare queries (unlabeled examples to predict)
n_query = 2
query_eeg = torch.randn(B, n_query, N, A, T)

# ICL inference
logits = model.forward_icl(demo_eeg, demo_labels, query_eeg)
# Output: (B, n_query, num_classes)

predictions = logits.argmax(dim=-1)
print(f"Query predictions: {predictions}")
```

**ICL Format**:
```
Input sequence: [demo_1 + label_1, demo_2 + label_2, ..., query_1, query_2, ...]
                 ↑ demonstrations with labels ↑    ↑ queries to predict ↑
```

## Complete Example

```python
import torch
from models.epilabram_extended import build_epilabram_extended

# Build model with all features
model = build_epilabram_extended(
    backbone_size='base',
    use_rope=True,           # RoPE for context extension
    use_lora=True,           # LoRA for efficient training
    lora_rank=8,
    lora_alpha=16.0,
    use_temporal=True,       # Temporal transformer
    temporal_size='base',
    temporal_num_classes=2,
    max_seq_len=512
)

# Stage 1: PADM continual pre-training with LoRA
model.freeze_backbone()
eeg = torch.randn(2, 23, 4, 200)
mask = torch.rand(2, 92) > 0.5
sym_mask = torch.rand(2, 92) > 0.5

logits, sym_logits = model.forward_stage1(eeg, mask, sym_mask)

# Stage 2: Multi-task fine-tuning
task_id = torch.tensor([0, 1])
results = model.forward_stage2(eeg, task_id)

# Temporal modeling for long-form EEG
eeg_sequence = torch.randn(2, 16, 23, 4, 200)  # 16 epochs
logits_temporal = model.forward_temporal(eeg_sequence, task_id=task_id)

# ICL inference
demo_eeg = torch.randn(2, 4, 23, 4, 200)
demo_labels = torch.randint(0, 2, (2, 4))
query_eeg = torch.randn(2, 2, 23, 4, 200)
logits_icl = model.forward_icl(demo_eeg, demo_labels, query_eeg)
```

## Training Recommendations

### Stage 1: PADM with LoRA
```python
# Freeze backbone except LoRA
model.freeze_backbone()
optimizer = torch.optim.AdamW(model.get_stage1_params(), lr=1e-4)

# Training loop
for eeg, mask, sym_mask in dataloader:
    logits, sym_logits = model.forward_stage1(eeg, mask, sym_mask)
    loss = compute_padm_loss(logits, sym_logits, targets)
    loss.backward()
    optimizer.step()
```

### Stage 2: Multi-task Fine-tuning
```python
# Keep backbone frozen, train prompts + adapters + heads
model.freeze_backbone()
optimizer = torch.optim.AdamW(model.get_stage2_params(), lr=1e-3)

for eeg, task_id, labels in dataloader:
    results = model.forward_stage2(eeg, task_id)
    loss = compute_multitask_loss(results, labels)
    loss.backward()
    optimizer.step()
```

### Temporal Transformer Training
```python
# Train temporal transformer on top of frozen backbone
model.freeze_backbone()
optimizer = torch.optim.AdamW(model.get_temporal_params(), lr=1e-3)

for eeg_sequence, labels in dataloader:
    logits = model.forward_temporal(eeg_sequence)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
```

## Parameter Efficiency Comparison

| Configuration | Total Params | Trainable (Stage 1) | Efficiency |
|--------------|--------------|---------------------|------------|
| Baseline     | ~50M         | ~50M                | 100%       |
| LoRA (r=8)   | ~50M         | ~0.5M               | ~1%        |
| LoRA (r=16)  | ~50M         | ~1M                 | ~2%        |

## Testing

Run the test suite to verify all features:

```bash
python test_extended_features.py
```

Tests include:
- RoPE and LoRA integration
- Temporal Transformer forward pass
- ICL inference
- Parameter efficiency
- Context length extension
- Full training pipeline

## File Structure

```
models/
├── rope.py                    # RoPE implementation
├── lora.py                    # LoRA adapters
├── temporal_transformer.py    # Temporal Transformer + ICL
├── epilabram_extended.py      # Extended EpiLaBraM model
├── labram_backbone.py         # Modified backbone (RoPE support)
└── ...

test_extended_features.py      # Feature tests
README_EXTENDED.md             # This file
```

## References

- **RoPE**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- **ICL**: Brown et al., "Language Models are Few-Shot Learners" (2020)
