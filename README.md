# EpiLaBraM

Evaluation and fine-tuning of [LaBraM](https://github.com/935963004/LaBraM) on epilepsy-related downstream tasks from the TUH EEG Corpus.

## Setup

```bash
conda activate epilabram
```

## Data Preprocessing

```bash
# Preprocess raw EEG data to h5 format
python preprocess_tuh.py

# Build memmap cache for fast data loading (one-time, reused across all runs)
python build_memmap.py \
    --tasks TUAB TUSZ TUEV TUEP \
    --num_workers 32 \
    --out_dir /projects/u6da/tuh_processed/memmap
```

## Linear Probe (Frozen Backbone)

```bash
python eval_labram_baseline.py \
    --ckpt checkpoints/labram-base.pth \
    --tasks TUAB TUSZ TUEV TUEP \
    --epochs 10 \
    --batch_size 2048 \
    --num_workers 32 \
    --bf16 \
    --memmap_dir /projects/u6da/tuh_processed/memmap \
    --output_dir experiments/labram_baseline
```

## Full Fine-tuning

```bash
python eval_labram_baseline.py \
    --ckpt checkpoints/labram-base.pth \
    --tasks TUAB TUSZ TUEV TUEP \
    --finetune \
    --epochs 30 \
    --patience 5 \
    --warmup_epochs 2 \
    --lr 3e-5 \
    --batch_size 128 \
    --num_workers 32 \
    --bf16 \
    --memmap_dir /projects/u6da/tuh_processed/memmap \
    --output_dir experiments/labram_finetune
```

---

## Results

### Linear Probe — LaBraM-Base

> Checkpoint: `labram-base.pth` | Mode: linear probe | GPU: NVIDIA GH200 96GB

#### TUAB (Normal / Abnormal Classification)

> To be updated

#### TUSZ (Seizure Detection)

> To be updated

#### TUEV (6-class Event Classification)

> To be updated

#### TUEP (Epilepsy Diagnosis)

> To be updated

---

### Full Fine-tuning — LaBraM-Base

> Checkpoint: `labram-base.pth` | Mode: finetune | GPU: NVIDIA GH200 96GB | `experiments/labram_finetune_v3`

#### TUAB (Normal / Abnormal Classification)

| Metric | Value |
|---|---|
| Accuracy | 0.8212 |
| Balanced Accuracy | 0.8189 |
| Macro F1 | 0.8196 |
| Weighted F1 | 0.8209 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal (0) | 0.8243 | 0.8492 | 0.8366 | 19,907 |
| Abnormal (1) | 0.8174 | 0.7885 | 0.8027 | 17,038 |

Confusion Matrix:

|  | Pred Normal | Pred Abnormal |
|---|---|---|
| True Normal | 16,905 | 3,002 |
| True Abnormal | 3,603 | 13,435 |

#### TUSZ (Seizure Detection)

| Metric | Value |
|---|---|
| Accuracy | 0.9516 |
| Balanced Accuracy | 0.6140 |
| Macro F1 | 0.6614 |
| Weighted F1 | 0.9404 |
| AUROC | 0.8989 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Non-seizure (0) | 0.9571 | 0.9933 | 0.9749 | 43,556 |
| Seizure (1) | 0.6716 | 0.2347 | 0.3479 | 2,535 |

Confusion Matrix:

|  | Pred Non-seizure | Pred Seizure |
|---|---|---|
| True Non-seizure | 43,265 | 291 |
| True Seizure | 1,940 | 595 |

#### TUEV (6-class Event Classification)

> To be updated

#### TUEP (Epilepsy Diagnosis)

| Metric | Value |
|---|---|
| Accuracy | 0.8889 |
| Balanced Accuracy | 0.7588 |
| Macro F1 | 0.7970 |
| Weighted F1 | 0.8793 |
| AUROC | 0.8618 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Non-epilepsy (0) | 0.8427 | 0.5428 | 0.6603 | 8,240 |
| Epilepsy (1) | 0.8957 | 0.9748 | 0.9336 | 33,199 |

Confusion Matrix:

|  | Pred Non-epilepsy | Pred Epilepsy |
|---|---|---|
| True Non-epilepsy | 4,473 | 3,767 |
| True Epilepsy | 835 | 32,364 |
