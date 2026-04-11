# EpiLaBraM

Evaluation and fine-tuning of [LaBraM](https://github.com/935963004/LaBraM) on epilepsy-related downstream tasks from the TUH EEG Corpus.

## Setup

```bash
conda activate epilabram
```

## Data Preprocessing

```bash
# Preprocess raw EEG data to h5 format (bandpass 0.1-75Hz, notch 50Hz, resample 200Hz, normalize /100)
python preprocess_tuh.py \
    --tuh_root /projects/u6da/tuh_eeg \
    --output_dir /projects/u6da/tuh_processed \
    --datasets tuab tusz tuev tuep \
    --workers 32

# Build memmap cache for fast data loading (one-time, reused across all runs)
python build_memmap.py \
    --tasks TUAB TUSZ TUEV TUEP \
    --num_workers 32 \
    --out_dir /projects/u6da/tuh_processed/memmap
```

> **Note — TUAR / TUEG**: These datasets are only listed as EDF file paths (no h5 conversion yet).
> Run `preprocess_tuh.py --datasets tuar tueg` before using them for any downstream task.

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

> ⚠️ Note: official BIOT split (official train 80/20 → train/val, official eval → test). AUROC computed only over classes present in val set.

| Metric | Value |
|---|---|
| Accuracy | 0.6175 |
| Balanced Accuracy | 0.4349 |
| Cohen's Kappa | 0.3897 |
| Macro F1 | 0.4207 |
| Weighted F1 | 0.5815 |
| AUROC | 0.7715 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| spsw (0) | 0.0000 | 0.0000 | 0.0000 | 239 |
| gped (1) | 0.5571 | 0.8612 | 0.6766 | 1,398 |
| pled (2) | 0.8966 | 0.6262 | 0.7374 | 1,038 |
| eyem (3) | 0.1829 | 0.0284 | 0.0491 | 1,128 |
| artf (4) | 0.3548 | 0.2798 | 0.3129 | 2,841 |
| bckg (5) | 0.6925 | 0.8139 | 0.7483 | 7,239 |

Confusion Matrix:

|  | spsw | gped | pled | eyem | artf | bckg |
|---|---|---|---|---|---|---|
| spsw | 0 | 0 | 9 | 0 | 0 | 230 |
| gped | 0 | 1204 | 0 | 5 | 124 | 65 |
| pled | 56 | 173 | 650 | 0 | 53 | 106 |
| eyem | 0 | 2 | 0 | 32 | 225 | 869 |
| artf | 0 | 657 | 6 | 37 | 795 | 1346 |
| bckg | 17 | 125 | 60 | 101 | 1044 | 5892 |

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

---

### Spatial-Aware Multi-Task Fine-tuning (coord_embed + GCN)

> Backbone: `labram-base.pth` + SpatialAwareLaBraM (coord_embed + GCN k=5) | 30 epochs | GPU: NVIDIA GH200 120GB

| Task | Balanced Acc | AUROC | AUC-PR | Weighted F1 | Cohen's Kappa |
|---|---|---|---|---|---|
| TUAB | 0.7954 | 0.8724 | 0.8658 | 0.7995 | 0.5961 |
| TUSZ | 0.6179 | 0.8178 | 0.1612 | 0.9737 | 0.2309 |
| TUEV | 0.2467 | — | — | 0.3277 | 0.0757 |
| TUEP | 0.7504 | 0.7910 | 0.9051 | 0.8553 | 0.5325 |

> Best checkpoint at epoch 24 (avg bal_acc). TUEV 6-class performance remains low — likely needs class-weighted loss or focal loss to handle imbalance.
