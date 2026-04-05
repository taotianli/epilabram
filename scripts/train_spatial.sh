#!/bin/bash
#SBATCH --job-name=spatial_eeg
#SBATCH --output=/projects/u6da/logs/spatial_%j.out
#SBATCH --error=/projects/u6da/logs/spatial_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --partition=workq

set -euo pipefail

mkdir -p /projects/u6da/logs

export PYTHONUNBUFFERED=1

source ~/miniforge3/bin/activate epilabram

cd ~/epilabram

echo "=========================================="
echo "Spatial-Aware EEG Training"
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"
echo "=========================================="

# ── 实验1: 坐标编码 + 图卷积（完整 spatial-aware）─────────────────────────
echo "=== Run: coord_embed + GCN (k=5) ==="
python train_spatial.py \
    --config    configs/stage2_mtpct.yaml \
    --stage1_ckpt checkpoints/labram-base.pth \
    --output_dir /projects/u6da/epilabram/spatial_gcn \
    --use_gcn \
    --gcn_k 5

# ── 实验2: 仅坐标编码（ablation）──────────────────────────────────────────
echo "=== Run: coord_embed only (no GCN) ==="
python train_spatial.py \
    --config    configs/stage2_mtpct.yaml \
    --stage1_ckpt checkpoints/labram-base.pth \
    --output_dir /projects/u6da/epilabram/spatial_coord_only \
    --no_gcn

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
