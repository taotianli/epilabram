#!/bin/bash
#SBATCH --job-name=epilabram
#SBATCH --output=/projects/u6da/logs/epilabram_%j.out
#SBATCH --error=/projects/u6da/logs/epilabram_%j.err
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
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"
echo "=========================================="

# ── Stage 1: PADM 续训（masked EEG modeling）──────────────────────────────
echo "=== Stage 1: PADM ==="
python train_stage1.py \
    --config configs/stage1_dacp.yaml \
    --output_dir /projects/u6da/epilabram/stage1

# ── Stage 2: 多任务微调（MTPCT）────────────────────────────────────────────
echo "=== Stage 2: MTPCT ==="
python train_stage2.py \
    --config    configs/stage2_mtpct.yaml \
    --stage1_ckpt /projects/u6da/epilabram/stage1/best.pth \
    --output_dir  /projects/u6da/epilabram/stage2

# ── Stage 3: CPA-DPO 偏好对齐 ──────────────────────────────────────────────
echo "=== Stage 3: CPA-DPO ==="
python train_dpo.py \
    --config      configs/stage3_cpa_dpo.yaml \
    --stage2_ckpt /projects/u6da/epilabram/stage2/stage2_best.pth \
    --output_dir  /projects/u6da/epilabram/stage3

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
