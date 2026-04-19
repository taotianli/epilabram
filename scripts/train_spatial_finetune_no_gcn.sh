#!/bin/bash
#SBATCH --job-name=formal_no_gcn
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
echo "Spatial-Aware EEG Training (finetune + no GCN)"
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"
echo "=========================================="

python train_spatial.py \
    --config    configs/stage2_mtpct.yaml \
    --stage1_ckpt checkpoints/labram-base.pth \
    --output_dir /projects/u6da/epilabram/spatial_finetune_no_gcn \
    --finetune \
    --no_gcn \
    --peak_lr 1e-4

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
