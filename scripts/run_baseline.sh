#!/bin/bash
#SBATCH --job-name=labram_baseline
#SBATCH --output=/projects/u6da/logs/baseline_%j.log
#SBATCH --error=/projects/u6da/logs/baseline_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=120G
#SBATCH --time=04:00:00

mkdir -p /projects/u6da/logs

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start:  $(date)"
echo "=========================================="

source ~/miniforge3/bin/activate epilabram

cd ~/epilabram

# H200 优化设置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

python eval_labram_baseline.py \
    --ckpt         checkpoints/labram-base.pth \
    --tasks        TUAB TUSZ TUEV TUEP \
    --epochs       10 \
    --batch_size   2048 \
    --lr           1e-3 \
    --num_workers  32 \
    --bf16 \
    --compile \
    --output_dir   experiments/labram_baseline

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
