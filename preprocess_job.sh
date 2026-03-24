#!/bin/bash
#SBATCH --job-name=tuh_preprocess
#SBATCH --output=/projects/u6da/logs/preprocess_%j.log
#SBATCH --error=/projects/u6da/logs/preprocess_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=24:00:00

mkdir -p /projects/u6da/logs

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "CPUs:   $SLURM_CPUS_PER_TASK"
echo "Start:  $(date)"
echo "=========================================="

source ~/.bashrc
conda activate epilabram

cd ~/epilabram

python preprocess_tuh.py \
    --tuh_root   /projects/u6da/tuh_eeg \
    --output_dir /projects/u6da/tuh_processed \
    --workers    48

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
