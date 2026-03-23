#!/bin/bash
# 评估脚本

set -e

CONFIG=${1:-configs/stage2_mtpct.yaml}
CKPT=${2:-experiments/stage2/stage2_best.pth}
OUTPUT_DIR=${3:-experiments/eval}
SPLIT=${4:-test}

echo "Config:  $CONFIG"
echo "Ckpt:    $CKPT"
echo "Output:  $OUTPUT_DIR"
echo "Split:   $SPLIT"

python evaluate.py \
    --config $CONFIG \
    --ckpt $CKPT \
    --output_dir $OUTPUT_DIR \
    --split $SPLIT \
    --save_vis \
    --tasks TUAB TUSZ TUEV TUEP
