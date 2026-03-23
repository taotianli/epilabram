#!/bin/bash
# Stage1 训练脚本（支持 EDF 和 H5 格式）

set -e

CONFIG=${1:-configs/stage1_dacp.yaml}
OUTPUT_DIR=${2:-experiments/stage1}
PRETRAINED=${3:-/home/taotl/Desktop/LaBraM/checkpoints/labram-base.pth}
DATA_FORMAT=${4:-edf}
EDF_PATH=${5:-/home/taotl/Desktop/TUAR/v3.0.1/edf}
GPUS=${6:-1}

echo "Config:      $CONFIG"
echo "Output:      $OUTPUT_DIR"
echo "Pretrained:  $PRETRAINED"
echo "Format:      $DATA_FORMAT"
echo "GPUs:        $GPUS"

EXTRA_ARGS=""
if [ "$DATA_FORMAT" = "edf" ]; then
    EXTRA_ARGS="--data_format edf --edf_path $EDF_PATH"
fi

if [ "$GPUS" -gt 1 ]; then
    torchrun --nproc_per_node=$GPUS \
        train_stage1.py \
        --config $CONFIG \
        --output_dir $OUTPUT_DIR \
        --pretrained_path $PRETRAINED \
        $EXTRA_ARGS
else
    python train_stage1.py \
        --config $CONFIG \
        --output_dir $OUTPUT_DIR \
        --pretrained_path $PRETRAINED \
        $EXTRA_ARGS
fi
