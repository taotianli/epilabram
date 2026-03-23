#!/bin/bash
# Stage2 训练脚本（支持 EDF 和 H5 格式）

set -e

CONFIG=${1:-configs/stage2_mtpct.yaml}
OUTPUT_DIR=${2:-experiments/stage2}
STAGE1_CKPT=${3:-experiments/stage1/stage1_ema_latest.pth}
DATA_FORMAT=${4:-edf}
EDF_PATH=${5:-/home/taotl/Desktop/TUAR/v3.0.1/edf}
GPUS=${6:-1}

echo "Config:      $CONFIG"
echo "Output:      $OUTPUT_DIR"
echo "Stage1 ckpt: $STAGE1_CKPT"
echo "Format:      $DATA_FORMAT"
echo "GPUs:        $GPUS"

EXTRA_ARGS=""
if [ "$DATA_FORMAT" = "edf" ]; then
    EXTRA_ARGS="--data_format edf --edf_path $EDF_PATH"
fi

if [ "$GPUS" -gt 1 ]; then
    torchrun --nproc_per_node=$GPUS \
        train_stage2.py \
        --config $CONFIG \
        --output_dir $OUTPUT_DIR \
        --stage1_ckpt $STAGE1_CKPT \
        $EXTRA_ARGS
else
    python train_stage2.py \
        --config $CONFIG \
        --output_dir $OUTPUT_DIR \
        --stage1_ckpt $STAGE1_CKPT \
        $EXTRA_ARGS
fi
