#!/bin/bash
# Stage3 训练脚本（支持 EDF 和 H5 格式）

set -e

CONFIG=${1:-configs/stage3_cpa_dpo.yaml}
OUTPUT_DIR=${2:-experiments/stage3}
STAGE2_CKPT=${3:-experiments/stage2/stage2_best.pth}
TASK=${4:-TUAB}
DATA_FORMAT=${5:-edf}
EDF_PATH=${6:-/home/taotl/Desktop/TUAR/v3.0.1/edf}

echo "Config:      $CONFIG"
echo "Output:      $OUTPUT_DIR"
echo "Stage2 ckpt: $STAGE2_CKPT"
echo "Task:        $TASK"
echo "Format:      $DATA_FORMAT"

EXTRA_ARGS=""
if [ "$DATA_FORMAT" = "edf" ]; then
    EXTRA_ARGS="--data_format edf --edf_path $EDF_PATH"
fi

python train_stage3.py \
    --config $CONFIG \
    --output_dir $OUTPUT_DIR \
    --stage2_ckpt $STAGE2_CKPT \
    --task $TASK \
    $EXTRA_ARGS
