#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi

export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
# NOTE(@laoconeth): For K-EXAONE, this environment variable seems critical for memory management.
# Reference: https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf
export PYTORCH_ALLOC_CONF=expandable_segments:True

# train eagle3 online for K-EXAONE-236B-A23B
TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-LGAI-EXAONE/K-EXAONE-236B-A23B}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:?TRAIN_DATA_PATH should be set}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/k-exaone}"

NUM_GPUS=${1:-8}
TP_SIZE=${2:-8}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-16}

# wandb logging (DO NOT commit API key to git)
USE_WANDB="${USE_WANDB:-false}"
REPORT_TO=""
if [ "$USE_WANDB" = true ]; then
    WANDB_ENTITY="${WANDB_ENTITY:-lgairesearch}"
    WANDB_PROJECT="${WANDB_PROJECT:-k-exaone-specdec}"
    WANDB_NAME="${WANDB_NAME:-}"
    REPORT_TO="--report-to wandb --wandb-entity $WANDB_ENTITY --wandb-project $WANDB_PROJECT"
    if [ -n "$WANDB_NAME" ]; then
        REPORT_TO="$REPORT_TO --wandb-name $WANDB_NAME"
    fi
else
    REPORT_TO="--report-to tensorboard"
fi

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path $TARGET_MODEL_PATH \
    --draft-model-config $ROOT_DIR/configs/k-exaone-236b-a23b-eagle3.json \
    --train-data-path $TRAIN_DATA_PATH \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $OUTPUT_DIR \
    --num-epochs 2 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --warmup-ratio 0.015 \
    --max-grad-norm 0.5 \
    --chat-template k-exaone \
    --save-interval 20000 \
    --eval-interval 20000 \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend fa3 \
    --sglang-mem-fraction-static 0.8 \
    --cache-dir $ROOT_DIR/cache \
    --dist-timeout 60 \
    $REPORT_TO
