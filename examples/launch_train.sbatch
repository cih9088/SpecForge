#!/bin/bash

#SBATCH --job-name=specforge
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=outputs/slurm/%x_%j/out.log
#SBATCH --error=outputs/slurm/%x_%j/err.log
#SBATCH --exclusive

set -x

CONTAINER_IMAGE=${CONTAINER_IMAGE:?Please provide CONTAINER_IMAGE to run}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-}
CONTAINER_ENV=${CONTAINER_ENV:-}
TARGET_TP_SIZE=${TARGET_TP_SIZE:-$SLURM_GPUS_ON_NODE}
SPECFORGE_DIR=${SPECFORGE_DIR:-$(git rev-parse --show-toplevel 2>/dev/null)}

_SCRIPT=${1:?}
shift

if [ ! -f "$_SCRIPT" ]; then
  echo "$_SCRIPT not found" >&2
  exit 1
fi
_SCRIPT=$(realpath "$_SCRIPT")

if [ -z "$SPECFORGE_DIR" ]; then
  echo "cannot infer SPECFORGE_DIR" >&2
  exit 1
fi
_SCRIPT=$(realpath --relative-to "$SPECFORGE_DIR" "$_SCRIPT")

if [ "$SPECFORGE_DIR" != "$SLURM_SUBMIT_DIR" ]; then
    echo "sbatch must be run from $SPECFORGE_DIR but $SLURM_SUBMIT_DIR" >&2
    exit 1
fi

_TOTAL_GPUS=$(( "$SLURM_GPUS_ON_NODE" * "$SLURM_JOB_NUM_NODES" ))
if [ "$TARGET_TP_SIZE" -gt "$_TOTAL_GPUS" ]; then
    echo "TARGET_TP_SIZE $TARGET_TP_SIZE cannot be greater than $SLURM_GPUS_ON_NODE * $SLURM_JOB_NUM_NODES = $_TOTAL_GPUS" >&2
    exit 1
fi

_MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
_MASTER_PORT=29500
_PRELUDE=$(
  cat <<EOF
pip3 install tensorboard --break-system-packages
export WANDB_NAME=${SLURM_JOB_NAME}_${SLURM_JOB_ID}
export OUTPUT_DIR=$(realpath --relative-to "$SPECFORGE_DIR" $(dirname $(scontrol show job $SLURM_JOB_ID | grep -oP 'StdOut=\K\S+')))
EOF
)

export PYTHONUNBUFFERED=0
CONTAINER_ENV="${CONTAINER_ENV},PYTHONUNBUFFERED"
export PET_RDZV_BACKEND=c10d
CONTAINER_ENV="${CONTAINER_ENV},PET_RDZV_BACKEND"
export PET_RDZV_ID=$SLURM_JOB_ID
CONTAINER_ENV="${CONTAINER_ENV},PET_RDZV_ID"
export PET_RDZV_ENDPOINT=$_MASTER_ADDR:$_MASTER_PORT
CONTAINER_ENV="${CONTAINER_ENV},PET_RDZV_ENDPOINT"
export PET_NNODES=$SLURM_JOB_NUM_NODES
CONTAINER_ENV="${CONTAINER_ENV},PET_NNODES"

CONTAINER_MOUNTS="${CONTAINER_MOUNTS},/var/tmp:/var/tmp"
CONTAINER_MOUNTS=${CONTAINER_MOUNTS},"${SPECFORGE_DIR}:/sgl-workspace/SpecForge"

CONTAINER_MOUNTS=${CONTAINER_MOUNTS#,}
CONTAINER_ENV=${CONTAINER_ENV#,}

srun \
  --unbuffered \
  --label \
  --container-mount-home \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --container-env="${CONTAINER_ENV}" \
  bash -c "
$_PRELUDE

bash $_SCRIPT $SLURM_GPUS_ON_NODE $TARGET_TP_SIZE $*
"
