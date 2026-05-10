#!/bin/bash
set -euo pipefail

# Usage:
#   ./reproduce_gld_on_waymo/run_train_stage1_mae_waymo.sh [NUM_GPUS] [RESUME_CKPT]

NUM_GPUS="${1:-4}"
RESUME_CKPT="${2:-}"
CONFIG="reproduce_gld_on_waymo/DA3_stage1_mae_waymo_336x224.yaml"
RESULTS_DIR="reproduce_gld_on_waymo/results/stage1-mae-waymo-336x224"

export WANDB_KEY="${WANDB_KEY:-}"
export ENTITY="${ENTITY:-gld}"
export PROJECT="${PROJECT:-RAE_stage1_waymo}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="src:${PYTHONPATH:-}"

RESUME_ARG=""
if [[ -n "${RESUME_CKPT}" ]]; then
  RESUME_ARG="--ckpt ${RESUME_CKPT}"
  echo "Resuming from: ${RESUME_CKPT}"
fi

echo "=========================================="
echo "  GLD Stage-1 MAE Decoder on Waymo"
echo "  GPUs: ${NUM_GPUS}"
echo "  Config: ${CONFIG}"
echo "  Results: ${RESULTS_DIR}"
echo "=========================================="

torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
  src/train_stage1_mae.py \
  --config "${CONFIG}" \
  --data-path ./ \
  --results-dir "${RESULTS_DIR}" \
  --precision bf16 \
  --wandb \
  ${RESUME_ARG}


