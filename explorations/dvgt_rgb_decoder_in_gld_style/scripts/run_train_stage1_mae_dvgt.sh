#!/bin/bash
set -euo pipefail

CONFIG="${1:-explorations/dvgt_rgb_decoder_in_gld_style/configs/DVGT_stage1_mae_waymo_672x448.yaml}"
RESUME_CKPT="${2:-}"
RESULTS_DIR="${RESULTS_DIR:-explorations/dvgt_rgb_decoder_in_gld_style/results/stage1-mae-dvgt-waymo-672x448}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="src:${PYTHONPATH:-}"

RESUME_ARG=""
if [[ -n "${RESUME_CKPT}" ]]; then
  RESUME_ARG="--ckpt ${RESUME_CKPT}"
fi

echo "=========================================="
echo "  GLD-style Stage-1 MAE Decoder on DVGT"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  Config: ${CONFIG}"
echo "  Results: ${RESULTS_DIR}"
echo "=========================================="

python explorations/dvgt_rgb_decoder_in_gld_style/scripts/train_stage1_mae_dvgt.py \
  --config "${CONFIG}" \
  --results-dir "${RESULTS_DIR}" \
  --precision bf16 \
  ${RESUME_ARG}
