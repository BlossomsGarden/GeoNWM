#!/usr/bin/env bash
set -eo pipefail

CODE_DIR="${CODE_DIR:-/data/wlh/GLD/code}"
OUT_DIR="${OUT_DIR:-/data/wlh/GLD/outputs/ar_diffusion_with_global_tokens/waymo_da3_cdit_formal_v2_7h1f_336x224}"
CONFIG="${CONFIG:-explorations/ar_diffusion_with_global_tokens/configs/waymo_da3_ar_cdit_formal_v2_7h1f_336x224.yaml}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")}"
MASTER_PORT="${MASTER_PORT:-29571}"
LAUNCHER="${LAUNCHER:-torchrun}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "${CODE_DIR}"
source /data/wlh/miniconda3/etc/profile.d/conda.sh
conda activate gld

mkdir -p "${OUT_DIR}"
export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"

PID_FILE="${OUT_DIR}/train_v2.pid"
LOG_FILE="${OUT_DIR}/train_v2.log"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}")"
  if [[ -n "${old_pid}" ]] && ps -p "${old_pid}" >/dev/null 2>&1; then
    echo "v2 training already running: ${old_pid}"
    exit 0
  fi
fi

if [[ "${LAUNCHER}" == "torch.distributed.launch" ]]; then
  LAUNCH_CMD=(python -m torch.distributed.launch --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" explorations/ar_diffusion_with_global_tokens/scripts/train_ar_da3.py)
else
  LAUNCH_CMD=(torchrun --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" explorations/ar_diffusion_with_global_tokens/scripts/train_ar_da3.py)
fi

nohup "${LAUNCH_CMD[@]}" \
  --config "${CONFIG}" \
  ${EXTRA_ARGS} \
  > "${LOG_FILE}" 2>&1 &

pid="$!"
echo "${pid}" > "${PID_FILE}"
sleep 2

if ! ps -p "${pid}" >/dev/null 2>&1; then
  echo "v2 training exited immediately; tailing log:" >&2
  tail -n 120 "${LOG_FILE}" >&2 || true
  exit 1
fi

echo "v2 training started: ${pid}"
echo "log: ${LOG_FILE}"
echo "cuda: ${CUDA_VISIBLE_DEVICES}"
echo "launcher: ${LAUNCHER}"
echo "nproc_per_node: ${NPROC_PER_NODE}"
echo "master_port: ${MASTER_PORT}"
