#!/usr/bin/env bash
# Formal entry: hif4 env + fixed cwd + python state machine.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

RUN_ID=""
THROUGH_STAGE=""
FROM_STAGE=""
MODEL_PATH="nvidia/Qwen3-30B-A3B-NVFP4"
PHASEA_ROOT="${ROOT}/Native_NVFP4_HiF4_Linear_Puncture/results/e2e_diag_reconstruction/phaseA_refactor_20260825T035730Z"
GPU_MEM=0.90

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --through-stage) THROUGH_STAGE="$2"; shift 2 ;;
    --from-stage) FROM_STAGE="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --phasea-root) PHASEA_ROOT="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEM="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${RUN_ID}" || -z "${THROUGH_STAGE}" ]]; then
  echo "usage: run_formal.sh --run-id ID --through-stage STAGE [--from-stage STAGE]" >&2
  exit 2
fi

cd "${ROOT}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export PYTHONUNBUFFERED=1
export PROJECT_GPU_POOL="${PROJECT_GPU_POOL:-0,1,2,3,4,5,6,7}"
export GPU_MIN_FREE_RATIO="${GPU_MIN_FREE_RATIO:-0.90}"
export GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"

# Detached process itself waits for 2 free GPUs; Cursor must not poll.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[run_formal] waiting for 2 free GPUs from pool=${PROJECT_GPU_POOL}"
  while true; do
    GPU_PAIR="$(conda run --no-capture-output -n hif4 python - <<'PY'
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability import gpu_pool
gpus = gpu_pool.available_gpus()
print(",".join(map(str, gpus[:2])) if len(gpus) >= 2 else "")
PY
)"
    if [[ -n "${GPU_PAIR}" ]]; then
      export CUDA_VISIBLE_DEVICES="${GPU_PAIR}"
      echo "[run_formal] using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
      break
    fi
    echo "[run_formal] no free GPU pair; sleep 60"
    sleep 60
  done
fi

EXTRA=()
if [[ -n "${FROM_STAGE}" ]]; then
  EXTRA+=(--from-stage "${FROM_STAGE}")
fi

exec conda run --no-capture-output -n hif4 python -u \
  "${PKG_DIR}/run_pipeline.py" \
  --run-id "${RUN_ID}" \
  --through-stage "${THROUGH_STAGE}" \
  --model-path "${MODEL_PATH}" \
  --phasea-root "${PHASEA_ROOT}" \
  --gpu-memory-utilization "${GPU_MEM}" \
  "${EXTRA[@]}"
