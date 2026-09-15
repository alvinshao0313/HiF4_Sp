#!/usr/bin/env bash
# Detached non-O4 recipe workers. Does not replace the formal S4 pipeline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ROOT=""
MODEL_PATH="nvidia/Qwen3-30B-A3B-NVFP4"
PHASEA_ROOT="${ROOT}/Native_NVFP4_HiF4_Linear_Puncture/results/e2e_diag_reconstruction/phaseA_refactor_20260825T035730Z"
GPUS="2,3"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --phasea-root) PHASEA_ROOT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${RUN_ROOT}" ]]; then
  echo "usage: launch_recipe_workers.sh --run-root PATH [--gpus 2,3]" >&2
  exit 2
fi

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
SHARDS="${#GPU_ARR[@]}"
mkdir -p "${RUN_ROOT}/logs"

cd "${ROOT}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

for i in "${!GPU_ARR[@]}"; do
  GPU="${GPU_ARR[$i]}"
  LOG="${RUN_ROOT}/logs/recipe_worker_${i}.log"
  PID_FILE="${RUN_ROOT}/logs/recipe_worker_${i}.pid"
  if [[ -f "${PID_FILE}" ]]; then
    OLD="$(cat "${PID_FILE}" || true)"
    if [[ -n "${OLD}" ]] && kill -0 "${OLD}" 2>/dev/null; then
      echo "REFUSE worker shard=${i} still alive pid=${OLD}" >&2
      exit 3
    fi
  fi
  echo "[launch_recipe_workers $(date -u +%Y-%m-%dT%H:%M:%SZ)] shard=${i}/${SHARDS} gpu=${GPU}" >>"${LOG}"
  CUDA_VISIBLE_DEVICES="${GPU}" nohup setsid conda run --no-capture-output -n hif4 python -u \
    "${PKG_DIR}/run_objective_recipe_worker.py" \
    --run-root "${RUN_ROOT}" \
    --model-path "${MODEL_PATH}" \
    --phasea-root "${PHASEA_ROOT}" \
    --shard "${i}" \
    --shards "${SHARDS}" \
    --exclude-o4 \
    </dev/null >>"${LOG}" 2>&1 &
  echo $! > "${PID_FILE}"
  echo "shard=${i} gpu=${GPU} pid=$(cat "${PID_FILE}") log=${LOG}"
done
