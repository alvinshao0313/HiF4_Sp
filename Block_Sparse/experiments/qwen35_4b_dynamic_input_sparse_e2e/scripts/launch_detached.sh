#!/usr/bin/env bash
# Detach full e2e from Cursor/interactive terminal.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p "${EXP_DIR}/logs"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
PID_FILE="${EXP_DIR}/logs/run_full_detached.pid"
LOG="${EXP_DIR}/logs/run_full_detached_${TS}.log"

export GPU_POOL="${GPU_POOL:-0,1,6,7}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
export CONDA_ENV="${CONDA_ENV:-hif4}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export VLLM_MAX_ATTEMPTS="${VLLM_MAX_ATTEMPTS:-5}"

if [[ -n "${RESUME_RUN_DIR:-}" ]]; then
  RUN_DIR="$(cd "${RESUME_RUN_DIR}" && pwd)"
else
  RUN_DIR="${EXP_DIR}/results/${TS}"
fi
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/telemetry"
echo "${RUN_DIR}" > "${EXP_DIR}/results/LATEST_RUN_DIR.txt"
export RUN_DIR

nohup bash "${SCRIPT_DIR}/run_full_matrix.sh" >>"${LOG}" 2>&1 </dev/null &
echo $! > "${PID_FILE}"
disown || true

echo "started pid=$(cat "${PID_FILE}")"
echo "RUN_DIR=${RUN_DIR}"
echo "log=${LOG}"
echo "tail -f ${LOG}"
