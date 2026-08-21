#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_native_nvfp4_hif4_linear_puncture}
CONFIG=${CONFIG:-Native_NVFP4_HiF4_Linear_Puncture/configs/qwen3_8b_native_nvfp4_linear_puncture.yaml}
DEVICE=${DEVICE:-cuda:0}

export RUN_ID CONFIG DEVICE
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[run_all] RUN_ID=${RUN_ID}"
echo "[run_all] Phase A: preflight"
bash "${SCRIPT_DIR}/run_preflight.sh"

echo "[run_all] Phase B: unit tests"
conda run -n hif4 --no-capture-output \
  pytest Native_NVFP4_HiF4_Linear_Puncture/tests -q

echo "[run_all] Phase C: smoke capture"
bash "${SCRIPT_DIR}/run_capture.sh" --mode smoke

echo "[run_all] Phase D: formal capture"
bash "${SCRIPT_DIR}/run_capture.sh" --mode formal

echo "[run_all] Phase E: linear cases"
bash "${SCRIPT_DIR}/run_linear_cases.sh"

echo "[run_all] Phase F: report"
bash "${SCRIPT_DIR}/build_report.sh"

echo "[run_all] done: ${RUN_ID}"
