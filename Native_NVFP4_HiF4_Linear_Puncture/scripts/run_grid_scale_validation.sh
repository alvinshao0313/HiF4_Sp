#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CAPTURE_RUN_ID=${CAPTURE_RUN_ID:-20260812T103800Z_native_nvfp4_hif4_linear_puncture}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_theory_grid_scale_validation}
CONFIG=${CONFIG:-Native_NVFP4_HiF4_Linear_Puncture/configs/qwen3_8b_native_nvfp4_linear_puncture.yaml}
DEVICE=${DEVICE:-cuda:0}

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

conda run -n hif4 --no-capture-output \
  python -m Native_NVFP4_HiF4_Linear_Puncture.src.grid_scale_validation \
  --config "$CONFIG" \
  --capture-run-id "$CAPTURE_RUN_ID" \
  --run-id "$RUN_ID" \
  --device "$DEVICE"
