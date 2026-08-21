#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CAPTURE_RUN_ID=${CAPTURE_RUN_ID:-20260812T103800Z_native_nvfp4_hif4_linear_puncture}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_diag_group_stats}
CONFIG=${CONFIG:-Native_NVFP4_HiF4_Linear_Puncture/configs/qwen3_8b_native_nvfp4_linear_puncture.yaml}

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

conda run -n hif4 --no-capture-output \
  python -m Native_NVFP4_HiF4_Linear_Puncture.src.diag_group_stats \
  --config "$CONFIG" \
  --capture-run-id "$CAPTURE_RUN_ID" \
  --run-id "$RUN_ID"
