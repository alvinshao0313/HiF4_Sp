#!/usr/bin/env bash
# H4 + 逐通道 DIAG 梯度实验
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

CAPTURE_RUN_ID=${CAPTURE_RUN_ID:-20260812T103800Z_native_nvfp4_hif4_linear_puncture}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_h4_channel_diag_gradient}
CONFIG=${CONFIG:-Native_NVFP4_HiF4_Linear_Puncture/configs/qwen3_8b_native_nvfp4_linear_puncture.yaml}
DEVICE=${DEVICE:-cuda:0}
LR=${LR:-0.05}
STEPS=${STEPS:-200}

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

conda run -n hif4 --no-capture-output \
  python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.diag_gradient.run_h4_channel \
  --config "$CONFIG" \
  --capture-run-id "$CAPTURE_RUN_ID" \
  --run-id "$RUN_ID" \
  --device "$DEVICE" \
  --lr "$LR" \
  --steps "$STEPS" \
  "$@"
