#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
cd "$ROOT"

CAPTURE_RUN_ID=${CAPTURE_RUN_ID:-20260812T103800Z_native_nvfp4_hif4_linear_puncture}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_activation_3d_viz}
CONFIG=${CONFIG:-Native_NVFP4_HiF4_Linear_Puncture/configs/qwen3_8b_native_nvfp4_linear_puncture.yaml}
DEVICE=${DEVICE:-cuda:0}
LAYERS=${LAYERS:-2 10 18}
STYLES=${STYLES:-coolwarm_max}

EXTRA=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA+=(--smoke)
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# shellcheck disable=SC2086
conda run -n hif4 --no-capture-output \
  python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.activation_3d_viz.plot_activation_3d \
  --config "$CONFIG" \
  --capture-run-id "$CAPTURE_RUN_ID" \
  --run-id "$RUN_ID" \
  --device "$DEVICE" \
  --layers $LAYERS \
  --styles $STYLES \
  "${EXTRA[@]}" \
  "$@"
