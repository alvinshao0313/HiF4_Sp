#!/usr/bin/env bash
# 关闭 z∈[-4,4] 钳位：依次跑 5 套 formal（仍用 d=2^z 保正）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CAPTURE_RUN_ID=${CAPTURE_RUN_ID:-20260812T103800Z_native_nvfp4_hif4_linear_puncture}
TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}
DEVICE=${DEVICE:-cuda:0}
LR=${LR:-0.05}
STEPS=${STEPS:-200}

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CAPTURE_RUN_ID DEVICE LR STEPS

echo "[nolimit] stamp=${TS}"

RUN_ID="${TS}_diag_gradient_nolimit" \
  bash "$HERE/run.sh" --no-log2-clamp

RUN_ID="${TS}_r64_channel_diag_gradient_nolimit" \
  bash "$HERE/run_r64_channel.sh" --no-log2-clamp

RUN_ID="${TS}_diag_then_r64_gradient_nolimit" \
  bash "$HERE/run_diag_then_r64.sh" --no-log2-clamp

RUN_ID="${TS}_h4_channel_diag_gradient_nolimit" \
  bash "$HERE/run_h4_channel.sh" --no-log2-clamp

RUN_ID="${TS}_diag_then_h4_gradient_nolimit" \
  bash "$HERE/run_diag_then_h4.sh" --no-log2-clamp

echo "[nolimit] DONE stamp=${TS}"
