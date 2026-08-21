#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_native_nvfp4_hif4_linear_puncture}

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

conda run -n hif4 --no-capture-output \
  python -m Native_NVFP4_HiF4_Linear_Puncture.src.report \
  --run-id "$RUN_ID" \
  "$@"
