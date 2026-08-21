#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/shaoyuantian/program/HiF4_Sp"
PY="/home/shaoyuantian/anaconda3/envs/hif4/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
RUN_ID="${1:-}"
if [[ -z "$RUN_ID" ]]; then
  echo "Usage: $0 <run_id>"; exit 1
fi
"$PY" -m Inference_Paradigm_Conversion.run_analysis ax-report --run-id "$RUN_ID"
