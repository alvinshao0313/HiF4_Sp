#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/shaoyuantian/program/HiF4_Sp"
PY="/home/shaoyuantian/anaconda3/envs/hif4/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
RUN_ID="${1:-${RUN_ID:-}}"
OUT="${OUT_DIR:-Inference_Paradigm_Conversion/results}"
if [[ -z "$RUN_ID" ]]; then
  echo "Usage: $0 <run_id>"; exit 1
fi
A2_DIR="${A2_RUN_DIR:-Inference_Paradigm_Conversion/results/20260811T032247Z_a2}"
"$PY" -m Inference_Paradigm_Conversion.run_analysis ax-merge --run-id "$RUN_ID" --a2-run-dir "$A2_DIR"
"$PY" -m Inference_Paradigm_Conversion.run_analysis ax-report --run-id "$RUN_ID"
