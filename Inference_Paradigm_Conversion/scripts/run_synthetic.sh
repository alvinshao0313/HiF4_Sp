#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/shaoyuantian/program/HiF4_Sp"
PY="/home/shaoyuantian/anaconda3/envs/hif4/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 TORCH_NUM_THREADS=4
OUT="${OUT_DIR:-Inference_Paradigm_Conversion/results}"
SEEDS="${SEEDS:-10}"
"$PY" -m Inference_Paradigm_Conversion.run_analysis --out-dir "$OUT" synthetic --seeds "$SEEDS"
