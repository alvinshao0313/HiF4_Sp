#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/shaoyuantian/program/HiF4_Sp"
PY="/home/shaoyuantian/anaconda3/envs/hif4/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m Inference_Paradigm_Conversion.run_analysis \
  --config Inference_Paradigm_Conversion/configs/qwen3_8b_nvfp4_qat_formal.yaml \
  --out-dir Inference_Paradigm_Conversion/results \
  preflight
