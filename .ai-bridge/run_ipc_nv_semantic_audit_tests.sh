#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/shaoyuantian/program/HiF4_Sp"
PY="/home/shaoyuantian/anaconda3/envs/hif4/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m pytest -q \
  Inference_Paradigm_Conversion/tests/test_nvfp4_adapter.py \
  Inference_Paradigm_Conversion/tests/test_activation_conversion.py \
  Inference_Paradigm_Conversion/tests/test_a2_counterfactual.py \
  Inference_Paradigm_Conversion/tests/test_activation_grid_occupancy.py \
  Inference_Paradigm_Conversion/tests/test_activation_scale_payload_factorization.py \
  Inference_Paradigm_Conversion/tests/test_activation_viz_pipeline.py \
  Inference_Paradigm_Conversion/tests/test_a5_and_l2.py \
  Inference_Paradigm_Conversion/tests/test_network_injection.py
