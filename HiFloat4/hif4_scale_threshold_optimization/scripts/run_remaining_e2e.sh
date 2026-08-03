#!/usr/bin/env bash
# Launch remaining phase6 schemes after batch1 finishes.
# Usage: bash scripts/run_remaining_e2e.sh [GPU_A] [GPU_B]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${HIF4_PY:-/home/shaoyuantian/anaconda3/envs/hif4/bin/python}"
GPU_A="${1:-5}"
GPU_B="${2:-6}"
COMMON=(
  --model Qwen/Qwen3.5-4B
  --device cuda
  --fixed-best-d 7.0
  --fixed-best-t8 3.9
  --fixed-best-t4 1.95
  --act-param-map results/20260730_phase5_act_calib/param_map_per_layer.pt
  --weight-updates results/20260730_phase4_weight_all/weight_recon_updates.pt
  --lm-batch-size 4
  --mmlu-pro-limit 300
  --ppl-max-length 2048
)
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU_A}" "$PY" scripts/evaluate_model.py \
  "${COMMON[@]}" --schemes act_calib_only \
  --out-dir results/20260730_phase6_e2e/act_calib_only \
  > /tmp/e2e_act_le.log 2>&1 &
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU_B}" "$PY" scripts/evaluate_model.py \
  "${COMMON[@]}" --schemes joint \
  --out-dir results/20260730_phase6_e2e/joint \
  > /tmp/e2e_joint_le.log 2>&1 &
wait
echo "remaining schemes done"
