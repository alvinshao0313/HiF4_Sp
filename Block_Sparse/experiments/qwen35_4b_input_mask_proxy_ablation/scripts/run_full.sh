#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="Block_Sparse/experiments/qwen35_4b_input_mask_proxy_ablation/results/${RUN_ID}"
# Free cards on this machine by default; override with DEVICES=0,1
DEVICES="${DEVICES:-0,1,6,7}"

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
conda run -n hif4 python -m Block_Sparse.input_mask_proxy_study.run_experiment \
  --config Block_Sparse/experiments/qwen35_4b_input_mask_proxy_ablation/configs/full.json \
  --output-dir "${OUT_DIR}" \
  --capture-cache Block_Sparse/experiments/qwen35_4b_input_mask_proxy_ablation/results/captured/layer15_up_proj_s8_t1024.pt \
  --devices "${DEVICES}"
