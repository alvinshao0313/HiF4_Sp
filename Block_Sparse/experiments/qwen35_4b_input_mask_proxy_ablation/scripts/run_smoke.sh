#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="Block_Sparse/experiments/qwen35_4b_input_mask_proxy_ablation/results/smoke_s0mean/${RUN_ID}"

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
conda run -n hif4 python -m Block_Sparse.input_mask_proxy_study.run_experiment \
  --config Block_Sparse/experiments/qwen35_4b_input_mask_proxy_ablation/configs/smoke.json \
  --output-dir "${OUT_DIR}" \
  --capture-cache Block_Sparse/experiments/qwen35_4b_input_mask_proxy_ablation/results/captured/layer15_up_proj_smoke_t1024.pt

echo "SMOKE_OUT_DIR=${OUT_DIR}"
