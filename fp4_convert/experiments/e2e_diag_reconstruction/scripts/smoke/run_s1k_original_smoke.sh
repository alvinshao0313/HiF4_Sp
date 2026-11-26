#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"

GPU_ID="${CUDA_VISIBLE_DEVICES:-1}"
RUN_ID="${RUN_ID:-s1k_original_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${RESULTS_ROOT}/${RUN_ID}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" e2e_train \
  --output_dir "${OUT}" \
  --use_r64 \
  --diag_epochs 1 \
  --calib_source s1k_original \
  --calib_nsamples 2 \
  --calib_val_nsamples 1 \
  --diag_batch_size 1 \
  --start_layer 0 \
  --end_layer 0
