#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"

GPU_ID="${CUDA_VISIBLE_DEVICES:-1}"
RUN_ID="${RUN_ID:-full_fusable_r64_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${RESULTS_ROOT}/${RUN_ID}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" e2e_train \
  --output_dir "${OUT}" \
  --use_r64 \
  --calib_source s1k_original
