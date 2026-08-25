#!/usr/bin/env bash
# Phase C1: layer_joint vs linear_independent on best online preset.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"

if [[ $# -lt 2 ]]; then
  echo "usage: $0 PHASE1_DIR BEST_ONLINE_PRESET" >&2
  exit 1
fi
PHASE1_DIR="$1"
BEST_ONLINE_PRESET="$2"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
BASE="${RESULTS_ROOT}/train_scope_${STAMP}"
CALIB="$(shared_calib_dir_for s1k_original)"
JOINT="${PHASE1_DIR}/$(phase1_run_name "${BEST_ONLINE_PRESET}")"
mkdir -p "${BASE}"
require_complete_train_run "${JOINT}"
require_shared_calib_cache "${CALIB}"
init_stage_gpu_pool
set_structure_args "${BEST_ONLINE_PRESET}"

echo "launch gpu=${AVAILABLE_GPUS[0]} linear_independent"
CUDA_VISIBLE_DEVICES="${AVAILABLE_GPUS[0]}" e2e_train \
  --output_dir "${BASE}/linear_independent" \
  --diag_batch_size "${DIAG_BATCH_SIZE}" \
  --calib_source s1k_original \
  --calib_cache_dir "${CALIB}" \
  --diag_train_scope linear_independent \
  "${STRUCTURE_ARGS[@]}"
require_complete_train_run "${BASE}/linear_independent"

require_fast_eval_pool
eval_pair="$(fast_eval_gpu 0)"
if [[ ! -f "${BASE}/linear_independent/eval/eval_summary.json" ]]; then
  fast_eval_run "${eval_pair}" "${BASE}/linear_independent" artifact \
    "${BASE}/linear_independent/checkpoint/final_model/conversion_state.pt"
fi
if [[ ! -f "${JOINT}/eval/eval_summary.json" ]]; then
  fast_eval_run "${eval_pair}" "${JOINT}" artifact \
    "${JOINT}/checkpoint/final_model/conversion_state.pt"
fi

python - <<PY > "${BASE}/run_map.json"
import json
print(json.dumps({
  "E0_native_nvfp4": "${PHASE1_DIR}/E0_native_nvfp4",
  "layer_joint": "${JOINT}",
  "linear_independent": "${BASE}/linear_independent",
}, indent=2))
PY
e2e_summarize --run_map "${BASE}/run_map.json" --output_json "${BASE}/summary.json" --output_md "${BASE}/report.md"
echo "train_scope_dir=${BASE}"
