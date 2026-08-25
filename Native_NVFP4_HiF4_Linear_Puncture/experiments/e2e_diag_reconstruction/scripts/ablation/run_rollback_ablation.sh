#!/usr/bin/env bash
# Phase C4: rollback ON vs OFF on best fusable.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"

if [[ $# -lt 2 ]]; then
  echo "usage: $0 PHASE1_DIR BEST_FUSABLE_PRESET" >&2
  exit 1
fi
PHASE1_DIR="$1"
BEST_FUSABLE_PRESET="$2"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
BASE="${RESULTS_ROOT}/rollback_${STAMP}"
CALIB="$(shared_calib_dir_for s1k_original)"
ON="${PHASE1_DIR}/$(phase1_run_name "${BEST_FUSABLE_PRESET}")"
mkdir -p "${BASE}"
require_complete_train_run "${ON}"
require_shared_calib_cache "${CALIB}"
init_stage_gpu_pool
set_structure_args "${BEST_FUSABLE_PRESET}"

CUDA_VISIBLE_DEVICES="${AVAILABLE_GPUS[0]}" e2e_train \
  --output_dir "${BASE}/rollback_off" \
  --diag_batch_size "${DIAG_BATCH_SIZE}" \
  --calib_source s1k_original \
  --calib_cache_dir "${CALIB}" \
  --loss_rollback on \
  --router_rollback off \
  "${STRUCTURE_ARGS[@]}"
require_complete_train_run "${BASE}/rollback_off"

require_fast_eval_pool
eval_pair="$(fast_eval_gpu 0)"
if [[ ! -f "${BASE}/rollback_off/eval/eval_summary.json" ]]; then
  fast_eval_run "${eval_pair}" "${BASE}/rollback_off" artifact \
    "${BASE}/rollback_off/checkpoint/final_model/conversion_state.pt"
fi
if [[ ! -f "${ON}/eval/eval_summary.json" ]]; then
  fast_eval_run "${eval_pair}" "${ON}" artifact \
    "${ON}/checkpoint/final_model/conversion_state.pt"
fi

python - <<PY > "${BASE}/run_map.json"
import json
print(json.dumps({
  "E0_native_nvfp4": "${PHASE1_DIR}/E0_native_nvfp4",
  "rollback_on": "${ON}",
  "rollback_off": "${BASE}/rollback_off",
}, indent=2))
PY
e2e_summarize --run_map "${BASE}/run_map.json" --output_json "${BASE}/summary.json" --output_md "${BASE}/report.md"
echo "rollback_dir=${BASE}"
