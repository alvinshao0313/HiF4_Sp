#!/usr/bin/env bash
# Phase C5: clamp -4,4 vs none on best fusable.
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
BASE="${RESULTS_ROOT}/clamp_${STAMP}"
CALIB="$(shared_calib_dir_for s1k_original)"
CLAMPED="${PHASE1_DIR}/$(phase1_run_name "${BEST_FUSABLE_PRESET}")"
mkdir -p "${BASE}"
require_complete_train_run "${CLAMPED}"
require_shared_calib_cache "${CALIB}"
init_stage_gpu_pool
set_structure_args "${BEST_FUSABLE_PRESET}"

CUDA_VISIBLE_DEVICES="${AVAILABLE_GPUS[0]}" e2e_train \
  --output_dir "${BASE}/clamp_none" \
  --diag_batch_size "${DIAG_BATCH_SIZE}" \
  --calib_source s1k_original \
  --calib_cache_dir "${CALIB}" \
  --diag_log2_clamp none \
  "${STRUCTURE_ARGS[@]}"
require_complete_train_run "${BASE}/clamp_none"

require_fast_eval_pool
eval_pair="$(fast_eval_gpu 0)"
if [[ ! -f "${BASE}/clamp_none/eval/eval_summary.json" ]]; then
  fast_eval_run "${eval_pair}" "${BASE}/clamp_none" artifact \
    "${BASE}/clamp_none/checkpoint/final_model/conversion_state.pt"
fi
if [[ ! -f "${CLAMPED}/eval/eval_summary.json" ]]; then
  fast_eval_run "${eval_pair}" "${CLAMPED}" artifact \
    "${CLAMPED}/checkpoint/final_model/conversion_state.pt"
fi

python - <<PY > "${BASE}/run_map.json"
import json
print(json.dumps({
  "E0_native_nvfp4": "${PHASE1_DIR}/E0_native_nvfp4",
  "clamp_m4_4": "${CLAMPED}",
  "clamp_none": "${BASE}/clamp_none",
}, indent=2))
PY
e2e_summarize --run_map "${BASE}/run_map.json" --output_json "${BASE}/summary.json" --output_md "${BASE}/report.md"
echo "clamp_dir=${BASE}"
