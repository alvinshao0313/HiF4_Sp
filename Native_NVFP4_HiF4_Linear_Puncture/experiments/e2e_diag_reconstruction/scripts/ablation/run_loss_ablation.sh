#!/usr/bin/env bash
# Phase C3: reconstruction loss ablation on best fusable and best online.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"

if [[ $# -lt 3 ]]; then
  echo "usage: $0 PHASE1_DIR BEST_FUSABLE_PRESET BEST_ONLINE_PRESET" >&2
  exit 1
fi
PHASE1_DIR="$1"
BEST_FUSABLE_PRESET="$2"
BEST_ONLINE_PRESET="$3"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
BASE="${RESULTS_ROOT}/loss_${STAMP}"
CALIB="$(shared_calib_dir_for s1k_original)"
FUSABLE="${PHASE1_DIR}/$(phase1_run_name "${BEST_FUSABLE_PRESET}")"
ONLINE="${PHASE1_DIR}/$(phase1_run_name "${BEST_ONLINE_PRESET}")"
mkdir -p "${BASE}"
require_complete_train_run "${FUSABLE}"
require_complete_train_run "${ONLINE}"
require_shared_calib_cache "${CALIB}"
init_stage_gpu_pool

JOB_PRESETS=("${BEST_FUSABLE_PRESET}" "${BEST_FUSABLE_PRESET}" "${BEST_ONLINE_PRESET}" "${BEST_ONLINE_PRESET}")
JOB_LOSSES=(block_output_nmse mse block_output_nmse mse)
JOB_NAMES=(fusable_block_output_nmse fusable_mse online_block_output_nmse online_mse)
i=0
n=4
while (( i < n )); do
  pids=()
  slot=0
  while (( slot < PARALLEL_SLOTS && i < n )); do
    gpu="${AVAILABLE_GPUS[${slot}]}"
    set_structure_args "${JOB_PRESETS[${i}]}"
    echo "launch gpu=${gpu} ${JOB_NAMES[${i}]}"
    CUDA_VISIBLE_DEVICES="${gpu}" e2e_train \
      --output_dir "${BASE}/${JOB_NAMES[${i}]}" \
      --diag_batch_size "${DIAG_BATCH_SIZE}" \
      --calib_source s1k_original \
      --calib_cache_dir "${CALIB}" \
      --recon_loss "${JOB_LOSSES[${i}]}" \
      "${STRUCTURE_ARGS[@]}" &
    pids+=("$!")
    slot=$((slot + 1))
    i=$((i + 1))
  done
  wait_gpu_wave "${pids[@]}"
done

for name in "${JOB_NAMES[@]}"; do
  require_complete_train_run "${BASE}/${name}"
done

EVAL_DIRS=("${BASE}/fusable_block_output_nmse" "${BASE}/fusable_mse" "${BASE}/online_block_output_nmse" "${BASE}/online_mse" "${FUSABLE}" "${ONLINE}")
require_fast_eval_pool
eval_slots="$(fast_eval_slots)"
i=0
n=${#EVAL_DIRS[@]}
while (( i < n )); do
  pids=()
  slot=0
  while (( slot < eval_slots && i < n )); do
    gpu="$(fast_eval_gpu "${slot}")"
    dir="${EVAL_DIRS[${i}]}"
    if [[ ! -f "${dir}/eval/eval_summary.json" ]]; then
      echo "launch gpu=${gpu} eval ${dir}"
      fast_eval_run "${gpu}" "${dir}" artifact "${dir}/checkpoint/final_model/conversion_state.pt" &
      pids+=("$!")
    fi
    slot=$((slot + 1))
    i=$((i + 1))
  done
  if [[ ${#pids[@]} -gt 0 ]]; then
    wait_gpu_wave "${pids[@]}"
  fi
done

python - <<PY > "${BASE}/run_map.json"
import json
print(json.dumps({
  "E0_native_nvfp4": "${PHASE1_DIR}/E0_native_nvfp4",
  "fusable_block_delta_nmse": "${FUSABLE}",
  "fusable_block_output_nmse": "${BASE}/fusable_block_output_nmse",
  "fusable_mse": "${BASE}/fusable_mse",
  "online_block_delta_nmse": "${ONLINE}",
  "online_block_output_nmse": "${BASE}/online_block_output_nmse",
  "online_mse": "${BASE}/online_mse",
}, indent=2))
PY
e2e_summarize --run_map "${BASE}/run_map.json" --output_json "${BASE}/summary.json" --output_md "${BASE}/report.md"
echo "loss_dir=${BASE}"
