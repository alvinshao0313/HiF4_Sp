#!/usr/bin/env bash
# Phase D: calibration source ablation on best fusable / best online.
# s1k_original baselines are reused from Phase A; new sources are prepared first.
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
BASE="${RESULTS_ROOT}/calib_ablation_${STAMP}"
FUSABLE="${PHASE1_DIR}/$(phase1_run_name "${BEST_FUSABLE_PRESET}")"
ONLINE="${PHASE1_DIR}/$(phase1_run_name "${BEST_ONLINE_PRESET}")"
mkdir -p "${BASE}"
require_complete_train_run "${FUSABLE}"
require_complete_train_run "${ONLINE}"

prepare_one() {
  local src="$1"
  local calib
  calib="$(shared_calib_dir_for "${src}")"
  mkdir -p "${calib}"
  local extra=()
  if [[ "${src}" == "wikitext2" || "${src}" == "c4" ]]; then
    extra+=(--calib_seqlen 1024)
  fi
  echo "=== prepare ${src} cache ==="
  e2e_prepare_calib \
    --calib_source "${src}" \
    --calib_nsamples 128 \
    --calib_val_nsamples 32 \
    --calib_seed 42 \
    --calib_cache_dir "${calib}" \
    "${extra[@]}"
  require_shared_calib_cache "${calib}"
}

prepare_one s1k_question
prepare_one wikitext2
prepare_one c4

init_stage_gpu_pool

NEW_SOURCES=(s1k_question wikitext2 c4)
JOB_STRUCTS=()
JOB_SOURCES=()
JOB_NAMES=()
for src in "${NEW_SOURCES[@]}"; do
  JOB_STRUCTS+=("${BEST_FUSABLE_PRESET}")
  JOB_SOURCES+=("${src}")
  JOB_NAMES+=("fusable_${src}")
done
for src in "${NEW_SOURCES[@]}"; do
  JOB_STRUCTS+=("${BEST_ONLINE_PRESET}")
  JOB_SOURCES+=("${src}")
  JOB_NAMES+=("online_${src}")
done

i=0
n=${#JOB_NAMES[@]}
while (( i < n )); do
  pids=()
  slot=0
  while (( slot < PARALLEL_SLOTS && i < n )); do
    gpu="${AVAILABLE_GPUS[${slot}]}"
    src="${JOB_SOURCES[${i}]}"
    calib="$(shared_calib_dir_for "${src}")"
    set_structure_args "${JOB_STRUCTS[${i}]}"
    echo "launch gpu=${gpu} train ${JOB_NAMES[${i}]}"
    CUDA_VISIBLE_DEVICES="${gpu}" e2e_train \
      --output_dir "${BASE}/${JOB_NAMES[${i}]}" \
      --diag_batch_size "${DIAG_BATCH_SIZE}" \
      --calib_cache_dir "${calib}" \
      --calib_source "${src}" \
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

EVAL_DIRS=()
for name in "${JOB_NAMES[@]}"; do
  EVAL_DIRS+=("${BASE}/${name}")
done
EVAL_DIRS+=("${FUSABLE}" "${ONLINE}")
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
  "fusable_s1k_original": "${FUSABLE}",
  "online_s1k_original": "${ONLINE}",
  "fusable_s1k_question": "${BASE}/fusable_s1k_question",
  "online_s1k_question": "${BASE}/online_s1k_question",
  "fusable_wikitext2": "${BASE}/fusable_wikitext2",
  "online_wikitext2": "${BASE}/online_wikitext2",
  "fusable_c4": "${BASE}/fusable_c4",
  "online_c4": "${BASE}/online_c4",
}, indent=2))
PY
e2e_summarize --run_map "${BASE}/run_map.json" --output_json "${BASE}/summary.json" --output_md "${BASE}/report.md"
echo "calib_ablation_dir=${BASE}"
