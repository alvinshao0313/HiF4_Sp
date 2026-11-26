#!/usr/bin/env bash
# Phase E: teacher trace policy ablation on the single best overall structure.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"

if [[ $# -lt 2 ]]; then
  echo "usage: $0 PHASE1_DIR BEST_OVERALL_PRESET" >&2
  exit 1
fi
PHASE1_DIR="$1"
BEST_OVERALL_PRESET="$2"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
BASE="${RESULTS_ROOT}/teacher_policy_${STAMP}"
ALL="${PHASE1_DIR}/$(phase1_run_name "${BEST_OVERALL_PRESET}")"
mkdir -p "${BASE}"
require_complete_train_run "${ALL}"
init_stage_gpu_pool
set_structure_args "${BEST_OVERALL_PRESET}"

POLICIES=(regenerate_correct replace_question_correct)
i=0
n=${#POLICIES[@]}
while (( i < n )); do
  pids=()
  slot=0
  while (( slot < PARALLEL_SLOTS && i < n )); do
    gpu="${AVAILABLE_GPUS[${slot}]}"
    policy="${POLICIES[${i}]}"
    calib="$(shared_calib_dir_for s1k_teacher_cot "${policy}")"
    echo "launch gpu=${gpu} policy ${policy}"
    CUDA_VISIBLE_DEVICES="${gpu}" e2e_train \
      --output_dir "${BASE}/${policy}" \
      --diag_batch_size "${DIAG_BATCH_SIZE}" \
      --calib_cache_dir "${calib}" \
      --calib_source s1k_teacher_cot \
      --teacher_trace_policy "${policy}" \
      "${STRUCTURE_ARGS[@]}" &
    pids+=("$!")
    slot=$((slot + 1))
    i=$((i + 1))
  done
  wait_gpu_wave "${pids[@]}"
done

for policy in "${POLICIES[@]}"; do
  require_complete_train_run "${BASE}/${policy}"
done

EVAL_DIRS=("${BASE}/regenerate_correct" "${BASE}/replace_question_correct" "${ALL}")
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
  "all": "${ALL}",
  "regenerate_correct": "${BASE}/regenerate_correct",
  "replace_question_correct": "${BASE}/replace_question_correct",
}, indent=2))
PY
e2e_summarize --run_map "${BASE}/run_map.json" --output_json "${BASE}/summary.json" --output_md "${BASE}/report.md"
echo "teacher_policy_dir=${BASE}"
