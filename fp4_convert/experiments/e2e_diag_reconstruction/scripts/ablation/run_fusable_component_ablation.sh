#!/usr/bin/env bash
# Phase B: fusable DIAG component ablation. B7 reuses Phase A E3.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 PHASE1_DIR" >&2
  exit 1
fi
PHASE1_DIR="$1"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
BASE="${RESULTS_ROOT}/fusable_components_${STAMP}"
CALIB="$(shared_calib_dir_for s1k_original)"
E3="${PHASE1_DIR}/$(phase1_run_name fusable)"
mkdir -p "${BASE}"
require_complete_train_run "${E3}"
require_shared_calib_cache "${CALIB}"
init_stage_gpu_pool

COMPONENTS=(qkv vo gu ud attn mlp)
i=0
n=${#COMPONENTS[@]}
while (( i < n )); do
  pids=()
  slot=0
  while (( slot < PARALLEL_SLOTS && i < n )); do
    gpu="${AVAILABLE_GPUS[${slot}]}"
    comp="${COMPONENTS[${i}]}"
    echo "launch gpu=${gpu} train B ${comp}"
    CUDA_VISIBLE_DEVICES="${gpu}" e2e_train \
      --output_dir "${BASE}/B_${comp}" \
      --diag_mode fusable \
      --diag_batch_size "${DIAG_BATCH_SIZE}" \
      --calib_source s1k_original \
      --calib_cache_dir "${CALIB}" \
      --fusable_diag_components "${comp}" &
    pids+=("$!")
    slot=$((slot + 1))
    i=$((i + 1))
  done
  wait_gpu_wave "${pids[@]}"
done

for comp in "${COMPONENTS[@]}"; do
  require_complete_train_run "${BASE}/B_${comp}"
done

EVAL_DIRS=()
EVAL_NAMES=()
for comp in "${COMPONENTS[@]}"; do
  EVAL_NAMES+=("B_${comp}")
  EVAL_DIRS+=("${BASE}/B_${comp}")
done
EVAL_NAMES+=("B_all")
EVAL_DIRS+=("${E3}")

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
      echo "launch gpu=${gpu} eval ${EVAL_NAMES[${i}]}"
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
base = "${BASE}"
e3 = "${E3}"
print(json.dumps({
  "E0_native_nvfp4": "${PHASE1_DIR}/E0_native_nvfp4",
  "B_qkv": f"{base}/B_qkv",
  "B_vo": f"{base}/B_vo",
  "B_gu": f"{base}/B_gu",
  "B_ud": f"{base}/B_ud",
  "B_attn": f"{base}/B_attn",
  "B_mlp": f"{base}/B_mlp",
  "B_all": e3,
}, indent=2))
PY
e2e_summarize --run_map "${BASE}/run_map.json" --output_json "${BASE}/summary.json" --output_md "${BASE}/report.md"
echo "fusable_components_dir=${BASE}"
