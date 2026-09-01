#!/usr/bin/env bash
# Resume Phase-A refactor after Cursor detach: E3/E4 from next incomplete layer,
# then E5–E7, then matrix eval/summarize with SKIP_TRAIN=1 on the same STAMP.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"

STAMP="${STAMP:?STAMP required, e.g. 20260825T035730Z}"
BASE="${RESULTS_ROOT}/phaseA_refactor_${STAMP}"
CALIB="$(shared_calib_dir_for s1k_original)"
GPU_POOL="${GPU_POOL:-6,7}"
export GPU_POOL

FORMAL_FUSABLE_POLICY=(
  --loss_rollback on
  --router_rollback on
  --router_align_loss_weight 0.0
  --optimizer AdamW
  --weight_decay 0.0
  --diag_scheduler cosine
)

next_start_layer() {
  local run_dir="$1"
  local layers_dir="${run_dir}/layers"
  local i=0
  while (( i < 48 )); do
    local d
    d="$(printf '%s/layer_%02d' "${layers_dir}" "${i}")"
    if [[ -f "${d}/metrics.json" && -f "${d}/best_diag.pt" && -f "${d}/train_log.jsonl" ]]; then
      i=$((i + 1))
      continue
    fi
    # Drop incomplete current-layer scratch before resume.
    if [[ -d "${d}" ]]; then
      echo "remove incomplete layer dir: ${d}"
      rm -rf "${d}"
    fi
    echo "${i}"
    return 0
  done
  echo 48
}

train_preset_from() {
  local gpu="$1"
  local preset="$2"
  local start_layer="$3"
  local name
  name="$(phase1_run_name "${preset}")"
  set_structure_args "${preset}"
  echo "launch gpu=${gpu} train ${preset} start_layer=${start_layer}"
  CUDA_VISIBLE_DEVICES="${gpu}" e2e_train \
    --output_dir "${BASE}/${name}" \
    --diag_batch_size "${DIAG_BATCH_SIZE}" \
    --calib_source s1k_original \
    --calib_cache_dir "${CALIB}" \
    --start_layer "${start_layer}" \
    "${STRUCTURE_ARGS[@]}" \
    "${FORMAL_FUSABLE_POLICY[@]}"
}

require_shared_calib_cache "${CALIB}"
init_stage_gpu_pool

echo "=== resume/continue Phase-A at ${BASE} ==="
E3_START="$(next_start_layer "${BASE}/E3_fusable")"
E4_START="$(next_start_layer "${BASE}/E4_fusable_r64")"
echo "E3_fusable start_layer=${E3_START}"
echo "E4_fusable_r64 start_layer=${E4_START}"

pids=()
if (( E3_START < 48 )); then
  train_preset_from "${AVAILABLE_GPUS[0]}" fusable "${E3_START}" &
  pids+=("$!")
else
  echo "E3_fusable already complete"
fi
if (( E4_START < 48 )); then
  train_preset_from "${AVAILABLE_GPUS[1]:-${AVAILABLE_GPUS[0]}}" fusable_r64 "${E4_START}" &
  pids+=("$!")
else
  echo "E4_fusable_r64 already complete"
fi
if ((${#pids[@]} > 0)); then
  wait_gpu_wave "${pids[@]}"
fi

REMAINING_PRESETS=(online online_diag_then_r64 online_r64_then_diag)
i=0
n=${#REMAINING_PRESETS[@]}
while (( i < n )); do
  pids=()
  slot=0
  while (( slot < PARALLEL_SLOTS && i < n )); do
    gpu="${AVAILABLE_GPUS[${slot}]}"
    preset="${REMAINING_PRESETS[${i}]}"
    name="$(phase1_run_name "${preset}")"
    if [[ -f "${BASE}/${name}/checkpoint/final_model/conversion_state.pt" ]]; then
      echo "skip already-complete train ${preset}"
    else
      train_preset_from "${gpu}" "${preset}" 0 &
      pids+=("$!")
    fi
    slot=$((slot + 1))
    i=$((i + 1))
  done
  if ((${#pids[@]} > 0)); then
    wait_gpu_wave "${pids[@]}"
  fi
done

echo "=== continue Phase-A eval/summarize (SKIP_TRAIN=1) ==="
STAMP="${STAMP}" SKIP_TRAIN=1 GPU_POOL="${GPU_POOL}" \
  bash "${SCRIPT_DIR}/run_phaseA_refactor_matrix.sh"

echo "continue_phaseA_done base=${BASE}"
