#!/usr/bin/env bash
# Corrected Phase-A after Router/rollback/artifact refactor.
# Explicit Fusable policy: KL=0, loss_rollback=on, router_rollback=on.
# E3–E7 always retrain from layer0; never reuse old phase1 DIAG.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"

STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
BASE="${RESULTS_ROOT}/phaseA_refactor_${STAMP}"
CALIB="$(shared_calib_dir_for s1k_original)"
# E2 is intentionally deferred. Set RUN_E2_WITH_ABI3=1 only when formally
# rerunning it with the optimized R64->HiF4 runtime ABI.
RUN_E2_WITH_ABI3="${RUN_E2_WITH_ABI3:-0}"
mkdir -p "${BASE}" "${CALIB}"

FORMAL_FUSABLE_POLICY=(
  --loss_rollback on
  --router_rollback on
  --router_align_loss_weight 0.0
  --optimizer AdamW
  --weight_decay 0.0
  --diag_scheduler cosine
)

echo "=== prepare s1k_original shared cache ==="
e2e_prepare_calib \
  --calib_source s1k_original \
  --calib_nsamples 128 \
  --calib_val_nsamples 32 \
  --calib_seed 42 \
  --calib_cache_dir "${CALIB}"
require_shared_calib_cache "${CALIB}"

init_stage_gpu_pool

train_preset() {
  local gpu="$1"
  local preset="$2"
  local name
  name="$(phase1_run_name "${preset}")"
  set_structure_args "${preset}"
  CUDA_VISIBLE_DEVICES="${gpu}" e2e_train \
    --output_dir "${BASE}/${name}" \
    --diag_batch_size "${DIAG_BATCH_SIZE}" \
    --calib_source s1k_original \
    --calib_cache_dir "${CALIB}" \
    "${STRUCTURE_ARGS[@]}" \
    "${FORMAL_FUSABLE_POLICY[@]}"
}

TRAIN_PRESETS=(fusable fusable_r64 online online_diag_then_r64 online_r64_then_diag)
SKIP_TRAIN="${SKIP_TRAIN:-0}"
if [[ "${SKIP_TRAIN}" == "1" ]]; then
  echo "=== Phase-A refactor E3–E7 train skipped (SKIP_TRAIN=1) ==="
else
  echo "=== Phase-A refactor E3–E7 train from layer0 ==="
  i=0
  n=${#TRAIN_PRESETS[@]}
  while (( i < n )); do
    pids=()
    slot=0
    while (( slot < PARALLEL_SLOTS && i < n )); do
      gpu="${AVAILABLE_GPUS[${slot}]}"
      echo "launch gpu=${gpu} train ${TRAIN_PRESETS[${i}]}"
      train_preset "${gpu}" "${TRAIN_PRESETS[${i}]}" &
      pids+=("$!")
      slot=$((slot + 1))
      i=$((i + 1))
    done
    wait_gpu_wave "${pids[@]}"
  done
fi

for preset in "${TRAIN_PRESETS[@]}"; do
  require_complete_train_run "${BASE}/$(phase1_run_name "${preset}")"
done
require_shared_calib_cache "${CALIB}"

mkdir -p "${BASE}/E0_native_nvfp4" "${BASE}/E1_direct_hif4" "${BASE}/E2_r64_only"

EVAL_NAMES=(
  E0_native_nvfp4
  E1_direct_hif4
)
EVAL_VARIANTS=(
  native_nvfp4
  direct_hif4
)
if [[ "${RUN_E2_WITH_ABI3}" == "1" ]]; then
  EVAL_NAMES+=(E2_r64_only)
  EVAL_VARIANTS+=(r64_only)
else
  echo "=== E2 eval deferred; existing pre-ABI3 result remains diagnostic-only ==="
fi
EVAL_NAMES+=(
  E3_fusable
  E4_fusable_r64
  E5_online
  E6_online_diag_then_r64
  E7_online_r64_then_diag
)
EVAL_VARIANTS+=(
  artifact
  artifact
  artifact
  artifact
  artifact
)

require_fast_eval_pool
eval_slots="$(fast_eval_slots)"
i=0
n=${#EVAL_NAMES[@]}
while (( i < n )); do
  pids=()
  slot=0
  while (( slot < eval_slots && i < n )); do
    gpu="$(fast_eval_gpu "${slot}")"
    name="${EVAL_NAMES[${i}]}"
    variant="${EVAL_VARIANTS[${i}]}"
    artifact=""
    if [[ "${variant}" == "artifact" ]]; then
      artifact="${BASE}/${name}/checkpoint/final_model/conversion_state.pt"
    fi
    echo "launch gpu=${gpu} eval ${name}"
    fast_eval_run "${gpu}" "${BASE}/${name}" "${variant}" "${artifact}" &
    pids+=("$!")
    slot=$((slot + 1))
    i=$((i + 1))
  done
  wait_gpu_wave "${pids[@]}"
done

# Task 14.5b: baseline LiveCodeBench. E2 stays deferred unless explicitly enabled.
LCB_NAMES=(E0_native_nvfp4 E1_direct_hif4)
LCB_VARIANTS=(native_nvfp4 direct_hif4)
if [[ "${RUN_E2_WITH_ABI3}" == "1" ]]; then
  LCB_NAMES+=(E2_r64_only)
  LCB_VARIANTS+=(r64_only)
fi
i=0
n=${#LCB_NAMES[@]}
while (( i < n )); do
  pids=()
  slot=0
  while (( slot < eval_slots && i < n )); do
    gpu="$(fast_eval_gpu "${slot}")"
    name="${LCB_NAMES[${i}]}"
    variant="${LCB_VARIANTS[${i}]}"
    echo "launch gpu=${gpu} livecodebench ${name}"
    livecodebench_eval_run "${gpu}" "${BASE}/${name}" "${variant}" &
    pids+=("$!")
    slot=$((slot + 1))
    i=$((i + 1))
  done
  wait_gpu_wave "${pids[@]}"
done

# Candidate diagnostic for E3/E4 (does not overwrite adopted eval/)
mkdir -p "${BASE}/diagnostics"
CAND_NAMES=(E3_fusable E4_fusable_r64)
i=0
n=${#CAND_NAMES[@]}
while (( i < n )); do
  pids=()
  slot=0
  while (( slot < eval_slots && i < n )); do
    gpu="$(fast_eval_gpu "${slot}")"
    name="${CAND_NAMES[${i}]}"
    artifact="${BASE}/${name}/checkpoint/final_model/conversion_state.pt"
    out="${BASE}/diagnostics/${name}_candidate"
    echo "launch gpu=${gpu} candidate-diag ${name}"
    fast_eval_artifact_variant_run "${gpu}" "${out}" "${artifact}" candidate &
    pids+=("$!")
    slot=$((slot + 1))
    i=$((i + 1))
  done
  wait_gpu_wave "${pids[@]}"
done

python - <<PY > "${BASE}/run_map.json"
import json
base = "${BASE}"
print(json.dumps({
  "E0_native_nvfp4": f"{base}/E0_native_nvfp4",
  "E1_direct_hif4": f"{base}/E1_direct_hif4",
  "E2_r64_only": f"{base}/E2_r64_only",
  "E3_fusable": f"{base}/E3_fusable",
  "E4_fusable_r64": f"{base}/E4_fusable_r64",
  "E5_online": f"{base}/E5_online",
  "E6_online_diag_then_r64": f"{base}/E6_online_diag_then_r64",
  "E7_online_r64_then_diag": f"{base}/E7_online_r64_then_diag",
}, indent=2))
PY
e2e_summarize --run_map "${BASE}/run_map.json" --output_json "${BASE}/summary.json" --output_md "${BASE}/report.md"
echo "phaseA_refactor_dir=${BASE}"
