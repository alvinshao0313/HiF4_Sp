#!/usr/bin/env bash
# Phase A structure matrix E0–E7. Calibration is s1k_original; no Teacher-CoT.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"

STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
BASE="${RESULTS_ROOT}/phase1_${STAMP}"
CALIB="$(shared_calib_dir_for s1k_original)"
mkdir -p "${BASE}" "${CALIB}"

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
    "${STRUCTURE_ARGS[@]}"
}

TRAIN_PRESETS=(fusable fusable_r64 online online_diag_then_r64 online_r64_then_diag)
SKIP_TRAIN="${SKIP_TRAIN:-0}"
if [[ "${SKIP_TRAIN}" == "1" ]]; then
  echo "=== Phase A E3–E7 train skipped (SKIP_TRAIN=1) ==="
else
  echo "=== Phase A E3–E7 train (shared s1k_original cache) ==="
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
  E2_r64_only
  E3_fusable
  E4_fusable_r64
  E5_online
  E6_online_diag_then_r64
  E7_online_r64_then_diag
)
EVAL_VARIANTS=(
  native_nvfp4
  direct_hif4
  r64_only
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
echo "phase1_dir=${BASE}"
