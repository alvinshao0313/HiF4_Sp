#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"
MODULE=Native_NVFP4_HiF4_Linear_Puncture.experiments.progressive_error_cancellation
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/Native_NVFP4_HiF4_Linear_Puncture/results/progressive_error_cancellation/$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_A="${GPU_A:?set GPU_A to an explicit physical GPU}"
GPU_B="${GPU_B:?set GPU_B to an explicit physical GPU}"
if [[ "${GPU_A}" == "${GPU_B}" ]]; then
  echo "GPU_A and GPU_B must be distinct" >&2
  exit 1
fi
if [[ -e "${RUN_ROOT}" && -n "$(find "${RUN_ROOT}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "RUN_ROOT must be new and empty: ${RUN_ROOT}" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

run_one() {
  local gpu="$1" name="$2" method="$3" lambda="$4"
  local out="${RUN_ROOT}/${name}"
  mkdir -p "${out}"
  local resume=()
  if [[ "${RESUME:-0}" == 1 && -f "${out}/manifest.json" ]]; then
    resume=(--resume)
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n hif4 python -m "${MODULE}.train" \
    --output_dir "${out}" --method "${method}" --lambda_direction "${lambda}" \
    "${resume[@]}" >"${out}/train.log" 2>&1
}

run_one "${GPU_A}" baseline baseline 0 & p1=$!
run_one "${GPU_B}" direct_l010 direct 0.1 & p2=$!
wait "${p1}" "${p2}"
run_one "${GPU_A}" direct_l030 direct 0.3 & p1=$!
run_one "${GPU_B}" jvp_l010 jvp 0.1 & p2=$!
wait "${p1}" "${p2}"
run_one "${GPU_A}" jvp_l030 jvp 0.3 & p1=$!
run_one "${GPU_B}" shuffled_l030 shuffled 0.3 & p2=$!
wait "${p1}" "${p2}"

for name in baseline direct_l010 direct_l030 jvp_l010 jvp_l030 shuffled_l030; do
  run="${RUN_ROOT}/${name}"
  CUDA_VISIBLE_DEVICES="${GPU_A}" conda run --no-capture-output -n hif4 python -m "${MODULE}.materialize" \
    --run_dir "${run}" --output_dir "${run}/model" --device cpu >"${run}/materialize.log" 2>&1
  CUDA_VISIBLE_DEVICES="${GPU_A},${GPU_B}" conda run --no-capture-output -n hif4 python -m "${MODULE}.capture" \
    --root "${run}" --output "${run}/captures/native" --native \
    --split holdout >"${run}/capture_native.log" 2>&1
  CUDA_VISIBLE_DEVICES="${GPU_A},${GPU_B}" conda run --no-capture-output -n hif4 python -m "${MODULE}.capture" \
    --root "${run}" --model_dir "${run}/model" --output "${run}/captures/candidate" \
    --split holdout >"${run}/capture.log" 2>&1
  CUDA_VISIBLE_DEVICES="${GPU_A},${GPU_B}" conda run --no-capture-output -n hif4 python -m "${MODULE}.evaluate" \
    --root "${run}" --model_dir "${run}/model" --output_dir "${run}/evaluation" \
    --downstream >"${run}/evaluate.log" 2>&1
  conda run --no-capture-output -n hif4 python -m "${MODULE}.summary" --run_root "${run}"
done
conda run --no-capture-output -n hif4 python -m "${MODULE}.summary" --matrix_root "${RUN_ROOT}"
