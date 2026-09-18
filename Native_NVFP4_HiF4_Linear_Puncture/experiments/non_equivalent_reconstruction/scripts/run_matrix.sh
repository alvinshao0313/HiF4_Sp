#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"
MODULE=Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/Native_NVFP4_HiF4_Linear_Puncture/results/non_equivalent_reconstruction/$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
if [[ "${GPU_A}" == "${GPU_B}" ]] || [[ ! "${GPU_A}" =~ ^[0-3]$ ]] || [[ ! "${GPU_B}" =~ ^[0-3]$ ]]; then
  echo 'Use two distinct GPUs from the project pool 0-3.' >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
export OMP_NUM_THREADS=8
run_train() {
  local gpu="$1" sharing="$2" loss="$3"
  local out="${RUN_ROOT}/${sharing}_${loss}"
  mkdir -p "${out}"
  local resume_args=()
  if [[ "${RESUME:-0}" == 1 && -f "${out}/manifest.json" ]]; then resume_args=(--resume); fi
  CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n hif4 python -m "${MODULE}.train" \
    --output_dir "${out}" --matrix_sharing "${sharing}" --router_loss "${loss}" \
    "${resume_args[@]}" >"${out}/train.log" 2>&1
}
for sharing in linear group; do
  run_train "${GPU_A}" "${sharing}" top_partial & pid_a=$!
  run_train "${GPU_B}" "${sharing}" top_mass & pid_b=$!
  # Wait for both jobs and propagate either failure; never start evaluation on a failed run.
  status=0
  wait "${pid_a}" || status=1
  wait "${pid_b}" || status=1
  if [[ "${status}" != 0 ]]; then exit 1; fi
done
for name in linear_top_partial linear_top_mass group_top_partial group_top_mass E4; do
  if [[ "${name}" == E4 ]]; then
    source_run="${RUN_ROOT}/linear_top_mass"
    extra=(--initialization_only)
  else
    source_run="${RUN_ROOT}/${name}"
    extra=()
  fi
  out="${RUN_ROOT}/${name}"
  mkdir -p "${out}"
  CUDA_VISIBLE_DEVICES="${GPU_A}" conda run --no-capture-output -n hif4 python -m "${MODULE}.materialize" \
    --run_dir "${source_run}" --output_dir "${out}/model" --device cuda "${extra[@]}" >"${out}/materialize.log" 2>&1
  for task in arc mmlu_pro; do
    CUDA_VISIBLE_DEVICES="${GPU_A},${GPU_B}" conda run --no-capture-output -n hif4 python -m "${MODULE}.evaluate" \
      --model_dir "${out}/model" --output_dir "${out}" --task "${task}" >"${out}/${task}.log" 2>&1
  done
done
conda run --no-capture-output -n hif4 python -m "${MODULE}.summarize" --run_root "${RUN_ROOT}"
