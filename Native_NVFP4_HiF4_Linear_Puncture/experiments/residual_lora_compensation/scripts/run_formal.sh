#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"
MODULE=Native_NVFP4_HiF4_Linear_Puncture.experiments.residual_lora_compensation
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/Native_NVFP4_HiF4_Linear_Puncture/results/residual_lora_compensation/$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"
if [[ "${GPU_A}" == "${GPU_B}" || ! "${GPU_A}" =~ ^[0-7]$ || ! "${GPU_B}" =~ ^[0-7]$ ]]; then
  echo "GPU_A and GPU_B must be distinct visible GPU ids" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
export OMP_NUM_THREADS=8

run_train() {
  local gpu="$1"
  local mode="$2"
  local out="${RUN_ROOT}/${mode}"
  mkdir -p "${out}"
  local resume_args=()
  if [[ "${RESUME:-0}" == 1 && -f "${out}/manifest.json" ]]; then
    resume_args=(--resume)
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n hif4 python -m "${MODULE}.train" \
    --output_dir "${out}" --matrix_sharing group --lora_mode "${mode}" \
    --router_loss top_mass "${resume_args[@]}" >"${out}/train.log" 2>&1
}

run_train "${GPU_A}" attention & pid_a=$!
run_train "${GPU_B}" moe & pid_b=$!
status=0
wait "${pid_a}" || status=1
wait "${pid_b}" || status=1
if [[ "${status}" != 0 ]]; then
  echo "attention/moe formal training failed" >&2
  exit 1
fi
run_train "${GPU_A}" both
echo "formal training complete: ${RUN_ROOT}"
