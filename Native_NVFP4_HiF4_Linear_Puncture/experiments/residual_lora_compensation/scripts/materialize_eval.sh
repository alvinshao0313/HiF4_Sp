#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"
MODULE=Native_NVFP4_HiF4_Linear_Puncture.experiments.residual_lora_compensation
RUN_ROOT="${1:?run root}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
BASELINE="${BASELINE_RUN:?existing group_top_mass run directory}"
E4="${E4_RUN:?existing E4 run directory}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
for mode in attention moe both; do
  out="${RUN_ROOT}/${mode}"
  CUDA_VISIBLE_DEVICES="${GPU_A}" conda run --no-capture-output -n hif4 python -m "${MODULE}.materialize" \
    --run_dir "${out}" --output_dir "${out}/model" --device cpu >"${out}/materialize.log" 2>&1
  for task in arc mmlu_pro lcb; do
    CUDA_VISIBLE_DEVICES="${GPU_A},${GPU_B}" conda run --no-capture-output -n hif4 python -m "${MODULE}.evaluate" \
      --model_dir "${out}/model" --output_dir "${out}" --task "${task}" >"${out}/${task}.log" 2>&1
  done
done
conda run --no-capture-output -n hif4 python -m "${MODULE}.summarize" \
  --run_root "${RUN_ROOT}" --baseline_run "${BASELINE}" --e4_run "${E4}"
