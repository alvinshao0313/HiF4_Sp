#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

# Invoke from a shell with hif4 already activated.
export CUDA_VISIBLE_DEVICES=2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
export OMP_NUM_THREADS=8
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.residual_lora_compensation.downstream \
  --run_root "${1:?run root}" \
  --baseline_run Native_NVFP4_HiF4_Linear_Puncture/results/non_equivalent_reconstruction/full48_20260915T084300Z/group_top_mass \
  --e4_run Native_NVFP4_HiF4_Linear_Puncture/results/non_equivalent_reconstruction/full48_20260915T084300Z/E4
