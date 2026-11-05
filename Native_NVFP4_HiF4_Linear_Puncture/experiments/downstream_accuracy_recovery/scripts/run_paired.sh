#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

# The caller activates hif4. run.py assigns the four authorized TP2 GPU pairs.
export OMP_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
export HF_ENDPOINT=https://huggingface.co
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.downstream_accuracy_recovery.run run \
  --run_root "${1:?run root}"
