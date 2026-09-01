#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"

if [[ $# -lt 2 ]]; then
  echo "usage: $0 OUT_DIR VARIANT [ARTIFACT_PATH]" >&2
  exit 1
fi
OUT_DIR="$1"
VARIANT="$2"
ARTIFACT="${3:-}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "${GPU_ID}" ]]; then
  echo "set CUDA_VISIBLE_DEVICES to two GPU ids from ${GPU_POOL:-${PROJECT_GPU_POOL}}, e.g. 0,1" >&2
  exit 1
fi
require_gpu_ids_in_pool "${GPU_ID}"
n_gpu=0
IFS=',' read -r -a _eval_gpus <<< "${GPU_ID}"
for tok in "${_eval_gpus[@]}"; do
  tok="${tok// /}"
  [[ -n "${tok}" ]] && n_gpu=$((n_gpu + 1))
done
if [[ "${n_gpu}" -lt 2 ]]; then
  echo "MMLU-Pro needs CUDA_VISIBLE_DEVICES with >=2 GPUs, got ${GPU_ID}" >&2
  exit 1
fi
FAST_EVAL_GROUPS="${FAST_EVAL_GROUPS:-arc,mmlu_pro_300}" \
  EVAL_SEED="${EVAL_SEED:-42}" \
  fast_eval_run "${GPU_ID}" "${OUT_DIR}" "${VARIANT}" "${ARTIFACT}"
