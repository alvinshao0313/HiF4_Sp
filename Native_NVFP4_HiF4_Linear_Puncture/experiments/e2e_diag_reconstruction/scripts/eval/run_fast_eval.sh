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
  echo "set CUDA_VISIBLE_DEVICES to two GPU ids, e.g. 1,6" >&2
  exit 1
fi
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
extra=()
if [[ -n "${ARTIFACT}" ]]; then
  extra+=(--artifact_path "${ARTIFACT}")
fi
CUDA_VISIBLE_DEVICES="${GPU_ID}" e2e_eval \
  --variant "${VARIANT}" \
  --output_dir "${OUT_DIR}" \
  --groups "${FAST_EVAL_GROUPS:-arc,mmlu_pro_300}" \
  --eval_seed "${EVAL_SEED:-42}" \
  "${extra[@]}"
