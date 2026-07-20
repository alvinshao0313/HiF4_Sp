#!/usr/bin/env bash
# Evaluate a pruned HF checkpoint with the repo's vLLM + lighteval main.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PRUNED_MODEL_DIR="${1:-}"
DATASETS="${DATASETS:-aime25_avg5}"
GPUS="${GPUS:-0,1,2,3}"
TP="${TP:-4}"
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"
# 1=关闭 Qwen thinking（MMLU 等短答案必须开）；0=保持模板默认
DISABLE_THINKING="${DISABLE_THINKING:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/Block_Sparse/results}"

if [[ -z "${PRUNED_MODEL_DIR}" ]]; then
  echo "Usage: $0 <pruned_model_dir> [extra main.py args...]" >&2
  echo "  Env: DATASETS GPUS TP MAX_MODEL_LENGTH MAX_NEW_TOKENS TEMPERATURE TOP_P TOP_K DISABLE_THINKING OUTPUT_DIR" >&2
  exit 1
fi

if [[ "${PRUNED_MODEL_DIR}" != /* ]]; then
  PRUNED_MODEL_DIR="${REPO_ROOT}/${PRUNED_MODEL_DIR}"
fi

if [[ ! -d "${PRUNED_MODEL_DIR}" ]]; then
  echo "pruned_model_dir does not exist: ${PRUNED_MODEL_DIR}" >&2
  exit 1
fi
if [[ ! -f "${PRUNED_MODEL_DIR}/config.json" ]]; then
  echo "config.json missing under ${PRUNED_MODEL_DIR}" >&2
  exit 1
fi

shift || true
mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

extra_args=()
if [[ "${DISABLE_THINKING}" -eq 1 ]]; then
  extra_args+=(--disable_thinking)
fi

echo "[eval] model=${PRUNED_MODEL_DIR} datasets=${DATASETS} gpus=${GPUS} tp=${TP} disable_thinking=${DISABLE_THINKING}"
exec conda run -n hif4 --no-capture-output env CUDA_VISIBLE_DEVICES="${GPUS}" \
  python main.py \
  --model_path "${PRUNED_MODEL_DIR}" \
  --datasets "${DATASETS}" \
  --tensor_parallel_size "${TP}" \
  --max_model_length "${MAX_MODEL_LENGTH}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --top_k "${TOP_K}" \
  --gpu_memory_utilization 0.9 \
  --output_dir "${OUTPUT_DIR}" \
  "${extra_args[@]}" \
  "$@"
