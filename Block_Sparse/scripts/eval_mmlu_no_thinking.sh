#!/usr/bin/env bash
# 对 magnitude / random 两个剪枝 ckpt 跑 MMLU（关 thinking，直接答选项）。
# 若指定 GPU 显存不足，会轮询等待直到空闲再开跑。
set -euo pipefail

cd /home/shaoyuantian/program/HiF4_Sp

if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate hif4
fi

MODELS=(
  Block_Sparse/experiments/wikitext2_calib/outputs/qwen35_27b_magnitude_s0.20_b128
  Block_Sparse/experiments/wikitext2_calib/outputs/qwen35_27b_random_s0.20_b128
)

GPUS="${GPUS:-7}"
TP="${TP:-1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-70000}"
POLL_SEC="${POLL_SEC:-60}"
LOG_DIR="${LOG_DIR:-Block_Sparse/experiments/wikitext2_calib/results/eval_mmlu_logs}"
mkdir -p "${LOG_DIR}"

wait_for_gpu() {
  local gpu="$1"
  while true; do
    local free
    free="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    echo "[wait] GPU${gpu} free=${free} MiB (need >= ${MIN_FREE_MIB})"
    if [[ "${free}" -ge "${MIN_FREE_MIB}" ]]; then
      return 0
    fi
    sleep "${POLL_SEC}"
  done
}

# 单卡号（CUDA_VISIBLE_DEVICES 里第一张）用于轮询
PRIMARY_GPU="${GPUS%%,*}"
wait_for_gpu "${PRIMARY_GPU}"

for model in "${MODELS[@]}"; do
  tag="$(basename "${model}")"
  log="${LOG_DIR}/${tag}_mmlu.log"
  echo "======== eval ${model} -> ${log} ========"
  DATASETS=mmlu \
  GPUS="${GPUS}" \
  TP="${TP}" \
  MAX_MODEL_LENGTH=4096 \
  MAX_NEW_TOKENS=32 \
  TEMPERATURE=0.0 \
  TOP_P=1.0 \
  TOP_K=20 \
  DISABLE_THINKING=1 \
  OUTPUT_DIR=Block_Sparse/results \
    bash Block_Sparse/scripts/eval_pruned.sh "${model}" 2>&1 | tee "${log}"
done

echo "All MMLU evals finished. logs: ${LOG_DIR}"
