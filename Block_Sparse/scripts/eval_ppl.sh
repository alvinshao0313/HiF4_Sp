#!/usr/bin/env bash
# WikiText-2 PPL for pruned (or dense) HF checkpoints.
set -euo pipefail

cd /home/shaoyuantian/program/HiF4_Sp

if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  echo "错误：当前 conda 环境不是 hif4，请先执行：conda activate hif4" >&2
  exit 1
fi

unset HF_ENDPOINT
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0

GPUS="${GPUS:-6}"
SEQ_LEN="${SEQ_LEN:-2048}"
DTYPE="${DTYPE:-bfloat16}"
OUT_DIR="${OUT_DIR:-Block_Sparse/experiments/wikitext2_calib/results/ppl}"
mkdir -p "${OUT_DIR}"

MODELS=()
if [[ "$#" -eq 0 ]]; then
  MODELS=(
    Block_Sparse/experiments/wikitext2_calib/outputs/qwen35_27b_magnitude_s0.20_b128
    Block_Sparse/experiments/wikitext2_calib/outputs/qwen35_27b_random_s0.20_b128
    Block_Sparse/experiments/wikitext2_calib/outputs/qwen35_27b_fisher_s0.20_b128
  )
else
  MODELS=("$@")
fi

echo "[eval_ppl] GPUS=${GPUS} SEQ_LEN=${SEQ_LEN}"
for model in "${MODELS[@]}"; do
  tag="$(basename "${model}")"
  out_json="${OUT_DIR}/${tag}_wikitext2_s${SEQ_LEN}.json"
  log="${OUT_DIR}/${tag}_wikitext2_s${SEQ_LEN}.log"
  echo "======== PPL ${model} -> ${out_json} ========"
  CUDA_VISIBLE_DEVICES="${GPUS}" python Block_Sparse/scripts/eval_ppl.py \
    --model_path "${model}" \
    --dataset wikitext2 \
    --sequence_length "${SEQ_LEN}" \
    --dtype "${DTYPE}" \
    --output_json "${out_json}" \
    2>&1 | tee "${log}"
done

echo "All PPL evals finished. results: ${OUT_DIR}"
