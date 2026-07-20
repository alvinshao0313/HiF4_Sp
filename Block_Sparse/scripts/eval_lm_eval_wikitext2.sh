#!/usr/bin/env bash
# lm_eval: arc_easy / arc_challenge / mmlu on wikitext2_calib pruned ckpts.
set -euo pipefail

cd /home/shaoyuantian/program/HiF4_Sp

if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate hif4
fi

unset HF_ENDPOINT
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false

GPUS="${GPUS:-4,5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TASKS="${TASKS:-arc_easy,arc_challenge,mmlu}"
OUT_DIR="${OUT_DIR:-Block_Sparse/experiments/wikitext2_calib/results/lm_eval}"
mkdir -p "${OUT_DIR}"

MODELS=()
if [[ "$#" -eq 0 ]]; then
  MODELS=(
    Block_Sparse/experiments/wikitext2_calib/outputs/qwen35_27b_magnitude_s0.20_b64
    Block_Sparse/experiments/wikitext2_calib/outputs/qwen35_27b_magnitude_s0.20_b128
    Block_Sparse/experiments/wikitext2_calib/outputs/qwen35_27b_random_s0.20_b64
    Block_Sparse/experiments/wikitext2_calib/outputs/qwen35_27b_random_s0.20_b128
    Block_Sparse/experiments/wikitext2_calib/outputs/qwen35_27b_fisher_s0.20_b64
    Block_Sparse/experiments/wikitext2_calib/outputs/qwen35_27b_fisher_s0.20_b128
    Block_Sparse/experiments/wikitext2_calib/outputs/qwen35_27b_fisher_s0.20_b64x32
  )
else
  MODELS=("$@")
fi

echo "[eval_lm_eval] GPUS=${GPUS} BATCH_SIZE=${BATCH_SIZE} TASKS=${TASKS}"
for model in "${MODELS[@]}"; do
  tag="$(basename "${model}")"
  out_json="${OUT_DIR}/${tag}_arc_mmlu.json"
  log="${OUT_DIR}/${tag}_arc_mmlu.log"
  if [[ -f "${out_json}" && "${FORCE:-0}" != "1" ]]; then
    echo "======== SKIP ${tag} (exists: ${out_json}; set FORCE=1 to rerun) ========"
    continue
  fi
  echo "======== lm_eval ${model} -> ${out_json} ========"
  CUDA_VISIBLE_DEVICES="${GPUS}" python Block_Sparse/scripts/eval_lm_eval.py \
    --model_path "${model}" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_json "${out_json}" \
    2>&1 | tee "${log}"
done

echo "All lm_eval runs finished. results: ${OUT_DIR}"
