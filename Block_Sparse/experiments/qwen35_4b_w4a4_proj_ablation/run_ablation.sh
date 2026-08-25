#!/usr/bin/env bash
# Qwen3.5-4B HiF4 W4A4 RTN 投影消融
# - arc_easy / arc_challenge / mmlu：lm_eval 0-shot（权重 RTN ckpt + HiF4 激活 fake quant）
# - mmlu_pro：lighteval，max_samples=300（对齐 Block_Sparse report.html §12 / dense_baseline）
# 临时 ckpt 评完即删；结果落在本目录 results/。
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

HIF4_PY="${HIF4_PY:-/home/shaoyuantian/anaconda3/envs/hif4/bin/python}"
if [[ ! -x "${HIF4_PY}" ]]; then
  echo "错误：找不到 hif4 python: ${HIF4_PY}" >&2
  exit 1
fi
export PATH="$(dirname "${HIF4_PY}"):${PATH}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate hif4
  set -u
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  echo "错误：需要 hif4 conda 环境" >&2
  exit 1
fi

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
GPUS="${GPUS:-7}"
TP="${TP:-1}"
DTYPE="${DTYPE:-bfloat16}"
HIF4_WEIGHT_FORMAT="${HIF4_WEIGHT_FORMAT:-hif4}"
FAKE_ACT_QUANT="${FAKE_ACT_QUANT:-hif4}"

# lm_eval（对齐 report：0-shot ARC/MMLU）
LM_EVAL_TASKS="${LM_EVAL_TASKS:-arc_easy,arc_challenge,mmlu}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-8}"
LM_EVAL_NUM_FEWSHOT="${LM_EVAL_NUM_FEWSHOT:-0}"

# lighteval mmlu_pro 300（对齐 dense_baseline / report §12）
MMLU_PRO_DATASET="${MMLU_PRO_DATASET:-mmlu_pro|0}"
MMLU_PRO_MAX_SAMPLES="${MMLU_PRO_MAX_SAMPLES:-300}"
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

VARIANTS="${VARIANTS:-full,skip_gate_up,skip_down,skip_o_proj,skip_mlp}"

RESULTS_ROOT="${EXP_DIR}/results"
TMP_ROOT="${EXP_DIR}/tmp"
LOG_ROOT="${EXP_DIR}/logs"
mkdir -p "${RESULTS_ROOT}" "${TMP_ROOT}" "${LOG_ROOT}"

exclude_for_variant() {
  local variant="$1"
  case "${variant}" in
    full) echo "lm_head" ;;
    skip_gate_up) echo "lm_head,gate_proj,up_proj" ;;
    skip_down) echo "lm_head,down_proj" ;;
    skip_o_proj) echo "lm_head,o_proj" ;;
    skip_mlp) echo "lm_head,gate_proj,up_proj,down_proj" ;;
    *)
      echo "未知变体: ${variant}" >&2
      exit 1
      ;;
  esac
}

exclude_to_hif4_args() {
  local csv="$1"
  local -a parts=()
  IFS=',' read -r -a parts <<< "${csv}"
  printf '%s\n' "${parts[@]}"
}

run_one() {
  local variant="$1"
  local exclude_csv
  exclude_csv="$(exclude_for_variant "${variant}")"
  local out_dir="${RESULTS_ROOT}/${variant}"
  local ckpt_dir="${TMP_ROOT}/${variant}"
  local log_file="${LOG_ROOT}/${variant}.log"
  mkdir -p "${out_dir}"

  echo "[$(date --iso-8601=seconds)] === ${variant} exclude=${exclude_csv} ===" | tee -a "${log_file}"

  rm -rf "${ckpt_dir}"
  mkdir -p "${ckpt_dir}"
  mapfile -t EXCLUDE_LAYERS < <(exclude_to_hif4_args "${exclude_csv}")

  echo "[$(date --iso-8601=seconds)] RTN quantize -> ${ckpt_dir}" | tee -a "${log_file}"
  CUDA_VISIBLE_DEVICES="${GPUS}" "${HIF4_PY}" HiFloat4/main.py \
    --model "${MODEL}" \
    --dtype "${DTYPE}" \
    --hif4w true \
    --hif4_weight_format "${HIF4_WEIGHT_FORMAT}" \
    --gptq false \
    --gptq_save_path "${ckpt_dir}" \
    --exclude-layers "${EXCLUDE_LAYERS[@]}" \
    >>"${log_file}" 2>&1

  if [[ ! -f "${ckpt_dir}/config.json" ]]; then
    echo "量化失败，缺少 config.json: ${ckpt_dir}" | tee -a "${log_file}" >&2
    exit 1
  fi

  "${HIF4_PY}" - <<PY
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("${MODEL}", trust_remote_code=True)
tok.save_pretrained("${ckpt_dir}")
print("tokenizer saved to ${ckpt_dir}")
PY

  # 1) lm_eval：arc / mmlu（0-shot）+ HiF4 激活
  local lm_json="${out_dir}/lm_eval_arc_mmlu.json"
  echo "[$(date --iso-8601=seconds)] lm_eval -> ${lm_json}" | tee -a "${log_file}"
  CUDA_VISIBLE_DEVICES="${GPUS}" "${HIF4_PY}" \
    "${EXP_DIR}/eval_lm_eval_hif4a.py" \
    --model_path "${ckpt_dir}" \
    --tasks "${LM_EVAL_TASKS}" \
    --num_fewshot "${LM_EVAL_NUM_FEWSHOT}" \
    --batch_size "${LM_EVAL_BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --fake_act_quant "${FAKE_ACT_QUANT}" \
    --fake_act_quant_exclude "${exclude_csv}" \
    --output_json "${lm_json}" \
    >>"${log_file}" 2>&1

  # 2) lighteval：mmlu_pro 300（对齐 report §12）
  local mmlu_pro_dir="${out_dir}/mmlu_pro"
  mkdir -p "${mmlu_pro_dir}"
  echo "[$(date --iso-8601=seconds)] lighteval mmlu_pro max_samples=${MMLU_PRO_MAX_SAMPLES} -> ${mmlu_pro_dir}" | tee -a "${log_file}"
  CUDA_VISIBLE_DEVICES="${GPUS}" "${HIF4_PY}" main.py \
    --model_path "${ckpt_dir}" \
    --datasets "${MMLU_PRO_DATASET}" \
    --max_samples "${MMLU_PRO_MAX_SAMPLES}" \
    --tensor_parallel_size "${TP}" \
    --max_model_length "${MAX_MODEL_LENGTH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --fake_act_quant "${FAKE_ACT_QUANT}" \
    --fake_act_quant_exclude "${exclude_csv}" \
    --disable_thinking \
    --output_dir "${mmlu_pro_dir}" \
    >>"${log_file}" 2>&1

  echo "[$(date --iso-8601=seconds)] remove ckpt ${ckpt_dir}" | tee -a "${log_file}"
  rm -rf "${ckpt_dir}"
  echo "[$(date --iso-8601=seconds)] done ${variant}" | tee -a "${log_file}"
}

IFS=',' read -r -a VARIANT_LIST <<< "${VARIANTS}"
for variant in "${VARIANT_LIST[@]}"; do
  variant="$(echo "${variant}" | xargs)"
  [[ -z "${variant}" ]] && continue
  run_one "${variant}"
done

echo "[$(date --iso-8601=seconds)] summarize"
"${HIF4_PY}" "${EXP_DIR}/summarize_results.py" --results_root "${RESULTS_ROOT}"

echo "全部完成。汇总: ${RESULTS_ROOT}/summary.json"
