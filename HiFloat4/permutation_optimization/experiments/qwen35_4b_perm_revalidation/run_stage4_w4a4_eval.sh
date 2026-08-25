#!/usr/bin/env bash
# Stage 4: paired W4A4 end-to-end evaluation.
#   bf16_identity / bf16_permuted  (from Stage 3, referenced for drift check)
#   w4a4_identity / w4a4_permuted  (RTN + activation fake quant, this stage)
# Plus MMLU-Pro with deterministic decoding (temperature=0.0, top_p=1.0).
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/Block_Sparse:${REPO_ROOT}/HiFloat4:${PYTHONPATH:-}"

HIF4_PY="${HIF4_PY:-/home/shaoyuantian/anaconda3/envs/hif4/bin/python}"
export PATH="$(dirname "${HIF4_PY}"):${PATH}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  set +u
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate hif4
  set -u
fi

GPUS="${GPUS:-4}"
DTYPE="${DTYPE:-bfloat16}"
IDENTITY="${EXP_DIR}/tmp/stage2_identity_bf16"
REORDERED="${EXP_DIR}/tmp/stage2_reordered_bf16"
CKPT_ID="${EXP_DIR}/tmp/stage4_rtn_identity"
CKPT_PERM="${EXP_DIR}/tmp/stage4_rtn_permuted"
OUT="${EXP_DIR}/results/stage4_w4a4"
LOG="${EXP_DIR}/logs/stage4_w4a4.log"
TASKS="${TASKS:-arc_easy,arc_challenge,mmlu,wikitext}"

MMLU_PRO_MAX_SAMPLES="${MMLU_PRO_MAX_SAMPLES:-1000}"
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

mkdir -p "${OUT}" "$(dirname "${LOG}")"
: > "${LOG}"

# 1) RTN both BF16 variants into HiF4 W4A4 ckpts.
for pair in "identity:${IDENTITY}:${CKPT_ID}" "permuted:${REORDERED}:${CKPT_PERM}"; do
  name="$(echo "${pair}" | cut -d: -f1)"
  src="$(echo "${pair}" | cut -d: -f2)"
  dst="$(echo "${pair}" | cut -d: -f3)"
  if [[ ! -f "${src}/config.json" ]]; then
    echo "missing ckpt ${src}; run Stage 2 first" >&2
    exit 1
  fi
  echo "[$(date --iso-8601=seconds)] Stage 4 RTN ${name}" | tee -a "${LOG}"
  rm -rf "${dst}"
  mkdir -p "${dst}"
  CUDA_VISIBLE_DEVICES="${GPUS}" "${HIF4_PY}" HiFloat4/main.py \
    --model "${src}" \
    --dtype "${DTYPE}" \
    --hif4w true \
    --hif4_weight_format hif4 \
    --gptq false \
    --gptq_save_path "${dst}" \
    --exclude-layers lm_head \
    --trust-remote-code true \
    2>&1 | tee -a "${LOG}"
  "${HIF4_PY}" - <<PY
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("${src}", trust_remote_code=True)
tok.save_pretrained("${dst}")
print("tokenizer saved")
PY
done

# 2) Paired lm_eval (same driver/tasks as Stage 3).
for pair in "identity:${CKPT_ID}" "permuted:${CKPT_PERM}"; do
  name="${pair%%:*}"
  ckpt="${pair##*:}"
  echo "[$(date --iso-8601=seconds)] Stage 4 w4a4 lm_eval ${name}" | tee -a "${LOG}"
  CUDA_VISIBLE_DEVICES="${GPUS}" "${HIF4_PY}" "${EXP_DIR}/eval_paired.py" \
    --model_path "${ckpt}" \
    --mode w4a4 \
    --tasks "${TASKS}" \
    --num_fewshot 0 \
    --batch_size 8 \
    --output_json "${OUT}/w4a4_${name}.json" \
    2>&1 | tee -a "${LOG}"
done

# 3) MMLU-Pro deterministic decoding (temperature=0.0, top_p=1.0).
for pair in "identity:${CKPT_ID}" "permuted:${CKPT_PERM}"; do
  name="${pair%%:*}"
  ckpt="${pair##*:}"
  echo "[$(date --iso-8601=seconds)] Stage 4 mmlu_pro ${name}" | tee -a "${LOG}"
  CUDA_VISIBLE_DEVICES="${GPUS}" "${HIF4_PY}" main.py \
    --model_path "${ckpt}" \
    --datasets "mmlu_pro|0" \
    --max_samples "${MMLU_PRO_MAX_SAMPLES}" \
    --tensor_parallel_size 1 \
    --max_model_length "${MAX_MODEL_LENGTH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature 0.0 \
    --top_p 1.0 \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --fake_act_quant hif4 \
    --fake_act_quant_exclude lm_head \
    --disable_thinking \
    --output_dir "${OUT}/mmlu_pro_${name}" \
    2>&1 | tee -a "${LOG}"
done

echo "[$(date --iso-8601=seconds)] Stage 4 done; ckpts kept at ${CKPT_ID} / ${CKPT_PERM}" | tee -a "${LOG}"
