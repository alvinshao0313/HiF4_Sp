#!/usr/bin/env bash
# Resume after successful MLP reorder: RTN → lm_eval → lighteval → summarize.
set -eo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/HiFloat4:${PYTHONPATH:-}"

HIF4_PY="${HIF4_PY:-/home/shaoyuantian/anaconda3/envs/hif4/bin/python}"
export PATH="$(dirname "${HIF4_PY}"):${PATH}"
# eval_lm_eval_hif4a.py lives under a symlink into Block_Sparse/experiments; resolve()
# would set REPO_ROOT wrong, so put Block_Sparse on PYTHONPATH explicitly.
export PYTHONPATH="${REPO_ROOT}/Block_Sparse:${REPO_ROOT}/HiFloat4:${PYTHONPATH:-}"

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
GPUS="${GPUS:-4}"
DTYPE="${DTYPE:-bfloat16}"
HIF4_WEIGHT_FORMAT="${HIF4_WEIGHT_FORMAT:-hif4}"
FAKE_ACT_QUANT="${FAKE_ACT_QUANT:-hif4}"

REORDERED="${EXP_DIR}/tmp/reordered_bf16"
CKPT_DIR="${EXP_DIR}/tmp/perm_rtn_ckpt"
OUT_DIR="${EXP_DIR}/results/perm_rtn"
LOG="${EXP_DIR}/logs/perm_rtn.log"
ABLATION_EVAL="${REPO_ROOT}/HiF4_exp/qwen35_4b_w4a4_proj_ablation/eval_lm_eval_hif4a.py"

if [[ ! -f "${REORDERED}/config.json" ]]; then
  echo "reordered model missing: ${REORDERED}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "[$(date --iso-8601=seconds)] === 2) HiF4 W4A4 RTN on reordered model (resume) ===" | tee -a "${LOG}"
if [[ -f "${CKPT_DIR}/config.json" ]]; then
  echo "Reusing existing RTN ckpt: ${CKPT_DIR}" | tee -a "${LOG}"
else
  rm -rf "${CKPT_DIR}"
  mkdir -p "${CKPT_DIR}"
  CUDA_VISIBLE_DEVICES="${GPUS}" "${HIF4_PY}" HiFloat4/main.py \
    --model "${REORDERED}" \
    --dtype "${DTYPE}" \
    --hif4w true \
    --hif4_weight_format "${HIF4_WEIGHT_FORMAT}" \
    --gptq false \
    --gptq_save_path "${CKPT_DIR}" \
    --exclude-layers lm_head \
    --trust-remote-code true \
    2>&1 | tee -a "${LOG}"
fi

if [[ ! -f "${CKPT_DIR}/config.json" ]]; then
  echo "RTN failed, missing config.json: ${CKPT_DIR}" >&2
  exit 1
fi

"${HIF4_PY}" - <<PY
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("${MODEL}", trust_remote_code=True)
tok.save_pretrained("${CKPT_DIR}")
print("tokenizer saved")
PY

echo "[$(date --iso-8601=seconds)] === 3) lm_eval arc/mmlu ===" | tee -a "${LOG}"
LM_JSON="${OUT_DIR}/lm_eval_arc_mmlu.json"
CUDA_VISIBLE_DEVICES="${GPUS}" "${HIF4_PY}" \
  "${ABLATION_EVAL}" \
  --model_path "${CKPT_DIR}" \
  --tasks arc_easy,arc_challenge,mmlu \
  --num_fewshot 0 \
  --batch_size 8 \
  --dtype "${DTYPE}" \
  --fake_act_quant "${FAKE_ACT_QUANT}" \
  --fake_act_quant_exclude lm_head \
  --output_json "${LM_JSON}" \
  2>&1 | tee -a "${LOG}"

echo "[$(date --iso-8601=seconds)] === 4) lighteval mmlu_pro ===" | tee -a "${LOG}"
MMLU_PRO_DIR="${OUT_DIR}/mmlu_pro"
mkdir -p "${MMLU_PRO_DIR}"
CUDA_VISIBLE_DEVICES="${GPUS}" "${HIF4_PY}" main.py \
  --model_path "${CKPT_DIR}" \
  --datasets "mmlu_pro|0" \
  --max_samples 300 \
  --tensor_parallel_size 1 \
  --max_model_length 32768 \
  --max_new_tokens 32768 \
  --temperature 0.7 \
  --top_p 0.8 \
  --top_k 20 \
  --gpu_memory_utilization 0.9 \
  --fake_act_quant "${FAKE_ACT_QUANT}" \
  --fake_act_quant_exclude lm_head \
  --disable_thinking \
  --output_dir "${MMLU_PRO_DIR}" \
  2>&1 | tee -a "${LOG}"

echo "[$(date --iso-8601=seconds)] === 5) summarize ===" | tee -a "${LOG}"
"${HIF4_PY}" "${EXP_DIR}/summarize_compare.py" \
  --perm_rtn_dir "${OUT_DIR}" \
  --baseline_summary "${REPO_ROOT}/HiF4_exp/qwen35_4b_w4a4_proj_ablation/results/summary.json" \
  --output_dir "${EXP_DIR}/results"

rm -rf "${CKPT_DIR}"
echo "[$(date --iso-8601=seconds)] done" | tee -a "${LOG}"
