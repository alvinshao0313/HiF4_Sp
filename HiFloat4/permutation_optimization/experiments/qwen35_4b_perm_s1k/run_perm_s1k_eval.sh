#!/usr/bin/env bash
# Apply s1k perm results → HiF4 W4A4 RTN → arc/mmlu + mmlu_pro (no re-search).
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/Block_Sparse:${REPO_ROOT}/HiFloat4:${PYTHONPATH:-}"

HIF4_PY="${HIF4_PY:-/home/shaoyuantian/anaconda3/envs/hif4/bin/python}"
export PATH="$(dirname "${HIF4_PY}"):${PATH}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate hif4
  set -u
fi

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
GPUS="${GPUS:-4}"
TP="${TP:-1}"
DTYPE="${DTYPE:-bfloat16}"
HIF4_WEIGHT_FORMAT="${HIF4_WEIGHT_FORMAT:-hif4}"
FAKE_ACT_QUANT="${FAKE_ACT_QUANT:-hif4}"

LM_EVAL_TASKS="${LM_EVAL_TASKS:-arc_easy,arc_challenge,mmlu}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-8}"
LM_EVAL_NUM_FEWSHOT="${LM_EVAL_NUM_FEWSHOT:-0}"

MMLU_PRO_DATASET="${MMLU_PRO_DATASET:-mmlu_pro|0}"
MMLU_PRO_MAX_SAMPLES="${MMLU_PRO_MAX_SAMPLES:-300}"
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

PERM_PT="${EXP_DIR}/results/perm_search/permutations.pt"
REORDERED="${EXP_DIR}/tmp/reordered_bf16"
CKPT_DIR="${EXP_DIR}/tmp/perm_rtn_ckpt"
OUT_DIR="${EXP_DIR}/results/perm_rtn"
LOG="${EXP_DIR}/logs/perm_s1k_eval.log"
ABLATION_EVAL="${REPO_ROOT}/HiF4_exp/qwen35_4b_w4a4_proj_ablation/eval_lm_eval_hif4a.py"
SUMMARIZE="${REPO_ROOT}/HiFloat4/permutation_optimization/experiments/qwen35_4b_perm_rtn/summarize_compare.py"

mkdir -p "${OUT_DIR}" "$(dirname "${LOG}")" "${EXP_DIR}/tmp"
: > "${LOG}"

if [[ ! -f "${PERM_PT}" ]]; then
  echo "missing ${PERM_PT}" >&2
  exit 1
fi

echo "[$(date --iso-8601=seconds)] === 1) Apply s1k permutations → BF16 ===" | tee -a "${LOG}"
rm -rf "${REORDERED}"
CUDA_VISIBLE_DEVICES="${GPUS}" "${HIF4_PY}" - <<PY
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from permutation_optimization.model_permutation import apply_permutations_from_file

model_id = "${MODEL}"
out = Path("${REORDERED}")
out.mkdir(parents=True, exist_ok=True)
print("Loading", model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cpu"
)
apply_permutations_from_file(model, "${PERM_PT}")
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
tok.save_pretrained(out)
print("saved", out)
PY

if [[ ! -f "${REORDERED}/config.json" ]]; then
  echo "reordered model missing" >&2
  exit 1
fi

echo "[$(date --iso-8601=seconds)] === 2) HiF4 W4A4 RTN ===" | tee -a "${LOG}"
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
  --tasks "${LM_EVAL_TASKS}" \
  --num_fewshot "${LM_EVAL_NUM_FEWSHOT}" \
  --batch_size "${LM_EVAL_BATCH_SIZE}" \
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
  --fake_act_quant_exclude lm_head \
  --disable_thinking \
  --output_dir "${MMLU_PRO_DIR}" \
  2>&1 | tee -a "${LOG}"

echo "[$(date --iso-8601=seconds)] === 5) summarize ===" | tee -a "${LOG}"
"${HIF4_PY}" "${SUMMARIZE}" \
  --perm_rtn_dir "${OUT_DIR}" \
  --baseline_summary "${REPO_ROOT}/HiF4_exp/qwen35_4b_w4a4_proj_ablation/results/summary.json" \
  --output_dir "${EXP_DIR}/results"

echo "[$(date --iso-8601=seconds)] remove ckpt ${CKPT_DIR}" | tee -a "${LOG}"
rm -rf "${CKPT_DIR}"
echo "[$(date --iso-8601=seconds)] done" | tee -a "${LOG}"
cat "${EXP_DIR}/results/summary.md"
