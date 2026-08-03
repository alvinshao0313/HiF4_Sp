#!/usr/bin/env bash
set -euo pipefail

# Block-pruned MLP recovery: peft Masked LoRA + S1K SFT
# 必须在 hif4 conda 环境中运行。实现均在 Block_Sparse 内。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLOCK_SPARSE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BLOCK_SPARSE_ROOT}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${BLOCK_SPARSE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

# ---------------------------------------------------------------------------
# 参数（只在这里改）
# ---------------------------------------------------------------------------
PRUNED_MODEL_DIR="${PRUNED_MODEL_DIR:-${BLOCK_SPARSE_ROOT}/outputs/qwen35_27b_fisher_budget_wanda_s0.20_b64_s1k_permwanda_shared}"
OUTPUT_DIR="${OUTPUT_DIR:-${BLOCK_SPARSE_ROOT}/outputs/qwen35_27b_mlp_lora_sft_s0.20_b64}"
DATASET_NAME="${DATASET_NAME:-simplescaling/s1K-1.1_tokenized}"
DATASET_PATH="${DATASET_PATH:-}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES
PARALLEL_MODE="${PARALLEL_MODE:-layer}"

LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
MAX_STEPS="${MAX_STEPS:-500}"
LR="${LR:-1e-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-32768}"
ALLOW_TRUNCATE="${ALLOW_TRUNCATE:-false}"
LOGIT_CHUNK_SIZE="${LOGIT_CHUNK_SIZE:-512}"

# 蒸馏（teacher=未剪基座；留空=纯 CE SFT）
TEACHER_MODEL_DIR="${TEACHER_MODEL_DIR:-Qwen/Qwen3.5-4B}"
TASK_ALPHA="${TASK_ALPHA:-0.05}"
EAKLD_ALPHA="${EAKLD_ALPHA:-2.0}"
LAFD_ALPHA="${LAFD_ALPHA:-0.5}"
TEMPERATURE="${TEMPERATURE:-1.0}"
LAFD_TOPK="${LAFD_TOPK:-3}"
KL_MODE="${KL_MODE:-eakld}"
KL_TOPK="${KL_TOPK:-0}"
KL_POST_ATTN="${KL_POST_ATTN:-false}"
# ---------------------------------------------------------------------------

if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  echo "ERROR: 请先 conda activate hif4（当前环境=${CONDA_DEFAULT_ENV:-none}）" >&2
  exit 1
fi

if [[ ! -d "${PRUNED_MODEL_DIR}" ]]; then
  echo "ERROR: PRUNED_MODEL_DIR 不存在: ${PRUNED_MODEL_DIR}" >&2
  exit 1
fi
if [[ ! -f "${PRUNED_MODEL_DIR}/pruning_artifacts/block_masks.pt" ]]; then
  echo "ERROR: 缺少 block_masks.pt: ${PRUNED_MODEL_DIR}/pruning_artifacts/" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${DATASET_PATH}" ]]; then
  EXTRA_ARGS+=(--dataset_path "${DATASET_PATH}")
fi
if [[ -n "${TEACHER_MODEL_DIR}" ]]; then
  EXTRA_ARGS+=(
    --teacher_model_dir "${TEACHER_MODEL_DIR}"
    --task_alpha "${TASK_ALPHA}"
    --eakld_alpha "${EAKLD_ALPHA}"
    --lafd_alpha "${LAFD_ALPHA}"
    --temperature "${TEMPERATURE}"
    --lafd_topk "${LAFD_TOPK}"
    --kl_mode "${KL_MODE}"
    --kl_topk "${KL_TOPK}"
    --kl_post_attn "${KL_POST_ATTN}"
  )
fi

echo "[mlp_lora_sft] pruned=${PRUNED_MODEL_DIR}"
echo "[mlp_lora_sft] output=${OUTPUT_DIR}"
echo "[mlp_lora_sft] GPUs=${CUDA_VISIBLE_DEVICES} parallel=${PARALLEL_MODE}"
echo "[mlp_lora_sft] lora_r=${LORA_R} alpha=${LORA_ALPHA} steps=${MAX_STEPS} lr=${LR}"
if [[ -n "${TEACHER_MODEL_DIR}" ]]; then
  echo "[mlp_lora_sft] distill teacher=${TEACHER_MODEL_DIR} kl_mode=${KL_MODE} alphas=${TASK_ALPHA}/${EAKLD_ALPHA}/${LAFD_ALPHA}"
else
  echo "[mlp_lora_sft] distill off (plain CE)"
fi

python "${BLOCK_SPARSE_ROOT}/tools/train_mlp_lora_sft.py" \
  --pruned_model_dir "${PRUNED_MODEL_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --max_steps "${MAX_STEPS}" \
  --learning_rate "${LR}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --allow_truncate "${ALLOW_TRUNCATE}" \
  --logit_chunk_size "${LOGIT_CHUNK_SIZE}" \
  --parallel_mode "${PARALLEL_MODE}" \
  "${EXTRA_ARGS[@]}"

echo "Done. Merged HF model: ${OUTPUT_DIR}"
echo "Eval example:"
echo "  bash Block_Sparse/scripts/eval_ppl.sh ${OUTPUT_DIR}"
