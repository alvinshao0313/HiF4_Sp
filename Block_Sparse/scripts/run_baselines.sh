#!/usr/bin/env bash
set -euo pipefail

cd /home/shaoyuantian/program/HiF4_Sp

if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  echo "错误：当前 conda 环境不是 hif4，请先执行：conda activate hif4" >&2
  exit 1
fi

# 去掉 HF 镜像，直连官方 Hub（镜像会导致 mmlu 等数据集加载失败）
unset HF_ENDPOINT
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
echo "[env] HF_ENDPOINT unset; using official huggingface.co"

# ---------------------------------------------------------------------------
# 参数（只在这里改）
# ---------------------------------------------------------------------------
MODEL_PATH=Qwen/Qwen3.5-27B
SPARSITY=0.20
MAX_PRUNE_RATIO_PER_MATRIX=0.30   # 每层/每个 Linear 稀疏率上限
BLOCK_SIZE=64            # 正方形写 128；矩形写 64x128
CALIBRATION_DATASET=s1k   # s1k | wikitext2 | c4 | ptb（fisher / fisher_budget_wanda）
CALIB_SAMPLES=128
SEQ_LEN=0               # 0=不截断（s1k 完整样本）；wiki/c4/ptb 需正整数
SEED=42
DTYPE=bfloat16

# 要跑的方法：magnitude / random / fisher / fisher_budget_wanda
# 单卡冒烟可先只跑 magnitude（不用反向）
METHODS=(magnitude random fisher)

# 是否跳过某阶段：0=执行，1=跳过
SKIP_PRUNE=0
SKIP_EVAL=0

# 剪枝可见卡（多卡会自动 device_map=auto 切分；单卡就写一张）
PRUNE_GPUS=4,5
# 评测可见卡（vLLM；与剪枝可不同）
EVAL_GPUS=4,5
TP=1
DATASETS=mmlu
# MMLU 是选择题，上下文和生成都不需要 32k
MAX_MODEL_LENGTH=4096
MAX_NEW_TOKENS=32
GPU_MEMORY_UTILIZATION=0.9
TEMPERATURE=0.0
TOP_P=1.0
TOP_K=20

OUTPUT_ROOT=./Block_Sparse/outputs
RESULT_ROOT=./Block_Sparse/results
# ---------------------------------------------------------------------------

mkdir -p "${OUTPUT_ROOT}" "${RESULT_ROOT}"

for method in "${METHODS[@]}"; do
  # 标签含校准集名，避免 wikitext2 / s1k 产物互相覆盖
  tag="qwen35_27b_${method}_s${SPARSITY}_b${BLOCK_SIZE}_${CALIBRATION_DATASET}"
  tag="${tag//\//_}"
  out_dir="${OUTPUT_ROOT}/${tag}"
  echo "======== method=${method} output=${out_dir} ========"

  if [[ "${SKIP_PRUNE}" -eq 0 ]]; then
    # 按 PRUNE_GPUS（即 CUDA_VISIBLE_DEVICES）在可见卡上自动切分
    echo "[prune] CUDA_VISIBLE_DEVICES=${PRUNE_GPUS}"
    CUDA_VISIBLE_DEVICES="${PRUNE_GPUS}" python \
      Block_Sparse/scripts/score_and_prune_mlp.py \
      --model_path "${MODEL_PATH}" \
      --output_dir "${out_dir}" \
      --score_type "${method}" \
      --target_block_sparsity "${SPARSITY}" \
      --max_prune_ratio_per_matrix "${MAX_PRUNE_RATIO_PER_MATRIX}" \
      --block_size "${BLOCK_SIZE}" \
      --calibration_dataset "${CALIBRATION_DATASET}" \
      --calibration_samples "${CALIB_SAMPLES}" \
      --sequence_length "${SEQ_LEN}" \
      --seed "${SEED}" \
      --dtype "${DTYPE}" \
      --device cuda
  fi

  if [[ "${SKIP_EVAL}" -eq 0 ]]; then
    CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" python main.py \
      --model_path "${out_dir}" \
      --output_dir "${RESULT_ROOT}" \
      --datasets "${DATASETS}" \
      --max_model_length "${MAX_MODEL_LENGTH}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --temperature "${TEMPERATURE}" \
      --top_p "${TOP_P}" \
      --top_k "${TOP_K}" \
      --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
      --tensor_parallel_size "${TP}"
  fi
done

echo "All methods finished."
echo "  pruned models: ${OUTPUT_ROOT}"
echo "  eval results:  ${RESULT_ROOT}"
