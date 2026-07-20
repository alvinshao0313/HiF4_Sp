#!/usr/bin/env bash
# 单独执行 MLP 块剪枝（不评测）。风格对齐 scripts/test.sh。
set -euo pipefail

cd /home/shaoyuantian/program/HiF4_Sp

if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  echo "错误：当前 conda 环境不是 hif4，请先执行：conda activate hif4" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 参数（只在这里改）
# ---------------------------------------------------------------------------
MODEL_PATH=Qwen/Qwen3.5-27B
SCORE_TYPE=fisher          # fisher | magnitude | random | fisher_budget_wanda
SPARSITY=0.30
BLOCK_SIZE=128             # 正方形写 128；矩形写 64x128（H=d_out, W=d_in）
CALIBRATION_DATASET=s1k   # s1k | wikitext2 | c4 | ptb（fisher / fisher_budget_wanda）
CALIB_SAMPLES=128
SEQ_LEN=0               # 0=不截断（s1k 完整样本）；wiki/c4/ptb 需正整数
SEED=42
DTYPE=bfloat16
DEVICE=cuda
# 多卡示例：CUDA_VISIBLE_DEVICES=6,7 bash ... （可见卡上自动 device_map=auto）
# CUDA_VISIBLE_DEVICES 由调用方导出；此处不改写

OUTPUT_DIR=./Block_Sparse/outputs/qwen35_27b_${SCORE_TYPE}_s${SPARSITY}_b${BLOCK_SIZE}_${CALIBRATION_DATASET}
# ---------------------------------------------------------------------------

echo "[prune_mlp] model=${MODEL_PATH}"
echo "[prune_mlp] score_type=${SCORE_TYPE} sparsity=${SPARSITY} block_size=${BLOCK_SIZE}"
echo "[prune_mlp] calib=${CALIBRATION_DATASET} n=${CALIB_SAMPLES} seq=${SEQ_LEN}"
echo "[prune_mlp] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} output=${OUTPUT_DIR}"

python Block_Sparse/scripts/score_and_prune_mlp.py \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --score_type "${SCORE_TYPE}" \
  --target_block_sparsity "${SPARSITY}" \
  --block_size "${BLOCK_SIZE}" \
  --calibration_dataset "${CALIBRATION_DATASET}" \
  --calibration_samples "${CALIB_SAMPLES}" \
  --sequence_length "${SEQ_LEN}" \
  --seed "${SEED}" \
  --dtype "${DTYPE}" \
  --device "${DEVICE}"

echo "[prune_mlp] done: ${OUTPUT_DIR}"
