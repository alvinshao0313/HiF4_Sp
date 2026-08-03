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
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
SCORE_TYPE="${SCORE_TYPE:-fisher_budget_wanda}"          # fisher | magnitude | random | fisher_budget_wanda
SPARSITY="${SPARSITY:-0.20}"                           # 全局块稀疏率（所有层 u/g/d 合计）
MAX_PRUNE_RATIO_PER_MATRIX="${MAX_PRUNE_RATIO_PER_MATRIX:-0.80}"         # 单个 Linear 最多可剪掉的块比例上限
# 空=现行全局跨 u/g/d 排序；示例: gate_proj=1,up_proj=1,down_proj=2
# 设了以后：在全局 SPARSITY 预算内按份额分配三类的剪块数
PROJECTION_PRUNE_SHARES="${PROJECTION_PRUNE_SHARES:-}"
BLOCK_SIZE="${BLOCK_SIZE:-64x32}"             # 正方形写 128；矩形写 64x128（H=d_out, W=d_in）
CALIBRATION_DATASET="${CALIBRATION_DATASET:-s1k}"   # s1k | wikitext2 | c4 | ptb（fisher / fisher_budget_wanda）
CALIB_SAMPLES="${CALIB_SAMPLES:-128}"
SEQ_LEN="${SEQ_LEN:-0}"               # 0=不截断（s1k 完整样本）；wiki/c4/ptb 需正整数
SEED="${SEED:-42}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
MLP_PERMUTATION="${MLP_PERMUTATION:-wanda_shared}"   # none | wanda_shared
RESIDUAL_PERMUTATION="${RESIDUAL_PERMUTATION:-none}"      # none | block_loss
RESIDUAL_PERM_SEARCH_STEPS="${RESIDUAL_PERM_SEARCH_STEPS:-2000}"
RESIDUAL_CHANNEL_AGG="${RESIDUAL_CHANNEL_AGG:-equal}"     # equal | layer_fisher | matrix_fisher | raw_wanda | sparsity_raw_wanda | density_raw_wanda
# 多卡示例：CUDA_VISIBLE_DEVICES=6,7 bash ... （可见卡上自动 device_map=auto）
# CUDA_VISIBLE_DEVICES 由调用方导出；此处不改写

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR=./Block_Sparse/outputs/qwen35_4b_${SCORE_TYPE}_s${SPARSITY}_b${BLOCK_SIZE}_${CALIBRATION_DATASET}_perm${MLP_PERMUTATION}_rperm${RESIDUAL_PERMUTATION}
  if [[ "${RESIDUAL_PERMUTATION}" == "block_loss" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}_s${RESIDUAL_PERM_SEARCH_STEPS}_agg${RESIDUAL_CHANNEL_AGG}"
  fi
fi
# ---------------------------------------------------------------------------

echo "[prune_mlp] model=${MODEL_PATH}"
echo "[prune_mlp] score_type=${SCORE_TYPE} sparsity=${SPARSITY} block_size=${BLOCK_SIZE}"
echo "[prune_mlp] max_prune_ratio_per_matrix=${MAX_PRUNE_RATIO_PER_MATRIX}"
echo "[prune_mlp] projection_prune_shares=${PROJECTION_PRUNE_SHARES:-<unset>}"
echo "[prune_mlp] calib=${CALIBRATION_DATASET} n=${CALIB_SAMPLES} seq=${SEQ_LEN}"
echo "[prune_mlp] mlp_permutation=${MLP_PERMUTATION}"
echo "[prune_mlp] residual_permutation=${RESIDUAL_PERMUTATION} search_steps=${RESIDUAL_PERM_SEARCH_STEPS} channel_agg=${RESIDUAL_CHANNEL_AGG}"
echo "[prune_mlp] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} output=${OUTPUT_DIR}"

EXTRA_ARGS=()
if [[ -n "${PROJECTION_PRUNE_SHARES}" ]]; then
  EXTRA_ARGS+=(--projection_prune_shares "${PROJECTION_PRUNE_SHARES}")
fi

python Block_Sparse/tools/score_and_prune_mlp.py \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --score_type "${SCORE_TYPE}" \
  --target_block_sparsity "${SPARSITY}" \
  --max_prune_ratio_per_matrix "${MAX_PRUNE_RATIO_PER_MATRIX}" \
  --block_size "${BLOCK_SIZE}" \
  --calibration_dataset "${CALIBRATION_DATASET}" \
  --calibration_samples "${CALIB_SAMPLES}" \
  --sequence_length "${SEQ_LEN}" \
  --seed "${SEED}" \
  --dtype "${DTYPE}" \
  --device "${DEVICE}" \
  --mlp_permutation "${MLP_PERMUTATION}" \
  --residual_permutation "${RESIDUAL_PERMUTATION}" \
  --residual_perm_search_steps "${RESIDUAL_PERM_SEARCH_STEPS}" \
  --residual_channel_agg "${RESIDUAL_CHANNEL_AGG}" \
  "${EXTRA_ARGS[@]}"

echo "[prune_mlp] done: ${OUTPUT_DIR}"
