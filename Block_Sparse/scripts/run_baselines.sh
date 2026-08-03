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
export TOKENIZERS_PARALLELISM=false
echo "[env] HF_ENDPOINT unset; using official huggingface.co"

# ---------------------------------------------------------------------------
# 参数（只在这里改）
# ---------------------------------------------------------------------------
MODEL_PATH=Qwen/Qwen3.5-27B
SPARSITY=0.20
MAX_PRUNE_RATIO_PER_MATRIX=0.50   # 每层/每个 Linear 稀疏率上限
BLOCK_SIZE=64            # 正方形写 128；矩形写 64x128
CALIBRATION_DATASET=s1k   # s1k | wikitext2 | c4 | ptb（fisher / fisher_budget_wanda）
CALIB_SAMPLES=128
# Fisher 多卡时最后一卡要扛 lm_head+logits；s1k 全长可达 ~2 万 token 会 OOM。
# 0=不截断；建议 4x80GB 用 8192。
SEQ_LEN=0
SEED=42
DTYPE=bfloat16
MLP_PERMUTATION=wanda_shared   # none | wanda_shared

# 要跑的方法：magnitude / random / fisher / fisher_budget_wanda
# 单卡冒烟可先只跑 magnitude（不用反向）
METHODS=(fisher_budget_wanda)

# 是否跳过某阶段：0=执行，1=跳过
SKIP_PRUNE=0
SKIP_EVAL=0

# 剪枝可见卡（多卡会自动 device_map=auto 切分；单卡就写一张）
PRUNE_GPUS=6,5,4
# 评测可见卡（lm_eval / ppl；与剪枝可不同）
EVAL_GPUS=4

# lm_eval：arc_easy / arc_challenge / mmlu
LM_EVAL_TASKS=arc_easy,arc_challenge,mmlu
LM_EVAL_BATCH_SIZE=8
LM_EVAL_NUM_FEWSHOT=0

# WikiText-2 PPL
PPL_DATASET=wikitext2
PPL_SEQ_LEN=2048

OUTPUT_ROOT=./Block_Sparse/outputs
RESULT_ROOT=./Block_Sparse/results
# ---------------------------------------------------------------------------

mkdir -p "${OUTPUT_ROOT}" "${RESULT_ROOT}"

for method in "${METHODS[@]}"; do
  # 标签含校准集名与 permutation，避免产物互相覆盖
  tag="qwen35_27b_${method}_s${SPARSITY}_b${BLOCK_SIZE}_${CALIBRATION_DATASET}_perm${MLP_PERMUTATION}"
  tag="${tag//\//_}"
  out_dir="${OUTPUT_ROOT}/${tag}"
  echo "======== method=${method} perm=${MLP_PERMUTATION} output=${out_dir} ========"

  if [[ "${SKIP_PRUNE}" -eq 0 ]]; then
    # 按 PRUNE_GPUS（即 CUDA_VISIBLE_DEVICES）在可见卡上自动切分
    echo "[prune] CUDA_VISIBLE_DEVICES=${PRUNE_GPUS}"
    CUDA_VISIBLE_DEVICES="${PRUNE_GPUS}" python \
      Block_Sparse/tools/score_and_prune_mlp.py \
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
      --device cuda \
      --mlp_permutation "${MLP_PERMUTATION}"
  fi

  if [[ "${SKIP_EVAL}" -eq 0 ]]; then
    lm_eval_json="${RESULT_ROOT}/${tag}_arc_mmlu.json"
    lm_eval_log="${RESULT_ROOT}/${tag}_arc_mmlu.log"
    echo "[lm_eval] CUDA_VISIBLE_DEVICES=${EVAL_GPUS} tasks=${LM_EVAL_TASKS}"
    CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" python \
      Block_Sparse/tools/eval_lm_eval.py \
      --model_path "${out_dir}" \
      --tasks "${LM_EVAL_TASKS}" \
      --num_fewshot "${LM_EVAL_NUM_FEWSHOT}" \
      --batch_size "${LM_EVAL_BATCH_SIZE}" \
      --dtype "${DTYPE}" \
      --output_json "${lm_eval_json}" \
      2>&1 | tee "${lm_eval_log}"

    ppl_json="${RESULT_ROOT}/${tag}_${PPL_DATASET}_s${PPL_SEQ_LEN}.json"
    ppl_log="${RESULT_ROOT}/${tag}_${PPL_DATASET}_s${PPL_SEQ_LEN}.log"
    echo "[ppl] CUDA_VISIBLE_DEVICES=${EVAL_GPUS} dataset=${PPL_DATASET} seq=${PPL_SEQ_LEN}"
    CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" python \
      Block_Sparse/tools/eval_ppl.py \
      --model_path "${out_dir}" \
      --dataset "${PPL_DATASET}" \
      --sequence_length "${PPL_SEQ_LEN}" \
      --dtype "${DTYPE}" \
      --output_json "${ppl_json}" \
      2>&1 | tee "${ppl_log}"
  fi
done

echo "All methods finished."
echo "  pruned models: ${OUTPUT_ROOT}"
echo "  eval results:  ${RESULT_ROOT}"
