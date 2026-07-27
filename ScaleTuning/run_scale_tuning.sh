#!/usr/bin/env bash
set -euo pipefail

# HiF4 S0 ScaleTuning：kd_top_1000 + adaptive_top_3 hidden alignment
# 必须在 hif4 conda 环境中运行。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VAELLM_ROOT="${VAELLM_ROOT:-/home/shaoyuantian/program/VAELLM}"

export PYTHONPATH="${SCRIPT_DIR}:${REPO_ROOT}/ChuanCi:${VAELLM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED="${PYTHONHASHSEED:-31}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
# Triton 编译 launcher 需要 -lcuda；本机只有 libcuda.so.1，用 CUDA stubs 供链接。
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/ScaleTuning/.result/scale_tuning}"
NPROC="${NPROC:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,4,5}"
export CUDA_VISIBLE_DEVICES

# 确认 conda 环境
if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  echo "ERROR: 请先 conda activate hif4（当前环境=${CONDA_DEFAULT_ENV:-none}）" >&2
  exit 1
fi

torchrun --standalone --nproc_per_node="${NPROC}" "${SCRIPT_DIR}/train_scale_tuning.py" \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --vaellm_root "${VAELLM_ROOT}" \
  --seed "31" \
  --deterministic "true" \
  --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --tune_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --target_layers "" \
  --tune_layers "" \
  --distill_dataset "edgerazor_ii_7m=0.676,edgerazor_ii_gen=0.133,edgerazor_tulu=0.055,edgerazor_am=0.127,vaellm_eval_task=0.009" \
  --distill_steps "2000" \
  --distill_batch_size "8" \
  --distill_lr "2e-5" \
  --distill_weight_decay "0.001" \
  --distill_log_every "10" \
  --distill_temperature "1.0" \
  --distill_loss_alpha "0.5" \
  --distill_loss_type "kd_top_1000" \
  --distill_eakld_confidence_k "16" \
  --distill_teacher_logits_cpu_staging "true" \
  --distill_hidden_loss_weight "0.1" \
  --distill_pre_mlp_hidden_loss_weight "0.0" \
  --distill_hidden_alignment_layer_weighting "adaptive_top_3" \
  --distill_gradient_accumulation_steps "2" \
  --distill_gradient_checkpointing "true" \
  --distill_gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --distill_optim "adamw_torch" \
  --distill_max_grad_norm "1.3" \
  --distill_warmup_ratio "0.05" \
  --distill_group_by_length "false" \
  --distill_lr_scheduler_type "constant_with_warmup" \
  --distill_model_max_length "1024" \
  --fp16 "false" \
  --bf16 "true" \
  --export_reconstructed_model "false" \
  "$@"
