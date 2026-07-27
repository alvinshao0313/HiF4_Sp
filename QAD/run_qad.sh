#!/usr/bin/env bash
set -euo pipefail

# HiF4 × EdgeRazor QAD：Qwen3.5-27B + S1K-1.1 + EAKLD/LAFD
# 必须在 hif4 conda 环境中运行。
# 默认并行：FSDP FULL_SHARD（ZeRO-3 等价）+ 数据并行。
# 备选：PARALLEL_MODE=layer 单进程按层 device_map（勿用多进程）。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCALE_TUNING_DIR="${REPO_ROOT}/ScaleTuning"

export PYTHONPATH="${SCRIPT_DIR}:${SCALE_TUNING_DIR}:${REPO_ROOT}/ChuanCi${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED="${PYTHONHASHSEED:-31}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-27B}"
# 可选：GPTQ/RTN 伪量化 HF 模型，仅用于初始化 frozen_b + s0；教师仍用 MODEL_PATH
PSEUDO_QUANT_MODEL_PATH="${PSEUDO_QUANT_MODEL_PATH:-Qmodel/Qwen3.5-27B-HiF4-1-RTN}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/QAD/.result/qad_qwen3_5_27b}"
DATASET_NAME="${DATASET_NAME:-simplescaling/s1K-1.1_tokenized}"
DATASET_PATH="${DATASET_PATH:-}"
NPROC="${NPROC:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
export CUDA_VISIBLE_DEVICES

# 实测：FSDP FULL_SHARD 在最长 S1K(~27k) 上 OOM；默认改按层切分。需要可再试 PARALLEL_MODE=fsdp
PARALLEL_MODE="${PARALLEL_MODE:-layer}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj,gate_proj,up_proj,down_proj}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-32768}"
# 可见 GPU≤4 默认 32；更多卡可用 512。只影响峰值显存，loss 与整段等价。
LOGIT_CHUNK_SIZE=32

if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  echo "ERROR: 请先 conda activate hif4（当前环境=${CONDA_DEFAULT_ENV:-none}）" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${DATASET_PATH}" ]]; then
  EXTRA_ARGS+=(--dataset_path "${DATASET_PATH}")
fi
if [[ -n "${PSEUDO_QUANT_MODEL_PATH}" ]]; then
  EXTRA_ARGS+=(--pseudo_quant_model_path "${PSEUDO_QUANT_MODEL_PATH}")
fi

COMMON=(
  --model_path "${MODEL_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --dataset_name "${DATASET_NAME}"
  --trace_source "deepseek"
  --seed "31"
  --deterministic "true"
  --target_modules "${TARGET_MODULES}"
  --tune_modules "${TARGET_MODULES}"
  --max_steps "2000"
  --per_device_train_batch_size "1"
  --gradient_accumulation_steps "8"
  --learning_rate "2e-5"
  --weight_decay "0.001"
  --logging_steps "10"
  --temperature "1.0"
  --task_alpha "0.05"
  --eakld_alpha "2.0"
  --lafd_alpha "0.5"
  --confidence_k "16"
  --lafd_topk "3"
  --logit_chunk_size "${LOGIT_CHUNK_SIZE}"
  --gradient_checkpointing "true"
  --gradient_checkpointing_kwargs '{"use_reentrant": false}'
  --optim "adamw_torch"
  --max_grad_norm "1.3"
  --warmup_ratio "0.05"
  --lr_scheduler_type "constant_with_warmup"
  --model_max_length "${MODEL_MAX_LENGTH}"
  --allow_truncate "false"
  --attn_implementation "sdpa"
  --fp16 "false"
  --bf16 "true"
  --export_reconstructed_model "false"
  --parallel_mode "${PARALLEL_MODE}"
)

if [[ "${PARALLEL_MODE}" == "layer" ]]; then
  python "${SCRIPT_DIR}/train_qad.py" "${COMMON[@]}" "${EXTRA_ARGS[@]}" "$@"
else
  torchrun --standalone --nproc_per_node="${NPROC}" "${SCRIPT_DIR}/train_qad.py" \
    "${COMMON[@]}" "${EXTRA_ARGS[@]}" "$@"
fi
