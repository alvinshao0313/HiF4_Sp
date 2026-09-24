#!/usr/bin/env bash
# 验收：先试 FSDP FULL_SHARD（ZeRO-3 等价）+ DP；OOM 则自动改按层 device_map。
set -euo pipefail

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
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/QAD/.result/fit_test}"
DATASET_NAME="${DATASET_NAME:-simplescaling/s1K-1.1_tokenized}"
NPROC="${NPROC:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-32768}"
MAX_STEPS="${MAX_STEPS:-2}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj,gate_proj,up_proj,down_proj}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  echo "ERROR: 请先 conda activate hif4（当前环境=${CONDA_DEFAULT_ENV:-none}）" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
RESULT_JSON="${OUTPUT_ROOT}/fit_result.json"
COMMON_ARGS=(
  --model_path "${MODEL_PATH}"
  --dataset_name "${DATASET_NAME}"
  --trace_source "deepseek"
  --seed "31"
  --deterministic "false"
  --target_modules "${TARGET_MODULES}"
  --tune_modules "${TARGET_MODULES}"
  --max_steps "${MAX_STEPS}"
  --per_device_train_batch_size "1"
  --gradient_accumulation_steps "1"
  --learning_rate "2e-5"
  --weight_decay "0.001"
  --logging_steps "1"
  --temperature "1.0"
  --task_alpha "0.05"
  --eakld_alpha "2.0"
  --lafd_alpha "0.5"
  --confidence_k "16"
  --lafd_topk "3"
  --logit_chunk_size "512"
  --gradient_checkpointing "true"
  --gradient_checkpointing_kwargs '{"use_reentrant": false}'
  --optim "adamw_torch"
  --max_grad_norm "1.3"
  --warmup_ratio "0.0"
  --lr_scheduler_type "constant"
  --model_max_length "${MODEL_MAX_LENGTH}"
  --allow_truncate "false"
  --attn_implementation "sdpa"
  --fp16 "false"
  --bf16 "true"
  --export_reconstructed_model "false"
  --dataloader_num_workers "0"
  --prefer_longest_sample "true"
)

echo "[fit] try FSDP FULL_SHARD nproc=${NPROC} max_len=${MODEL_MAX_LENGTH}"
FSDP_OUT="${OUTPUT_ROOT}/fsdp"
set +e
torchrun --standalone --nproc_per_node="${NPROC}" "${SCRIPT_DIR}/train_qad.py" \
  --output_dir "${FSDP_OUT}" \
  --parallel_mode "fsdp" \
  "${COMMON_ARGS[@]}" \
  "$@" 2>&1 | tee "${OUTPUT_ROOT}/fsdp.log"
FSDP_RC=${PIPESTATUS[0]}
set -e

if [[ ${FSDP_RC} -eq 0 ]]; then
  python - <<PY
import json
from pathlib import Path
p = Path(${RESULT_JSON@Q})
p.write_text(json.dumps({
    "ok": True,
    "mode": "fsdp",
    "nproc": int(${NPROC@Q}),
    "model_max_length": int(${MODEL_MAX_LENGTH@Q}),
    "max_steps": int(${MAX_STEPS@Q}),
    "output_dir": ${FSDP_OUT@Q},
}, indent=2), encoding="utf-8")
print("FSDP fit OK ->", p)
PY
  exit 0
fi

echo "[fit] FSDP failed (rc=${FSDP_RC}); checking OOM then fallback to layer split"
if ! grep -qiE "out of memory|CUDA OOM|CUDA out of memory" "${OUTPUT_ROOT}/fsdp.log"; then
  python - <<PY
import json
from pathlib import Path
p = Path(${RESULT_JSON@Q})
p.write_text(json.dumps({
    "ok": False,
    "mode": "fsdp",
    "fallback": None,
    "reason": "fsdp_failed_non_oom",
    "rc": int(${FSDP_RC}),
    "log": ${OUTPUT_ROOT@Q} + "/fsdp.log",
}, indent=2), encoding="utf-8")
print("FSDP failed (non-OOM) ->", p)
PY
  exit "${FSDP_RC}"
fi

echo "[fit] fallback: parallel_mode=layer (single process, device_map=auto)"
LAYER_OUT="${OUTPUT_ROOT}/layer"
set +e
python "${SCRIPT_DIR}/train_qad.py" \
  --output_dir "${LAYER_OUT}" \
  --parallel_mode "layer" \
  "${COMMON_ARGS[@]}" \
  "$@" 2>&1 | tee "${OUTPUT_ROOT}/layer.log"
LAYER_RC=${PIPESTATUS[0]}
set -e

python - <<PY
import json
from pathlib import Path
ok = int(${LAYER_RC}) == 0
p = Path(${RESULT_JSON@Q})
p.write_text(json.dumps({
    "ok": ok,
    "mode": "layer" if ok else "failed",
    "fsdp_rc": int(${FSDP_RC}),
    "layer_rc": int(${LAYER_RC}),
    "nproc_fsdp": int(${NPROC@Q}),
    "model_max_length": int(${MODEL_MAX_LENGTH@Q}),
    "max_steps": int(${MAX_STEPS@Q}),
    "output_dir": ${LAYER_OUT@Q} if ok else None,
    "fsdp_log": ${OUTPUT_ROOT@Q} + "/fsdp.log",
    "layer_log": ${OUTPUT_ROOT@Q} + "/layer.log",
}, indent=2), encoding="utf-8")
print("layer fit", "OK" if ok else "FAIL", "->", p)
PY
exit "${LAYER_RC}"
