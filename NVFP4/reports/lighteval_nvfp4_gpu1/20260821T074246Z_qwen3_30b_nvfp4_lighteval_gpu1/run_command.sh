#!/usr/bin/env bash
set -o pipefail

REPO_ROOT=/home/shaoyuantian/program/HiF4_Sp
CKPT=/home/shaoyuantian/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3
RUN_ROOT=/home/shaoyuantian/program/HiF4_Sp/NVFP4/reports/lighteval_nvfp4_gpu1/20260821T074246Z_qwen3_30b_nvfp4_lighteval_gpu1

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

echo "===== experiment start ====="
date -u +"UTC=%Y-%m-%dT%H:%M:%SZ"
echo "requested_model=nvidia/Qwen3-30B-A3B-NVFP4"
echo "resolved_model=${CKPT}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "conda_env=hif4"
echo "tasks=mmlu_pro|0,lcb:codegeneration_v6|0,custom|aime25_avg5|0"
echo "max_samples=300"
echo "max_new_tokens=32768"
echo "max_model_length=40960"
echo "linear_backend=emulation"
echo "moe_backend=emulation"
echo "enforce_eager=true"
echo "===== command output ====="

exec conda run -n hif4 --no-capture-output python main.py \
  --model_path "${CKPT}" \
  --datasets "mmlu_pro|0,lcb:codegeneration_v6|0,custom|aime25_avg5|0" \
  --custom_tasks "${REPO_ROOT}/tasks/custom_tasks.py" \
  --max_samples 300 \
  --tensor_parallel_size 1 \
  --max_model_length 40960 \
  --max_new_tokens 32768 \
  --temperature 0.7 \
  --top_p 0.8 \
  --top_k 20 \
  --gpu_memory_utilization 0.90 \
  --linear_backend emulation \
  --moe_backend emulation \
  --enforce_eager \
  --output_dir "${RUN_ROOT}/results"
