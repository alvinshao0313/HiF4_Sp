#!/usr/bin/env bash
set -euo pipefail

cd /home/shaoyuantian/program/HiF4_Sp

GPUS="${GPUS:-4,5,6,7}"
REPEATS="${REPEATS:-5}"

COMMON_ARGS=(
  --datasets aime25
  --max_model_length 32768
  --max_new_tokens 32768
  --temperature 0.7
  --top_p 0.8
  --top_k 20
  --gpu_memory_utilization 0.9
  --tensor_parallel_size 4
)

run_many() {
  local model_path="$1"
  local output_dir="$2"
  local fake_act_quant="$3"
  local kv_quant_format="$4"

  for run_idx in $(seq 1 "${REPEATS}"); do
    echo "[$(date --iso-8601=seconds)] run ${run_idx}/${REPEATS}: model=${model_path}, act=${fake_act_quant}, kv=${kv_quant_format}"
    if [[ "${kv_quant_format}" == "none" ]]; then
      CUDA_VISIBLE_DEVICES="${GPUS}" python main.py \
        --model_path "${model_path}" \
        "${COMMON_ARGS[@]}" \
        --output_dir "${output_dir}" \
        --fake_act_quant "${fake_act_quant}"
    else
      CUDA_VISIBLE_DEVICES="${GPUS}" python main.py \
        --model_path "${model_path}" \
        "${COMMON_ARGS[@]}" \
        --output_dir "${output_dir}" \
        --fake_act_quant "${fake_act_quant}" \
        --kv_quant_format "${kv_quant_format}"
    fi
  done
}

# hif4-1 权重 + hif4-1 激活
run_many \
  /home/shaoyuantian/program/HiF4_Sp/Qmodel/Qwen3.5-27B-HiF4-1-RTN \
  ./results/hif4_1 \
  hif4-1 \
  none

# hif4-1 权重 + hif4-1 激活 + hif4-1 KV cache
run_many \
  /home/shaoyuantian/program/HiF4_Sp/Qmodel/Qwen3.5-27B-HiF4-1-RTN \
  ./results/hif4_1_kv \
  hif4-1 \
  hif4-1

# NVFP4-BF16 -> hif4 RTN 权重 + hif4 激活
run_many \
  /home/shaoyuantian/program/HiF4_Sp/Qmodel/Qwen3.5-27B-NVFP4-BF16-HiF4-RTN \
  ./results/nvfp4_bf16_hif4 \
  hif4 \
  none

# NVFP4-BF16 -> hif4 RTN 权重 + hif4 激活 + hif4 KV cache
run_many \
  /home/shaoyuantian/program/HiF4_Sp/Qmodel/Qwen3.5-27B-NVFP4-BF16-HiF4-RTN \
  ./results/nvfp4_bf16_hif4_kv \
  hif4 \
  hif4
  
# hif4-1 GPTQ 权重 + hif4-1 激活
run_many \
  /home/shaoyuantian/program/HiF4_Sp/Qmodel/Qwen3.5-27B-HiF4-1-GPTQ \
  ./results/hif4_1 \
  hif4-1 \
  none

# hif4-1 GPTQ 权重 + hif4-1 激活 + hif4-1 KV cache
run_many \
  /home/shaoyuantian/program/HiF4_Sp/Qmodel/Qwen3.5-27B-HiF4-1-GPTQ \
  ./results/hif4_1_kv \
  hif4-1 \
  hif4-1

# NVFP4-BF16 -> hif4 GPTQ 权重 + hif4 激活
run_many \
  /home/shaoyuantian/program/HiF4_Sp/Qmodel/Qwen3.5-27B-NVFP4-BF16-HiF4-GPTQ \
  ./results/nvfp4_bf16_hif4 \
  hif4 \
  none

# NVFP4-BF16 -> hif4 GPTQ 权重 + hif4 激活 + hif4 KV cache
run_many \
  /home/shaoyuantian/program/HiF4_Sp/Qmodel/Qwen3.5-27B-NVFP4-BF16-HiF4-GPTQ \
  ./results/nvfp4_bf16_hif4_kv \
  hif4 \
  hif4

# mmlu_pro,gpqa:diamond,math_500,aime24_avg,aime25_avg,lcb:codegeneration_v6,ifeval,musr,hellaswag_fixed,winogrande
