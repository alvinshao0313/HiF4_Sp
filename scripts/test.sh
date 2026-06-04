#!/usr/bin/env bash
set -euo pipefail

cd /home/shaoyuantian/program/HiF4_Sp

if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  echo "错误：当前 conda 环境不是 hif4，请先执行：conda activate hif4" >&2
  exit 1
fi

# HiF4 RTN 权重 + HiF4-1 激活 + HiF4-1 KV cache + MXFP8 Q。
# kv_quant_chunk_size 仅对 NVFP4 生效，HiF4-1 会忽略。
CUDA_VISIBLE_DEVICES=0,1,2,7 python main.py \
  --model_path /home/shaoyuantian/program/HiF4_Sp/Qmodel/Qwen3.5-27B-HiF4-RTN \
  --output_dir ./results/hif4_rtn_act_hif4_1_kv_hif4_1_q_mxfp8_sink16_recent128 \
  --datasets aime25 \
  --max_model_length 32768 \
  --max_new_tokens 32768 \
  --temperature 0.7 \
  --top_p 0.8 \
  --top_k 20 \
  --gpu_memory_utilization 0.9 \
  --tensor_parallel_size 4 \
  --fake_act_quant hif4-1 \
  --kv_quant_format hif4-1 \
  --kv_quant_chunk_size 64 \
  --kv_quant_sink_size 16 \
  --kv_quant_recent_size 128 \
  --kv_quant_target kv \
  --kv_quant_query mxfp8
