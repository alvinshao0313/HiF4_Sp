CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --model_path /home/chenyuanteng/.cache/modelscope/hub/models/Qwen/Qwen3-Next-80B-A3B-Instruct \
    --datasets ifeval \
    --gpu_memory_utilization 0.9

CUDA_VISIBLE_DEVICES=4,5 python main.py \
    --model_path /home/chenyuanteng/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --datasets musr \
    --gpu_memory_utilization 0.9

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --model_path /home/chenyuanteng/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --datasets winogrande \
    --gpu_memory_utilization 0.9

CUDA_VISIBLE_DEVICES=4,5 python main.py \
    --model_path /home/chenyuanteng/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --datasets hellaswag_fixed \
    --gpu_memory_utilization 0.9

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --model_path /home/chenyuanteng/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --datasets lcb:codegeneration_v6 \
    --gpu_memory_utilization 0.9

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --model_path /home/chenyuanteng/.cache/modelscope/hub/models/Qwen/Qwen3.5-35B-A3B \
    --datasets aime25,lcb:codegeneration_v6 \
    --gpu_memory_utilization 0.9

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py \
    --model_path /home/chenyuanteng/.cache/modelscope/hub/models/Qwen/Qwen3.5-35B-A3B \
    --datasets mmlu_pro \
    --gpu_memory_utilization 0.9

# mmlu_pro,gpqa:diamond,math_500,aime24_avg,aime25_avg,lcb:codegeneration_v6,ifeval,musr,hellaswag_fixed,winogrande