# GPU 评测与量化命令参考

返回 [仓库首页](../../README.md) · [目录与文件归属](repository_layout.md)。所有命令从仓库根目录、已激活的 `hif4` 环境执行。

本页保留通用评测 CLI 和已有 Qwen3.5 RTN / GPTQ 用法。它们是功能参考；NVFP4→HiF4 / MXFP4 的正式实验配置以[转换实验导航](../../fp4_convert/experiments/README.md)及对应协议为准。示例 GPU 编号不构成当前资源授权，示例采样参数不替代已确认的评测协议。

普通下游任务使用 `lm_eval`；reasoning 任务通过根目录 `main.py` 使用 vLLM + lighteval。MMLU-Pro 默认 300 样本；完整评测或其他协议须以具体实验授权为准。安装与环境版本见[安装指南](install.md)。

## 评测

最小冒烟测试：

```bash
conda activate hif4

CUDA_VISIBLE_DEVICES=0 python main.py \
  --datasets gsm8k \
  --model_path Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --max_samples 2
```

Qwen3.5 HiFloat4 模型评测示例：

```bash
conda activate hif4

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
  --model_path Qmodel/Qwen3.5-27B-HiF4-RTN \
  --datasets aime25,mmlu_pro,lcb:codegeneration_v6 \
  --tensor_parallel_size 4 \
  --max_model_len 32768 \
  --max_new_tokens 32768 \
  --temperature 0.7 \
  --top_p 0.8 \
  --top_k 20 \
  --gpu_memory_utilization 0.9
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--datasets` | lighteval 任务名，逗号分隔 |
| `--custom_tasks` | 自定义任务文件，默认 `tasks/custom_tasks.py` |
| `--tensor_parallel_size` | vLLM TP。默认等于当前可见 GPU 数 |
| `--pipeline_parallel_size` | vLLM PP |
| `--data_parallel_size` | vLLM DP |
| `--max_model_len` / `--max_model_length` | vLLM 最大上下文长度 |
| `--max_new_tokens` | 最大生成长度 |
| `--max_samples` | 每个任务最多评测样本数，冒烟测试建议设小 |
| `--batch_size` | 传给 vLLM 的 `max_num_seqs`，不设则使用 lighteval / vLLM 默认值 |
| `--gpu_memory_utilization` | vLLM 显存利用率 |
| `--enforce_eager` | 关闭 CUDAGraph / torch.compile，排错用 |
| `--cpu_offload_gb` | 每卡向 CPU 卸载多少 GiB 权重 |
| `--num_experts_per_tok` | 覆盖本地 MoE 模型 `config.json` 里的每 token 激活专家数 |
| `--load_multilingual_tasks` | 加载 lighteval 多语言任务注册表 |
| `--fake_act_quant` | dense linear 输入激活 fake quant，可选 `none` / `hif4` / `hif4-1` / `nvfp4`。`nvfp4` 要求本地模型目录有 `nvfp4_activation_scales.safetensors` |
| `--kv_quant_format` | KV cache fake quant，可选 `none` / `nvfp4` / `hif4` / `hif4-1` |
| `--kv_quant_chunk_size` | NVFP4 KV fake quant 的 non-sink token chunk 大小，默认 `64`；HiF4 KV 会忽略 |
| `--kv_quant_sink_size` | 前多少个 token 的 KV cache 保持原精度，默认 `4` |
| `--kv_quant_recent_size` | HiF4 / HiF4-1 KV cache 最后多少个 token 保持原精度，默认 `0`；启用后自动关闭 prefix caching |
| `--kv_quant_target` | KV fake quant 目标，可选 `kv` / `k` / `v` |
| `--kv_quant_query` | RoPE 后、QK 点积前的 Q fake quant，可选 `none` / `enabled` / `mxfp8`；`enabled` 跟随 KV 格式，`mxfp8` 使用 OCP MXFP8 E4M3（每 32 个 head_dim 元素共享 E8M0 scale） |

`--kv_quant_query mxfp8` 要求 `head_dim` 能被 32 整除。Q 量化不能在 `--kv_quant_format none` 下单独启用。

注意任务名：

- MMLU-Pro 是 `mmlu_pro`，不是 `mmlu-pro`。
- LiveCodeBench code generation 是 `lcb:codegeneration_v6`，不是 `livecodebench`。

## 量化

默认量化 `Qwen/Qwen3.5-27B`，保存到 `Qmodel/`：

```bash
conda activate hif4
bash HiFloat4/quantize_qwen3_5_27b.sh
```

显式指定本地模型和输出目录：

```bash
MODEL=/path/to/Qwen3.5-27B \
OUTPUT=/data/Qwen3.5-27B-HiF4-RTN \
bash HiFloat4/quantize_qwen3_5_27b.sh
```

使用 GPTQ 路径：

```bash
GPTQ=true \
GPTQ_CAL_DATASET=c4 \
GPTQ_CAL_NSAMPLES=512 \
GPTQ_CAL_SEQLEN=512 \
OUTPUT=/data/Qwen3.5-27B-HiF4-GPTQ \
bash HiFloat4/quantize_qwen3_5_27b.sh
```

切换 HiF4 权重量化格式：

```bash
HIF4_WEIGHT_FORMAT=hif4-1 \
OUTPUT=/data/Qwen3.5-27B-HiF4-1-RTN \
bash HiFloat4/quantize_qwen3_5_27b.sh
```

保存成功后，输出目录应包含 `config.json`、`generation_config.json`、分片权重文件和 index 文件。中断或磁盘写满留下的目录不能当作可用 checkpoint。

## 自定义任务

| 任务名 | 文件 | 备注 |
|--------|------|------|
| `aime24_avg5`, `aime25_avg5` | `tasks/aime.py` | avg@5 |
| `triviaqa_em` | `tasks/triviaqa.py` | TriviaQA EM 修正 |
| `simpleqa_v2` | `tasks/simpleqa.py` | SimpleQA 新列名适配 |
| `hellaswag_fixed` | `tasks/hellaswag.py` | HellaSwag 修复版 |
| `ifeval_pass_at_n`, `ifbench_test_pass_at_n` | `tasks/if_pass_at_n.py` | pass@n |
