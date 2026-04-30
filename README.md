# HiF4_Sp

本仓库用于两件事：

- 用本仓库自带的 vLLM + lighteval 跑评测。
- 用 HiFloat4 对 Qwen3.5 做 RTN / GPTQ fake-quant，并保存 Hugging Face 格式模型。

当前只维护一个 conda 环境：`hif4`。不再需要单独的 `qhif4` / `qwen35` 环境。

## 关键原则

- vLLM 适配代码已经放在 `3rdparty/vllm` 内置源码目录里。用户 clone 后不需要再手工改 vLLM。
- lighteval 适配代码已经放在 `3rdparty/lighteval` 内置源码目录里。根目录 `main.py` 会优先导入这份本地源码。
- 安装脚本会源码编译本仓库的 vLLM，并编译 HiFloat4 CUDA 扩展。编译完成后即可运行评测或量化。
- 大模型权重、评测结果、日志和编译产物不提交到 git。

## 环境版本

| 组件 | 版本 / 来源 |
|------|-------------|
| Python | 3.11 |
| conda env | `hif4` |
| torch | `2.10.0` |
| torchvision / torchaudio | `0.25.0` / `2.10.0` |
| CUDA toolkit | `12.8`，优先安装在 `hif4` 环境内 |
| transformers | `5.6.2` |
| vLLM | `3rdparty/vllm` 内置源码目录，editable source build |
| lighteval | `3rdparty/lighteval` 内置源码目录，editable install |
| runtime deps | `accelerate` / `datasets` / `safetensors` / `tqdm` / `inspect-ai` 等 |

机器必须有可用 NVIDIA driver。conda 里的 CUDA toolkit 只解决编译期 `nvcc` / headers，不能替代系统驱动。

## 安装

```bash
git clone <this-repo-url>
cd HiF4_Sp

conda create -n hif4 python=3.11 -y
conda activate hif4

bash install.sh
```

`install.sh` 会做这些事：

1. 确认当前环境是 `hif4`。
2. 检查 `3rdparty/vllm` 和 `3rdparty/lighteval` 内置源码目录。
3. 安装 conda CUDA toolkit 12.8。
4. 安装 torch / transformers / vLLM build 依赖。
5. 用 `--no-build-isolation` 源码编译安装 `3rdparty/vllm`。
6. editable 安装 `3rdparty/lighteval`，不安装 `[vllm]` extras，避免覆盖本地 vLLM。
7. 编译 `HiFloat4/hif4_gpu` CUDA 扩展。
8. 安装量化和评测运行依赖，并导入检查 vLLM、HiFloat4、lighteval、Qwen3.5 适配。

安装脚本最后会自动执行导入检查。手动复查可以运行：

```bash
python -c "import torch, vllm, vllm._C; import HiFloat4.main as h; from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM; print(torch.__version__, vllm.__version__, Qwen3_5ForCausalLM.__name__, 'ok')"
```

## 目录

```text
.
├── 3rdparty/
│   ├── vllm/        # 已包含本项目需要的 Qwen3.5 / HiF4 适配
│   └── lighteval/   # 已包含本项目需要的评测适配
├── HiFloat4/
│   ├── main.py
│   ├── quantize_qwen3_5_27b.sh
│   ├── hif4_gpu/
│   └── hif4gptq/
├── tasks/           # 自定义 lighteval 任务
├── main.py          # vLLM + lighteval 评测入口
└── install.sh
```

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
| `--gpu_memory_utilization` | vLLM 显存利用率 |
| `--enforce_eager` | 关闭 CUDAGraph / torch.compile，排错用 |
| `--cpu_offload_gb` | 每卡向 CPU 卸载多少 GiB 权重 |
| `--hif4_fake_act` | 打开 vLLM dense linear 输入激活 HiF4 fake quant |

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

保存成功后，输出目录应包含 `config.json`、`generation_config.json`、分片权重文件和 index 文件。中断或磁盘写满留下的目录不能当作可用 checkpoint。

## 自定义任务

| 任务名 | 文件 | 备注 |
|--------|------|------|
| `aime24_avg5`, `aime25_avg5` | `tasks/aime.py` | avg@5 |
| `triviaqa_em` | `tasks/triviaqa.py` | TriviaQA EM 修正 |
| `simpleqa_v2` | `tasks/simpleqa.py` | SimpleQA 新列名适配 |
| `hellaswag_fixed` | `tasks/hellaswag.py` | HellaSwag 修复版 |
| `ifeval_pass_at_n`, `ifbench_test_pass_at_n` | `tasks/if_pass_at_n.py` | pass@n |

## 提交约定

- 提交主仓库代码、`HiFloat4/`、`tasks/`、安装脚本和文档。
- 提交 `3rdparty/vllm` / `3rdparty/lighteval` 内置源码目录里的源码改动。
- 不提交 `Qmodel/`、`results/`、日志、offload 目录、编译产物、缓存目录。
