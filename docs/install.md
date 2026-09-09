# 环境安装文档

本文说明如何从零搭好本仓库唯一主线环境：`hif4`。

装完后可以做两件事：

1. 用根目录 `main.py`（vLLM + lighteval）跑评测。
2. 用 `HiFloat4/` 对模型做 RTN / GPTQ fake-quant。

更细的 vLLM ABI / 编译踩坑见：

- [vllm_install_notes.md](./vllm_install_notes.md)
- [vllm_0_17_source_build_guide.md](./vllm_0_17_source_build_guide.md)（历史迁移记录，不要按它切回旧版本）

---

## 1. 前置条件

| 项 | 要求 |
|----|------|
| OS | Linux x86_64 |
| GPU | NVIDIA GPU |
| 驱动 | 系统 NVIDIA driver 可用（`nvidia-smi` 能通） |
| conda | 已安装 Anaconda / Miniconda |
| 磁盘 | 建议预留 ≥ 50 GB（torch + vLLM 编译产物 + 依赖） |
| 内存 | 源码编译 vLLM 时建议 ≥ 64 GB；内存紧张时降低 `MAX_JOBS` |

说明：

- conda 里的 CUDA toolkit 只解决编译期 `nvcc` / headers，**不能替代系统驱动**。
- 没有可用驱动时，仍可能完成编译和 `import` 检查；真正 GPU 推理必须在驱动正常的机器上跑。

---

## 2. 固定版本

| 组件 | 版本 / 来源 |
|------|-------------|
| conda env | `hif4` |
| Python | 3.11 |
| torch | `2.10.0` |
| torchvision / torchaudio | `0.25.0` / `2.10.0` |
| CUDA toolkit | `12.8`（优先装进 `hif4`） |
| transformers | `5.6.2` |
| vLLM | `3rdparty/vllm` 内置源码，editable **源码编译**（主线按 v0.19.1 适配） |
| lighteval | `3rdparty/lighteval` 内置源码，editable（约 v0.13.0 + 本仓库补丁） |
| HiFloat4 | `HiFloat4/hif4_gpu` CUDA 扩展 |

禁止事项：

- 不要用 `VLLM_USE_PRECOMPILED=1`。
- 不要装 `lighteval[vllm]` extras（会覆盖本地 editable vLLM）。
- 不要再找旧的 `install_qwen35.sh` / `qhif4` / `qwen35` 环境入口；当前只有 `install.sh`。

---

## 3. 一键安装

```bash
git clone <this-repo-url>
cd HiF4_Sp

# 确认源码目录完整
ls 3rdparty/vllm 3rdparty/lighteval HiFloat4/hif4_gpu

conda create -n hif4 python=3.11 -y
conda activate hif4

bash install.sh
```

编译压力大时：

```bash
MAX_JOBS=8 bash install.sh
```

默认 `MAX_JOBS=32`。高配机器大约 30–45 分钟；低配会更久。

---

## 4. `install.sh` 实际做什么

按顺序：

1. 强制检查当前 conda env 是 `hif4`。
2. 检查 `3rdparty/vllm`、`3rdparty/lighteval` 存在。
3. 若本机没有 CUDA 12.8 的 `nvcc`，则：
   `conda install -n hif4 -c nvidia cuda-toolkit=12.8`
4. 设置 conda CUDA 编译变量：
   `CUDA_HOME`、`CUDAToolkit_ROOT`、`PATH`、`LD_LIBRARY_PATH`、`NVCC_PREPEND_FLAGS`
5. 安装 torch / torchvision / torchaudio / transformers。
6. 安装 vLLM build 依赖（`3rdparty/vllm/requirements/build.txt`）。
7. 在 `3rdparty/vllm` 执行：
   `pip install --editable . --no-build-isolation`
8. 在 `3rdparty/lighteval` 执行：
   `pip install --editable .`（不带 `[vllm]`）。
9. 编译 `HiFloat4/hif4_gpu` CUDA 扩展（`bash build.sh`）。
10. 安装运行依赖：`accelerate`、`datasets`、`safetensors`、`tqdm`、`inspect-ai`、`more_itertools`、`langdetect`。
11. 跑导入检查：torch / vllm / `_C` / lighteval / HiFloat4 / Qwen3.5 适配类。

---

## 5. 安装后验证

### 5.1 导入检查

脚本末尾会自动跑。也可手动：

```bash
conda activate hif4

python -c "
import torch, vllm, vllm._C
import HiFloat4.main as h
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM
print(torch.__version__, vllm.__version__, Qwen3_5ForCausalLM.__name__, 'ok')
"
```

期望：

- `torch` 类似 `2.10.0+cu128`
- `vllm` 指向仓库内 `3rdparty/vllm/...`
- 无 `ImportError` / `undefined symbol`

再确认路径：

```bash
python -m pip show vllm torch | sed -n '1,20p'
```

关键字段应类似：

```text
Name: vllm
Editable project location: .../HiF4_Sp/3rdparty/vllm

Name: torch
Version: 2.10.0
```

### 5.2 GPU 检查

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

若这里是 `False 0`，说明驱动不可用；不等于编译失败。

### 5.3 评测冒烟

```bash
conda activate hif4

CUDA_VISIBLE_DEVICES=0 python main.py \
  --datasets gsm8k \
  --model_path Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --max_samples 2
```

能完成模型加载和少量生成即通过。

---

## 6. 日常使用约定

所有 Python / 评测 / 量化命令都在 `hif4` 里跑：

```bash
conda activate hif4
# 或
conda run -n hif4 --no-capture-output python ...
```

评测工具选择：

| 任务类型 | 工具 |
|----------|------|
| 普通下游（ARC、标准 MMLU 等） | `lm_eval` |
| reasoning（MMLU-Pro、GSM8K 等） | 根目录 `main.py` + lighteval |

禁止用 `lm_eval` 跑 MMLU-Pro。

量化示例：

```bash
conda activate hif4
bash HiFloat4/quantize_qwen3_5_27b.sh
```

---

## 7. 常见失败与处理

### 7.1 当前 env 不是 `hif4`

```text
[install.sh] 错误：当前 conda env 不是 hif4
```

先 `conda activate hif4` 再跑。

### 7.2 `import vllm._C` 报 `undefined symbol: MessageLogger...`

根因通常是预编译 wheel 与本机 torch ABI 不一致。

处理：

- 不要用 `VLLM_USE_PRECOMPILED=1`
- 按 `install.sh` 用当前环境 torch 做 `--no-build-isolation` 源码编译

详见 [vllm_install_notes.md](./vllm_install_notes.md)。

### 7.3 CMake 找不到 CUDA headers / `cicc: not found` / `cuda_runtime.h` 缺失

conda CUDA 布局不是 `/usr/local/cuda`。`install.sh` 已设置：

```bash
CUDA_HOME=$CONDA_PREFIX
CUDAToolkit_ROOT=$CONDA_PREFIX/targets/x86_64-linux
PATH=$CONDA_PREFIX/bin:$CONDA_PREFIX/nvvm/bin:$PATH
NVCC_PREPEND_FLAGS="-I$CUDAToolkit_ROOT/include ..."
```

若手工编译，必须沿用同一套变量。

### 7.4 编译内存爆了 / 机器卡死

```bash
MAX_JOBS=4 bash install.sh
# 或
MAX_JOBS=8 bash install.sh
```

### 7.5 装完 lighteval 后本地 vLLM 被覆盖

原因：执行了 `pip install -e "3rdparty/lighteval[vllm]"`。

处理：只装本体 `pip install --editable .`，然后重新按第 4 节编译安装本地 vLLM。

### 7.6 缺少 `3rdparty/vllm` 或 `3rdparty/lighteval`

仓库必须带这两份内置源码。重新完整 clone，不要只拷贝部分目录。

---

## 8. 重装 / 清环境

彻底重来：

```bash
conda deactivate
conda env remove -n hif4 -y
conda create -n hif4 python=3.11 -y
conda activate hif4
cd /path/to/HiF4_Sp
bash install.sh
```

只重编 vLLM：

```bash
conda activate hif4
cd 3rdparty/vllm
MAX_JOBS=8 pip install --editable . --no-build-isolation
```

只重编 HiFloat4 CUDA 扩展：

```bash
conda activate hif4
cd HiFloat4/hif4_gpu
bash build.sh
```

---

## 9. 相关文档

| 文档 | 内容 |
|------|------|
| [../README.md](../README.md) | 仓库总览、评测与量化用法 |
| [vllm_install_notes.md](./vllm_install_notes.md) | vLLM 预编译 ABI 问题与方案选择 |
| [vllm_0_17_source_build_guide.md](./vllm_0_17_source_build_guide.md) | 历史源码编译报错与变量说明 |
| [hif4_gpu_quant_flow.md](./hif4_gpu_quant_flow.md) | HiFloat4 GPU 量化流程 |
