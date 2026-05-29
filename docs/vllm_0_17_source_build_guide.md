# vLLM 0.17.0 Editable 源码编译安装指南

这是历史文档。当前主线已升级到 `3rdparty/vllm@v0.19.1`，安装入口仍是
`install.sh`，不要按本文切回 v0.17.0 配置新环境。

本文档记录本仓库把 `hif4` 环境里的本地 editable vLLM 从 `v0.10.2` 升级到
`v0.17.0` 时遇到的主要报错、根因，以及最终采用的适配。目标不是泛泛说明
“如何安装 vLLM”，而是解释为什么本仓库必须这样装。

## 历史目标

| 项 | 目标值 |
|----|--------|
| Conda env | `hif4` |
| Python | 3.11 |
| vLLM | `3rdparty/vllm@v0.17.0`，editable，源码编译 |
| torch | `2.10.0` |
| torchvision / torchaudio | `0.25.0` / `2.10.0` |
| CUDA toolkit | `12.8`，安装在 `hif4` conda env 内 |
| lighteval | `3rdparty/lighteval` editable，不安装 `[vllm]` extras |

关键约束：

- 不使用 `VLLM_USE_PRECOMPILED=1`。
- 不拉 `wheels.vllm.ai` 的 rolling wheel。
- vLLM 必须从 `3rdparty/vllm` editable 安装，因为后续会改 vLLM 源码。
- 所有安装命令只作用于 `hif4`，不能影响其他 conda 环境。

## 当前安装命令

当前主线只执行脚本：

```bash
conda activate hif4
bash install.sh
```

下面是当时 v0.17.0 迁移时的历史关键步骤，只用于理解问题，不是当前 `install.sh` 的等价展开：

```bash
conda activate hif4

# 历史记录：当时目标是 v0.17.0。当前主线是 v0.19.1，不要照抄这一行。
git -C 3rdparty/vllm checkout v0.17.0
conda install -n hif4 -c nvidia cuda-toolkit=12.8 -y

pip install "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0"
pip install -r 3rdparty/vllm/requirements/build.txt

export CUDA_HOME="$CONDA_PREFIX"
export CUDA_ROOT="$CUDA_HOME"
export CUDA_PATH="$CUDA_HOME"
export CUDAToolkit_ROOT="$CONDA_PREFIX/targets/x86_64-linux"
export CUDA_TOOLKIT_ROOT_DIR="$CUDAToolkit_ROOT"
export PATH="$CONDA_PREFIX/bin:$CONDA_PREFIX/nvvm/bin:$PATH"
export LD_LIBRARY_PATH="$CUDAToolkit_ROOT/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export NVCC_PREPEND_FLAGS="-I$CUDAToolkit_ROOT/include ${NVCC_PREPEND_FLAGS:-}"

cd 3rdparty/vllm
MAX_JOBS=32 \
CMAKE_ARGS="-DCUDAToolkit_ROOT=$CUDAToolkit_ROOT -DCUDA_TOOLKIT_ROOT_DIR=$CUDAToolkit_ROOT -DCUDA_INCLUDE_DIRS=$CUDAToolkit_ROOT/include -DCUDA_CUDART_LIBRARY=$CUDAToolkit_ROOT/lib/libcudart.so" \
    pip install --editable . --no-build-isolation
```

如果 `MAX_JOBS=32` 导致内存压力过大或机器负载过高，用：

```bash
MAX_JOBS=8 bash install.sh
```

## 为什么不是沿用 v0.10.2 的安装方式

旧的 `v0.10.2` 路线不是“在 conda 里装最新 CUDA toolkit 再编译”。它依赖的是
vLLM 0.10.2 的预编译二进制和 torch 2.8 CUDA wheel 的 ABI 匹配。

这条路线可以跑旧模型，但有两个限制：

- vLLM 0.10.2 不支持 Qwen3.5 这类更新 MoE 架构。
- editable + precompiled wheel 会把源码可改性和二进制 ABI 问题混在一起，升级到
  0.17.0 后风险更高。

vLLM 0.17.0 这次改为源码编译，是为了同时满足：

- 能支持 Qwen3.5。
- 仍然保留 `3rdparty/vllm` editable 源码修改能力。
- 编译出的 `vllm._C` 和当前环境里的 torch 2.10.0 ABI 对齐。

## 报错 1：`VLLM_USE_PRECOMPILED=1` 后 `undefined symbol: MessageLogger`

错误形态：

```text
ImportError: .../vllm/_C.abi3.so: undefined symbol:
_ZN3c1013MessageLoggerC1ENS_14SourceLocationEib
```

根因：

- `VLLM_USE_PRECOMPILED=1` 在 editable 安装时会去拉 `wheels.vllm.ai` 的 rolling
  wheel。
- 这个 wheel 可能是用更新的 torch/nightly ABI 编出来的。
- 本环境里的 stable `torch==2.10.0` 提供的是另一代 `c10::MessageLogger` 符号。

处理方式：

- 禁止使用 `VLLM_USE_PRECOMPILED=1`。
- 使用 `pip install --editable . --no-build-isolation` 源码编译。
- 先安装目标 torch，再编译 vLLM，确保 ABI 跟运行时 torch 一致。

## 报错 2：`Could NOT find CUDA (missing: CUDA_INCLUDE_DIRS)`

错误形态：

```text
Could NOT find CUDA (missing: CUDA_INCLUDE_DIRS) (found version "12.8")
Your installed Caffe2 version uses CUDA but I cannot find the CUDA libraries.
```

根因：

conda 安装的 `cuda-toolkit=12.8` 不是传统 `/usr/local/cuda` 布局。关键文件在：

```text
$CONDA_PREFIX/targets/x86_64-linux/include/cuda_runtime.h
$CONDA_PREFIX/targets/x86_64-linux/lib/libcudart.so
$CONDA_PREFIX/bin/nvcc
```

CMake 能看到 CUDA 版本，但没有自动找到 headers/libs。

处理方式：

```bash
export CUDA_HOME="$CONDA_PREFIX"
export CUDAToolkit_ROOT="$CONDA_PREFIX/targets/x86_64-linux"
export CUDA_TOOLKIT_ROOT_DIR="$CUDAToolkit_ROOT"
```

并在 vLLM build 时传：

```bash
CMAKE_ARGS="-DCUDAToolkit_ROOT=$CUDAToolkit_ROOT -DCUDA_TOOLKIT_ROOT_DIR=$CUDAToolkit_ROOT -DCUDA_INCLUDE_DIRS=$CUDAToolkit_ROOT/include -DCUDA_CUDART_LIBRARY=$CUDAToolkit_ROOT/lib/libcudart.so"
```

## 报错 3：`cuda_runtime.h: No such file or directory`

错误形态：

```text
<command-line>: fatal error: cuda_runtime.h: No such file or directory
```

根因：

`nvcc` 被找到后，内部调用 host compiler 时没有把 conda CUDA 的 include 目录带上。

处理方式：

```bash
export NVCC_PREPEND_FLAGS="-I$CONDA_PREFIX/targets/x86_64-linux/include ${NVCC_PREPEND_FLAGS:-}"
```

这个变量会让 `nvcc` 在每次调用时自动加上 include path。

## 报错 4：`cicc: not found`

错误形态：

```text
sh: 1: cicc: not found
```

根因：

conda CUDA 的 `cicc` 在：

```text
$CONDA_PREFIX/nvvm/bin/cicc
```

但默认 `PATH` 里不一定包含 `$CONDA_PREFIX/nvvm/bin`。

处理方式：

```bash
export PATH="$CONDA_PREFIX/bin:$CONDA_PREFIX/nvvm/bin:$PATH"
```

## 报错 5：`crt/link.stub: No such file or directory`

错误形态：

```text
cc1plus: fatal error:
/.../targets/x86_64-linux/bin/crt/link.stub: No such file or directory
```

根因：

如果把 `CUDA_HOME` 直接设成 `$CONDA_PREFIX/targets/x86_64-linux`，`nvcc` 会去
`targets/x86_64-linux/bin/crt/link.stub` 找文件。但 conda CUDA 实际位置是：

```text
$CONDA_PREFIX/bin/crt/link.stub
```

处理方式：

- `CUDA_HOME` 设为 `$CONDA_PREFIX`。
- `CUDAToolkit_ROOT` 单独设为 `$CONDA_PREFIX/targets/x86_64-linux`。
- include/lib 通过 `NVCC_PREPEND_FLAGS` 和 `CMAKE_ARGS` 显式传入。

也就是本仓库最终采用的组合：

```bash
export CUDA_HOME="$CONDA_PREFIX"
export CUDAToolkit_ROOT="$CONDA_PREFIX/targets/x86_64-linux"
export PATH="$CONDA_PREFIX/bin:$CONDA_PREFIX/nvvm/bin:$PATH"
export NVCC_PREPEND_FLAGS="-I$CUDAToolkit_ROOT/include ${NVCC_PREPEND_FLAGS:-}"
```

## 报错 6：当前节点没有可用 NVIDIA driver

错误形态：

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

或：

```text
torch.cuda.is_available() -> False
torch.cuda.device_count() -> 0
```

根因：

conda CUDA toolkit 只能提供编译期的 `nvcc`、headers 和用户态 CUDA 库，不能替代
内核 NVIDIA driver。

处理方式：

- 源码编译和 `import vllm._C` 可以在没有可用 driver 的节点上完成。
- 真正的 GPU 推理、Qwen3.5 初始化和 generation smoke test 必须去 driver 正常的节点运行。

## 报错 7：lighteval 重新覆盖本地 editable vLLM

风险：

```bash
pip install -e "3rdparty/lighteval[vllm]"
```

可能触发 pip 重新解析 vLLM 依赖，覆盖刚刚编译好的本地 editable vLLM。

处理方式：

只安装 lighteval 本体：

```bash
cd 3rdparty/lighteval
pip install --editable .
pip install more_itertools langdetect
```

本仓库 `3rdparty/lighteval` 已经有本地补丁放宽 `is_package_available` 的版本检查，
否则 lighteval 可能把 vLLM backend 视为不可用。

## 环境隔离适配

为了避免误影响其他环境，`install.sh` 现在强制检查：

```bash
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "hif4" ]]; then
    echo "当前 conda env 不是 hif4"
    exit 1
fi
```

CUDA toolkit 也通过：

```bash
conda install -n hif4 -c nvidia cuda-toolkit=12.8 -y
```

只安装到 `hif4`。这不会修改 `qwen35`、base 或其他 conda env。

## 安装成功后的验证

源码编译完成后运行：

```bash
conda run -n hif4 python -c "import vllm, vllm._C, torch; print(vllm.__version__, vllm.__file__, torch.__version__, torch.version.cuda)"
```

期望输出类似：

```text
0.17.0 /home/shaoyuantian/program/HiF4_Sp/3rdparty/vllm/vllm/__init__.py 2.10.0+cu128 12.8
```

再确认 pip metadata：

```bash
conda run -n hif4 python -m pip show vllm torch
```

关键字段应包含：

```text
Name: vllm
Version: 0.17.0+cu128
Editable project location: /home/shaoyuantian/program/HiF4_Sp/3rdparty/vllm

Name: torch
Version: 2.10.0
```

GPU 检查：

```bash
conda run -n hif4 python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi
```

如果 driver 不可用，这一步会失败或返回 `False 0`，但不代表 vLLM 编译失败。

## Qwen3.5 验证建议

到 driver 正常的节点后，再跑小样本验证：

```bash
conda activate hif4
python main.py \
  --datasets gsm8k \
  --model_path /path/to/Qwen3.5-35B-A3B \
  --max_samples 1 \
  --max_model_length 4096 \
  --max_new_tokens 128
```

成功标准：

- 模型架构能被 vLLM 0.17.0 识别。
- 初始化完成。
- 至少完成一次 generation。

## 当时改动摘要

- 当时 `3rdparty/vllm` submodule 指到 `v0.17.0`；当前主线已是 `v0.19.1`。
- `install.sh` 改为：
  - 只允许在 `hif4` 运行。
  - 安装 `cuda-toolkit=12.8` 到 `hif4`。
  - 固定 torch 栈到 `2.10.0 / 0.25.0 / 2.10.0`。
  - 安装 vLLM build requirements。
  - 使用 `pip install --editable . --no-build-isolation` 源码编译。
  - 设置 conda CUDA 所需的 `CUDA_HOME`、`CUDAToolkit_ROOT`、`PATH`、
    `LD_LIBRARY_PATH`、`NVCC_PREPEND_FLAGS` 和 `CMAKE_ARGS`。
  - 保留 lighteval editable 安装，但不安装 `[vllm]` extras。
- README 和 `docs/vllm_install_notes.md` 当前以 `v0.19.1` 主线为准；本文只保留 v0.17.0 迁移过程。
