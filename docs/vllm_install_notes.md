# vLLM 安装踩坑与备选方案

本文档记录在搭建本仓库环境过程中遇到的 vLLM 安装问题、根因分析、以及当前采用的
安装方案。主要目的：

1. 开源后如果有人在 CUDA 12.4 / driver 550 这类“相对老旧”的系统上复现环境失败，
   可以直接查到现象与解决方案。
2. 后续继续升级 vLLM 时，知道之前是卡在哪一步。

当前主线已升级到 `3rdparty/vllm@v0.19.1`，仍采用 editable source build。
下面关于 v0.17.0 / v0.19.0 预编译 wheel 的内容是历史踩坑记录。

---

## 环境背景

| 组件 | 版本 |
|------|------|
| OS | Linux 5.15.0-164-generic |
| GPU Driver | 550.54.14（声明支持 CUDA 12.4） |
| nvcc | CUDA 12.4 |
| Conda env | `hif4`（Python 3.11） |

目标：在 `3rdparty/vllm` 内置源码目录中固定 vLLM 版本。当前主线固定为
`v0.19.1`，并采用 `pip install --editable . --no-build-isolation` 做真正源码
编译，避免 `VLLM_USE_PRECOMPILED=1` 拉取 `wheels.vllm.ai` rolling wheel 后造成
torch c10 ABI 不匹配。

当前主线构建栈：

| 组件 | 版本 |
|------|------|
| vLLM | `v0.19.1`（`3rdparty/vllm` 内置源码目录, editable source build） |
| torch | `2.10.0` |
| torchvision / torchaudio | `0.25.0` / `2.10.0` |
| transformers | `5.6.2` |
| CUDA toolkit | `12.8`（优先安装在 `hif4` conda env 内） |

注意：conda CUDA toolkit 只负责编译期的 nvcc/headers 和用户态 CUDA 库，不能替代
内核 NVIDIA driver。若 `nvidia-smi` 无法访问驱动，vLLM 可以完成 CPU 侧 import
验证，但 GPU 推理仍需在 driver 正常的节点上测试。

---

## 现象：vLLM v0.17.0 / v0.19.0 预编译安装后 `import vllm._C` 失败

执行：

```bash
cd 3rdparty/vllm
git checkout v0.17.0          # 或 v0.19.0
VLLM_USE_PRECOMPILED=1 pip install --editable .
python -c "import vllm"
```

报错（关键行）：

```
File ".../3rdparty/vllm/vllm/platforms/cuda.py", line 19, in <module>
    import vllm._C
ImportError: .../vllm/_C.abi3.so: undefined symbol:
    _ZN3c1013MessageLoggerC1ENS_14SourceLocationEib
```

C++ demangle 之后是：

```
c10::MessageLogger::MessageLogger(c10::SourceLocation, int, bool)
```

v0.17.0 和 v0.19.0 的预编译 wheel 均会触发该错误。

---

## 根因：vLLM 预编译 wheel 的 torch ABI 与本机 stable torch 不一致

> **TL;DR**：`vllm/_C.abi3.so` 是 C++ 扩展，必须和当前进程里的 `torch/lib/libc10.so`
> 用同一份 ABI。`c10::MessageLogger` 的构造函数签名在 torch 历史上至少改过 3 次，
> 我们之前用 `wheels.vllm.ai` 滚动 wheel 时，它跟着 torch nightly 编，引用了
> 一个本机 stable torch 还没暴露的新签名，dlopen 时找不到符号就报 undefined symbol。

1. 从 `wheels.vllm.ai` 上实际装下来的版本是类似 `0.17.1.dev0+gb31e9326a.d20260423.precompiled`，
   **每天滚动重编**；wheel 的 `Requires-Dist` 虽然写着 `torch==2.10.0`，
   但它的 `_C.abi3.so` 是按更新的 torch 源码（甚至 nightly）编译的，已经切换到了
   `c10::MessageLogger(c10::SourceLocation, int, bool)` 这个新签名。
2. PyPI 默认 `torch==2.10.0+cu128` 以及 pytorch.org cu129 索引上的
   `torch==2.10.0+cu129`，它们 `torch/lib/libc10.so` 里只有 4 参数的老签名
   `c10::MessageLogger(char const*, int, int, bool)`。
3. 所以加载 `vllm._C` 时找不到符号，报 undefined symbol。

验证：

```bash
nm -D --defined-only $CONDA_PREFIX/lib/python3.11/site-packages/torch/lib/libc10.so \
    | c++filt | grep MessageLogger
# 结果：只有 c10::MessageLogger::MessageLogger(char const*, int, int, bool)
# 没有 c10::MessageLogger::MessageLogger(c10::SourceLocation, int, bool)
```

`c10::MessageLogger` 构造函数在 torch 上看到过的 3 个版本（按时间从老到新）：

| torch 版本 | ABI symbol（mangled） | demangled |
|------------|------------------------|-----------|
| ≤ 2.8.x    | `_ZN3c1013MessageLoggerC1EPKcii`         | `MessageLogger(char const*, int, int)` |
| 2.10.x     | `_ZN3c1013MessageLoggerC1EPKciib`        | `MessageLogger(char const*, int, int, bool)` |
| ≥ 2.11 / nightly | `_ZN3c1013MessageLoggerC1ENS_14SourceLocationEib` | `MessageLogger(c10::SourceLocation, int, bool)` |

vLLM 预编译 wheel 必须按某一代 torch 编出来，运行时 torch 必须能提供同一代签名。

## 官方 v0.17.0 release notes 里的 known issue 能解决吗？

v0.17.0 release 说明里提到：

> Known Issue: If you are on CUDA 12.9+ and encounter a CUBLAS_STATUS_INVALID_VALUE
> error, this is caused by a CUDA library mismatch. To resolve, try one of the following:
>
> 1. Remove the path to system CUDA shared library files (e.g. /usr/local/cuda) from
>    LD_LIBRARY_PATH, or simply unset LD_LIBRARY_PATH.
> 2. Install vLLM with `uv pip install vllm --torch-backend=auto`.
> 3. Install vLLM with
>    `pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129`.

**不能**解决我们这个错误。对比：

| | release notes 描述的问题 | 本次我们遇到的问题 |
|---|---|---|
| 触发时机 | runtime（调用 cuBLAS 时） | import 时（加载 `_C.abi3.so`） |
| 错误形式 | `CUBLAS_STATUS_INVALID_VALUE` | `undefined symbol: _ZN3c1013MessageLogger...` |
| 根因 | libcublas 版本不匹配 | **C++ ABI 不匹配**（torch 的 c10 符号） |

1、2、3 都是在不同的 CUDA / torch 变体中挑选，但所有 stable `torch==2.10.0`
的 `libc10.so` 都不含新签名，所以换变体没用。实测 cu128 → cu129 确认仍然报同一个
undefined symbol。

---

## 当前方案 A：vLLM 0.19.1 源码编译（主线采用）

`install.sh` 当前采用源码编译流程：

```bash
conda activate hif4
git -C 3rdparty/vllm checkout v0.19.1
conda install -n hif4 -c nvidia cuda-toolkit=12.8 -y
pip install "cmake>=3.26.1" ninja "packaging>=24.2" \
            "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8.0" wheel jinja2 \
            "grpcio-tools==1.78.0"
pip install "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0"
pip install "transformers==5.6.2"

pushd 3rdparty/vllm
CUDA_HOME="$CONDA_PREFIX" \
CUDAToolkit_ROOT="$CONDA_PREFIX/targets/x86_64-linux" \
CUDA_TOOLKIT_ROOT_DIR="$CONDA_PREFIX/targets/x86_64-linux" \
PATH="$CONDA_PREFIX/bin:$CONDA_PREFIX/nvvm/bin:$PATH" \
LD_LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
NVCC_PREPEND_FLAGS="-I$CONDA_PREFIX/targets/x86_64-linux/include ${NVCC_PREPEND_FLAGS:-}" \
MAX_JOBS=32 pip install --editable . --no-build-isolation
popd
```

要点：

- `--no-build-isolation` 强制让构建过程用当前环境里的 torch，这样编出来的 `_C.so`
  一定和运行时 torch 的 ABI 匹配。
- 需要机器上有 `nvcc`。如果系统 CUDA 偏旧，脚本会在 `hif4` 里安装
  `cuda-toolkit=12.8`，并把 `CUDA_HOME` 指向 `$CONDA_PREFIX`；conda 的 CUDA
  headers/libs 位于 `$CONDA_PREFIX/targets/x86_64-linux`，所以还要显式设置
  `CUDAToolkit_ROOT` 和 `NVCC_PREPEND_FLAGS`。
- 编译时间按机器 `MAX_JOBS` 设置，单机 256 核 1.5TiB 内存上 MAX_JOBS=32 大约
  30–45 分钟；低配机器要更久。如果编译压力太大，用 `MAX_JOBS=8` 重试。
- **开源友好**：别人拿到仓库后，只要他有 nvcc，就能复现；不依赖某个特定 torch
  nightly 的存档。

## 历史方案 B：回退到 vLLM v0.10.2 + 预编译（**已不再是主线**）

v0.10.2 是 2025 年发布的稳定版，那个时候 vLLM 还没有跳到新的 c10 ABI；它的预编译
wheel 能正常搭配 PyPI stable torch 2.8.0 系列。**但是不能直接裸跑
`VLLM_USE_PRECOMPILED=1`**——这样 setup.py 会去拉 `wheels.vllm.ai` 上的 rolling
dev build，那个 URL 虽然 tag 还叫 v0.10.2，底层二进制却是每天滚动重编的，有
可能和我们本机 torch 不匹配，实测会在 `import vllm._C` 时直接 abort 成
`std::bad_alloc`（一个 C++ 构造函数里的 bad_alloc，dlopen 阶段就挂，Python 无法
捕获）。

正确的做法是**显式告诉 setup.py 去 PyPI 拉 v0.10.2 发布时那份静态 wheel**：

```bash
cd 3rdparty/vllm
git checkout v0.10.2
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION="https://files.pythonhosted.org/packages/a2/1a/365479f413e7408b314c0237d6c929569874d5c002bc7c8b5a7fbf40c7d9/vllm-0.10.2-cp38-abi3-manylinux1_x86_64.whl" \
    pip install --editable .
```

`VLLM_PRECOMPILED_WHEEL_LOCATION` 是 vLLM 的 setup.py 提供的显式开关
（见 `setup.py` 里 `os.getenv("VLLM_PRECOMPILED_WHEEL_LOCATION", None)` 分支）。
指向 PyPI 上 v0.10.2 的不可变 wheel 之后，每次装都能拿到相同的二进制，开源复现
友好。

代价：

- 不是最新的 vLLM。某些模型（比如 DeepSeek V3 / Qwen3 更新的 MoE 结构）可能缺
  支持；后续 lighteval 升级时如果开始要求更新的 vLLM，再切回方案 A（源码编译）。
- 和 `local_lighteval` 老版本一致，复现起来和原仓库 `Half-Experts-Candoall`
  的已知可用组合几乎一样，风险最低。

### 排查日志（留给以后的自己看）

- 现象：`import vllm._C` 触发 `terminate called after throwing an instance of
  'std::bad_alloc'`，exit 134（core dumped）。没有 Python traceback，是 C++ 构造
  函数里抛的；改用 `ctypes.CDLL(...)` 直接 dlopen 也一样会 abort。
- 验证是二进制问题而不是环境问题：在一个**同机、同 torch 2.8.0+cu128、同驱动**
  的另一个 conda env（`halfexperts`，Python 3.10）里装的 PyPI `vllm==0.10.2`
  能正常 `from vllm import LLM`；两边 `.so` md5 不同——我们 `VLLM_USE_PRECOMPILED=1`
  抓到的是 wheels.vllm.ai 今天重编的 406MB 版本，halfexperts 里那份是 4 月 17 日
  从 PyPI 装的 363MB 版本。把 halfexperts 里的 `_C.abi3.so / _moe_C.abi3.so /
  _flashmla_C.abi3.so` 原样拷到我们的 submodule 目录下覆盖，立刻就能 import 了。
- 最终的正式修复：让 setup.py 指向 PyPI URL（上面那个 `VLLM_PRECOMPILED_WHEEL_LOCATION`
  开关），装完的 `.so` 与 halfexperts 那份 md5 完全一致。

---

## 备选方案 C（事后发现的最简单方案）：直接 `pip install vllm==X`，**不要 editable**

如果你**不打算改 vLLM 源码**（只想用它做推理 / lighteval 后端），那根本不用走
submodule + editable + `VLLM_USE_PRECOMPILED=1` 这条路。**最简单可靠的做法是
让 pip 自己解析依赖**：

```bash
conda create -n <some_env> python=3.11 -y
conda activate <some_env>
pip install vllm==0.17.0      # 或 0.10.2 / 0.19.0 / 任意 PyPI 上发布过的版本
```

这条命令会让 pip：

1. 拿 PyPI 上 v0.17.0 的**正式发布 wheel**——是不变的二进制，不是 wheels.vllm.ai
   每天滚动重编的版本。
2. 解析 wheel 的 `Requires-Dist`，把 torch 装到 vLLM **发布时实际用来编译**的那个
   版本（v0.17.0 → torch 2.10.0+cu128；v0.10.2 → torch 2.8.0+cu128）。
3. C++ ABI 自动对齐，import 一次过。

我们之前没意识到这条路，是因为本仓库一开始就把 vLLM 作为 submodule editable 装，
为的是“万一以后要改 vLLM 源码”。这个权衡仍然成立：主线继续保留 editable，但从
旧方案 B 改为当前方案 A（源码编译）；如果你只是想跑实验、不改 vLLM，方案 C 仍然
是最干净的。

详见下面"补遗：在 qwen35 环境里的对照实验"。

## 后续升级路径

按优先级：

1. **不改 vLLM 源码的场景**：直接走方案 C（`pip install vllm==X`），让 pip 帮你
   把 torch 选到匹配的版本。这是开源用户最不容易踩坑的入口。
2. 需要改 vLLM 源码的场景：使用当前主线方案 A，固定 submodule tag，并用当前环境的
   torch 做 `--no-build-isolation` 源码编译。
3. 如果后续硬件和 driver 允许，也可以重新评估 vLLM 0.18/0.19 的 PyPI wheel 或
   CUDA 12.9/13.0 wheel，但不要混用 editable 源码和 rolling precompiled wheel。
4. 任何时候都避免“`VLLM_USE_PRECOMPILED=1` + 过老的 torch”的组合，因为 vLLM 的
   rolling build 随时可能引入 torch nightly 才有的符号。

---

## 补遗：在 qwen35 环境里的对照实验（2026-04-23/24）

事后我们在另一个干净的 conda env (`qwen35`) 里做了一次 sanity check——**只跑
`pip install vllm==0.17.0`，什么 editable / `VLLM_USE_PRECOMPILED` / wheel URL
锚定都不做**。

```bash
conda create -n qwen35 python=3.11 -y
conda activate qwen35
pip install vllm==0.17.0
python -c "import vllm; import vllm._C; from vllm import LLM, SamplingParams; print(vllm.__version__)"
# -> 0.17.0   (无任何 ImportError，无 std::bad_alloc)
```

竟然**完全没问题**。和我们之前在旧 `expertpruning` 环境里 v0.17.0 屡次失败的现象
形成强烈反差。

进一步对照 `_C.abi3.so` 引用的 `MessageLogger` 符号（`U` = undefined / 需要外部
提供，`T` = defined / 自己提供）和对应 torch 的 `libc10.so` 实际暴露的符号：

| env | torch | torch libc10 提供的 MessageLogger ctor | 装的 vllm | vllm `_C.abi3.so` 引用的 ctor | 结果 |
|-----|-------|----------------------------------------|-----------|---------------------------------|------|
| 旧 `expertpruning` | **2.8.0+cu128** | `MessageLogger(char*, int, int)` | 0.10.2 PyPI wheel | `EPKcii`（3 参数） | ✅ 匹配 |
| `qwen35` | **2.10.0+cu128** | `MessageLogger(char*, int, int, bool)` | 0.17.0 PyPI wheel | `EPKciib`（4 参数） | ✅ 匹配 |
| 旧 `expertpruning`（之前失败的实验） | 2.10.0+cu128 *(当时)* | 4 参数 | 0.17.0 / 0.19.0 **wheels.vllm.ai rolling** | `NS_14SourceLocationEib`（新 ABI） | ❌ undefined symbol |

收获：

- **`pip install vllm==X` 比我们之前以为的更智能**：它会按 PyPI wheel 的
  `Requires-Dist` 元数据，自动把 torch 解析到 vLLM 发布时编译用的那个版本。
  v0.10.2 → torch 2.8.0、v0.17.0 → torch 2.10.0 ——pip 是透明地帮我们做了**ABI
  代际对齐**。
- 我们之前对方案 B（`VLLM_PRECOMPILED_WHEEL_LOCATION` 锚定 PyPI wheel）的功劳
  评价偏高了：它确实让 wheel 来源稳定可复现，但**真正让 v0.10.2 在
  旧 `expertpruning` 跑通的关键，其实是 pip 把 torch 一并降到了 2.8.0**。如果
  当时 torch 没被降，光锚定 wheel 也救不了——因为 0.10.2 wheel 需要的 3 参数
  ABI 在 torch 2.10.0 里同样不存在。
- **`wheels.vllm.ai` 的 rolling 滚动 wheel 是真正的元凶**：它跟着 torch nightly
  编，引用的 `MessageLogger(SourceLocation,…)` 在任何 stable torch 里都不存在，
  跟版本号 (0.17 / 0.19) 没关系，跟 vLLM 自己的代码也没关系，纯粹是 **torch
  nightly 改了 c10 内部接口**。

`nm` 证据：

```bash
# qwen35 (干净 PyPI 装的 0.17.0)
nm -D ~/anaconda3/envs/qwen35/lib/python3.11/site-packages/vllm/_C.abi3.so \
    | grep MessageLogger
#   U _ZN3c1013MessageLogger6streamB5cxx11Ev
#   U _ZN3c1013MessageLoggerC1EPKciib                  <-- 4 参数 (torch 2.10 风格)
#   U _ZN3c1013MessageLoggerD1Ev

nm -D ~/anaconda3/envs/qwen35/lib/python3.11/site-packages/torch/lib/libc10.so \
    | grep "T .*MessageLoggerC"
#   T _ZN3c1013MessageLoggerC1EPKciib                  <-- 正好对得上 ✅

# 旧 expertpruning (历史方案 B 装的 0.10.2)
nm -D /home/chenyuanteng/An-Empirical-Study-on-Expert-Pruning/3rdparty/vllm/vllm/_C.abi3.so \
    | grep MessageLogger
#   U _ZN3c1013MessageLoggerC1EPKcii                   <-- 3 参数 (torch 2.8 风格)
#   U _ZN3c1013MessageLoggerD1Ev

nm -D ~/anaconda3/envs/expertpruning/lib/python3.11/site-packages/torch/lib/libc10.so \
    | grep "T .*MessageLoggerC"
#   T _ZN3c1013MessageLoggerC1EPKcii                   <-- 也对得上 ✅
```

**结论**：历史方案 B 解释了为什么旧 0.10.2 环境能工作；当前
`install.sh` 已切到方案 A，用 v0.19.1 源码编译继续保留 editable。
不再推荐用旧 wheel 路径配置新环境。

---

## 两个 install 脚本的分工

基于上面的讨论，本仓库当前只把 `install.sh` 作为主线安装入口。
`install_qwen35.sh` 是旧 baseline 路径，保留用于历史复现。

| 脚本 | conda env | vLLM 来源 | 对应上面的方案 | 何时选 |
|------|-----------|-----------|----------------|--------|
| `install.sh` | `hif4` | `3rdparty/vllm` 内置源码目录 (`v0.19.1`) + `pip install --editable . --no-build-isolation` 源码编译。torch 固定到 `2.10.0`，Transformers 固定到 `5.6.2`，CUDA toolkit 默认用 conda `12.8`。 | **方案 A**：editable source build | 当前主线。Qwen3.5 / HiF4 适配已在内置源码目录内，用户不需要再手工改 vLLM；ABI 靠当前环境 torch 源码编译保证。|
| `install_qwen35.sh`        | `qwen35`        | 旧 PyPI wheel 路径。 | 历史 baseline | 只用于复现旧环境，不作为当前主线。|

`install.sh` 会从 `3rdparty/lighteval`（本仓库 fork，
`expert-pruning-mods` 分支）editable 安装 lighteval，本仓库打的
`[ExpertPruning-mod]` 补丁会在 `hif4` 生效。

**为什么仍然源码编译？**

- `hif4` 不能用 `VLLM_USE_PRECOMPILED=1`：editable 会拉 `wheels.vllm.ai` 的 rolling
  dev wheel，就是本文开头那个 ABI 不匹配的元凶。主线必须源码编译。
- `install_qwen35.sh` 不能替代 `install.sh`：它是旧 wheel 路径，
  不是 editable，不能用于修改 vLLM 源码。
