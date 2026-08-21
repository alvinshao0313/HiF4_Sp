# vLLM v0.27.0 NVFP4 Emulation Backport Inventory

## Upstream source freeze

| Item | Value |
|---|---|
| Tag | `v0.27.0` |
| Tag commit SHA | `4bdc8a788d2e2ce9165d552b3d4d8b72604626bf` |
| Fetch date | `2026-08-21T02:26:57+00:00` |
| Reference path | `/tmp/vllm-v0.27.0-reference` (workspace 外，未入库) |
| Historical note | PR `#44667` merge `552a9db` 仅作历史定位；shallow clone 未包含该对象，最终以 tag `v0.27.0` 文件内容为准 |

## Local vendored baseline

| Item | Value |
|---|---|
| Path | `3rdparty/vllm` |
| Git HEAD | `0cdc52596045680b7174f19befab726371fe1f5b` |
| `vllm.__version__` (hif4) | `0.19.2.dev0+gb1388b1fb.d20260430` |
| Repo branch | `block-sparse` (ahead 1) |

## Baseline import / test (Task 0 Step 4)

```text
conda run -n hif4 python -c "import sys; sys.path.insert(0, '3rdparty/vllm'); import vllm; print(vllm.__version__)"
→ 0.19.2.dev0+gb1388b1fb.d20260430

conda run -n hif4 pytest -q 3rdparty/vllm/tests/kernels/quantization/test_nvfp4_quant.py
→ FAIL (pre-existing): ModuleNotFoundError: No module named 'tblib'
  (tests/conftest.py imports tblib)

conda run -n hif4 pytest -q --noconftest -rs 3rdparty/vllm/tests/kernels/quantization/test_nvfp4_quant.py
→ SKIPPED: Nvfp4 Requires compute capability of 10 or above.
```

Task 0 不修复 `tblib` / SM 门禁；后续 emulation 相关测试需避免依赖 SM100 native quant kernel skip，且尽量绕开有问题的全局 conftest，或在测试环境补齐 `tblib`（仅当后续任务跑 pytest 必需时再装，不算“为迁就 v0.27 升级依赖栈”）。

## Per-file classification

### A. 当前缺失、必须新增

| Upstream path | 说明 |
|---|---|
| `vllm/model_executor/kernels/linear/nvfp4/__init__.py` | NVFP4 linear kernel 包入口 |
| `vllm/model_executor/kernels/linear/nvfp4/base.py` | `NvFp4LinearKernel` / `NvFp4LinearLayerConfig` |
| `vllm/model_executor/kernels/linear/nvfp4/emulation.py` | `EmulationNvFp4LinearKernel` |
| `vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py` | `Nvfp4QuantizationEmulationTritonExperts` |
| `tests/kernels/quantization/test_nvfp4_emulation.py` | upstream emulation 数值/MoE 测试（按本地 fixture 适配） |
| `tests/config/test_kernel_config.py` | 本地无此文件；按 Task 1 新建，仅覆盖 backend 字段 |

**条件新增（仅当 registry/selection 硬依赖，且无法复用本地旧 native 路径时）：**

| Upstream path | 判定 |
|---|---|
| `vllm/model_executor/kernels/linear/nvfp4/cutlass.py` | emulation 注册若引用同列表则可能需要 stub/最小类；优先只导出 emulation 所需 selection 路径 |
| `vllm/model_executor/kernels/linear/nvfp4/marlin.py` | 同上；正式实验禁止目标层落到 Marlin，但 selection 代码可能 import 该类 |
| `vllm/model_executor/kernels/linear/nvfp4/flashinfer.py` | 非 emulation 必需则不回移 |
| `vllm/model_executor/kernels/linear/nvfp4/fbgemm.py` | 非 emulation 必需则不回移 |
| `vllm/model_executor/kernels/linear/nvfp4/humming.py` | 非 emulation 必需则不回移 |

### B. 当前已有旧实现、必须按 v0.27.0 局部升级

| Local path | 现状 | 回移要点 |
|---|---|---|
| `vllm/config/kernel.py` | 仅有 `MoEBackend`，无 `emulation`；无 `LinearBackend` / `linear_backend` | 最小加入 `LinearBackend`（至少含 `auto`+现有可用项+`emulation`）、`moe_backend` 增加 `emulation`、validators；**不**回移 `IrOpPriorityConfig` / JIT warmup 等无关字段 |
| `vllm/engine/arg_utils.py` | 有 `moe_backend` CLI，无 `linear_backend` | 对齐 v0.27：`EngineArgs.linear_backend` + `--linear-backend` + 写入 `KernelConfig` |
| `vllm/entrypoints/llm.py` | 无显式 `linear_backend`/`moe_backend` 字段（经 EngineArgs kwargs） | 仅在确认 Python `LLM(**kwargs)` 无法透传时再改；当前优先依赖 EngineArgs |
| `vllm/model_executor/kernels/linear/__init__.py` | 无 NVFP4 package / `select_nvfp4_linear_kernel` | **局部**接入 NVFP4 selection（`_get_linear_backend` + `select_nvfp4_linear_kernel` + emulation 映射）；禁止整文件替换为 v0.27（含 mxfp* 大重构） |
| `vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py` | 旧 Python 路径（约 4.6KB），缺 Triton QDQ / `kE2M1ToFloat_handle` / `ref_nvfp4_quant_dequant` | 对齐 v0.27.0 数学与 API；旧调用点可留兼容 wrapper |
| `vllm/model_executor/layers/quantization/utils/nvfp4_utils.py` | 旧 `NvFp4LinearBackend` + 环境变量选择 + 内联 apply | 改为调用新 `kernels/linear/nvfp4` selection；保留旧 env 入口兼容，正式入口改为 `linear_backend=emulation` |
| `vllm/model_executor/layers/quantization/modelopt.py` | Dense/MoE 走旧 `select_nvfp4_linear_backend` / 现有 MoE oracle（无 EMULATION） | Dense 交给新 linear kernel；MoE 在 `moe_backend=emulation` 时选 `Nvfp4QuantizationEmulationTritonExperts` |
| `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py` | 旧 backend enum 路径 | 按 v0.27 接新 kernel 接口，保留 CT reciprocal scale 语义 |
| `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_nvfp4.py` | **本地有、upstream v0.27 无同路径** | 仅当仍被 CT NVFP4 路径引用时做最小适配；不按“删除本地文件”处理 |
| `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py` | 本地单文件；upstream 已拆成 `compressed_tensors_moe/` 包 | **不整体换包**；只在本地单文件内接 `NvFp4MoeBackend.EMULATION` |
| `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` | 无 `EMULATION` | 回移 enum / map / convert / quant_config 的 emulation 分支（含 `a13_scale.max()` 倒数） |
| `vllm/model_executor/layers/fused_moe/utils.py` | 与 v0.27 有差异 | 仅补 emulation 所需 symbol（如 quantize 接口差异）；禁止整文件替换 |
| `vllm/model_executor/layers/fused_moe/config.py` | 有差异 | 仅补 emulation 硬依赖字段/helper |
| `vllm/model_executor/layers/fused_moe/experts/__init__.py` | 空文件（与 upstream 相同） | 按需 export；可保持空若直接路径 import |
| `main.py` | 无 `--linear_backend` / `--moe_backend` / `--kv_cache_dtype` | Task 7 透传；禁止与 `--fake_act_quant nvfp4` 混用 |
| `tests/engine/test_arg_utils.py` | 存在 | 扩展 backend/emulation 断言 |
| `tests/kernels/quantization/nvfp4_utils.py` | 存在 | 复用为 test helper |

### C. v0.27 有变化但本次明确不回移

| 区域 | 原因 |
|---|---|
| `IrOpPriorityConfig` / IR op priority | 与 NVFP4 emulation 无直接依赖 |
| `enable_cutedsl_warmup` / `enable_jit_warmup` / `enable_bf16x3_router_gemm` | warmup/实验特性 |
| MoEBackend 中 `batched_triton` / `deep_gemm_mega_moe` / `humming` / `flydsl` / `hpc` / `flashinfer_b12x` 等新增项 | 非 emulation 必需；本地保留现有字面量，仅追加 `emulation` |
| LinearBackend 中除 `emulation`（及 selection 硬依赖名）外的大量新 backend | 不全量扩字面量；最小集合以本地 EngineArgs 可解析 + emulation 可用为准 |
| `kernels/linear` 的 mxfp4/mxfp6/mxfp8 重组 | 无关 |
| upstream CT `compressed_tensors_moe/` 包化拆分 | 用本地单文件适配 |
| Quark / online NVFP4 / humming / Qutlass transform | 非 ModelOpt/CT 验收范围 |
| E2E gsm8k YAML、SM100 native scaled_mm / flashinfer NVFP4 tests | 非 emulation backport |
| 整仓 PyTorch/CUDA/FlashInfer/Transformers 升级 | 禁止 |
| 整体替换 `3rdparty/vllm` | 禁止 |

## Dependency closure whitelist（允许修改/新增）

后续任务若需新增 whitelist 外文件，必须先在本文件追加“缺失 symbol / 调用链 / 为何必需”，再改单个文件。

### Vendored vLLM — config / args

- `3rdparty/vllm/vllm/config/kernel.py`
- `3rdparty/vllm/vllm/engine/arg_utils.py`
- `3rdparty/vllm/vllm/entrypoints/llm.py`（仅当 kwargs 透传不足时）

### Vendored vLLM — Dense NVFP4 linear

- `3rdparty/vllm/vllm/model_executor/kernels/linear/__init__.py`（局部）
- `3rdparty/vllm/vllm/model_executor/kernels/linear/nvfp4/__init__.py` **(new)**
- `3rdparty/vllm/vllm/model_executor/kernels/linear/nvfp4/base.py` **(new)**
- `3rdparty/vllm/vllm/model_executor/kernels/linear/nvfp4/emulation.py` **(new)**
- `3rdparty/vllm/vllm/model_executor/kernels/linear/nvfp4/{cutlass,marlin,flashinfer,fbgemm,humming}.py` **(条件 new)**
- `3rdparty/vllm/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py`
- `3rdparty/vllm/vllm/model_executor/layers/quantization/utils/nvfp4_utils.py`
- `3rdparty/vllm/vllm/model_executor/layers/quantization/modelopt.py`
- `3rdparty/vllm/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py`
- `3rdparty/vllm/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_nvfp4.py`（条件）

### Vendored vLLM — MoE NVFP4 emulation

- `3rdparty/vllm/vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py` **(new)**
- `3rdparty/vllm/vllm/model_executor/layers/fused_moe/experts/__init__.py`（条件）
- `3rdparty/vllm/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`
- `3rdparty/vllm/vllm/model_executor/layers/fused_moe/utils.py`（条件局部）
- `3rdparty/vllm/vllm/model_executor/layers/fused_moe/config.py`（条件局部）
- `3rdparty/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py`

### Project root / NVFP4 scripts & tests / reports

- `main.py`
- `3rdparty/lighteval/src/lighteval/models/vllm/vllm_model.py`（仅当 LLM kwargs 未透传且调用链需要）
- `3rdparty/vllm/tests/config/test_kernel_config.py` **(new)**
- `3rdparty/vllm/tests/engine/test_arg_utils.py`
- `3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py` **(new/adapt)**
- `3rdparty/vllm/tests/kernels/quantization/test_nvfp4_quant.py`（仅当 API 断言需跟 v0.27）
- `3rdparty/vllm/tests/kernels/quantization/nvfp4_utils.py`
- `NVFP4/tests/test_vllm_emulation_cli_plumbing.py` **(new)**
- `NVFP4/tests/test_vllm_nvfp4_backend_contract.py` **(new)**
- `NVFP4/scripts/inspect_native_nvfp4_checkpoint.py` **(new)**
- `NVFP4/scripts/run_nvfp4_emulation_puncture.py` **(new)**
- `NVFP4/scripts/run_qwen3_30b_nvfp4_emulation_smoke.sh` **(new)**
- `NVFP4/scripts/run_qwen3_30b_nvfp4_emulation_mmlu_pro.sh` **(new)**
- `NVFP4/reports/vllm_v027_nvfp4_backport/*`

### Explicitly out of scope（勿改）

- `NVFP4/torch_fake.py` / `NVFP4/triton_fake.py` / 现有 `--fake_act_quant nvfp4` sidecar 路径（保留独立）
- 无关 vLLM 子系统、依赖栈、整体替换

## Key semantic gaps (summary)

1. **Dense：** 本地仍用 `NvFp4LinearBackend` + `VLLM_USE_NVFP4_CT_EMULATIONS` / `VLLM_NVFP4_GEMM_BACKEND`；v0.27 正式入口是 `KernelConfig.linear_backend="emulation"` → `EmulationNvFp4LinearKernel`。
2. **MoE：** 本地 oracle 无 `EMULATION`，无 `nvfp4_emulation_moe.py`；A800 上 auto 会落到 Marlin W4A16。
3. **数值原语：** 本地 `nvfp4_emulation_utils` 缺少 v0.27 Triton QDQ / device LUT handle；需对齐后再做 Dense/MoE。
4. **CLI：** `main.py` 尚未透传 `linear_backend` / `moe_backend` / `kv_cache_dtype`。
5. **验收 checkpoint：** `.../models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3/` 已存在于本机 cache。

## Adaptation principles for later tasks

1. 只回移 emulation 最小闭包；native CUTLASS/FlashInfer 类仅在 registry import 硬依赖时加入。
2. CT MoE 保持本地单文件，不跟 upstream 包化。
3. 不改变默认 `auto` 行为；正式实验显式 `emulation`；目标 NVFP4 层请求失败必须报错，禁止静默 Marlin。
4. 数学规则（含 MoE `a13_scale/a2_scale` 的 `max()` 倒数）严格跟 v0.27.0，禁止“精度修正”。
5. **A800 (SM80) Triton 门禁：** v0.27 Triton dequant/QDQ 使用 `tl.float8e4nv`，在 SM80 上无法编译。`nvfp4_emulation_utils` 增加 `_nvfp4_triton_fp8e4nv_supported()`（`has_device_capability(89)`），不支持时走 upstream 已有 Python 参考路径；**不改变数值规则**，只做硬件能力选择。
