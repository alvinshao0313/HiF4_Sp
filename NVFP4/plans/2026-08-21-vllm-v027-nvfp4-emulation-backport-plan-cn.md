# vLLM v0.27.0 NVFP4 Dense + MoE Emulation 局部回移实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不整体升级 vendored vLLM 的前提下，将 vLLM v0.27.0 的 NVFP4 Dense Linear 与 FusedMoE emulation 能力局部回移到当前仓库，使 A800 能直接加载 ModelOpt / compressed-tensors 原生 NVFP4 checkpoint，并显式运行 W4A4 emulation；KV cache 可在 checkpoint 配置与 BF16 之间切换。

**Architecture:** 以 vLLM tag `v0.27.0` 为唯一功能/数值基线。Dense 侧回移新版 `linear_backend="emulation"` kernel abstraction；MoE 侧回移 `Nvfp4QuantizationEmulationTritonExperts` 与 `moe_backend="emulation"`。ModelOpt 和 compressed-tensors 保持各自 loader，仅把两种 checkpoint 归一到同一 upstream backend。packed NVFP4 weight 常驻，禁止全量 BF16 权重常驻。项目根 `main.py` 只负责透传 backend 与 KV dtype，不承载量化数学实现。

**Tech Stack:** Python, PyTorch, Triton, vendored vLLM, CUDA/A800(SM80), pytest；所有 Python/test/inference 命令在 `hif4` conda 环境执行。

**Spec:** `NVFP4/plans/2026-08-21-vllm-v027-nvfp4-emulation-backport-design-cn.md`

## Global Constraints

1. **Upstream 唯一基线：vLLM `v0.27.0`。** PR `#44667` / merge commit `552a9db` 仅用于历史定位；如果 PR 内容与 v0.27.0 最终代码不同，以 v0.27.0 为准。
2. 不整体替换 `3rdparty/vllm`，不升级整个依赖栈，不从 `main` 随机摘代码。
3. 不修改 upstream NVFP4 数值规则。尤其 MoE emulation 中 `a13_scale/a2_scale` 的 `max()` 聚合、倒数、W13 scale 合并等全部保持 v0.27.0 语义。
4. 不实现自定义 per-expert activation scale 修正版。
5. 不将原始 packed NVFP4 weight 重新量化，不把全模型/全 expert 权重展开成 BF16 常驻。
6. 不改变 vLLM 默认 `auto` backend 行为；正式实验显式指定 `linear_backend=emulation`、`moe_backend=emulation`。
7. 对目标 NVFP4 层，显式 emulation 请求失败时必须报错；正式验收不允许静默使用 Marlin W4A16。
8. router、top-k、expert combine、非 NVFP4/excluded module 保持 checkpoint/current-vLLM 原行为。
9. ModelOpt 与 compressed-tensors 必须共用 backend 实现，不复制两份 emulation 数学代码。
10. 现有 `NVFP4/torch_fake.py`、`NVFP4/triton_fake.py`、`--fake_act_quant nvfp4` 不删除、不重定义；本计划的新 native-checkpoint emulation 与旧 sidecar fake-activation 路径相互独立。
11. 当前工作区已有大量无关修改。每一步只编辑本计划明确列出的文件；禁止格式化整个目录、批量重写文件或顺手修无关问题。
12. 不创建新 branch/worktree。若执行 git commit，只能 `git add` 本任务明确文件，绝不能 `git add .`。
13. 每个任务遵循：先写/回移测试并确认预期失败，再做最小实现，再跑定向测试；出现失败先定位原因，禁止 fallback patch。

---

## Task 0：冻结当前基线并建立 upstream v0.27.0 差异清单

**Files:**
- Create: `NVFP4/reports/vllm_v027_nvfp4_backport/upstream_inventory.md`
- Read only: `3rdparty/vllm/vllm/config/kernel.py`
- Read only: `3rdparty/vllm/vllm/model_executor/layers/quantization/modelopt.py`
- Read only: `3rdparty/vllm/vllm/model_executor/layers/quantization/utils/nvfp4_utils.py`
- Read only: `3rdparty/vllm/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py`
- Read only: `3rdparty/vllm/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py`
- Read only: `3rdparty/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py`
- Read only: `3rdparty/vllm/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`
- Read only: `3rdparty/vllm/vllm/model_executor/layers/fused_moe/utils.py`
- Read only: `main.py`

- [ ] **Step 1: 固定 upstream source**

只读获取 vLLM `v0.27.0` 源码到 workspace 外的临时目录，例如 `/tmp/vllm-v0.27.0-reference`。不得把 upstream checkout 放进仓库，不得切换当前仓库分支。

记录：

- tag：`v0.27.0`
- 解析到的 tag commit SHA
- 获取日期
- PR `#44667` merge commit `552a9db` 仅作为历史备注

- [ ] **Step 2: 做逐文件差异分类**

在 `upstream_inventory.md` 中建立三类：

A. 当前缺失、必须新增；
B. 当前已有旧实现、必须按 v0.27.0 局部升级；
C. v0.27.0 有变化但与 NVFP4 emulation 无直接依赖，本次明确不回移。

至少核对这些 upstream 路径是否存在/变化：

- `vllm/config/kernel.py`
- `vllm/engine/arg_utils.py`
- `vllm/entrypoints/llm.py`（仅检查 Python `LLM` 参数传递所需变化）
- `vllm/model_executor/kernels/linear/__init__.py`
- `vllm/model_executor/kernels/linear/nvfp4/` 全部文件
- `vllm/model_executor/layers/quantization/modelopt.py`
- `vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py`
- `vllm/model_executor/layers/quantization/utils/nvfp4_utils.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_nvfp4.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py`
- `vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py`
- `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`
- `vllm/model_executor/layers/fused_moe/utils.py`
- `vllm/model_executor/layers/fused_moe/config.py`
- upstream NVFP4 emulation tests。

- [ ] **Step 3: 明确依赖闭包白名单**

`upstream_inventory.md` 必须列出最终允许修改/新增文件。后续如果发现一个 import/symbol 依赖不在白名单中，只能先把“缺失 symbol、调用链、为什么属于 NVFP4 emulation 必需依赖”补到 inventory，再增加单个必要文件；禁止以“版本差太大”为理由扩大到整个 vLLM 子系统。

- [ ] **Step 4: 基线 import/test**

运行：

```bash
conda run -n hif4 python -c "import sys; sys.path.insert(0, '3rdparty/vllm'); import vllm; print(vllm.__version__)"
conda run -n hif4 pytest -q 3rdparty/vllm/tests/kernels/quantization/test_nvfp4_quant.py
```

把当前结果记录到 inventory。若存在 pre-existing failure，只记录，不在 Task 0 修。

- [ ] **Step 5: Commit only report**

```bash
git add NVFP4/reports/vllm_v027_nvfp4_backport/upstream_inventory.md
git commit -m "docs: inventory vllm nvfp4 emulation backport"
```

如果仓库策略禁止 commit，跳过 commit，但后续所有 `git add` 仍须保持显式文件列表。

---

## Task 1：先建立 backend 配置接口测试

**Files:**
- Modify: `3rdparty/vllm/vllm/config/kernel.py`
- Modify only if v0.27.0 requires it: `3rdparty/vllm/vllm/engine/arg_utils.py`
- Modify only if v0.27.0 requires it: `3rdparty/vllm/vllm/entrypoints/llm.py`
- Create/Modify: `3rdparty/vllm/tests/config/test_kernel_config.py`
- Modify: `3rdparty/vllm/tests/engine/test_arg_utils.py`

- [ ] **Step 1: 写失败测试**

测试必须固定以下行为：

```python
KernelConfig(linear_backend="emulation").linear_backend == "emulation"
KernelConfig(moe_backend="emulation").moe_backend == "emulation"
```

并验证大小写/连字符 normalization 仅按 upstream v0.27.0 实现，不新增自定义别名。

如果当前仓库没有 `test_kernel_config.py`，创建该文件，只覆盖本任务的 backend 字段，不搬运无关 v0.27.0 config tests。

- [ ] **Step 2: 运行并确认失败原因正确**

```bash
conda run -n hif4 pytest -q 3rdparty/vllm/tests/config/test_kernel_config.py 3rdparty/vllm/tests/engine/test_arg_utils.py -k "backend or emulation"
```

Expected：当前 `LinearBackend` 不存在或 `emulation` 不被接受；MoE `emulation` 不被接受。

- [ ] **Step 3: 最小回移 KernelConfig**

从 v0.27.0 回移：

- `LinearBackend` 类型及 `linear_backend` 字段；
- `MoEBackend` 中 `emulation`；
- 对应 validator；
- 仅为这两个字段所需的 CLI/Python LLM 参数 plumbing。

不要回移 v0.27.0 中与本任务无关的 IR priority、其它 backend、新 warmup feature 等，除非是上述两个字段的硬依赖。

- [ ] **Step 4: 跑测试**

```bash
conda run -n hif4 pytest -q 3rdparty/vllm/tests/config/test_kernel_config.py 3rdparty/vllm/tests/engine/test_arg_utils.py -k "backend or emulation"
```

Expected：PASS。

- [ ] **Step 5: Commit**

只 stage 本任务文件。

---

## Task 2：升级共享 NVFP4 emulation 数值原语

**Files:**
- Modify: `3rdparty/vllm/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py`
- Create/Modify: `3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py`
- Reuse: `3rdparty/vllm/tests/kernels/quantization/nvfp4_utils.py`

- [ ] **Step 1: 从 v0.27.0 回移/适配 reference tests**

测试覆盖至少：

- E2M1 unpack table；
- group=16；
- zero block；
- 正负值；
- E2M1 分界/rounding 边界；
- E4M3 block-scale 极值/普通值；
- BF16 input；
- FP16 input（若 v0.27.0 支持）；
- `run_nvfp4_emulations` 对 packed weight + scale + global scale 的结果。

Golden 只使用 v0.27.0 upstream test/reference 实现，不引用项目自己的 HiF4/NVFP4 转换实验代码作为 oracle。

- [ ] **Step 2: 先运行测试**

```bash
conda run -n hif4 pytest -q 3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py
```

记录当前旧实现与 v0.27.0 test 的差异。

- [ ] **Step 3: 将 `nvfp4_emulation_utils.py` 对齐 v0.27.0**

规则：

- 保持 API/数学逻辑与 v0.27.0 一致；
- 如果当前仓库已有同名函数但接口较旧，优先做最小兼容 wrapper，核心计算只保留一份；
- E2M1 lookup tensor 的 device placement 必须按 v0.27.0；
- 不增加本项目自定义 rounding 或 scale 分支。

- [ ] **Step 4: 定向测试通过**

再次运行同一 pytest，Expected：PASS。

- [ ] **Step 5: 回归当前 NVFP4 quant test**

```bash
conda run -n hif4 pytest -q 3rdparty/vllm/tests/kernels/quantization/test_nvfp4_quant.py 3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py
```

Expected：PASS；若旧 test 断言旧 API，按 v0.27.0 API 改 test，不改变数值逻辑来迁就旧断言。

---

## Task 3：回移新版 Dense NVFP4 linear kernel backend

**Files:**
- Create: `3rdparty/vllm/vllm/model_executor/kernels/linear/nvfp4/`（文件集合严格以 v0.27.0 inventory 为准）
- Modify: `3rdparty/vllm/vllm/model_executor/kernels/linear/__init__.py`
- Modify: `3rdparty/vllm/vllm/model_executor/layers/quantization/modelopt.py`
- Modify: `3rdparty/vllm/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py`
- Modify if v0.27.0 uses same kernel path: `3rdparty/vllm/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_nvfp4.py`
- Modify/retire only as needed: `3rdparty/vllm/vllm/model_executor/layers/quantization/utils/nvfp4_utils.py`
- Extend: `3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py`

- [ ] **Step 1: 写 Dense backend selection + numerical failure tests**

至少构造两条 scheme：

1. ModelOpt-style layer metadata；
2. compressed-tensors W4A4 NVFP4-style layer metadata。

验证显式 `linear_backend="emulation"` 时：

- 选择新版 `EmulationNvFp4LinearKernel`（类名以 v0.27.0 实际名称为准）；
- packed `uint8` weight 在 `process_weights_after_loading` 后仍是 packed representation，不出现完整 BF16 weight Parameter；
- output 对 upstream reference `run_nvfp4_emulations` 在 v0.27.0 test tolerance 内一致；
- bias 路径一致。

- [ ] **Step 2: 运行并确认当前失败**

当前仓库没有 `kernels/linear/nvfp4/`，显式 `linear_backend` 也尚未接入，因此 selection test 应失败。

- [ ] **Step 3: 回移 v0.27.0 NVFP4 linear kernel 包**

只回移 inventory 中 NVFP4 linear package 需要的文件。对 package 内 native CUTLASS/FlashInfer/Marlin 类：

- 若是 emulation registry 的硬依赖，则按 v0.27.0 回移；
- 若当前仓库旧 native backend 已可独立工作且 v0.27.0 emulation 不依赖其新实现，不为了“版本统一”扩大修改。

最终必须形成 upstream 风格的 kernel selection，而不是继续用环境变量强制 `NvFp4LinearBackend.EMULATION` 作为正式入口。

- [ ] **Step 4: 适配 ModelOpt Dense**

`ModelOptNvFp4LinearMethod`：

- loader tensor 语义不变；
- checkpoint packed weight/weight_scale/weight_scale_2/input_scale 仍由原 loader 创建；
- process-after-load 后交给新版 NVFP4 linear kernel；
- `linear_backend=emulation` 走 emulation kernel；
- 不复制 `run_nvfp4_emulations` 到 ModelOpt 文件。

- [ ] **Step 5: 适配 compressed-tensors Dense**

当前 `CompressedTensorsW4A4Fp4` 已创建 `weight_packed`、`weight_global_scale`、`weight_scale`、`input_global_scale`。按 v0.27.0 改成新版 kernel 接口；保留 CT scale reciprocal 语义，禁止重新估计 scale。

- [ ] **Step 6: 数值测试**

```bash
conda run -n hif4 pytest -q 3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py -k "linear or dense"
```

Expected：ModelOpt/CT 两种模拟 layer 均 PASS。

- [ ] **Step 7: backend 强制性测试**

增加断言：显式 `linear_backend="emulation"` 对受支持 NVFP4 layer 不得返回 Marlin。不要通过日志字符串唯一判断；优先断言实例 kernel/backend 类型。

---

## Task 4：回移 NVFP4 MoE Emulation TritonExperts

**Files:**
- Create: `3rdparty/vllm/vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py`
- Modify if required by v0.27.0 interface: `3rdparty/vllm/vllm/model_executor/layers/fused_moe/experts/__init__.py`
- Modify: `3rdparty/vllm/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`
- Modify if required: `3rdparty/vllm/vllm/model_executor/layers/fused_moe/config.py`
- Modify only required symbols: `3rdparty/vllm/vllm/model_executor/layers/fused_moe/utils.py`
- Extend: `3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py`

- [ ] **Step 1: 回移 upstream MoE correctness test，先不实现**

从 v0.27.0 的 NVFP4 emulation test 中提取 MoE correctness 部分，适配当前测试 fixture/import，而不是改数学期望。

必须覆盖：

- gated W13 + W2；
- top-k routing；
- 1/多 token；
- 至少 2 个 expert；
- packed uint8 W13/W2；
- FP8 E4M3 block scales；
- weight global scales；
- activation global scales；
- non-uniform scale case，以确认实现确实遵循 upstream `max()` 聚合结果；
- router weight combine；
- BF16 output；
- 无 NaN/Inf。

- [ ] **Step 2: 确认失败**

```bash
conda run -n hif4 pytest -q 3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py -k "moe"
```

Expected：当前没有 emulation MoE expert/backend。

- [ ] **Step 3: 原样回移 `Nvfp4QuantizationEmulationTritonExperts` 核心**

核心要求：

- packed NVFP4 W13/W2 weight 在 Triton GEMM 中按 v0.27.0 解码；
- 不在初始化阶段创建完整 `[E,...]` BF16 weight Parameter；
- 只使用 upstream v0.27.0 的 E2M1 LUT / scale 规则；
- activation QDQ 位置和次数与 v0.27.0 一致；
- 不增加项目私有 scale 变体。

- [ ] **Step 4: 防止 double-QDQ**

这是硬门禁。检查 v0.27.0 与当前 `moe_kernel_quantize_input()`、`FusedMoEExperts.apply()` 的接口差异，确保 W13 input 和 W2 input 各发生 upstream 规定的一次 QDQ，不因当前旧 modular kernel 父类再量化一次。

测试需构造一个能区分“一次 QDQ”和“两次 QDQ”的输入，并断言 emulation output 与单次-QDQ upstream reference 相符。

- [ ] **Step 5: test pass**

```bash
conda run -n hif4 pytest -q 3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py -k "moe"
```

Expected：PASS。

---

## Task 5：接入 `NvFp4MoeBackend.EMULATION`

**Files:**
- Modify: `3rdparty/vllm/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`
- Modify: `3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py`
- Modify/add a focused oracle test under `3rdparty/vllm/tests/model_executor/` only if existing test layout requires it。

- [ ] **Step 1: 写 backend mapping test**

固定：

```text
moe_backend="emulation" -> NvFp4MoeBackend.EMULATION -> Nvfp4QuantizationEmulationTritonExperts
```

并验证 unsupported quant key 不会误选 emulation。

- [ ] **Step 2: 回移 v0.27.0 oracle 分支**

至少包括：

- Enum `EMULATION`；
- `backend_to_kernel_cls()` 分支；
- `map_nvfp4_backend()` 的 `"emulation"`；
- explicit selection path；
- `convert_to_nvfp4_moe_kernel_format()` 的 emulation 分支；
- `make_nvfp4_moe_quant_config()` 的 emulation 分支。

**重要：** emulation 分支的 `a13_scale = 1.0 / a13_scale.max()`、`a2_scale = 1.0 / a2_scale.max()` 等逻辑必须按 v0.27.0 原样回移，不“优化”。

- [ ] **Step 3: 不改变 auto 选择策略**

本任务目标是让 explicit `moe_backend="emulation"` 可用。不要因为 A800 无 native FP4 就自动改变全局默认。若 v0.27.0 的 auto list 包含 emulation，可保留 upstream selection 顺序，但本项目正式实验仍必须显式指定 emulation。

- [ ] **Step 4: 测试 explicit selection**

运行相关测试并断言 backend 类型，不接受“日志看起来像 emulation”作为唯一证据。

---

## Task 6：让 ModelOpt 与 compressed-tensors MoE 共用新 backend

**Files:**
- Modify: `3rdparty/vllm/vllm/model_executor/layers/quantization/modelopt.py`
- Modify: `3rdparty/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py`
- Extend: `3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py`

- [ ] **Step 1: 写两种 loader contract test**

ModelOpt 和 CT 分别创建最小 FusedMoE layer，验证：

- W13/W2 packed weight shape/dtype 正确；
- group size 16；
- block scale E4M3；
- global weight/input scale shape 正确；
- `process_weights_after_loading()` 后通过同一个 `NvFp4MoeBackend.EMULATION`；
- 两种格式最终交给 emulation expert 的 logical quant config 一致。

- [ ] **Step 2: 按 v0.27.0 适配 ModelOpt**

只回移 emulation 所需改动。不得改变 ModelOpt checkpoint key 定义或为 Qwen3 加特殊 key alias。

- [ ] **Step 3: 按 v0.27.0 适配 compressed-tensors**

保留 CT 自身 param 命名/reciprocal semantics，在 kernel-format conversion 边界转换。不要把 CT 转成 ModelOpt checkpoint，也不要反过来。

- [ ] **Step 4: test**

```bash
conda run -n hif4 pytest -q 3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py -k "modelopt or compressed or moe"
```

Expected：PASS。

---

## Task 7：项目根 `main.py` 暴露 upstream backend 与 KV dtype

**Files:**
- Modify: `main.py`
- Modify: `3rdparty/lighteval/src/lighteval/models/vllm/vllm_model.py` **仅当当前 wrapper 未透传 Python LLM kwargs 且实际调用链需要**
- Modify: `3rdparty/lighteval/src/lighteval/models/transformers/transformers_model.py` **禁止修改，除非证明本任务调用链实际经过该文件；正常情况下不应需要**
- Create: `NVFP4/tests/test_vllm_emulation_cli_plumbing.py`

- [ ] **Step 1: 写 CLI plumbing test**

新增根参数：

```text
--linear_backend
--moe_backend
--kv_cache_dtype
```

命名保持当前项目 Python CLI 的下划线风格；传给 vLLM 时分别映射为：

```python
linear_backend=args.linear_backend
moe_backend=args.moe_backend
kv_cache_dtype=args.kv_cache_dtype
```

默认值：

- `linear_backend="auto"`
- `moe_backend="auto"`
- `kv_cache_dtype="auto"`

`kv_cache_dtype` 至少允许 `auto`、`bfloat16`、`fp8`、`fp8_e4m3`、`fp8_e5m2`；不要在项目层复制整个未来 vLLM choice 列表，如果当前 LLM API 本身做 validation，则项目 parser 只做字符串透传也可以，但 test 必须覆盖 `auto` 与 `bfloat16`。

- [ ] **Step 2: 显式禁止与旧 fake activation 混用**

当 `--fake_act_quant nvfp4` 与 `--linear_backend emulation` 同时请求时直接报 `ValueError`，说明一个是 sidecar/unquantized fake path，一个是 native packed checkpoint emulation path。不要静默选择其一。

已有其它 `fake_act_quant`（如 HiF4）若与 quantized native checkpoint 本来就不兼容，保持现有行为；本任务不扩展组合空间。

- [ ] **Step 3: KV 两个实验模式**

不新增 fake KV 算法。项目脚本通过：

```text
--kv_cache_dtype auto       # checkpoint/model config 模式
--kv_cache_dtype bfloat16   # 关闭 FP8 KV，单独测 W4A4
```

实现只透传。

- [ ] **Step 4: test parser/LLM kwargs**

`NVFP4/tests/test_vllm_emulation_cli_plumbing.py` mock `vllm.LLM`，断言三个参数被准确透传；不得启动真实模型。

运行：

```bash
conda run -n hif4 pytest -q NVFP4/tests/test_vllm_emulation_cli_plumbing.py
```

---

## Task 8：增加 A800 backend 身份与“禁止 Marlin 混入”门禁

**Files:**
- Create: `NVFP4/tests/test_vllm_nvfp4_backend_contract.py`
- Optionally modify relevant vLLM tests from Tasks 3–6，禁止新增生产 fallback 代码。

- [ ] **Step 1: config-level contract**

在模拟 CUDA capability 8.0 / explicit backend 的情况下，验证配置允许：

```text
linear_backend=emulation
moe_backend=emulation
```

- [ ] **Step 2: layer-level contract**

对最小 ModelOpt 与 CT NVFP4 dense/MoE layer，断言：

- dense kernel class 是 emulation；
- MoE backend enum 是 EMULATION；
- expert class 是 `Nvfp4QuantizationEmulationTritonExperts`；
- 目标 layer 不使用 Marlin。

- [ ] **Step 3: packed residency contract**

测试不得简单比较总显存。直接检查 layer parameters：

- packed weight 保持 uint8/上游 kernel-format packed representation；
- 不新增与完整 logical W13/W2/linear weight 同 shape 的 persistent BF16/FP16 Parameter。

Triton 临时 tile/activation workspace 不属于违规。

---

## Task 9：真实 checkpoint preflight（不做完整 benchmark）

**Files:**
- Create: `NVFP4/scripts/inspect_native_nvfp4_checkpoint.py`
- Create: `NVFP4/reports/vllm_v027_nvfp4_backport/qwen3_30b_preflight.md`

**Target checkpoint:**

`/home/shaoyuantian/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3/`

- [ ] **Step 1: 只读 metadata 检查**

脚本只读 `config.json` / quant config / safetensors index/header，不加载全部 tensor 到 CPU/GPU。输出：

- quantization method；
- quant algo；
- group size；
- KV quant algo；
- exclude modules；
- 至少一个 dense NVFP4 tensor key/shape/dtype；
- 至少一个 expert W13/W2 packed tensor key/shape/dtype；
- 对应 block/global/input scale key/shape/dtype。

禁止 hardcode “第 N 层一定存在”；通过 index 中真实匹配的第一组完整 NVFP4 layer 作为报告样例。

- [ ] **Step 2: gate**

若目标 checkpoint metadata 与计划假设不一致，停止真实 smoke，先更新报告和适配 loader；禁止在 smoke 中猜 key。

---

## Task 10：A800 单层/小 MoE 数值 puncture

**Files:**
- Create: `NVFP4/scripts/run_nvfp4_emulation_puncture.py`
- Create: `NVFP4/reports/vllm_v027_nvfp4_backport/qwen3_30b_puncture.md`

- [ ] **Step 1: Dense puncture**

从真实 checkpoint 选 preflight 找到的一个 NVFP4 dense layer：

- 读取其 packed weight/scales；
- 构造固定 seed BF16 activation；
- upstream-style Python/reference emulation 计算 output；
- 新 Dense emulation backend 计算 output；
- 报告 max abs、mean abs、relative L2/NMSE（至少三项）；
- tolerance 使用 v0.27.0 对应 upstream test tolerance，不自行放宽到“看起来差不多”。

- [ ] **Step 2: MoE puncture**

从真实 checkpoint 选一个完整 FusedMoE layer 或抽取若干 expert tensor，构造固定 top-k routing，比较 upstream reference 与 Triton emulation。

必须覆盖 W13 -> activation -> W2 的完整 expert 数学流程，而不是只测 weight dequant。

- [ ] **Step 3: 判定**

只有 Dense 与 MoE 均通过 upstream tolerance，才进入 E2E smoke。

---

## Task 11：A800 真实 vLLM smoke，分别验证 BF16 KV 与 checkpoint KV

**Files:**
- Create: `NVFP4/scripts/run_qwen3_30b_nvfp4_emulation_smoke.sh`
- Create/Update: `NVFP4/reports/vllm_v027_nvfp4_backport/qwen3_30b_smoke.md`

- [ ] **Step 1: 固定 smoke 参数**

模型路径使用 Task 9 固定 snapshot。第一轮使用：

```text
TP=1
linear_backend=emulation
moe_backend=emulation
kv_cache_dtype=bfloat16
enforce_eager=true
固定 2~4 个短 prompt
固定 max_new_tokens
```

`enforce_eager=true` 用于首轮排除 CUDA graph 干扰；通过后再补一次默认 graph-mode smoke（若当前项目默认支持）。

- [ ] **Step 2: BF16 KV smoke**

运行根 `main.py`/当前 lighteval-vLLM 初始化链路，要求：

- model load 成功；
- prefill + decode 成功；
- 输出无 NaN/Inf；
- 日志可确认 `linear_backend=emulation`、`moe_backend=emulation`；
- 不出现 NVFP4 target layer 使用 Marlin W4A16 的 warning/selection。

如果 TP=1 真实 OOM：必须记录峰值/报错证据，然后选择**最小能运行的 TP**；不得因为速度慢就主动改 TP。

- [ ] **Step 3: checkpoint KV smoke**

仅将：

```text
kv_cache_dtype=auto
```

其余参数完全不变。确认该 checkpoint 实际解析为 FP8 KV；若解析结果不是 FP8，直接判定 checkpoint-KV plumbing 未完成，不得手动硬改成 fp8 后宣称“auto 已支持”。

- [ ] **Step 4: graph-mode smoke**

若 eager 两组均通过，再用同样两个 KV 模式各做一次短 graph-mode initialization/decode。若 emulation backend upstream 明确不支持某种 compile/graph 模式，报告该 upstream 限制，不自行 fallback 到 Marlin。

---

## Task 12：端到端精度实验入口与结果记录

**Files:**
- Create: `NVFP4/scripts/run_qwen3_30b_nvfp4_emulation_mmlu_pro.sh`
- Create: `NVFP4/reports/vllm_v027_nvfp4_backport/qwen3_30b_accuracy.md`

- [ ] **Step 1: 固定两组唯一变量**

两组都必须：

```text
linear_backend=emulation
moe_backend=emulation
```

只改变：

A. `kv_cache_dtype=bfloat16`
B. `kv_cache_dtype=auto`

其它 tokenizer、prompt、sampling、seed、max length、batch 参数完全一致。

- [ ] **Step 2: MMLU-Pro 300**

按仓库约定使用根 `main.py` + lighteval，`max_samples=300`。本实验用于观察结果，不设置拍脑袋的 accuracy pass threshold。

- [ ] **Step 3: 报告**

至少记录：

- 模型 snapshot；
- GPU 型号/数量；
- TP；
- vLLM 当前 vendored revision；
- upstream backport 基线 `v0.27.0` 与解析到的 tag SHA；
- linear/moe backend；
- KV dtype 实际解析值；
- MMLU-Pro 300 accuracy；
- BF16-KV vs checkpoint-KV 的绝对差值；
- 是否出现 backend fallback warning。

不要将这 300 条结果包装成完整 benchmark 结论。

---

## Task 13：回归测试、代码清理与最终验收

**Files:**
- All files in Task 0 inventory whitelist
- `NVFP4/plans/2026-08-21-vllm-v027-nvfp4-emulation-backport-design-cn.md`
- `NVFP4/plans/2026-08-21-vllm-v027-nvfp4-emulation-backport-plan-cn.md`
- `NVFP4/reports/vllm_v027_nvfp4_backport/*`

- [ ] **Step 1: 定向完整测试**

至少运行：

```bash
conda run -n hif4 pytest -q 3rdparty/vllm/tests/kernels/quantization/test_nvfp4_quant.py
conda run -n hif4 pytest -q 3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py
conda run -n hif4 pytest -q NVFP4/tests/test_vllm_emulation_cli_plumbing.py
conda run -n hif4 pytest -q NVFP4/tests/test_vllm_nvfp4_backend_contract.py
```

再运行 Task 0 inventory 中识别出的、因公共接口改动可能受影响的原有 vLLM quantization/MoE config tests。

- [ ] **Step 2: source scan**

检查本任务新增/修改代码：

- 无 `TODO`/`pass`/placeholder；
- 无 Qwen3 专用 layer-name hack；
- 无 `except Exception: fallback_to_marlin`；
- 无全量 BF16 expert weight cache；
- 无复制出的第二份 NVFP4 数学实现；
- 无修改 upstream scale 算法的注释/代码；
- 无 `git add .` 产生的无关文件。

- [ ] **Step 3: 行为矩阵最终确认**

必须在最终报告给出：

| Checkpoint | Dense | MoE | KV=bfloat16 | KV=auto |
|---|---|---|---|---|
| ModelOpt NVFP4 | emulation | emulation | supported | checkpoint-driven |
| compressed-tensors NVFP4 | emulation | emulation | supported | checkpoint-driven |

若某一格未验证，写“未验证”，不能写“支持”。

- [ ] **Step 4: final diff review**

只检查本计划白名单内 diff。任何与 NVFP4 emulation 无关的 vLLM 大范围改动都应撤回。

- [ ] **Step 5: 最终 commit**

只 stage 本任务文件，commit message 示例：

```text
feat: backport vllm nvfp4 emulation backends
```

---

# 完成标准

只有同时满足以下条件才算完成：

1. 当前仓库能显式接受 `linear_backend=emulation` 和 `moe_backend=emulation`；
2. Dense NVFP4 使用 v0.27.0-style emulation kernel，而不是旧环境变量作为正式入口；
3. MoE NVFP4 使用 `Nvfp4QuantizationEmulationTritonExperts`；
4. ModelOpt 与 compressed-tensors 均通过同一 backend；
5. packed NVFP4 权重常驻，没有全量 BF16 weight materialization；
6. upstream v0.27.0 scale/QDQ 数学逻辑未被修改；
7. MoE 不存在 double-QDQ；
8. A800 上目标 checkpoint 在 BF16 KV 和 checkpoint/FP8 KV 两种模式都能完成 smoke；
9. 正式 smoke 中 NVFP4 Dense/MoE 均无 Marlin W4A16 fallback；
10. 真实 layer puncture 与 v0.27.0 reference 在 upstream tolerance 内通过；
11. MMLU-Pro 300 两组实验能闭环记录；
12. 没有破坏现有 `--fake_act_quant nvfp4` 独立实验路径；
13. 最终 diff 仅包含本计划允许的 NVFP4 emulation/backport/测试/脚本/报告相关改动。
