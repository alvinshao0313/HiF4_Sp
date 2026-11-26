# vLLM NVFP4 Emulation torch.compile / CUDA Graph Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变 NVFP4 数学语义、packed 权重驻留和 eager 数值结果的前提下，使当前 A800 NVFP4 Dense/MoE emulation 能在 `enforce_eager=False` 下通过 vLLM `torch.compile` 与 CUDA Graph capture，并量化实际性能收益。

**Architecture:** 第一层只修当前明确的根因：将 SM80/SM89+ capability 判定从 forward 图内移到 Dense kernel / MoE experts 初始化阶段，并把结果作为固定 bool 显式传入 emulation primitive。真实 graph smoke 若仍在 NVFP4 helper 内 graph break，则按固定升级规则使用 vLLM `direct_register_custom_op` 隔离 QDQ/weight-dequant；若之后只剩 MoE direct Triton launch graph break，再单独封装 MoE GEMM custom op。禁止重写 NVFP4 数学或静默 fallback 到 Marlin。

**Tech Stack:** Python 3.11、PyTorch/TorchDynamo、vLLM V1、CUDA Graph、Triton、ModelOpt NVFP4、A800 SM80、pytest、lighteval。

**Spec:** `NVFP4/plans/2026-08-21-vllm-nvfp4-emulation-cudagraph-design-cn.md`

## Global Constraints

- 目标 GPU 固定以 NVIDIA A800 80GB（SM80）完成正式验收。
- 所有 Python/test/inference 命令在 `hif4` conda 环境运行。
- 不修改 NVFP4 group size=16、E2M1 payload、E4M3FN block scale、global scale、MoE `a13/a2 max()` 等数学语义。
- 不修改当前 ModelOpt / compressed-tensors loader 的 scale 解释。
- 不把 packed NVFP4 权重全量反量化后常驻 BF16/FP16。
- 不允许 `linear_backend=emulation` 或 `moe_backend=emulation` 静默 fallback 到 Marlin W4A16。
- 不整体升级 `3rdparty/vllm`。
- 不修改 Qwen3 专用模型 forward；所有修复放在通用 NVFP4 emulation / vLLM compile 边界。
- `--enforce_eager` 必须保留，继续作为 correctness reference/debug 模式。
- graph 支持的 correctness 优先级高于性能；任何性能优化都不得改变 eager 数值门禁。
- 当前工作区可能存在其它实验生成文件；只 stage 本任务白名单文件，不使用 `git add .`。

---

## File Structure / 修改白名单

### 预计必须修改

- `3rdparty/vllm/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py`
  - 将 runtime GPU capability 查询改成显式参数驱动；保留现有数学实现。
- `3rdparty/vllm/vllm/model_executor/kernels/linear/nvfp4/emulation.py`
  - Dense kernel 初始化时冻结 capability；forward 只消费固定 bool。
- `3rdparty/vllm/vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py`
  - MoE experts 初始化时冻结 capability；两次 activation QDQ 和 block-scale 准备不再 runtime 查询设备。
- `3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py`
  - 新增 capability-free forward、compile/custom-op 数值测试。
- `NVFP4/reports/vllm_nvfp4_emulation_cudagraph/*`
  - 本任务独立报告目录。

### 仅当 Task 5 的实际 graph break 命中 Level 1 条件时修改

- `3rdparty/vllm/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py`
  - 注册 `vllm::nvfp4_emulation_qdq` 与 `vllm::nvfp4_emulation_dequant_weight`。
- `3rdparty/vllm/vllm/model_executor/kernels/linear/nvfp4/emulation.py`
  - graph 路径调用上述 op。
- `3rdparty/vllm/vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py`
  - 两次 QDQ 复用同一 op。

### 仅当 Task 7 的实际 graph break 命中 Level 2 条件时修改

- `3rdparty/vllm/vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py`
  - 注册 `vllm::nvfp4_emulation_moe_gemm`，只封装单次 expert GEMM。

### 新增验证脚本

- `NVFP4/scripts/benchmark_qwen3_30b_nvfp4_emulation_graph.py`
  - eager vs graph steady-state benchmark。
- `NVFP4/tests/test_nvfp4_emulation_graph_contract.py`
  - 本项目层面的 graph contract / CLI 行为门禁。

### 复用但不修改的现有验证工具

- `NVFP4/scripts/run_qwen3_30b_nvfp4_emulation_smoke.py`
- `NVFP4/scripts/run_qwen3_30b_nvfp4_emulation_smoke.sh`
- `NVFP4/scripts/run_nvfp4_emulation_puncture.py`

### 明确禁止修改

- `3rdparty/vllm/vllm/model_executor/models/qwen3_moe.py`
- `NVFP4/torch_fake.py`
- `NVFP4/triton_fake.py`
- 当前 NVFP4 scale 生成/搜索实验代码
- 其它量化格式的 kernel
- vLLM scheduler/router/attention 算法主体；若真实 stack 的第一处错误落在这些区域，立即停止并记录，必须重新审计划后才能扩范围。

---

### Task 0: 冻结 baseline、复现 graph break、建立独立报告目录

**Files:**
- Read: `NVFP4/reports/vllm_v027_nvfp4_backport/qwen3_30b_smoke.md`
- Read: `NVFP4/reports/vllm_v027_nvfp4_backport/smoke_logs/graph_bf16_tp1.log`
- Create: `NVFP4/reports/vllm_nvfp4_emulation_cudagraph/baseline.md`

**Interfaces:**
- Consumes: 当前真实 checkpoint、现有 smoke script。
- Produces: 本任务唯一 baseline stack、GPU、commit、eager 数值/graph failure 证据。

- [ ] **Step 1: 记录代码与环境 baseline**

在 `baseline.md` 固定写入：

```text
checkpoint=/home/shaoyuantian/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3
GPU=NVIDIA A800 80GB PCIe
SM=80
TP=1
linear_backend=emulation
moe_backend=emulation
kv_cache_dtype=bfloat16
```

并记录当前 git HEAD / vendored vLLM version，但不要修改版本文件。

- [ ] **Step 2: 重新跑 eager smoke，确认 baseline 仍可用**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n hif4 python NVFP4/scripts/run_qwen3_30b_nvfp4_emulation_smoke.py \
  --kv-cache-dtype bfloat16 \
  --tp 1 \
  --enforce-eager \
  --tag cudagraph_baseline_eager_bf16 \
  --out-json NVFP4/reports/vllm_nvfp4_emulation_cudagraph/baseline_eager_bf16.json
```

Expected:

```text
PASS
linear=emulation
moe=emulation
no_marlin=True
```

如果 eager baseline 失败，停止本计划；先修复/定位已有 regression，不能继续 graph 修改。

- [ ] **Step 3: 重新复现 graph failure**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n hif4 python NVFP4/scripts/run_qwen3_30b_nvfp4_emulation_smoke.py \
  --kv-cache-dtype bfloat16 \
  --tp 1 \
  --tag cudagraph_baseline_graph_bf16 \
  --out-json NVFP4/reports/vllm_nvfp4_emulation_cudagraph/baseline_graph_bf16.json
```

Expected baseline failure stack 必须包含：

```text
EmulationNvFp4LinearKernel.apply_weights
run_nvfp4_emulations
ref_nvfp4_quant_dequant
_nvfp4_triton_fp8e4nv_supported
current_platform.has_device_capability
pynvml.nvmlInit
```

若第一处错误不是这条链，停止并把新 stack 写入 `baseline.md`，不要按后续 Task 猜修。

- [ ] **Step 4: Commit baseline 文档（不提交大日志）**

```bash
git add NVFP4/reports/vllm_nvfp4_emulation_cudagraph/baseline.md
git commit -m "docs: record nvfp4 graph baseline"
```

---

### Task 1: 为“forward 不得查询 GPU capability”写失败测试

**Files:**
- Modify: `3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py`

**Interfaces:**
- Consumes: `_nvfp4_triton_fp8e4nv_supported`, `EmulationNvFp4LinearKernel`。
- Produces: 两个 red tests，锁死 capability 只能初始化一次。

- [ ] **Step 1: 新增 Dense runtime-query 禁止测试**

测试意图必须是：kernel 初始化允许调用 detection；初始化之后把 detection monkeypatch 成会抛异常的函数，再执行 `apply_weights`，forward 不能触发异常。

先在同一测试文件新增一个**只复用现有 `_fill_modelopt_style_layer` / `_pack_e2m1_pairs` 的固定 helper**：

```python
def _make_dense_modelopt_emulation_case(device: str = "cpu"):
    from torch.nn.parameter import Parameter
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.kernels.linear.nvfp4.select import (
        init_nvfp4_linear_kernel,
    )

    torch.manual_seed(123)
    m, n, k = 2, 32, 64
    dtype = torch.bfloat16
    x = torch.randn(m, k, dtype=dtype, device=device)
    mag = E2M1_TABLE[torch.randint(0, 8, (n, k))]
    signs = torch.where(torch.rand(n, k) < 0.5, 1.0, -1.0)
    weight_u8 = _pack_e2m1_pairs((mag * signs).float()).to(device)
    weight_scale = torch.ones(n, k // 16, dtype=torch.float32).to(
        torch.float8_e4m3fn
    ).to(device)
    nvfp4_emulation_utils.kE2M1ToFloat_handle.val = E2M1_TABLE.clone().to(device)

    with set_current_vllm_config(_emulation_vllm_config()):
        kernel = init_nvfp4_linear_kernel()

    layer = torch.nn.Module()
    _fill_modelopt_style_layer(
        layer,
        n=n,
        k=k,
        weight_u8=weight_u8,
        weight_scale=weight_scale,
        input_gs=0.05,
        weight_gs=0.05,
    )
    input_global_scale = layer.input_scale.max().to(device=device, dtype=torch.float32)
    layer.input_global_scale = Parameter(input_global_scale, requires_grad=False)
    del layer.input_scale
    weight_global_scale = layer.weight_scale_2.max().to(device=device, dtype=torch.float32)
    layer.weight_global_scale = Parameter(weight_global_scale, requires_grad=False)
    del layer.weight_scale_2
    layer.alpha = Parameter(
        layer.input_global_scale * layer.weight_global_scale,
        requires_grad=False,
    )
    layer.input_global_scale_inv = Parameter(
        (1.0 / layer.input_global_scale).to(torch.float32),
        requires_grad=False,
    )
    kernel.process_weights_after_loading(layer)
    return kernel, layer, x
```

然后新增：

```python
def test_dense_emulation_does_not_query_device_capability_in_forward(monkeypatch):
    import vllm.model_executor.kernels.linear.nvfp4.emulation as dense_emulation

    calls = 0

    def detect_once():
        nonlocal calls
        calls += 1
        return False

    monkeypatch.setattr(
        dense_emulation,
        "_nvfp4_triton_fp8e4nv_supported",
        detect_once,
    )
    kernel, layer, x = _make_dense_modelopt_emulation_case()
    assert calls == 1

    def forbidden():
        raise AssertionError("runtime capability query")

    monkeypatch.setattr(
        dense_emulation,
        "_nvfp4_triton_fp8e4nv_supported",
        forbidden,
    )
    _ = kernel.apply_weights(layer, x)
```

不要新建第二套 NVFP4 reference；helper 只负责构造已存在测试同款的 ModelOpt packed layer。

- [ ] **Step 2: 新增 primitive 显式 backend-flag 测试**

新增：

```python
def test_ref_qdq_explicit_sm80_flag_never_redetects(monkeypatch):
    def forbidden():
        raise AssertionError("runtime capability query")

    monkeypatch.setattr(
        nvfp4_emulation_utils,
        "_nvfp4_triton_fp8e4nv_supported",
        forbidden,
    )
    out = nvfp4_emulation_utils.ref_nvfp4_quant_dequant(
        x,
        global_scale,
        block_size=16,
        use_triton_fp8e4nv=False,
    )
    q, block_scale = nvfp4_emulation_utils.ref_nvfp4_quant(
        x.clone(), global_scale, 16
    )
    expected = (
        q.reshape(x.shape[0], x.shape[1] // 16, 16)
        * (block_scale.unsqueeze(-1) / global_scale)
    ).reshape_as(x).to(x.dtype)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
```

- [ ] **Step 3: 运行测试确认 RED**

Run:

```bash
conda run -n hif4 pytest -q --noconftest \
  3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py \
  -k "does_not_query_device_capability_in_forward or explicit_sm80_flag"
```

Expected: FAIL，原因必须是当前函数无 `use_triton_fp8e4nv` 参数或 forward 仍调用 detection。

- [ ] **Step 4: Commit tests**

```bash
git add 3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py
git commit -m "test: guard nvfp4 capability queries from forward"
```

---

### Task 2: 给 NVFP4 primitive 增加显式 capability 参数，不改数学

**Files:**
- Modify: `3rdparty/vllm/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py`
- Test: `3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py`

**Interfaces:**
- Produces:
  - `ref_nvfp4_quant_dequant(x: torch.Tensor, global_scale: torch.Tensor, block_size: int, *, use_triton_fp8e4nv: bool | None = None) -> torch.Tensor`
  - `dequantize_to_dtype(tensor_fp4: torch.Tensor, tensor_sf: torch.Tensor, global_scale: torch.Tensor, dtype: torch.dtype, block_size: int = 16, swizzle: bool | None = True, *, use_triton_fp8e4nv: bool | None = None) -> torch.Tensor`
  - `run_nvfp4_emulations(x: torch.Tensor, input_global_scale: torch.Tensor, weight: torch.Tensor, weight_scale_swizzled: torch.Tensor, weight_global_scale: torch.Tensor, swizzle: bool | None = True, *, use_triton_fp8e4nv: bool | None = None) -> torch.Tensor`
- `None` 只用于兼容非-forward旧调用；正式 Dense/MoE forward 必须显式传 bool。

- [ ] **Step 1: 修改 `ref_nvfp4_quant_dequant` 签名**

固定实现形式：

```python
def ref_nvfp4_quant_dequant(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    block_size: int,
    *,
    use_triton_fp8e4nv: bool | None = None,
) -> torch.Tensor:
    if use_triton_fp8e4nv is None:
        use_triton_fp8e4nv = _nvfp4_triton_fp8e4nv_supported()
    if use_triton_fp8e4nv:
        return _triton_nvfp4_quant_dequant(x, global_scale, block_size)
    # 以下保持现有 Python reference 数学原样
```

禁止改变 rounding threshold、scale clamp、dtype cast。

- [ ] **Step 2: 修改 `dequantize_to_dtype`**

固定增加 keyword-only 参数：

```python
use_triton_fp8e4nv: bool | None = None
```

现有：

```python
if not swizzle and _nvfp4_triton_fp8e4nv_supported():
```

改成：

```python
if use_triton_fp8e4nv is None:
    use_triton_fp8e4nv = _nvfp4_triton_fp8e4nv_supported()
if not swizzle and use_triton_fp8e4nv:
```

其它 dequant 数学逐行保持不变。

- [ ] **Step 3: 修改 `run_nvfp4_emulations`**

固定签名：

```python
def run_nvfp4_emulations(
    x: torch.Tensor,
    input_global_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_swizzled: torch.Tensor,
    weight_global_scale: torch.Tensor,
    swizzle: bool | None = True,
    *,
    use_triton_fp8e4nv: bool | None = None,
):
```

并把同一个 bool 同时传给 activation QDQ 和 weight dequant。

- [ ] **Step 4: 运行 primitive tests**

Run:

```bash
conda run -n hif4 pytest -q --noconftest \
  3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py \
  -k "explicit_sm80_flag or quant or dequant"
```

Expected: 新增 explicit flag test PASS；原有 NVFP4 数值测试不退化。

- [ ] **Step 5: Commit**

```bash
git add 3rdparty/vllm/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py \
        3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py
git commit -m "fix: make nvfp4 emulation backend choice explicit"
```

---

### Task 3: Dense kernel 初始化阶段冻结 A800 capability

**Files:**
- Modify: `3rdparty/vllm/vllm/model_executor/kernels/linear/nvfp4/emulation.py`
- Test: `3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py`

**Interfaces:**
- Produces: `EmulationNvFp4LinearKernel.use_triton_fp8e4nv: bool`
- Consumes: `_nvfp4_triton_fp8e4nv_supported()`，只在 `__init__` 调一次。

- [ ] **Step 1: Override `__init__`**

固定结构：

```python
class EmulationNvFp4LinearKernel(NvFp4LinearKernel):
    def __init__(self, config: NvFp4LinearLayerConfig) -> None:
        super().__init__(config)
        self.use_triton_fp8e4nv = _nvfp4_triton_fp8e4nv_supported()
```

在 import 中显式引入 detection helper。

不要在 `process_weights_after_loading` 重新检测，避免每 layer/model reload 重复查询。

- [ ] **Step 2: `apply_weights` 显式传 flag**

固定：

```python
out = run_nvfp4_emulations(
    x=x,
    input_global_scale=layer.input_global_scale_inv,
    weight=layer.weight,
    weight_scale_swizzled=layer.weight_scale,
    weight_global_scale=layer.weight_global_scale,
    swizzle=False,
    use_triton_fp8e4nv=self.use_triton_fp8e4nv,
)
```

- [ ] **Step 3: 跑 Task 1 Dense red test，应转 GREEN**

Run:

```bash
conda run -n hif4 pytest -q --noconftest \
  3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py \
  -k "dense_emulation_does_not_query_device_capability_in_forward"
```

Expected: PASS。

- [ ] **Step 4: Dense puncture regression**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n hif4 python NVFP4/scripts/run_nvfp4_emulation_puncture.py \
  --checkpoint /home/shaoyuantian/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3 \
  --report NVFP4/reports/vllm_nvfp4_emulation_cudagraph/task3_dense_puncture.md
```

Expected:

```text
Dense max_abs=0
PASS
```

- [ ] **Step 5: Commit**

```bash
git add 3rdparty/vllm/vllm/model_executor/kernels/linear/nvfp4/emulation.py \
        3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py
git commit -m "fix: freeze nvfp4 dense capability before forward"
```

---

### Task 4: MoE emulation 初始化阶段冻结 capability，并绕开 generic runtime dispatch

**Files:**
- Modify: `3rdparty/vllm/vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py`
- Test: `3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py`

**Interfaces:**
- Produces: `Nvfp4QuantizationEmulationTritonExperts.use_triton_fp8e4nv: bool`
- 两次 A4 QDQ 直接调用 `ref_nvfp4_quant_dequant(A, A_scale, block_size=16, use_triton_fp8e4nv=self.use_triton_fp8e4nv)`；不再经过 `moe_kernel_quantize_input` 的 generic dtype dispatch。
- `_prepare_nvfp4_moe_block_scales(B_scale, *, use_triton_fp8e4nv: bool)` 不再自行检测 GPU。

- [ ] **Step 1: 写 MoE runtime-query failing test**

测试核心：构造 emulation experts 后，将 detection helper monkeypatch 为 raising；执行已有 micro MoE apply，不能再次检测设备。

若完整 experts fixture 成本过高，至少对 `_prepare_nvfp4_moe_block_scales` 和两次 QDQ helper 建立显式 flag test；但真实 MoE puncture 是最终硬门禁，不能用 unit test 替代。

- [ ] **Step 2: `__init__` 固定 capability**

在 `Nvfp4QuantizationEmulationTritonExperts.__init__` 末尾增加：

```python
self.use_triton_fp8e4nv = _nvfp4_triton_fp8e4nv_supported()
```

A800 test 必须断言 `False`。

- [ ] **Step 3: 修改 `_prepare_nvfp4_moe_block_scales`**

固定签名：

```python
def _prepare_nvfp4_moe_block_scales(
    B_scale: torch.Tensor,
    *,
    use_triton_fp8e4nv: bool,
) -> tuple[torch.Tensor, bool]:
```

删除函数内部：

```python
_nvfp4_triton_fp8e4nv_supported()
```

判断只用显式 bool。

- [ ] **Step 4: 精简并修改 `invoke_fused_moe_nvfp4_emulation_kernel`**

当前参数 `act_global_scale` 在函数体第一行即被 `del`，没有参与任何计算。固定删除这个死参数，并增加 keyword-only：

```python
use_triton_fp8e4nv: bool
```

因此函数签名中保留 `A/B/C/B_scale/w_global_scale/topk_weights/sorted_token_ids/expert_ids/num_tokens_post_padded/mul_routed_weight/top_k/config/compute_type`，删除 `act_global_scale`；随后把 `use_triton_fp8e4nv` 传给 `_prepare_nvfp4_moe_block_scales`。W13/W2 两个调用点同步删除无效的 activation-scale 实参。

- [ ] **Step 5: MoE 两次 QDQ 直接调用统一 NVFP4 primitive**

当前代码块为：

```python
hidden_states_qdq, _ = moe_kernel_quantize_input(
    A=hidden_states,
    A_scale=self.quant_config.a1_gscale,
    quant_dtype="nvfp4",
    per_act_token_quant=False,
    quantization_emulation=True,
)
```

固定改为：

```python
hidden_states_qdq = ref_nvfp4_quant_dequant(
    hidden_states,
    self.quant_config.a1_gscale,
    block_size=16,
    use_triton_fp8e4nv=self.use_triton_fp8e4nv,
)
```

W2 前同理：

```python
intermediate_cache2_qdq = ref_nvfp4_quant_dequant(
    intermediate_cache2,
    self.quant_config.a2_gscale,
    block_size=16,
    use_triton_fp8e4nv=self.use_triton_fp8e4nv,
)
```

这样避免进入 `moe_kernel_quantize_input` 中 `current_platform.fp8_dtype()` 等 generic runtime dispatch；数学仍是原来同一个 `ref_nvfp4_quant_dequant`。

- [ ] **Step 6: 两次 GEMM launch 显式传 capability**

W13/W2 两个 `invoke_fused_moe_nvfp4_emulation_kernel` 调用都增加：

```python
use_triton_fp8e4nv=self.use_triton_fp8e4nv
```

- [ ] **Step 7: 跑 MoE unit + puncture**

Run:

```bash
conda run -n hif4 pytest -q --noconftest \
  3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py \
  -k "moe or scale"
```

Run:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n hif4 python NVFP4/scripts/run_nvfp4_emulation_puncture.py \
  --checkpoint /home/shaoyuantian/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3 \
  --report NVFP4/reports/vllm_nvfp4_emulation_cudagraph/task4_moe_puncture.md
```

Expected:

```text
atol <= 0.05 gate
rtol <= 0.01 gate
PASS
```

并确认 warning 仍是 upstream `a13_scale/a2_scale max()` 语义。

- [ ] **Step 8: Commit**

```bash
git add 3rdparty/vllm/vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py \
        3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py
git commit -m "fix: freeze nvfp4 moe capability before forward"
```

---

### Task 5: Level 0 真实 torch.compile / CUDA Graph 门禁

**Files:**
- Reuse unchanged: `NVFP4/scripts/run_qwen3_30b_nvfp4_emulation_smoke.py`
- Create: `NVFP4/reports/vllm_nvfp4_emulation_cudagraph/level0_graph.md`

**Interfaces:**
- Consumes: Task 2–4。
- Produces: 明确判定：PASS，或进入 Level 1/Level 2 的实际 stack 分类。

- [ ] **Step 1: 运行 BF16-KV graph smoke**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n hif4 python NVFP4/scripts/run_qwen3_30b_nvfp4_emulation_smoke.py \
  --kv-cache-dtype bfloat16 \
  --tp 1 \
  --tag cudagraph_level0_graph_bf16 \
  --out-json NVFP4/reports/vllm_nvfp4_emulation_cudagraph/level0_graph_bf16.json
```

- [ ] **Step 2: 按以下固定规则分类，不允许自由发挥**

**Case A：PASS**

同时满足：

```text
Engine initialized
compile completed
CUDA Graph capture completed
generation completed
linear=emulation
moe=emulation
no_marlin=True
```

则：

- Task 6、Task 7 标记 `N/A: Level 0 sufficient`；
- 直接进入 Task 8。

**Case B：FAIL，第一处 graph break 位于以下任一函数内部**

```text
run_nvfp4_emulations
ref_nvfp4_quant_dequant
dequantize_to_dtype
cast_to_fp4
break_fp4_bytes
```

则：

- 将完整 stack 写入 `level0_graph.md`；
- 必须执行 Task 6；
- 不允许改其它 vLLM 子系统。

**Case C：FAIL，Dense 已通过，但第一处 graph break 位于**

```text
Nvfp4QuantizationEmulationTritonExperts.apply
invoke_fused_moe_nvfp4_emulation_kernel
fused_moe_nvfp4_emulation_kernel launch wrapper
```

且不属于 QDQ helper，则：

- Task 6 可标记 `N/A: primitive tracing passed`；
- 必须执行 Task 7。

**Case D：FAIL 在上述白名单外**

停止实现；把 stack、第一处 user-code frame、是否由本任务引入写入报告。不能修改 scheduler、Qwen、attention、TorchDynamo 配置来“绕过去”。

- [ ] **Step 3: Commit report**

```bash
git add NVFP4/reports/vllm_nvfp4_emulation_cudagraph/level0_graph.md
git commit -m "test: classify nvfp4 graph capture after capability fix"
```

---

### Task 6: Level 1 — 仅当 Task 5 Case B 时，把 NVFP4 primitive 注册为 vLLM custom op

**Files:**
- Modify: `3rdparty/vllm/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py`
- Modify: `3rdparty/vllm/vllm/model_executor/kernels/linear/nvfp4/emulation.py`
- Modify: `3rdparty/vllm/vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py`
- Test: `3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py`

**Interfaces:**
- Produces:
  - `torch.ops.vllm.nvfp4_emulation_qdq(Tensor x, Tensor global_scale) -> Tensor`
  - `torch.ops.vllm.nvfp4_emulation_dequant_weight(Tensor weight, Tensor weight_scale, Tensor weight_global_scale, Tensor dtype_ref) -> Tensor`
- `dtype_ref` 只用于确定输出 device/dtype，正式调用固定传当前 activation `x`；不得读取其数值。
- 两个 op 均无 mutable tensor side effect。

- [ ] **Step 1: 写 custom-op fake/meta tests，先 RED**

测试：

```python
def test_nvfp4_emulation_qdq_custom_op_fake_shape_dtype():
    out = torch.ops.vllm.nvfp4_emulation_qdq(x, global_scale)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_nvfp4_emulation_weight_dequant_custom_op_shape_dtype():
    dtype_ref = torch.empty(1, dtype=torch.bfloat16, device=weight.device)
    out = torch.ops.vllm.nvfp4_emulation_dequant_weight(
        weight, weight_scale, weight_global_scale, dtype_ref
    )
    assert out.shape == (weight.shape[0], weight.shape[1] * 2)
    assert out.dtype == dtype_ref.dtype
```

Run expected RED：op 不存在。

- [ ] **Step 2: 注册 QDQ op**

使用仓库现有：

```python
from vllm.utils.torch_utils import direct_register_custom_op
```

固定实现：

```python
def nvfp4_emulation_qdq(
    x: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    return ref_nvfp4_quant_dequant(
        x,
        global_scale,
        block_size=16,
        use_triton_fp8e4nv=False,
    )


def nvfp4_emulation_qdq_fake(x, global_scale):
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="nvfp4_emulation_qdq",
    op_func=nvfp4_emulation_qdq,
    mutates_args=[],
    fake_impl=nvfp4_emulation_qdq_fake,
)
```

注意：Level 1 是 A800/SM80 graph fallback；op 内固定 `False`，不能重新查询 NVML。

- [ ] **Step 3: 注册 Dense weight dequant op**

固定实现，不传 `torch.dtype` scalar，而是传一个 `dtype_ref` tensor：

```python
def nvfp4_emulation_dequant_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    dtype_ref: torch.Tensor,
) -> torch.Tensor:
    return dequantize_to_dtype(
        weight.data.view(torch.uint8),
        weight_scale.data,
        weight_global_scale,
        dtype_ref.dtype,
        block_size=16,
        swizzle=False,
        use_triton_fp8e4nv=False,
    )


def nvfp4_emulation_dequant_weight_fake(
    weight,
    weight_scale,
    weight_global_scale,
    dtype_ref,
):
    return torch.empty(
        *weight.shape[:-1],
        weight.shape[-1] * 2,
        dtype=dtype_ref.dtype,
        device=weight.device,
    )


direct_register_custom_op(
    op_name="nvfp4_emulation_dequant_weight",
    op_func=nvfp4_emulation_dequant_weight,
    mutates_args=[],
    fake_impl=nvfp4_emulation_dequant_weight_fake,
)
```

`dtype_ref` 正式固定传当前 Dense activation `x`，仅用于 dtype/device metadata；禁止读取其元素。

- [ ] **Step 4: Dense apply 改用 op**

仅在 `self.use_triton_fp8e4nv is False` 的 A800 path：

```python
x_dq = torch.ops.vllm.nvfp4_emulation_qdq(
    x,
    layer.input_global_scale_inv,
)
w_dq = torch.ops.vllm.nvfp4_emulation_dequant_weight(
    layer.weight,
    layer.weight_scale,
    layer.weight_global_scale,
    x,
)
out = torch.matmul(x_dq, w_dq.t())
if bias is not None:
    out = out + bias
return out
```

SM89+ 保留原有 Triton/upstream emulation path，不受此次 A800 graph 修复影响。

- [ ] **Step 5: MoE 两次 QDQ 调用同一 custom op**

A800 path：

```python
hidden_states_qdq = torch.ops.vllm.nvfp4_emulation_qdq(
    hidden_states,
    self.quant_config.a1_gscale,
)
intermediate_cache2_qdq = torch.ops.vllm.nvfp4_emulation_qdq(
    intermediate_cache2,
    self.quant_config.a2_gscale,
)
```

不得复制新的 QDQ 数学代码。

- [ ] **Step 6: 数值回归**

对每个 op 与原 Python reference 做：

```python
torch.testing.assert_close(actual, expected, atol=0, rtol=0)
```

至少覆盖：random、全零、E2M1 rounding boundaries、block amax extreme。

- [ ] **Step 7: 再跑真实 graph smoke**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n hif4 python NVFP4/scripts/run_qwen3_30b_nvfp4_emulation_smoke.py \
  --kv-cache-dtype bfloat16 \
  --tp 1 \
  --tag cudagraph_level1_graph_bf16 \
  --out-json NVFP4/reports/vllm_nvfp4_emulation_cudagraph/level1_graph_bf16.json
```

若 PASS → Task 7 标记 N/A，进入 Task 8。  
若第一处 graph break 明确落在 MoE direct Triton GEMM wrapper → 执行 Task 7。  
其它位置 → 停止，写报告，不扩范围。

- [ ] **Step 8: Commit**

```bash
git add 3rdparty/vllm/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py \
        3rdparty/vllm/vllm/model_executor/kernels/linear/nvfp4/emulation.py \
        3rdparty/vllm/vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py \
        3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py
git commit -m "fix: isolate nvfp4 emulation primitives for graph capture"
```

---

### Task 7: Level 2 — 仅当 MoE direct Triton launch 是剩余唯一 graph break 时注册 MoE GEMM custom op

**Files:**
- Modify: `3rdparty/vllm/vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py`
- Test: `3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py`

**Interfaces:**
- Produces 一个固定 inplace custom op：
  `torch.ops.vllm.nvfp4_emulation_moe_gemm(output, A, B, B_scale, w_global_scale, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, mul_routed_weight, top_k, block_size_m, block_size_n, block_size_k, group_size_m) -> None`。
- `mutates_args=["output"]`，其它 tensor 只读。
- 只封装已有 `invoke_fused_moe_nvfp4_emulation_kernel`，不封装 router/top-k/activation/reduce。

- [ ] **Step 1: 写 fake shape test，先 RED**

fake output 必须与当前 `intermediate_cache1` / `intermediate_cache3` shape 完全一致。不要在 fake impl 做 routing 或量化。

- [ ] **Step 2: 把单次 GEMM launch 抽为纯函数**

固定职责：

```text
input QDQ tensor + packed B + block scale + w global scale
+ sorted ids/expert ids/padded count + optional routed weights
-> output tensor
```

它内部只：

1. 接收调用方已经分配好的 `output`；
2. 调当前 `invoke_fused_moe_nvfp4_emulation_kernel` 写入 `output`；
3. 返回 `None`。

所有 config 标量必须在 op 调用前已经由 Python 确定；禁止 op 内查 GPU capability。

- [ ] **Step 3: 注册固定 inplace `nvfp4_emulation_moe_gemm`**

使用 `direct_register_custom_op`，固定设置：

```python
direct_register_custom_op(
    op_name="nvfp4_emulation_moe_gemm",
    op_func=nvfp4_emulation_moe_gemm,
    mutates_args=["output"],
    fake_impl=nvfp4_emulation_moe_gemm_fake,
)
```

`nvfp4_emulation_moe_gemm_fake` 只返回 `None`；接口唯一允许 mutation 的 tensor 是 `output`，packed weights/scales/input/routing tensors 全部只读。

- [ ] **Step 4: W13/W2 都调用同一 op**

W13：`mul_routed_weight=False`。  
W2：保持当前 `not apply_router_weight_on_input` 语义。

- [ ] **Step 5: MoE 数值测试**

复用已有 micro MoE reference：

```text
atol=0.05
rtol=0.01
```

并确认 eager custom-op path 和原 eager implementation 的结果在相同门禁内。

- [ ] **Step 6: 真实 graph smoke**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n hif4 python NVFP4/scripts/run_qwen3_30b_nvfp4_emulation_smoke.py \
  --kv-cache-dtype bfloat16 \
  --tp 1 \
  --tag cudagraph_level2_graph_bf16 \
  --out-json NVFP4/reports/vllm_nvfp4_emulation_cudagraph/level2_graph_bf16.json
```

Expected: 必须 PASS；若仍在白名单外 graph break，停止，不继续封装整个 MoE/Qwen 模型。

- [ ] **Step 7: Commit**

```bash
git add 3rdparty/vllm/vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py \
        3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py
git commit -m "fix: make nvfp4 moe emulation graph opaque"
```

---

### Task 8: 增加项目级 graph contract tests，锁死不再回归到 eager-only

**Files:**
- Create: `NVFP4/tests/test_nvfp4_emulation_graph_contract.py`
- Modify: `3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py`
- Reuse unchanged: `NVFP4/scripts/run_qwen3_30b_nvfp4_emulation_smoke.py`

**Interfaces:**
- Produces: 不依赖完整 30B E2E 的快速 contract tests。

- [ ] **Step 1: capability source contract**

在 `NVFP4/tests/test_nvfp4_emulation_graph_contract.py` 写固定结构测试：

```python
import inspect

from vllm.model_executor.kernels.linear.nvfp4.emulation import (
    EmulationNvFp4LinearKernel,
)
from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
    Nvfp4QuantizationEmulationTritonExperts,
    _prepare_nvfp4_moe_block_scales,
)


def test_nvfp4_forward_paths_do_not_query_device_capability():
    dense_src = inspect.getsource(EmulationNvFp4LinearKernel.apply_weights)
    moe_src = inspect.getsource(Nvfp4QuantizationEmulationTritonExperts.apply)
    assert "_nvfp4_triton_fp8e4nv_supported" not in dense_src
    assert "has_device_capability" not in dense_src
    assert "_nvfp4_triton_fp8e4nv_supported" not in moe_src
    assert "has_device_capability" not in moe_src


def test_moe_block_scale_helper_requires_frozen_capability_flag():
    sig = inspect.signature(_prepare_nvfp4_moe_block_scales)
    assert "use_triton_fp8e4nv" in sig.parameters
    assert sig.parameters["use_triton_fp8e4nv"].default is inspect.Parameter.empty
```

同时保留 Task 1/4 的 monkeypatch behavior tests；source test 不能替代行为测试。

- [ ] **Step 2: torch.compile micro Dense test**

在 `3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py` 中新增 A800 CUDA test，直接复用 Task 1 的 helper：

```python
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 0),
    reason="A800/SM80 graph contract test",
)
def test_dense_emulation_torch_compile_fullgraph_sm80():
    kernel, layer, x = _make_dense_modelopt_emulation_case(device="cuda")

    def forward(inp: torch.Tensor) -> torch.Tensor:
        return kernel.apply_weights(layer, inp, bias=None)

    y_eager = forward(x)
    compiled = torch.compile(forward, fullgraph=True)
    y_graph = compiled(x)
    torch.testing.assert_close(y_graph, y_eager, atol=0, rtol=0)
```

这里 `torch.cuda.get_device_capability` 只出现在 pytest skip/测试图外，不能出现在生产 forward helper 中。

- [ ] **Step 3: MoE micro compile test**

先把现有 `test_nvfp4_moe_emulation_correctness` 中 synthetic case 的构造部分抽成 `_make_nvfp4_moe_emulation_case(num_tokens: int = 4, top_k: int = 2)`，保持原 seed=0、hidden_dim=64、intermediate_size=32、BF16、packed weights、scale 和 routing 生成逻辑不变；现有 correctness test 改为调用该 helper，先确认原测试仍 PASS。

helper 返回一个 dict，固定包含：

```python
{
    "experts": experts,
    "hidden_states": hidden_states,
    "w1": w1,
    "w2": w2,
    "topk_weights": topk_weights,
    "topk_ids": topk_ids,
    "num_experts": num_experts,
    "top_k": top_k,
    "N": N,
    "K": K,
    "intermediate_size": intermediate_size,
}
```

然后新增：

```python
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 0),
    reason="A800/SM80 graph contract test",
)
def test_nvfp4_moe_emulation_torch_compile_fullgraph_sm80():
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    case = _make_nvfp4_moe_emulation_case(num_tokens=4, top_k=2)
    experts = case["experts"]
    num_tokens = case["hidden_states"].size(0)

    def forward(hidden_states, topk_weights, topk_ids):
        output = torch.zeros(
            num_tokens, case["K"], dtype=hidden_states.dtype, device=hidden_states.device
        )
        ws13 = torch.zeros(
            num_tokens * case["top_k"] * max(case["intermediate_size"], case["K"]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        ws2 = torch.zeros(
            num_tokens * case["top_k"] * max(case["N"], case["K"]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        experts.apply(
            output=output,
            hidden_states=hidden_states,
            w1=case["w1"],
            w2=case["w2"],
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=case["num_experts"],
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=ws13,
            workspace2=ws2,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )
        return output

    y_eager = forward(
        case["hidden_states"], case["topk_weights"], case["topk_ids"]
    )
    compiled = torch.compile(forward, fullgraph=True)
    y_graph = compiled(
        case["hidden_states"], case["topk_weights"], case["topk_ids"]
    )
    torch.testing.assert_close(y_graph, y_eager, atol=0.05, rtol=0.01)
```

如果这个 test 的第一处 graph break 明确位于 `invoke_fused_moe_nvfp4_emulation_kernel` / direct Triton launch wrapper，则按 Task 5 Case C / Task 7 处理；若第一处错误是 `try_get_optimal_moe_config` 或其它未列入 Case C 的 Python orchestration，则按 Task 5 Case D 停止并记录完整 stack，不能用 skip/xfail 隐藏，也不能自行扩大 custom-op 边界。

- [ ] **Step 4: 跑 contract suite**

```bash
conda run -n hif4 pytest -q \
  NVFP4/tests/test_nvfp4_emulation_graph_contract.py \
  NVFP4/tests/test_vllm_nvfp4_backend_contract.py \
  NVFP4/tests/test_vllm_emulation_cli_plumbing.py
```

Expected: PASS/硬件相关 skip 只能有明确 reason，不允许 xfail 掩盖 A800 graph test。

- [ ] **Step 5: Commit**

```bash
git add NVFP4/tests/test_nvfp4_emulation_graph_contract.py
git commit -m "test: add nvfp4 emulation graph contracts"
```

---

### Task 9: 真实 ModelOpt Qwen3-30B BF16-KV / FP8-KV CUDA Graph 验收

**Files:**
- Create: `NVFP4/reports/vllm_nvfp4_emulation_cudagraph/qwen3_30b_graph_smoke.md`
- Runtime logs under: `NVFP4/reports/vllm_nvfp4_emulation_cudagraph/smoke_logs/`

**Interfaces:**
- Consumes: 完成后的 graph-capable emulation。
- Produces: 两种 KV 模式真实 E2E 证据。

- [ ] **Step 1: 用现有 wrapper 一次性运行四组正式 smoke**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 \
REPORT_DIR=/home/shaoyuantian/program/HiF4_Sp/NVFP4/reports/vllm_nvfp4_emulation_cudagraph \
REPORT=/home/shaoyuantian/program/HiF4_Sp/NVFP4/reports/vllm_nvfp4_emulation_cudagraph/qwen3_30b_graph_smoke.md \
LOG_DIR=/home/shaoyuantian/program/HiF4_Sp/NVFP4/reports/vllm_nvfp4_emulation_cudagraph/smoke_logs \
TP=1 \
bash NVFP4/scripts/run_qwen3_30b_nvfp4_emulation_smoke.sh
```

该现有 wrapper 固定运行：

```text
eager_bf16_tp1
eager_auto_tp1
graph_bf16_tp1
graph_auto_tp1
```

并分别保存 `.log` 与 `.json`，无需修改 smoke 脚本。

- [ ] **Step 2: 四组 JSON 行为门禁**

四个 JSON 都必须 `passed=true`。另外：

```text
resolved_linear_backend = emulation
resolved_moe_backend = emulation
no_marlin = true
```

BF16 两组：`resolved_kv_cache_dtype=bfloat16`。  
auto 两组：resolved KV 必须包含 `fp8`，目标 checkpoint 当前预期 `fp8_e4m3`。

注意 checkpoint 可能继续打印 K/V scale=1.0 warning；本任务只记录，不修改 KV 数学。

- [ ] **Step 3: graph 原始日志必须给出真正 capture 证据**

对 `graph_bf16_tp1.log` 和 `graph_auto_tp1.log` 同时检查：

```text
Using EmulationNvFp4LinearKernel
Using Nvfp4QuantizationEmulationTritonExperts
Capturing CUDA graphs
```

并确认日志中没有：

```text
torch._dynamo.exc.Unsupported
fallback to Marlin
W4A16 fallback
```

不能仅凭 JSON `passed=true` 宣称 CUDA Graph 已支持；必须有 vLLM graph-capture 日志证据。

- [ ] **Step 4: 报告中记录 graph/eager 四组结果**

`qwen3_30b_graph_smoke.md` 至少记录：backend、resolved KV dtype、graph capture evidence、Marlin hit、generation 是否完成。现有 wrapper 自动生成的报告可作为基础；如信息不足，只增补该报告文本，不修改 smoke 程序。

- [ ] **Step 5: Commit report**

```bash
git add NVFP4/reports/vllm_nvfp4_emulation_cudagraph/qwen3_30b_graph_smoke.md
git commit -m "test: validate nvfp4 cuda graph on a800"
```

---

### Task 10: 数值一致性总门禁

**Files:**
- Reuse: `NVFP4/scripts/run_nvfp4_emulation_puncture.py`
- Create: `NVFP4/reports/vllm_nvfp4_emulation_cudagraph/numerical_parity.md`
- Runtime output: `NVFP4/reports/vllm_nvfp4_emulation_cudagraph/final_puncture.md`

**Interfaces:**
- Produces: eager vs graph / reference 数值结论。

- [ ] **Step 1: primitive + graph micro suite**

Run:

```bash
conda run -n hif4 pytest -q --noconftest \
  3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py
```

Run:

```bash
conda run -n hif4 pytest -q \
  NVFP4/tests/test_nvfp4_emulation_graph_contract.py
```

必须明确报告：

```text
QDQ exact parity: PASS/FAIL
weight dequant exact parity: PASS/FAIL
Dense graph-vs-eager micro parity: PASS/FAIL
MoE graph-vs-eager micro parity: PASS/FAIL
```

- [ ] **Step 2: 真实 checkpoint puncture 回归**

现有 puncture script 本身是 eager/reference 数值 oracle，不承担 CUDA Graph capture；graph parity 已由 Step 1 的 compile micro test 与 Task 9 真实 graph E2E 共同覆盖。Run:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n hif4 python NVFP4/scripts/run_nvfp4_emulation_puncture.py \
  --checkpoint /home/shaoyuantian/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3 \
  --report NVFP4/reports/vllm_nvfp4_emulation_cudagraph/final_puncture.md
```

Acceptance：

```text
Dense reference parity: max_abs=0
MoE: atol=0.05, rtol=0.01, finite=True
Overall: PASS
```

- [ ] **Step 3: double-QDQ gate**

确认 `expects_unquantized_inputs=True` 仍然存在，且 W13/W2 前每次只执行一次 activation QDQ。该结论必须来自现有/新增 behavior test，而不是只看注释。

- [ ] **Step 4: Commit report**

```bash
git add NVFP4/reports/vllm_nvfp4_emulation_cudagraph/numerical_parity.md \
        NVFP4/reports/vllm_nvfp4_emulation_cudagraph/final_puncture.md
git commit -m "test: verify nvfp4 eager graph parity"
```

---

### Task 11: eager vs CUDA Graph 性能 A/B

**Files:**
- Create: `NVFP4/scripts/benchmark_qwen3_30b_nvfp4_emulation_graph.py`
- Create: `NVFP4/reports/vllm_nvfp4_emulation_cudagraph/performance.md`

**Interfaces:**
- CLI:

```text
--mode {eager,graph}
--workload {prefill,decode}
--repeat 3
--warmup 1
--kv-cache-dtype bfloat16
--tp 1
```

- Produces JSON with wall time / input tokens / output tokens / tok/s。

- [ ] **Step 1: benchmark script 固定模型与 sampling**

模型：

```text
/home/shaoyuantian/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3
```

固定：

```text
linear_backend=emulation
moe_backend=emulation
TP=1
kv_cache_dtype=bfloat16
temperature=0
seed=1234
```

`mode=eager` 设置 `enforce_eager=True`；`mode=graph` 设置 `False`。

- [ ] **Step 2: prefill-heavy 工作负载**

固定 32 prompts，目标实际 tokenizer 长度约 512；`max_new_tokens=1`。

每个 mode：

1. 构建 engine；
2. warmup 1 次；
3. 正式重复 3 次；
4. 只统计正式 generate wall time；
5. 记录真实 prompt token 总数；
6. 报 median input tok/s。

compile/capture 首次开销不得计入 steady-state median。

- [ ] **Step 3: decode-heavy 工作负载**

32 prompts，输入约 64 tokens，`max_new_tokens=128`。

记录 median output tok/s。

- [ ] **Step 4: 运行 4 组 benchmark**

```bash
conda run -n hif4 python NVFP4/scripts/benchmark_qwen3_30b_nvfp4_emulation_graph.py --mode eager --workload prefill
conda run -n hif4 python NVFP4/scripts/benchmark_qwen3_30b_nvfp4_emulation_graph.py --mode graph --workload prefill
conda run -n hif4 python NVFP4/scripts/benchmark_qwen3_30b_nvfp4_emulation_graph.py --mode eager --workload decode
conda run -n hif4 python NVFP4/scripts/benchmark_qwen3_30b_nvfp4_emulation_graph.py --mode graph --workload decode
```

- [ ] **Step 5: 报告 gain**

固定计算：

```text
prefill_speedup = graph_input_tok_s / eager_input_tok_s
decode_speedup = graph_output_tok_s / eager_output_tok_s
```

解释规则写死：

```text
speedup >= 1.02  -> 有可测 steady-state 收益
0.98 <= speedup < 1.02 -> 无显著差异
speedup < 0.98 -> graph 在该 workload 反而更慢，需要报告，但不得因此改 NVFP4 数学
```

不允许只挑最好一次；使用 3 次 median。

- [ ] **Step 6: Commit**

```bash
git add NVFP4/scripts/benchmark_qwen3_30b_nvfp4_emulation_graph.py \
        NVFP4/reports/vllm_nvfp4_emulation_cudagraph/performance.md
git commit -m "bench: compare nvfp4 eager and cuda graph"
```

---

### Task 12: 完整回归与最终验收

**Files:**
- Create: `NVFP4/reports/vllm_nvfp4_emulation_cudagraph/final_acceptance.md`
- Review all modified whitelist files。

**Interfaces:**
- Produces: 最终是否“支持静态图”的唯一结论文件。

- [ ] **Step 1: 定向完整测试**

Run:

```bash
conda run -n hif4 pytest -q \
  3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py \
  NVFP4/tests/test_nvfp4_emulation_graph_contract.py \
  NVFP4/tests/test_vllm_nvfp4_backend_contract.py \
  NVFP4/tests/test_vllm_emulation_cli_plumbing.py \
  3rdparty/vllm/tests/config/test_kernel_config.py
```

若全局 vLLM conftest 因当前环境缺包失败，必须使用 `hif4` 环境补齐/使用项目已有正确测试入口；禁止用系统 Python 结果代替。任何 skip 必须在 final report 列原因。

- [ ] **Step 2: Source-scan 禁止项**

检查本任务修改后的 forward reachable 代码：

```text
没有 pynvml.nvmlInit
没有 current_platform.has_device_capability
没有 torch.cuda.get_device_capability
没有 torch._dynamo.disable
没有 graph_break()
没有 fallback_to_marlin
没有 BF16 full-weight persistent cache
没有 Qwen3 layer-name 分支
```

初始化/selection 阶段允许 capability detection；forward 图内禁止。

- [ ] **Step 3: 最终行为矩阵**

`final_acceptance.md` 必须有：

| Mode | Dense NVFP4 | MoE NVFP4 | BF16 KV | auto/FP8 KV | CUDA Graph |
|---|---|---|---|---|---|
| eager | emulation | emulation | PASS | PASS | N/A |
| graph | emulation | emulation | PASS | PASS | PASS |

若任一格未真实验证，写 `UNVERIFIED/FAIL`，不得写 supported。

- [ ] **Step 4: 最终结论必须区分 correctness 与 performance**

格式固定：

```text
CUDA Graph support: PASS/FAIL
Numerical parity: PASS/FAIL
Prefill speedup: X.XXx
Decode speedup: X.XXx
Primary remaining cost: software QDQ / packed-weight decode / BF16 GEMM
```

- [ ] **Step 5: 最终 diff review**

只允许本计划白名单文件改动。任何 scheduler/Qwen/其它 quant format 无关 diff 必须撤回。

- [ ] **Step 6: Commit final report**

```bash
git add NVFP4/reports/vllm_nvfp4_emulation_cudagraph/final_acceptance.md
git commit -m "docs: accept nvfp4 cuda graph support"
```

---

# 完成标准

只有同时满足以下条件才算本计划完成：

1. 真实 A800 graph smoke 不再因 `_nvfp4_triton_fp8e4nv_supported -> NVML` 失败；
2. Dense emulation forward 不做 runtime device capability 查询；
3. MoE emulation forward 不做 runtime device capability 查询；
4. `enforce_eager=False` 的真实 ModelOpt Qwen3-30B-A3B-NVFP4 完成 engine compile + CUDA Graph capture + generation；
5. Dense backend 仍为 `EmulationNvFp4LinearKernel`；
6. MoE backend 仍为 `Nvfp4QuantizationEmulationTritonExperts`；
7. BF16 KV graph smoke 通过；
8. `auto -> FP8` KV graph smoke 通过；
9. 无 Marlin W4A16 fallback；
10. QDQ / weight dequant 数学与改动前 reference 一致；
11. Dense 真实层 puncture 保持 exact parity；
12. MoE 真实层 puncture 继续满足 `atol=0.05, rtol=0.01`；
13. eager 模式继续可用；
14. 完成 prefill-heavy 和 decode-heavy eager-vs-graph 3-repeat median A/B；
15. 性能报告如实记录收益，无论是否达到 2%；
16. 没有为了图捕获修改 NVFP4 scale/QDQ 算法；
17. 没有 Qwen3 专用 hack、全量 BF16 weight cache 或无关 vLLM 大范围改动。
