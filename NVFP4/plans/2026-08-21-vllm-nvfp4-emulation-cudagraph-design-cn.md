# vLLM NVFP4 Emulation torch.compile / CUDA Graph 支持设计

**日期：** 2026-08-21  
**目标仓库：** `/home/shaoyuantian/program/HiF4_Sp`  
**目标硬件：** NVIDIA A800 80GB（SM80）  
**前置能力：** 当前仓库已完成 vLLM v0.27.0 风格 NVFP4 Dense/MoE emulation backport，ModelOpt Qwen3-30B-A3B-NVFP4 在 `--enforce_eager` 下已通过真实模型 smoke 与单层数值穿刺。

## 1. 目标

在**不改变 NVFP4 数学语义、不改变 packed 权重驻留方式、不改变 eager 数值结果**的前提下，使当前 A800 NVFP4 emulation 路径能够被 vLLM 的 `torch.compile` / CUDA Graph 正常捕获并运行，从而消除目前必须使用 `--enforce_eager` 的限制。

本任务不是重新设计 NVFP4 fake quant，也不是性能 kernel 重写。它只处理 compile/capture 兼容性。

## 2. 当前已确认根因

当前真实 `graph_bf16_tp1` smoke 的第一个失败发生在 attention Dense `qkv_proj`，并未进入 MoE：

```text
qwen3_moe.self_attn
  -> qkv_proj
  -> EmulationNvFp4LinearKernel.apply_weights
  -> run_nvfp4_emulations
  -> ref_nvfp4_quant_dequant
  -> _nvfp4_triton_fp8e4nv_supported
  -> current_platform.has_device_capability(89)
  -> pynvml.nvmlInit()
  -> torch._dynamo.exc.Unsupported: _FuncPtr.__call__
```

因此当前失败**不是 MoE 架构导致的**，而是 forward 图内存在运行时 GPU capability/NVML 查询。

同类风险还存在于 MoE emulation：

```text
Nvfp4QuantizationEmulationTritonExperts.apply
  -> moe_kernel_quantize_input(... quantization_emulation=True)
  -> ref_nvfp4_quant_dequant
  -> _nvfp4_triton_fp8e4nv_supported
```

以及：

```text
invoke_fused_moe_nvfp4_emulation_kernel
  -> _prepare_nvfp4_moe_block_scales
  -> _nvfp4_triton_fp8e4nv_supported
```

这些硬件判定必须全部移出 forward/capture 路径。

## 3. 固定设计原则

### 3.1 GPU capability 只允许在初始化阶段确定一次

A800 是否支持 Triton `tl.float8e4nv` 是设备属性，不是运行时张量属性。必须在 Dense kernel / MoE experts 对象初始化或 `process_weights_after_loading` 阶段计算一次，并保存为普通 Python `bool`。

禁止在以下函数的 forward 路径再次调用：

```python
current_platform.has_device_capability(...)
pynvml.nvmlInit()
torch.cuda.get_device_capability(...)
```

### 3.2 数学实现不变

SM80 仍然使用当前已经验证过的 Python reference 数学路径：

- group size = 16；
- E2M1 payload；
- E4M3FN block scale；
- ModelOpt/CT global scale 语义保持现状；
- MoE `a13_scale = 1 / max(a13_scale)` / `a2_scale = 1 / max(a2_scale)` 保持 vLLM v0.27.0 逻辑；
- packed uint8 NVFP4 权重保持常驻；
- 不预先全量反量化为 BF16 Parameter；
- 不允许 graph 模式静默 fallback 到 Marlin W4A16。

### 3.3 先最小修复，再按实际 graph break 逐级封装

不一开始重写成 custom op。固定升级顺序如下：

1. **Level 0：初始化阶段冻结 capability。**
2. 运行真实 fullgraph smoke。
3. 如果下一处 graph break 位于 NVFP4 QDQ / dense weight dequant Python helper，则进入 **Level 1：NVFP4 emulation primitive custom op**。
4. 再运行真实 fullgraph smoke。
5. 如果下一处 graph break 位于 `Nvfp4QuantizationEmulationTritonExperts.apply` 中的 direct Triton MoE GEMM launch / Python orchestration，则进入 **Level 2：MoE emulation GEMM custom op**。
6. 若错误不属于上述白名单调用链，停止扩改并记录堆栈；禁止 Cursor 自己扩大范围。

这是一个由实际错误栈驱动的固定状态机，不是自由选择实现方案。

## 4. Level 0：冻结 capability

### Dense

`EmulationNvFp4LinearKernel` 在初始化阶段保存：

```python
self.use_triton_fp8e4nv: bool
```

其值只由当前 worker 已选定 GPU 的 capability 决定。

之后调用链显式传递这个布尔值：

```text
EmulationNvFp4LinearKernel.apply_weights
  -> run_nvfp4_emulations(..., use_triton_fp8e4nv=...)
  -> ref_nvfp4_quant_dequant(..., use_triton_fp8e4nv=...)
  -> dequantize_to_dtype(..., use_triton_fp8e4nv=...)
```

helper 内不得再自行查询 GPU。

### MoE

`Nvfp4QuantizationEmulationTritonExperts` 初始化时同样保存：

```python
self.use_triton_fp8e4nv: bool
```

该值显式传到：

```text
moe_kernel_quantize_input
ref_nvfp4_quant_dequant
_prepare_nvfp4_moe_block_scales
invoke_fused_moe_nvfp4_emulation_kernel
```

A800 必须固定为 `False`。

## 5. Level 1：仅在需要时注册 NVFP4 primitive custom op

当前 vLLM 已提供 `vllm.utils.torch_utils.direct_register_custom_op`，且 quantization / MoE / attention 已大量使用该模式。因此如果 Level 0 后 Dynamo 仍尝试追踪 emulation helper 内部 Python 实现，则使用仓库原生 custom-op 机制，而不是 `torch._dynamo.disable` 或 graph break。

固定注册两个无副作用 op：

```text
vllm::nvfp4_emulation_qdq
vllm::nvfp4_emulation_dequant_weight
```

职责：

- `nvfp4_emulation_qdq(x, global_scale) -> Tensor`：执行现有 `ref_nvfp4_quant_dequant` 数学；
- `nvfp4_emulation_dequant_weight(weight, weight_scale, weight_global_scale) -> Tensor`：执行现有 `dequantize_to_dtype(swizzle=False)` 数学。

A800 实现固定走已经验证的 Python/Torch 路径；fake implementation 只声明 shape/dtype，不执行真实量化。

Dense graph 内变成：

```text
x
 -> torch.ops.vllm.nvfp4_emulation_qdq
 -> torch.ops.vllm.nvfp4_emulation_dequant_weight
 -> torch.matmul
 -> bias(optional)
```

MoE 的两次 activation QDQ 同样复用 `nvfp4_emulation_qdq`，避免出现 Dense/MoE 两套数学实现。

## 6. Level 2：仅在需要时封装 MoE direct Triton launch

如果 Level 1 后 fullgraph 只剩 MoE direct Triton launch / Python orchestration graph break，则将**单次 expert GEMM**封装为 vLLM custom op，而不是把整个 router/MoE 层封装掉。

固定 op 名：

```text
vllm::nvfp4_emulation_moe_gemm
```

它只负责当前 `invoke_fused_moe_nvfp4_emulation_kernel` 的计算，输入必须显式包含：

- QDQ 后 activation；
- packed weight；
- block scale；
- weight global scale；
- top-k weights（W13 传 None，W2 使用）；
- sorted token ids；
- expert ids；
- num tokens post padded；
- output tensor / output shape 所需标量；
- 已经在图外确定的 Triton config 标量。

router、top-k、`moe_align_block_size`、SiLU-and-mul、最终 reduce 继续沿用 vLLM 原逻辑，不重复实现。

禁止封装整个 Qwen3Moe 层，禁止写 Qwen3 专用 custom op。

## 7. compile / CUDA Graph 成功定义

不能只以“torch.compile 不报错”作为完成标准。必须同时满足：

1. `enforce_eager=False` 的真实 ModelOpt Qwen3-30B-A3B-NVFP4 能完成 Engine 初始化；
2. vLLM compilation mode 不被项目代码主动关闭；
3. CUDA Graph capture 完成；
4. 至少完成一次 prefill + decode generation；
5. Dense backend 仍为 `EmulationNvFp4LinearKernel`；
6. MoE backend 仍为 `Nvfp4QuantizationEmulationTritonExperts`；
7. 无 Marlin fallback；
8. BF16 KV 与 `auto -> FP8` 两种 KV 模式各跑通一次。

## 8. 数值回归门禁

静态图支持不得改变 accuracy semantics。

### primitive

固定随机 seed 和边界值，对比改动前 reference：

- QDQ：`atol=0, rtol=0`；
- Dense weight dequant：`atol=0, rtol=0`；
- Dense linear output：保持现有 puncture `max_abs=0` 目标。

### MoE

复用当前真实 layer puncture，并增加同一 synthetic MoE case 的 eager-vs-fullgraph micro parity：

- 真实 puncture tolerance：`atol=0.05, rtol=0.01`；
- micro eager-vs-graph tolerance：`atol=0.05, rtol=0.01`；
- 不允许新增 double-QDQ。

### E2E

固定 prompts、seed、KV dtype，分别完成 eager 与 graph generation smoke。E2E 用于验证完整编译/capture/生成链路；数值 acceptance 由 primitive exact parity、Dense/MoE micro graph parity 和真实 layer puncture共同承担，生成文本只作为附加观察，不要求逐 token 完全一致。

## 9. 性能验证

支持 CUDA Graph 的目的之一是降低 Python / kernel launch 开销，但 emulation 的主要成本仍包括：

- activation NVFP4 software QDQ；
- packed weight on-the-fly decode；
- BF16/FP16 GEMM。

因此不预设收益幅度，也不把“必须提升 X%”作为 correctness 门禁。

固定做两类 A/B：

### Prefill-heavy

- 同一模型；
- `kv_cache_dtype=bfloat16`；
- TP=1；
- 32 条输入；
- 输入长度约 512 tokens；
- `max_new_tokens=1`；
- eager / graph 各 warmup 后测 3 次；
- 报告 median wall time、input tok/s。

### Decode-heavy

- 32 条输入；
- 输入约 64 tokens；
- `max_new_tokens=128`；
- 同样 warmup + 3 次；
- 报告 median wall time、output tok/s。

只比较 steady-state，不把 graph compile/capture 首次开销计入推理吞吐。

若提升 <2%，报告“静态图已支持，但该工作负载无显著性能收益”；禁止为了追求性能继续修改 NVFP4 数值算法。

## 10. 非目标

本任务明确不做：

- 不修改 NVFP4 格式/scale/QDQ 公式；
- 不优化 MoE routing 算法；
- 不添加 A800 专用近似量化；
- 不切换到 W4A16；
- 不整体升级 vLLM；
- 不处理 compressed-tensors 的新 E2E 支持问题（已有 contract 保持回归即可）；
- 不处理 MMLU-Pro 当前运行脚本；
- 不改变 KV cache quantization 数学；
- 不为性能引入 BF16 全量权重缓存。

## 11. 最终验收

完成后应当允许：

```bash
python main.py \
  --model_path <native ModelOpt NVFP4 checkpoint> \
  --linear_backend emulation \
  --moe_backend emulation \
  --kv_cache_dtype bfloat16
```

**不再要求 `--enforce_eager`。**

同时 `--enforce_eager` 仍必须保持可用，并作为数值 reference / debug 模式。
