# Qwen3-8B 原生 NVFP4 QAT → HiF4 Linear 穿刺实验 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于原生 packed NVFP4 QAT checkpoint，严格复现“每个 Linear 输入先做在线分块旋转，再做 NVFP4 activation QDQ，再与 NVFP4 权重相乘”的 source 推理语义；只选择多层代表性 Linear，保存**旋转后、量化前**的真实激活，离线快速比较 NVFP4/MXFP8/HiF4 激活、HiF4 权重直接转换、权重 S0/e8/e4 三级层级 scale 联合搜索，以及逐通道对角等价缩放搜索等方案的 Linear 输出损失。

**Architecture:** 新实验必须位于仓库根目录下独立目录 `Native_NVFP4_HiF4_Linear_Puncture/`，不得写入或依赖 `Inference_Paradigm_Conversion` 的实验 pipeline。source-of-truth 是 HuggingFace 默认缓存中的原生 packed checkpoint `DASLab/Qwen3-8B-FPQuant-QAT-NVFP4`；正式 forward 用一个纯 PyTorch semantic runtime：从 packed checkpoint 原位读取/解码 NVFP4 权重、逐 Linear 读取或重建 checkpoint 指定的 block-Hadamard/rotation、在所有 252 个目标 Linear 上执行“rotation → NVFP4 A4 QDQ → BF16 GEMM”，只在指定代表层记录 post-rotation/pre-quant activation。完成一次真实 capture 后，六组主要反事实全部离线计算，避免重复跑 8B 全模型。

**Tech Stack:** Python 3.11、PyTorch、transformers、accelerate、huggingface_hub、safetensors、NumPy、pandas、matplotlib；复用仓库已有 `NVFP4.dequantize` / `NVFP4.torch_fake` 和 `HiFloat4/hif4_scale_threshold_optimization/src/{quantizer,weight_search}.py` 的数学实现；conda 环境固定 `hif4`。

## Global Constraints

- 新实验根目录固定：`Native_NVFP4_HiF4_Linear_Puncture/`。
- **禁止修改** `Inference_Paradigm_Conversion/` 现有实验代码来完成本任务；新实验运行时也禁止 import `Inference_Paradigm_Conversion.*`。
- 允许只读复用：`NVFP4/` 与 `HiFloat4/hif4_scale_threshold_optimization/src/` 中已经存在的格式数学实现。
- 原生 checkpoint 默认模型 ID 固定为 `DASLab/Qwen3-8B-FPQuant-QAT-NVFP4`；若实际 HuggingFace repo ID 有命名差异，只允许通过配置项 `model_id` 显式修改，禁止自动 fallback 到旧 `Qmodel/Qwen3-8B-FPQuant-QAT-NVFP4-Dequant-BF16-NoHadamard`。
- checkpoint 未下载完整时，preflight **立即失败并退出**；脚本不负责后台等待或自动下载。
- source 权重必须直接从 packed checkpoint 的 `weight_packed + weight_scale + weight_global_scale` 解码；禁止把旧 fake-dequant BF16 checkpoint 当 source。
- source activation 必须保留 checkpoint 自带的 per-Linear FP32 activation global scale。
- source forward 中，所有目标 Linear 都必须执行 checkpoint 指定的在线 block rotation 后再 NVFP4 activation QDQ；禁止只在被测层量化。
- **捕获点固定为 rotation 后、activation quantization 前。** 记录变量命名统一为 `X_rot`。
- 被测 Linear 自己在正式 source forward 中也继续执行 NVFP4 A4 QDQ；记录 `X_rot` 不能改变 source 输出。
- 离线反事实中的 MXFP8/HiF4 activation 均直接量化已经保存的 `X_rot`；禁止再次施加在线 rotation，避免 double rotation。
- 所有主要方案共享相同的 online rotation。当前实验只研究“旋转坐标系中的格式转换”，不研究“移除 rotation”的影响。
- HiF4 direct RTN 固定使用 `group_size=64, group_dim=-1, s0_divisor=7.0, e8_threshold=4.0, e4_threshold=2.0, s0_mode="hardware"`。
- HiF4 权重“scale 搜索”允许同时优化 **三级层级 scale：S0 + e8 + e4**。实现固定调用现有 `search_weight_groups(..., budget="full", enumerate_e8_e4=True)`：对每个 64-weight group 搜索 S0 的 E6M2 邻域候选，并在每个候选 S0 下，对每个 8-element 子块枚举 e8 与左右两个 4-element e4 的 8 种组合，选择最小 weight squared error 的组合；最后再在 S0 候选中选择组误差最小者。报告中统一称为“HiF4 三级层级 scale 联合搜索”，不得再写成 S0-only。
- 对角缩放优化不是 SmoothQuant `alpha` 搜索。正式变量是每个输入通道独立的正数 `d_j`，组成 `D=diag(d_1,...,d_K)`；使用 `X'=X_rot D^{-1}, W'=W_N D`。每个通道都可以独立调整，禁止退化成一个全局 `alpha` 或 SmoothQuant 解析式。
- 对角阵搜索只做 Linear-local 等价变换，属于“逐通道可恢复损失穿刺实验”；不得在报告中直接声称已解决全模型 foldability。
- 对角阵搜索必须严格使用 calibration split 选择 `D`，validation split 只评估固定 `D`；禁止同一批 activation 既搜索又报告。
- 正式结果以 validation split 的 aggregate error energy 为主，不以逐模块 NMSE 简单算术平均代替。
- **本计划严格停留在 Linear 层级。** 全模型 Qwen3 forward 的唯一用途，是在真实原生 NVFP4 上游传播语义下生成被测 Linear 的 `X_rot`；不得把最终 logits、生成结果、perplexity、KL、ARC、MMLU、MMLU-Pro、AIME 等任何模型级/下游指标作为本实验结果。
- 不做 kernel latency，不 materialize 完整 HiF4 模型；所有正式比较、搜索目标、恢复率和最终结论都只基于被测 Linear 的局部输出 `Y_variant` 相对 `Y_NN` 的误差。

---

# 0. 先锁死本实验真正回答的问题

本实验基准不是 BF16 Linear，而是原生 QAT checkpoint 的 NVFP4 source Linear：

```text
X_pre
  ↓ checkpoint 指定的 block rotation / Hadamard
X_rot                         # 本实验唯一正式 capture 点
  ↓ NVFP4 activation QDQ
A_N
  ↓ packed NVFP4 weight 语义（离线用 W_N=Dequant(packed weight) 等价计算）
Y_NN
```

定义：

```text
W_N = 从原生 packed NVFP4 checkpoint 严格反量化得到的权重
X_rot = 当前 Linear 在线 rotation 后、activation quantization 前的 BF16 激活
A_N = Q_NVFP4(X_rot; current Linear act_global_scale)
A_M = Q_MXFP8(X_rot)
A_H = Q_HiF4(X_rot)
W_H = Q_HiF4_RTN(W_N)
W_Hg = Q_HiF4_HierarchicalScaleSearch(W_N)
```

baseline：

```text
Y_NN = Linear(A_N, W_N, bias)
```

所有 headline 损失都相对于 `Y_NN`。

## 0.1 六个用户要求的主要实验

固定 variant ID，不允许 Cursor 自行改名：

| ID | 权重 | 激活 | 优化 | 用户问题 |
|---|---|---|---|---|
| `E1_WN_AM` | `W_N` | `A_M` | 无 | W NVFP4 / A MXFP8 会比 W NVFP4 / A NVFP4 损失多少？ |
| `E2_WH_AM_RTN` | `W_H` direct | `A_M` | 无 | W HiF4 / A MXFP8 相比 baseline 损失多少？ |
| `E3_WH_AM_GREEDY` | `W_Hg` | `A_M` | 权重三级层级 scale 联合搜索 | S0/e8/e4 联合搜索能降低 E2 多少？ |
| `E4_WH_AH_RTN` | `W_H` direct | `A_H` | 无 | W HiF4 / A HiF4 直接转换损失多少？ |
| `E5_WH_AH_DIAG` | `Q_H(W_N·D)` | `Q_H(X_rot·D^{-1})` | 逐通道对角阵搜索 | 每个输入通道都可独立缩放后，HH Linear 损失能恢复多少？ |
| `E6_WH_AH_GREEDY` | `W_Hg` | `A_H` | 权重三级层级 scale 联合搜索 | HH 下三级 weight-scale 搜索能降低多少？ |

## 0.2 必须额外计算的低成本诊断 control

这些 control 不增加模型 forward，必须一起算，帮助解释总损失：

```text
C0_FP        = Linear(X_rot, W_N)             # source activation 不量化，只作诊断
C1_WN_AH     = Linear(A_H, W_N)               # 单独看 A_N -> A_H
C2_WH_AN_RTN = Linear(A_N, W_H)               # 单独看 W_N -> W_H direct
C3_WH_AN_G   = Linear(A_N, W_Hg)              # 单独看 weight greedy
```

这样最终可以同时回答：

```text
activation-only MXFP8 loss
activation-only HiF4 loss
weight-only direct HiF4 loss
weight-only greedy HiF4 loss
total HM loss
total HH loss
Diagonal-search recovery
Greedy recovery
```

---

# 1. 固定代表层、projection、数据量

## 1.1 Formal representative layers

Qwen3-8B 36 层固定选择：

```python
FORMAL_LAYERS = [2, 10, 18, 26, 34]
```

每层全部七个目标 projection：

```python
PROJECTIONS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
```

formal 共：

```text
5 layers × 7 = 35 Linear
```

选择理由：覆盖 early / early-middle / middle / late-middle / late，不只挑单层；同时避免 36×7 全扫导致离线存储和 weight search 时间不必要膨胀。

## 1.2 Smoke modules

正式 capture 前只跑 3 个穿刺点：

```text
model.layers.2.self_attn.q_proj
model.layers.18.mlp.down_proj
model.layers.34.self_attn.o_proj
```

它们分别覆盖：普通 hidden→attention、intermediate→hidden 的大 K down_proj、深层 attention output。

## 1.3 Prompt bank

新目录中创建独立固定 prompt bank，不 import IPC prompt 文件。

固定 4 类 × 8 = 32 prompts：

```text
natural_language: 8
knowledge_reasoning: 8
math_reasoning: 8
instruction_code: 8
```

每类：

```text
前 4 条 -> calibration/search
后 4 条 -> validation/report
```

固定：

```text
max_seq_len = 256
padding = False
truncation = True
add_generation_prompt = True（若 tokenizer chat template 可用）
formal phase = prefill only
```

本轮 intentionally 不做 decode：目标是快速、可信的 Linear 格式穿刺；若后续需要 decode distribution，单独增量实验，不在本计划偷加。

## 1.4 Token row 采样

每个 prompt × module 最多保存 `64` 个 token row。

规则：

1. 去掉 padding；
2. sequence length `<=64`：全保存；
3. `>64`：使用固定等距 index：`torch.linspace(0, T-1, 64).round().long()`；
4. 禁止随机 token sampling，保证完全可复现；
5. calibration / validation 独立保存。

预计每 module 每 split 最多：

```text
16 prompts × 64 rows = 1024 rows
```

35 modules 的 BF16 `X_rot` 存储量可控，且足够做逐通道对角阵搜索和 Linear 输出统计。

---

# 2. 新目录结构

Cursor 必须创建以下结构，不得把文件散落到 IPC：

```text
Native_NVFP4_HiF4_Linear_Puncture/
├── README.md
├── configs/
│   └── qwen3_8b_native_nvfp4_linear_puncture.yaml
├── plans/
│   └── 2026-08-12-qwen3-8b-native-nvfp4-to-hif4-linear-puncture-plan-cn.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── checkpoint.py
│   ├── rotation.py
│   ├── native_nvfp4.py
│   ├── semantic_model.py
│   ├── prompts.py
│   ├── capture.py
│   ├── formats.py
│   ├── weight_variants.py
│   ├── diagonal_search.py
│   ├── linear_cases.py
│   ├── metrics.py
│   ├── io_utils.py
│   └── report.py
├── scripts/
│   ├── run_preflight.sh
│   ├── run_capture.sh
│   ├── run_linear_cases.sh
│   ├── build_report.sh
│   └── run_all.sh
├── tests/
│   ├── test_checkpoint.py
│   ├── test_rotation.py
│   ├── test_native_nvfp4.py
│   ├── test_semantic_model.py
│   ├── test_capture.py
│   ├── test_formats.py
│   ├── test_weight_variants.py
│   ├── test_diagonal_search.py
│   ├── test_linear_cases.py
│   └── test_metrics.py
└── results/
```

新目录代码可 import：

```text
NVFP4.dequantize
NVFP4.torch_fake
HiFloat4.hif4_scale_threshold_optimization.src.quantizer
HiFloat4.hif4_scale_threshold_optimization.src.weight_search
```

正式 runtime 中禁止 import：

```text
Inference_Paradigm_Conversion.*
```

---

# 3. Task 1：配置、HF snapshot 解析与 preflight

**Files:**

- Create: `Native_NVFP4_HiF4_Linear_Puncture/src/config.py`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/src/checkpoint.py`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/configs/qwen3_8b_native_nvfp4_linear_puncture.yaml`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/scripts/run_preflight.sh`
- Test: `tests/test_checkpoint.py`

## 3.1 Config 固定内容

YAML：

```yaml
model:
  model_id: DASLab/Qwen3-8B-FPQuant-QAT-NVFP4
  expected_architecture: Qwen3ForCausalLM
  expected_num_layers: 36
  target_projections: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

experiment:
  formal_layers: [2, 10, 18, 26, 34]
  max_seq_len: 256
  token_rows_per_prompt: 64
  seed: 20260812
  phase: prefill

nvfp4:
  activation_group_size: 16
  expected_payload: e2m1
  expected_local_scale: e4m3fn

mxfp8:
  block_size: 32
  payload: e4m3fn
  shared_scale: e8m0

hif4:
  group_size: 64
  s0_divisor: 7.0
  e8_threshold: 4.0
  e4_threshold: 2.0
  s0_mode: hardware

weight_greedy:
  budget: full
  enumerate_e8_e4: true
  memory_budget_fraction: 0.25

diagonal_search:
  parameterization: log2
  coarse_log2_offsets: [-1.0, -0.5, 0.0, 0.5, 1.0]
  refine_log2_offsets: [-0.25, 0.0, 0.25]
  num_coarse_sweeps: 1
  num_refine_sweeps: 1
  log2_scale_min: -4.0
  log2_scale_max: 4.0
  search_token_rows_per_module: 128
  search_output_channels_per_module: 128
  full_calibration_rescore: true
  eps: 1.0e-8
```

## 3.2 HF snapshot 解析

实现：

```python
def resolve_local_snapshot(model_id: str) -> Path:
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=model_id, local_files_only=True)
    return Path(path)
```

若 checkpoint 还未下载完整，捕获 `LocalEntryNotFoundError` / missing shard，输出：

```text
Native NVFP4 checkpoint is not fully available in local HF cache.
model_id=...
No fallback checkpoint is allowed.
```

并 exit non-zero。

**禁止：**

```text
自动改用 Qmodel/*Dequant-BF16*
自动联网下载
自动改 repo id
```

## 3.3 Preflight 必查 config 语义

读取 `config.json`、`model.safetensors.index.json`，输出 `preflight.json`。

至少记录：

```json
{
  "model_id": "...",
  "snapshot_path": "...",
  "config_sha256": "...",
  "architecture": "Qwen3ForCausalLM",
  "num_layers": 36,
  "quantization_config": {},
  "forward_dtype": "nvfp4",
  "hadamard_group_size": 16,
  "target_linear_count": 252,
  "packed_weight_coverage": 252,
  "weight_scale_coverage": 252,
  "weight_global_scale_coverage": 252,
  "activation_global_scale_coverage": 252,
  "rotation_source": "checkpoint_tensor|reconstructed_from_official_config",
  "rotation_coverage": 252
}
```

正式 gate：

```text
architecture != Qwen3ForCausalLM -> FAIL
num_layers != 36 -> FAIL
forward_dtype != nvfp4 -> FAIL
252 target 任何 packed/scale/global-scale 缺失 -> FAIL
rotation 语义无法确定 -> FAIL
```

不要假定新的 checkpoint 和旧 dequant checkpoint key 名完全相同；`checkpoint.py` 先枚举每个 target prefix 的真实 auxiliary keys，并在 preflight 中写 `key_schema.json`。一旦 schema 确定，后续代码只使用该 schema，禁止多套 silent fallback。

## 3.4 Rotation 来源审计

优先级固定：

1. checkpoint 为每个 Linear 保存 exact `forward_hadamard_matrix` / transform tensor：直接读取；
2. 若 checkpoint 不存矩阵，但 config 明确 `transform_init=hadamard` 等官方可重建信息：只调用官方 FP-Quant/transformers 对应 helper 重建；
3. 禁止用本项目其他 rotation helper“猜一个 H16”作为 formal source。

preflight 对每个 rotation 矩阵记录：

```text
shape
sha256
max_abs(H H^T - I)
symmetry_error = max_abs(H-H^T)
```

若是标准 normalized Hadamard，orthogonality error 应接近浮点误差；若 checkpoint 定义不是正交归一形式，以官方 runtime 语义为准，不自行 normalize。

## 3.5 TDD

至少测试：

```python
def test_missing_local_snapshot_never_falls_back_to_old_dequant_ckpt(): ...
def test_preflight_requires_packed_weight_triplet_for_every_target_linear(): ...
def test_preflight_requires_activation_global_scale_for_every_target_linear(): ...
def test_rotation_source_must_be_explicit(): ...
def test_config_rejects_non_nvfp4_forward_dtype(): ...
```

---

# 4. Task 2：packed NVFP4 权重解码与 rotation 数学 oracle

**Files:**

- Create: `src/rotation.py`
- Create: `src/native_nvfp4.py`
- Test: `tests/test_rotation.py`
- Test: `tests/test_native_nvfp4.py`

## 4.1 Packed weight source-of-truth

定义：

```python
@dataclass(frozen=True)
class PackedNVFP4LinearState:
    module_name: str
    weight_packed: torch.Tensor
    weight_scale: torch.Tensor
    weight_global_scale: torch.Tensor
    input_global_scale: torch.Tensor
    rotation_matrix: torch.Tensor
    bias: torch.Tensor | None
```

权重反量化只允许调用：

```python
from NVFP4.dequantize import dequantize_nvfp4_weight

w_n = dequantize_nvfp4_weight(
    weight_packed=state.weight_packed,
    weight_scale=state.weight_scale,
    weight_global_scale=state.weight_global_scale,
    dtype=torch.bfloat16,
    group_size=16,
)
```

`W_N` 是**从当前 packed checkpoint 现场解码**得到的 semantic weight，不保存成新的 source checkpoint。

## 4.2 Rotation 接口

```python
def apply_block_rotation(
    x: torch.Tensor,
    rotation_matrix: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    ...
```

要求：

- 只沿最后一维 K 分块；
- 输入 `[..., K]`；
- `K % group_size == 0`，否则 fail；
- reshape 为 `[-1, K/group_size, group_size]`；
- 按 checkpoint/官方 helper 已确认的乘法方向执行；
- formal 输出 dtype 固定 BF16；
- 禁止在这里做任何 activation quantization。

## 4.3 Activation NVFP4 QDQ：输入已经 rotation

定义：

```python
def qdq_nvfp4_post_rotation(
    x_rot_bf16: torch.Tensor,
    input_global_scale: torch.Tensor,
) -> torch.Tensor:
    from NVFP4.torch_fake import fake_quant_nvfp4_activation
    return fake_quant_nvfp4_activation(
        x_rot_bf16,
        input_global_scale=input_global_scale,
        group_size=16,
        output_dtype=torch.bfloat16,
    )
```

注意：这里**不能**调用会再次施加 Hadamard 的 fused source op，因为 `x_rot_bf16` 已经是 post-rotation。

## 4.4 Source Linear oracle

```python
def source_linear_semantic(
    x_pre: torch.Tensor,
    state: PackedNVFP4LinearState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_rot = apply_block_rotation(x_pre, state.rotation_matrix, group_size=...)
    a_n = qdq_nvfp4_post_rotation(x_rot, state.input_global_scale)
    w_n = dequantize_packed_weight(state)
    y = F.linear(a_n, w_n, state.bias)
    return y, x_rot, a_n
```

返回顺序固定，后续 capture 使用同一接口，防止 observer 和 source forward 写出两套数学。

## 4.5 可选 native kernel cross-check

如果当前 GPU capability >= 10.0 且官方 FPQuant fused NV op 可加载：

- 对 smoke 3 modules 各生成 8 组真实/随机输入；
- 比较 native source output 与 `source_linear_semantic()`；
- 优先要求 `NMSE <= 1e-6`、cosine >= `0.999999`；
- 若不满足，**停止 formal capture，先定位 rotation/scale/rounding 语义**。

若硬件不支持 native FPQuant kernel：

```text
native_kernel_crosscheck = skipped_unsupported_hardware
```

允许继续 semantic-oracle 实验，但报告必须写“不是 kernel bit-exact benchmark”。

## 4.6 Tests

```python
def test_rotation_operates_on_last_dim_blocks_only(): ...
def test_rotation_does_not_quantize(): ...
def test_post_rotation_nvfp4_qdq_does_not_rotate_again(): ...
def test_packed_weight_dequant_uses_weight_global_scale(): ...
def test_source_linear_matches_manual_rotate_qdq_linear(): ...
def test_zero_bias_and_nonzero_bias_both_match_manual_reference(): ...
```

---

# 5. Task 3：构建全模型 NVFP4 semantic source runtime

**Files:**

- Create: `src/semantic_model.py`
- Test: `tests/test_semantic_model.py`

这是本计划最关键的实现。**不能为了只测 35 个 Linear，就只给 35 个 Linear 加 NVFP4 A4。**

## 5.1 Loader 原则

原始 packed checkpoint 不能先整体转换成历史 BF16 source checkpoint。

实现：

```python
def load_native_nvfp4_semantic_model(
    snapshot_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, NativeCheckpointIndex]:
    ...
```

建议实现路径：

1. 从 checkpoint config 复制一份内存 config；
2. **只在内存 config 中**移除 quantization loader 配置，使 transformers 能创建普通 Qwen3 architecture；不得写回 checkpoint；
3. 使用 `accelerate.init_empty_weights()` 在 meta device 构建 Qwen3ForCausalLM；
4. 对非 target Linear 的标准参数（embedding、norm、lm_head 等）从 packed checkpoint 原 shard 精确加载；
5. 对 252 target Linear `.weight`，现场调用 `dequantize_nvfp4_weight()` 得到 BF16 `W_N` 后 materialize；
6. 读取/关联每个 Linear 的 `input_global_scale` 和 rotation metadata；
7. 把每个 target `nn.Linear` 包装为 `NativeNVFP4SemanticLinear`。

## 5.2 Wrapper

```python
class NativeNVFP4SemanticLinear(torch.nn.Module):
    def __init__(
        self,
        base_linear: torch.nn.Linear,
        module_name: str,
        input_global_scale: torch.Tensor,
        rotation_matrix: torch.Tensor,
        rotation_group_size: int,
    ): ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_rot = apply_block_rotation(x, self.rotation_matrix, self.rotation_group_size)
        self._maybe_record(x_rot)
        a_n = qdq_nvfp4_post_rotation(x_rot, self.input_global_scale)
        return F.linear(a_n, self.weight, self.bias)
```

顺序必须是：

```text
rotate
→ record X_rot
→ NVFP4 QDQ
→ GEMM
```

不允许：

```text
record X_pre
record A_N 冒充 X_rot
QDQ 后再 rotation
double rotation
```

## 5.3 全覆盖 gate

模型包装完成后：

```text
expected target = 252
wrapped target = 252
```

少 1 个都 fail。

## 5.4 q/k/v 与 gate/up 的独立 rotation

同一个 `X_pre` 可能进入 q/k/v 三个 Linear，也可能进入 gate/up 两个 Linear。

禁止：

```text
先对 x 做一次 rotation，然后把同一个 x_rot 给 q/k/v 共用
```

除非 preflight 已逐元素证明它们 rotation matrix 完全一致。

formal 实现一律让每个 wrapper 根据**自己的 rotation metadata**独立生成 `X_rot`。

## 5.5 Source trajectory smoke

用 2 个短 prompt 跑完整 model prefill：

- layer2/18/34 wrapper 调用次数 >0；
- 252 wrappers 均被注册；
- selected observer on/off 时最终 logits 必须 `torch.equal`；
- 临时 disable 任一早期 layer A4 时，layer18/34 `X_rot` 必须发生非零变化，用来证明上游 A4 确实传播到深层。

测试：

```python
def test_observer_records_after_rotation_before_qdq(): ...
def test_observer_does_not_change_output(): ...
def test_each_linear_uses_its_own_rotation(): ...
def test_upstream_nvfp4_qdq_changes_downstream_capture(): ...
def test_all_expected_target_linears_are_wrapped(): ...
```

---

# 6. Task 4：真实 post-rotation activation capture

**Files:**

- Create: `src/prompts.py`
- Create: `src/capture.py`
- Create: `scripts/run_capture.sh`
- Test: `tests/test_capture.py`

## 6.1 Recorder 只保存 selected 35 modules

全模型 252 Linear 都执行 source QDQ，但 recorder 仅允许：

```text
layer ∈ [2,10,18,26,34]
projection ∈ seven target projections
```

保存的数据就是 wrapper 内部的 `X_rot`。

## 6.2 文件格式

每个 module 每 split 独立一个 `.pt`：

```text
results/<RUN_ID>/captures/
  layer02_q_proj_cal.pt
  layer02_q_proj_val.pt
  ...
```

内容：

```python
{
    "schema_version": 1,
    "module_name": str,
    "layer_idx": int,
    "projection": str,
    "split": "cal|val",
    "x_rot_bf16": Tensor[N,K],
    "input_global_scale_fp32": Tensor[],
    "rotation_sha256": str,
    "rotation_group_size": int,
    "sample_rows": [
        {
            "prompt_id": str,
            "row_start": int,
            "row_end": int,
            "original_token_count": int,
            "selected_token_indices": list[int]
        }
    ]
}
```

不保存 `A_N/A_M/A_H`，这些离线按同一 `X_rot` 重算，避免重复存储和语义漂移。

## 6.3 Capture manifest

```text
results/<RUN_ID>/capture_manifest.json
```

必须记录：

```text
model_id
snapshot hash
source semantic version = native_nvfp4_rot_a4_v1
formal layer list
35/35 capture coverage
252/252 wrapped coverage
prompt IDs by split
max_seq_len
sampling policy
capture point = post_rotation_pre_activation_quant
```

## 6.4 强制审计

随机抽 10 个 capture row，在线同时保存一次小型 audit：

```text
X_pre_sample
X_rot_sample
A_N_sample
```

离线检查：

```text
X_rot_sample == apply_block_rotation(X_pre_sample)
A_N_sample == qdq_nvfp4_post_rotation(X_rot_sample)
```

必须 exact / torch.equal（相同 dtype/math path）。

---

# 7. Task 5：本地 MXFP8 / HiF4 activation 格式 oracle

**Files:**

- Create: `src/formats.py`
- Test: `tests/test_formats.py`

## 7.1 不依赖 IPC 的 MXFP8

实现 local MXFP8 QDQ，语义固定为：

```text
block32 along last dim
shared E8M0 power-of-two scale
E4M3FN payload
```

数学：

```python
amax = group.abs().amax(-1, keepdim=True)
scale_exp = ceil(log2(amax / 448.0))
scale_exp = clamp(scale_exp, -127, 127)
scale = where(amax == 0, 1, 2**scale_exp)
payload = cast_to_fp8_e4m3fn(group / scale)
recon = payload * scale
```

使用 `NVFP4.torch_fake.cast_to_fp8_e4m3fn`，不要复制另一套 E4M3 rounding。

接口：

```python
def qdq_mxfp8_post_rotation(x_rot_bf16: torch.Tensor) -> torch.Tensor: ...
```

## 7.2 HiF4 direct activation/weight

统一 wrapper：

```python
def qdq_hif4_direct(x: torch.Tensor) -> torch.Tensor:
    cfg = HiF4QuantConfig(
        group_size=64,
        group_dim=-1,
        s0_divisor=7.0,
        e8_threshold=4.0,
        e4_threshold=2.0,
        s0_mode="hardware",
    )
    return quantize_hif4(x, config=cfg).reconstruction
```

activation 与 weight 都用同一个正式 HiF4 基础格式，只是输入 tensor 不同。

## 7.3 Tests

```python
def test_mxfp8_block_size_is_32(): ...
def test_mxfp8_zero_block_stays_zero(): ...
def test_hif4_group_size_is_64(): ...
def test_hif4_grouping_is_last_dim_for_both_activation_and_weight(): ...
def test_offline_formats_never_apply_rotation(): ...
```

---

# 8. Task 6：HiF4 权重 direct 与 S0 greedy scale 版本

**Files:**

- Create: `src/weight_variants.py`
- Test: `tests/test_weight_variants.py`

## 8.1 Direct

```python
def build_hif4_direct_weight(w_n: torch.Tensor) -> WeightVariant:
    result = quantize_hif4(w_n, config=STANDARD_HIF4_CONFIG)
    return WeightVariant(
        variant_id="hif4_rtn",
        reconstruction=result.reconstruction,
        metadata={...},
    )
```

## 8.2 HiF4 三级层级 scale 联合搜索（S0 + e8 + e4）

调用现有：

```python
from HiFloat4.hif4_scale_threshold_optimization.src.weight_search import search_weight_groups

result = search_weight_groups(
    w_n,
    budget="full",
    memory_budget_fraction=0.25,
    s0_mode="hardware",
    s0_divisor=7.0,
    enumerate_e8_e4=True,
    e8_threshold=4.0,
    e4_threshold=2.0,
    device=device,
)
```

固定解释：

- `budget="full"` -> S0 E6M2 邻域 offsets `(-2,-1,0,1,2)`；
- `enumerate_e8_e4=True` -> 对每个 candidate S0，在每个 8-element block 内枚举 `(e8, e4_left, e4_right)` 的 8 种组合并选 block squared error 最小者；8 个 block 在给定 S0 下相互独立，因此这一局部枚举等价于该 S0 条件下的全局最优 e8/e4 组合，而不需要暴力枚举 `8^8` 个整组模式；
- objective -> 每个 64-weight group 的 weight squared error；
- 因此 E3/E6 的收益统一称为“**HiF4 三级层级 scale 联合搜索**”：它联合优化 S0/e8/e4，但 objective 仍然只使用 static weight squared error，不混入 activation-aware/output-aware search。

## 8.3 每个 module 保存搜索统计

```text
weight_variants.csv
```

字段：

```text
module_name
numel
rtn_weight_nmse
greedy_weight_nmse
weight_nmse_recovery
greedy_elapsed_s
groups_per_second
group_chunk_size
fraction_s0_index_0..4
```

定义：

```text
weight_nmse_recovery = 1 - greedy_error_energy / rtn_error_energy
```

不 clamp，允许负值。

## 8.4 Test

```python
def test_hierarchical_scale_search_enables_e8_e4_enumeration(): ...
def test_greedy_weight_nmse_never_uses_activation_data(): ...
def test_direct_and_greedy_shapes_match_source_weight(): ...
def test_weight_grouping_is_along_k_dimension(): ...
```

---

# 9. Task 7：逐通道对角阵 D 搜索

**Files:**

- Create: `src/diagonal_search.py`
- Test: `tests/test_diagonal_search.py`

这一任务**不实现 SmoothQuant alpha**。优化变量直接是长度为 K 的正向量：

```text
D = diag(d_1, d_2, ..., d_K),  d_j > 0
```

每个输入通道都有独立自由度。

## 9.1 等价变换定义

在 rotation 后的 Linear-local 坐标系：

```text
Y = X_rot @ W_N^T
```

定义：

```text
X_D = X_rot / d[None, :]
W_D = W_N * d[None, :]
```

未量化时严格满足：

```text
X_D @ W_D^T == X_rot @ W_N^T
```

注意：

- `d` 作用在输入通道/K 维；
- activation 除以 `d`；
- weight 对应列乘以 `d`；
- bias 不变；
- 不在本任务处理 D 如何跨 RMSNorm、attention、MLP 或 residual fold；这里只测 Linear-local 可恢复上界。

## 9.2 为什么按 64-channel group 搜索，但组内每个通道仍独立

HiF4 沿 K 维每 64 个元素共享 S0，因此同一个 64-group 内各 `d_j` 会通过 S0/e8/e4 相互耦合。不能把 4096/12288 个通道当成完全独立 scalar quantizer 搜索。

因此搜索单元固定为：

```text
K-group g = [64*g : 64*(g+1)]
```

每个 group 内维护：

```text
z_g[64] = log2(d_g[64])
```

64 个 `z_j` 都能独立变化；不同 64-group 因 HiF4 K-group 不共享 scale，可独立搜索并最终拼接成完整 `d[K]`。

初始化固定：

```text
z_j = 0  -> d_j = 1
```

禁止用 SmoothQuant alpha 公式初始化正式结果；这样 E5 的收益完全来自实际搜索，而不是预置解析规则。

## 9.3 搜索目标：直接最小化该 64-channel group 的 Linear 输出贡献误差

对第 g 个 K-group：

```python
Xg = X_rot_cal[:, k0:k1]
ANg = A_N_cal[:, k0:k1]
Wg = W_N[:, k0:k1]
```

source group contribution：

```python
Y_ref_g = ANg @ Wg.T
```

候选对角向量 `d_g`：

```python
Xg_D = Xg / d_g
Wg_D = Wg * d_g
AHg_D = qdq_hif4_direct(Xg_D)   # exactly one 64-group
WHg_D = qdq_hif4_direct(Wg_D)   # each output row exactly one 64-group
Y_D_g = AHg_D @ WHg_D.T
```

搜索目标：

```text
E_g(d_g) = ||Y_D_g - Y_ref_g||_F^2
```

**不是 activation MSE，也不是 weight MSE。**

这直接针对最终 Linear 输出贡献做搜索。

## 9.4 用 Gram identity 精确计算 sampled group-output error，避免反复生成 N×O 大矩阵

对搜索子集，禁止每个候选都真正 materialize `Y_D_g[N,O]`。使用恒等式：

```text
||A B^T - C D^T||_F^2
= tr((A^T A)(B^T B))
+ tr((C^T C)(D^T D))
- 2 tr((A^T C)(B^T D))
```

这里：

```text
A = AHg_D
B = WHg_D
C = ANg
D = Wg
```

所有 Gram 都只有 `64×64`。

实现：

```python
def group_output_error_via_gram(
    a_target: torch.Tensor,
    w_target: torch.Tensor,
    a_ref: torch.Tensor,
    w_ref: torch.Tensor,
) -> torch.Tensor:
    ...
```

单测必须与显式 `A@B.T-C@D.T` 的 FP64 squared error 对齐，relative error <= `1e-10`。

## 9.5 固定搜索子集

为了让 35 个 Linear 快速完成逐通道搜索，每个 module 的 calibration split 固定选择：

```text
search token rows = min(128, N_cal)，等距确定性抽样
search output channels = min(128, O)，等距确定性抽样
```

搜索期间只用这个固定子集。

**禁止：**每个 channel/candidate 重采样不同 token 或 output channel。

搜索完成后，必须在**全部 calibration rows + 全部 output channels**上重算最终 D 和 identity D=I 的真实 group contribution error，用作过拟合保护；validation 始终使用全保存 rows + 全 output channels。

## 9.6 两阶段逐通道 coordinate search

参数空间固定在 log2 domain：

```text
z_j = log2(d_j)
clip z_j to [-4, 4]
```

### Stage A：coarse sweep

对每个 64-group，通道顺序固定 `j=0..63`。

对当前通道 j 批量评价：

```text
z_j + {-1.0, -0.5, 0.0, +0.5, +1.0}
```

其他 63 个通道保持当前值。

把 5 个候选一次 stack/batch，禁止 Python 逐 candidate 单独调用 quantizer/GEMM。

选择 sampled `E_g` 最小者并立即更新 `z_j`。

整个 64 通道只做 1 个 coarse sweep。

### Stage B：refine sweep

基于 coarse 结果，再按 `j=0..63` 搜索：

```text
z_j + {-0.25, 0.0, +0.25}
```

同样 batch 评价，做 1 个 refine sweep。

### Tie-break

若多个候选 error 在 FP64 下完全相同：

1. 优先 `|z_j|` 更小；
2. 再优先数值更小的 `z_j`。

这样避免无收益时无意义漂移到极端 scale。

## 9.7 Full-calibration group rollback gate

每个 64-group 搜索结束后，用全部 calibration rows + 全部 output channels分别计算：

```text
E_identity_g = E_g(d_g = 1)
E_search_g   = E_g(d_g = searched)
```

若：

```text
E_search_g > E_identity_g
```

则该 group 整体回滚：

```text
d_g = ones(64)
```

这样搜索子集上的偶然过拟合不会恶化 formal validation。

记录：

```text
num_groups_searched
num_groups_kept
num_groups_rolled_back
```

## 9.8 搜索 API

```python
@dataclass
class DiagonalSearchResult:
    d: torch.Tensor                 # [K], FP32 CPU output
    log2_d: torch.Tensor            # [K]
    group_cal_error_identity: torch.Tensor
    group_cal_error_searched: torch.Tensor
    group_kept_mask: torch.Tensor
    elapsed_s: float


def search_channelwise_diagonal(
    x_rot_cal: torch.Tensor,
    a_n_cal: torch.Tensor,
    w_n: torch.Tensor,
    config: DiagonalSearchConfig,
) -> DiagonalSearchResult:
    ...
```

函数**不接受 validation tensor**，也不接受一个 `alpha`。

## 9.9 Equivalence gate

搜索完成后，在 calibration 前 32 rows 做未量化等价性验证：

```python
y0 = F.linear(x_rot.float(), w_n.float(), bias_float)
y1 = F.linear(
    (x_rot / d).float(),
    (w_n * d).float(),
    bias_float,
)
```

要求：

```text
relative_l2 <= 1e-6
cosine >= 0.9999999
```

失败说明 D 方向/广播轴写错，立即停止该 module。

## 9.10 Validation E5

D 完全由 calibration 固定后：

```python
x_d_val = x_rot_val / d
w_d = w_n * d

a_h_d_val = qdq_hif4_direct(x_d_val)
w_h_d = qdq_hif4_direct(w_d)
y_e5 = F.linear(a_h_d_val, w_h_d, bias)
```

然后统一与：

```text
Y_NN = Linear(A_N_val, W_N)
```

比较。

E5 的正式名称：

```text
E5_WH_AH_DIAG
```

报告用“逐通道对角等价缩放搜索”，禁止写“Smooth alpha”。如果为了口头解释提到 Smooth，只能写“与 Smooth 类似地使用等价对角缩放，但这里每个通道直接独立搜索，不使用 SmoothQuant alpha 参数化”。

## 9.11 输出

`diagonal_search.csv` 每 module 一行：

```text
module_name
k_dim
num_k_groups
num_groups_kept
num_groups_rolled_back
cal_identity_error_energy
cal_searched_error_energy
cal_recovery
diag_scale_min
diag_scale_p10
diag_scale_median
diag_scale_p90
diag_scale_max
fraction_d_eq_1
fraction_d_at_lower_bound
fraction_d_at_upper_bound
elapsed_s
```

另保存：

```text
diagonal_scales/<module_name>.pt
```

内容：

```python
{
    "d": d.cpu(),
    "log2_d": log2_d.cpu(),
    "group_kept_mask": group_kept_mask.cpu(),
    "config": {...},
}
```

## 9.12 Tests

```python
def test_diagonal_transform_is_exact_before_quantization(): ...
def test_search_api_has_no_alpha_and_no_validation_tensor(): ...
def test_each_channel_has_independent_log2_parameter(): ...
def test_coordinate_candidate_grid_is_deterministic(): ...
def test_group_output_error_gram_matches_explicit_matmul(): ...
def test_search_operates_within_one_hif4_k64_group(): ...
def test_full_calibration_worse_group_rolls_back_to_identity(): ...
def test_diagonal_search_returns_finite_positive_scales(): ...
def test_validation_is_never_used_during_search(): ...
```

---

# 10. Task 8：统一 Linear case runner

**Files:**

- Create: `src/linear_cases.py`
- Create: `scripts/run_linear_cases.sh`
- Test: `tests/test_linear_cases.py`

## 10.1 输入

每个 module：

```text
X_rot_cal
X_rot_val
input_global_scale
W_N
W_H_RTN
W_H_GREEDY
bias
```

## 10.2 先统一生成 activation variants

```python
A_N_cal = qdq_nvfp4_post_rotation(X_rot_cal, input_global_scale)
A_N_val = qdq_nvfp4_post_rotation(X_rot_val, input_global_scale)

A_M_cal = qdq_mxfp8_post_rotation(X_rot_cal)
A_M_val = qdq_mxfp8_post_rotation(X_rot_val)

A_H_cal = qdq_hif4_direct(X_rot_cal)
A_H_val = qdq_hif4_direct(X_rot_val)
```

所有 activation variant 必须来自**同一保存的 `X_rot`**。

## 10.3 Validation headline outputs

固定计算：

```python
Y_NN = F.linear(A_N_val, W_N, bias)

Y_E1 = F.linear(A_M_val, W_N, bias)
Y_E2 = F.linear(A_M_val, W_H_RTN, bias)
Y_E3 = F.linear(A_M_val, W_H_GREEDY, bias)
Y_E4 = F.linear(A_H_val, W_H_RTN, bias)
Y_E5 = diagonal_validation_output(...searched_d_from_cal...)
Y_E6 = F.linear(A_H_val, W_H_GREEDY, bias)
```

control：

```python
Y_C0 = F.linear(X_rot_val, W_N, bias)
Y_C1 = F.linear(A_H_val, W_N, bias)
Y_C2 = F.linear(A_N_val, W_H_RTN, bias)
Y_C3 = F.linear(A_N_val, W_H_GREEDY, bias)
```

## 10.4 Primary metrics

每个 `module × variant` 保存：

```text
num_output_elements
reference_energy = sum(Y_NN^2)
error_energy = sum((Y_variant-Y_NN)^2)
nmse = error_energy/reference_energy
sqnr_db = -10*log10(nmse)
relative_l2 = sqrt(nmse)
mae
max_abs_error
cosine
bias_mean = mean(Y_variant-Y_NN)
```

同时保存 activation-level：

```text
NMSE(A_M vs A_N)
NMSE(A_H vs A_N)
zero_rate A_N/A_M/A_H
```

和 weight-level：

```text
NMSE(W_H_RTN vs W_N)
NMSE(W_H_GREEDY vs W_N)
```

## 10.5 Recovery 定义

总输出 error energy：

```text
R_greedy_HM = (E2_error - E3_error) / E2_error
R_smooth_HH = (E4_error - E5_error) / E4_error
R_greedy_HH = (E4_error - E6_error) / E4_error
```

其中所有 `E*_error` 都是相对于 `Y_NN` 的 error energy。

另外输出更纯的 weight-only recovery：

```text
R_weight_only =
  1 - Error(C3 vs NN) / Error(C2 vs NN)
```

但报告必须说明 `C2/C3 vs NN` 中 source activation 仍然是 `A_N`，因此是“固定 source activation 的 weight conversion output recovery”。

若 direct error=0，recovery 写 NaN，并单独 `direct_error_zero=true`；禁止除零写 0。

## 10.6 Aggregate 不能简单平均 NMSE

全局 variant aggregate：

```python
global_nmse = sum(error_energy_i) / sum(reference_energy_i)
```

同时报告：

```text
median(module_nmse)
q90(module_nmse)
max(module_nmse)
```

但 headline 使用 global energy-weighted NMSE。

## 10.7 Tests

```python
def test_all_six_headline_variants_use_same_validation_rows(): ...
def test_mxfp8_and_hif4_do_not_rotate_saved_x_again(): ...
def test_baseline_is_wn_an_not_fp(): ...
def test_recovery_uses_error_energy_against_same_nn_reference(): ...
def test_global_nmse_is_energy_aggregated_not_mean_of_module_nmse(): ...
```

---

# 11. Task 9：结果表、图和中文报告

**Files:**

- Create: `src/metrics.py`
- Create: `src/io_utils.py`
- Create: `src/report.py`
- Create: `scripts/build_report.sh`
- Test: `tests/test_metrics.py`

## 11.1 结果目录

```text
Native_NVFP4_HiF4_Linear_Puncture/results/<RUN_ID>/
├── preflight.json
├── key_schema.json
├── capture_manifest.json
├── captures/
├── audits/
├── weight_variants.csv
├── diagonal_search.csv
├── linear_results.csv
├── global_summary.csv
├── summary.json
├── report_cn.md
├── figures/
└── logs/
```

## 11.2 `linear_results.csv` 一行一个 module × variant

字段至少：

```text
run_id
split
layer_idx
projection
module_name
variant_id
weight_format
activation_format
weight_optimization
diagonal_search_enabled
diagonal_scale_min
diagonal_scale_median
diagonal_scale_max
num_rows
num_output_elements
reference_energy
error_energy
nmse
sqnr_db
relative_l2
mae
max_abs_error
cosine
bias_mean
```

## 11.3 `global_summary.csv`

固定按 variant 输出：

```text
BASE_WN_AN
E1_WN_AM
E2_WH_AM_RTN
E3_WH_AM_GREEDY
E4_WH_AH_RTN
E5_WH_AH_DIAG
E6_WH_AH_GREEDY
C0_FP
C1_WN_AH
C2_WH_AN_RTN
C3_WH_AN_GREEDY
```

## 11.4 必须生成的图

1. `fig01_global_nmse_by_variant.png`
   - 六个 headline variant；
   - y 用 log scale；
   - baseline 不画 0 bar，只在标题说明 reference=WN+AN。

2. `fig02_sqnr_by_variant.png`
   - global SQNR；
   - 越高越好。

3. `fig03_layer_projection_nmse_heatmap_e1.png`
   - WN+AM；
   - 5×7。

4. `fig04_layer_projection_nmse_heatmap_e2.png`
   - WH+AM direct。

5. `fig05_layer_projection_nmse_heatmap_e4.png`
   - WH+AH direct。

6. `fig06_greedy_recovery_by_module.png`
   - HM greedy 与 HH greedy 两条；
   - y recovery，可出现负值。

7. `fig07_diagonal_recovery_by_module.png`
   - HH direct → 逐通道对角阵搜索。

8. `fig08_diagonal_scale_distribution_heatmap.png`
   - 5×7 展示各 module 的 `median(|log2 d|)`；并在图注给出 p10/p50/p90 与边界命中率，不能再画单个 alpha。

9. `fig09_weight_nmse_vs_output_nmse.png`
   - direct / greedy；
   - 看 weight MSE 改善是否真的传递到 output。

10. `fig10_activation_only_loss.png`
    - E1 `WN+AM` 与 C1 `WN+AH`；
    - 直接比较 MXFP8、HiF4 activation 相对 source NVFP4 A4 的局部损失。

11. `fig11_error_decomposition_controls.png`
    - C1 activation-only、C2 weight-only、E4 total HH；
    - 不声称线性可加，只用于直观比较。

12. `fig12_per_projection_boxplot.png`
    - q/k/v/o/gate/up/down 各 variant NMSE 分布。

## 11.5 报告固定结构

`report_cn.md`：

```text
1. 实验目的
2. 原生 NVFP4 QAT checkpoint 的真实 Linear 推理语义
3. 为什么必须捕获 rotation 后、quantization 前 X_rot
4. representative layers / prompts / split / token sampling
5. source semantic oracle 与 native kernel（若可用）cross-check
6. WN+AM：只替换 activation 为 MXFP8 的损失
7. WH+AM direct：weight 转 HiF4 后总损失
8. WH+AM greedy：S0/e8/e4 三级层级 scale 联合搜索能恢复多少
9. WH+AH direct：完整 HiF4 W4A4 local loss
10. WH+AH DIAG：逐通道对角等价缩放搜索能恢复多少
11. WH+AH greedy：三级 weight-scale 搜索在 HIF4 A4 下恢复多少
12. activation-only / weight-only controls
13. layer / projection 敏感性
14. 对角阵 `d_j` 的分布、边界命中率与 group rollback 统计
15. 最终回答与下一步建议
16. 限制：Linear-local semantic oracle，不等于端到端部署精度
```

## 11.6 最终必须用一句话回答六个问题

summary 中固定字段：

```json
{
  "questions": {
    "wn_am_loss_vs_wn_an": {},
    "wh_am_rtn_loss_vs_wn_an": {},
    "wh_am_greedy_recovery": {},
    "wh_ah_rtn_loss_vs_wn_an": {},
    "wh_ah_diagonal_recovery": {},
    "wh_ah_greedy_recovery": {}
  }
}
```

每个对象至少有：

```text
global_nmse
sqnr_db
median_module_nmse
q90_module_nmse
worst_module
```

recovery 问题额外：

```text
recovery_fraction
absolute_nmse_delta
number_modules_improved / 35
```

---

# 12. Task 10：Smoke → Capture → Offline Formal 的执行脚本

**Files:**

- Create: `scripts/run_all.sh`
- 所有现有 scripts 串联

## 12.1 Run ID

```bash
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_native_nvfp4_hif4_linear_puncture}
```

## 12.2 Phase A：preflight

```bash
conda run -n hif4 --no-capture-output \
  python -m Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint \
  --config Native_NVFP4_HiF4_Linear_Puncture/configs/qwen3_8b_native_nvfp4_linear_puncture.yaml \
  --run-id "$RUN_ID"
```

checkpoint 未下载完就停在这里，不得继续。

## 12.3 Phase B：unit tests

```bash
conda run -n hif4 --no-capture-output pytest \
  Native_NVFP4_HiF4_Linear_Puncture/tests -q
```

要求 `0 failed`。

## 12.4 Phase C：3-module source smoke

固定只启用 3 smoke observers，但仍 wrap 全 252 target linears。

验收：

```text
wrapped=252/252
capture=3/3
observer on/off logits torch.equal
X_rot audit exact
A_N audit exact
```

若 native kernel 可用，cross-check gate 同样必须通过。

## 12.5 Phase D：formal capture

```bash
DEVICE=cuda:0 \
RUN_ID="$RUN_ID" \
conda run -n hif4 --no-capture-output \
  bash Native_NVFP4_HiF4_Linear_Puncture/scripts/run_capture.sh
```

期望：

```text
35 cal files
35 val files
35/35 coverage
```

## 12.6 Phase E：释放 8B model 后再做离线 case

capture 结束后必须：

```python
del model
gc.collect()
torch.cuda.empty_cache()
```

离线 runner 重新只加载 35 个 packed weights/aux metadata + captures，不再加载完整模型。

这样三级 weight-scale 搜索 / 逐通道对角阵搜索的显存和运行速度都可控。

## 12.7 Phase F：offline formal

顺序固定：

```text
1. 重算 A_N/A_M/A_H
2. build W_H_RTN
3. build W_H_GREEDY
4. 先跑 controls 与 E1/E2/E3/E4/E6
5. calibration 上搜索逐通道对角阵 D
6. 固定 D 后只在 validation 运行 E5
7. aggregate
8. figures/report
```

禁止先看 validation 结果再修改对角搜索候选网格、搜索 sweep 数或回滚规则。

---

# 13. 结果判读规则

这部分直接写进 report builder，避免 Cursor 事后自由发挥。

## 13.1 WN+AM

若：

```text
E1 global NMSE 很小，SQNR 很高
```

说明在保留在线 rotation 后，把 source NVFP4 activation 换成 MXFP8 对 Linear output 影响有限，`W NVFP4 / A MXFP8` 是有希望的中间部署路径。

若 E1 已很大，则后续 `WH+AM` 的总损失不能简单归咎于 weight HiF4。

## 13.2 WH+AM direct vs greedy

重点同时看：

```text
E2 total loss
E3 total loss
R_greedy_HM
weight NMSE recovery
```

若 weight NMSE 大幅降低但 Linear output 几乎不恢复，说明单纯 weight-MSE greedy 不是最合适目标；后续才有理由做 activation-aware/output-aware scale search。

本轮**不要直接升级成 output-aware greedy**，避免一轮实验同时改变两个问题。

## 13.3 WH+AH direct

E4 是最接近“原生 NVFP4 QAT Linear 直接转换到 HiF4 W4A4”的 local counterfactual。

必须和：

```text
C1 WN+AH activation-only
C2 WH+AN weight-only
```

一起看。

## 13.4 逐通道对角阵 D

若 E5 相比 E4 大幅恢复，说明 post-rotation 坐标下仍存在明显的 per-channel weight/activation range imbalance，而且这种失衡不是一个全局 SmoothQuant alpha 就能充分描述；HiF4 可以从逐通道等价 balancing 中获得明显收益。

报告只能写：

```text
Linear-local per-channel diagonal search demonstrates recoverable loss.
```

不能写：

```text
已得到可直接部署的全模型对角变换方案。
```

因为跨 q/k/v、gate/up、residual/norm 的 foldability 未在本计划验证，而且当前 D 是每个 Linear 独立搜索的。

## 13.5 Greedy HH

E6 与 E4 直接比较，回答“只优化 static weight 的 HiF4 三级层级 scale，在 activation 也换 HiF4 时还能恢复多少”。

逐通道 D 搜索和三级 weight-scale 搜索是两条独立机制，本轮不组合 `DIAG + GREEDY`，否则难判断收益来源。

---

# 14. 自检：本计划写完后的新风险与强制防护

以下是针对本设计重新做的数据流自检。Cursor 实现时不得删减这些 gate。

## 风险 A：新 checkpoint 仍在下载，脚本偷用旧 source

**防护：** `snapshot_download(local_files_only=True)` 失败就退出；无任何 fallback。

## 风险 B：把 packed 权重先导出成历史 NoHadamard BF16 再实验

**防护：** 正式 `W_N` 每次从当前 packed triplet 现场解码；manifest 记录 packed shard/hash。

## 风险 C：捕获位置错在 rotation 前

**防护：** recorder 内嵌到 `NativeNVFP4SemanticLinear`，固定顺序 `rotate → record → QDQ → GEMM`；audit 保存 X_pre/X_rot/A_N 小样本 exact 验证。

## 风险 D：offline variant 又做一次 rotation

**防护：** 所有离线函数参数命名 `x_rot_bf16`；MXFP8/HiF4/NVFP4-post-rotation adapter 内禁止接受/调用 rotation metadata。

## 风险 E：只给 35 个被测 Linear 做 A4，深层 distribution 仍是 BF16

**防护：** 全 252 wrappers；observer 仅 35。layer18/34 upstream-A4 perturbation smoke 必须证明深层 capture 会随早期 A4 改变。

## 风险 F：忽略 per-Linear activation global scale

**防护：** `PackedNVFP4LinearState` 强制包含 scalar `input_global_scale`；252/252 coverage gate；不同 Linear 不共享 scale。

## 风险 G：q/k/v 或 gate/up 被错误共享 rotation

**防护：** 每个 wrapper 持有自己的 rotation hash；默认逐 Linear 执行，只有 exact hash 相同时才可能等价，但 formal 不做共享优化。

## 风险 H：rotation matrix 的方向/归一化猜错

**防护：** 优先 exact checkpoint tensor / 官方 helper；可用 native kernel 时必须 cross-check；禁止用本项目其他 Hadamard helper 猜 formal 语义。

## 风险 I：权重和 post-rotation activation 不在同一坐标系

**防护：** source linear smoke 直接验证 `rotate → A_N → W_N`；若 native runtime 可用，与官方 output 对齐后才能 formal。

## 风险 J：三级 scale 搜索的实现或命名与真实搜索空间不一致

**防护：** `enumerate_e8_e4=True` 写死，并在 test monkeypatch/spy 验证调用参数。额外构造单个 64-element group 的小型 brute-force reference：对 5 个 S0 邻域候选逐一枚举每个 8-element block 的 8 种 `(e8,e4_left,e4_right)` 组合，验证生产 `search_weight_groups()` 返回的 reconstruction、S0、e8、e4 和最小 squared error 与 reference 完全一致。报告只能称“三级层级 scale 联合搜索”，不得误写成 S0-only，也不得误称为 activation-aware 搜索。

## 风险 K：对角阵 D 的方向或广播轴写反

**防护：** 每 module quantization 前检查 `F.linear(X,W) ≈ F.linear(X/D,W*D)`，要求 relative_l2 <= 1e-6；并测试 D 只作用 K/输入通道轴，不允许作用 output channel 轴。

## 风险 L：逐通道 D 搜索发生 validation leakage

**防护：** `search_channelwise_diagonal()` API 根本不接收 validation；cal/val 文件分开；候选网格、sweep 次数、search row/output-channel 子集规则都在看 validation 前固定；report 记录 prompt IDs。

## 风险 M：逐通道 D 搜索出现极端 scale、NaN/Inf 或搜索子集过拟合

**防护：** `log2(d)` 强制 clamp 到 `[-4,4]`，即 `d∈[2^-4,2^4]`；任何非 finite 立即 fail；每个 K64 group 搜索后必须用全部 calibration rows + 全 output channels重评分，若比 identity 更差则整组回滚。报告必须给出边界命中率和 rollback group 数。

## 风险 N：E2/E3 与 E4/E6 使用了不同 token row

**防护：** 所有 variant 都从同一个 `x_rot_val` tensor 构造，runner 禁止 variant 自己采样。

## 风险 O：用逐模块 NMSE 平均给全局结论，小模块权重过大

**防护：** headline global NMSE = `Σerror_energy / Σreference_energy`；模块 median/q90 只作为分布指标。

## 风险 P：down_proj K=intermediate size，group64 或 activation slice 错轴

**防护：** 所有 format wrapper `group_dim=-1`；测试分别覆盖 K=4096 和 K=12288；assert `K%64==0`。

## 风险 Q：bias 处理不一致

**防护：** baseline 和所有 variants 使用同一个 source bias；Qwen 当前多为 no-bias，但代码不能依赖此假设。

## 风险 R：MXFP8 定义与用户想要的 P1 不一致

**防护：** report/config 明确写 `OCP MXFP8: E4M3FN payload + E8M0 shared scale, block32, applied after original online rotation`；不使用含糊“FP8”。

## 风险 S：source baseline 误用 FP output

**防护：** baseline ID 固定 `BASE_WN_AN`；`C0_FP` 只能 control，所有 headline error reference 指针单测检查为 `Y_NN`。

## 风险 T：逐通道 D 与三级 weight-scale 搜索同时使用，归因混乱

**防护：** E5 只使用 direct HiF4 weight + 逐通道 D；E6 只使用三级 weight-scale 搜索 + ordinary HiF4 activation。主六组不加入 `DIAG + GREEDY` 联合项。

## 风险 U：把 local semantic oracle 当真实低比特 kernel 数值/速度

**防护：** report 首页写：`semantic QDQ + BF16 GEMM, not a kernel latency benchmark`；native kernel cross-check 仅用于语义验证。

## 风险 V：从旧 IPC 结果或 helper 不小心带回错误 source 语义

**防护：** 新 runtime 禁止 import `Inference_Paradigm_Conversion.*`；测试扫描 `Native_NVFP4_HiF4_Linear_Puncture/src/**/*.py`，若出现该 import 直接 fail。

---

# 15. Cursor 执行顺序（禁止自由调整）

```text
1. 创建新目录骨架，只改 Native_NVFP4_HiF4_Linear_Puncture/
2. 实现 config + local HF snapshot + checkpoint key schema preflight
3. checkpoint 下载未完成则停止，不得 fallback
4. 实现 packed weight decoder + exact rotation loader/rebuilder + unit tests
5. 实现 post-rotation NVFP4 A4 QDQ + source Linear semantic oracle
6. 若 native kernel 可用，先过 semantic cross-check；不可用则记录 skip
7. 实现 full Qwen3 semantic loader + 252 Linear wrapper
8. 跑 3-module source trajectory smoke；失败禁止继续
9. 实现 35-module post-rotation capture，生成 cal/val files
10. 卸载完整模型
11. 实现本地 MXFP8 + HiF4 direct format adapters
12. 实现 W_H_RTN 与三级层级 scale 联合搜索 W_Hg（S0 + e8 + e4）
13. 实现 controls + E1/E2/E3/E4/E6 offline runner
14. 实现逐通道对角阵 D 的 calibration coordinate search + Gram objective + full-calibration rollback + equivalence gate
15. 固定 calibration 得到的 D 后只在 validation 运行 E5
16. 聚合 energy-weighted metrics
17. 生成全部 figures + report_cn.md
18. 全新目录 pytest 0 failed
19. 跑 formal run
20. 检查 summary 是否逐项回答六个用户问题
```

任何 gate 失败就停在该步修复；禁止先跑出数字再补语义测试。

---

# 16. 最终完成条件

只有全部满足才算实验闭环：

- [ ] source 是本地 HF cache 中的原生 packed `DASLab/Qwen3-8B-FPQuant-QAT-NVFP4`，没有 fallback 到 dequant checkpoint；
- [ ] 252/252 target Linear packed weight / scale / activation scale / rotation 语义可解析；
- [ ] `W_N` 直接从 packed weight triplet 解码；
- [ ] 全 252 target Linear source forward 都执行 checkpoint 指定 rotation + NVFP4 A4 QDQ；
- [ ] formal capture 只取 post-rotation / pre-quant `X_rot`；
- [ ] 35/35 representative Linear cal/val capture 完整；
- [ ] offline 所有 activation variant 都从相同 `X_rot` 生成且不重复 rotation；
- [ ] baseline 固定 `W_N + A_N`；
- [ ] E1–E6 六个 headline variant 完整；
- [ ] C0–C3 controls 完整；
- [ ] HiF4 权重搜索明确为三级层级 scale 联合搜索，`enumerate_e8_e4=True`，并通过单组 brute-force reference 校验；
- [ ] 逐通道对角阵 D 搜索 cal/val 无泄漏，64-group 内每个通道可独立调整，且对角等价变换 pre-quant gate 通过；
- [ ] global NMSE 使用 energy aggregation；
- [ ] 所有 figures 和 `report_cn.md` 生成；
- [ ] report 明确在线 rotation 对所有方案都保留；
- [ ] report 明确本实验是 Linear-local semantic oracle，不是完整端到端模型精度；
- [ ] 新目录 pytest 0 failed；
- [ ] 最终 summary 对用户六个问题分别给出 global NMSE、SQNR、module median/q90、worst module，以及三级 weight-scale recovery / per-channel diagonal recovery。
