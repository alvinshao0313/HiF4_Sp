# HiF4 MLP Hierarchical Permutation

对每层 SwiGLU MLP 的中间通道做离线层级排列，使 `down_proj` 的输入激活与权重列在 HiF4 的 `4 → 8 → 64` 分组下重构误差更低。排列吸收进 `gate/up/down` 权重后，浮点输出与原模型等价，推理无额外算子。

## 为什么浮点等价

```text
A = SiLU(X W_g^T) ⊙ (X W_u^T)
Y = A W_d^T
```

SiLU 与逐元素乘法都是逐通道的。对中间维做排列 `perm` 后：

```python
up.weight   = up.weight[perm, :]
gate.weight = gate.weight[perm, :]
down.weight = down.weight[:, perm]
```

浮点路径上 `A` 的通道被同步重排，再被 `down` 的列重排抵消，输出不变（到浮点舍入误差）。

## 为什么只优化 down 激活 / down 权重

排列只交换 `up/gate` 的**输出行**，不改变它们沿输入维的 HiF4 分组。直接受中间维排列影响的量化对象是：

1. `A`（`down_proj` 输入激活）
2. `W_d`（`down_proj.weight` 的输入通道列）

## HiF4 的 4 / 8 / 64

- **G4**：共享最底层 S1P2 scale；理想非零动态范围约 `1.75/0.25 = 7`
- **G8**：两个 G4 共享一级微指数（约允许 2× 峰值差）
- **G64**：八个 G8 共享基础 scale；完整代价走真实 `hifx4` fake quant

搜索用局部邻居召回 + 小束宽贪心构造层级，再用真实 HiF4 误差做局部交换。不用梯度、随机搜索或在线 gather。

## 公共 API（供日后挂 main.py）

推荐挂点：`HiFloat4/main.py` 在 `from_pretrained` 之后、`hif4_rtn_quant` / `gptq_fwrd` **之前**。

```python
from permutation_optimization import reorder_model_mlps, SearchConfig
# 或仅应用已有排列：
from permutation_optimization import apply_permutations_from_file
```

本次改动**不修改** `main.py`；用独立 CLI 跑搜索。

## CLI

```bash
conda run -n hif4 --no-capture-output python -m permutation_optimization.run_mlp_reorder \
  --model Qwen/Qwen3.5-4B \
  --calibration-dataset wikitext2 \
  --calibration-nsamples 128 \
  --calibration-seqlen 2048 \
  --activation-rows 512 \
  --weight-rows 512 \
  --device cuda \
  --output-dir /path/to/out \
  --save-reordered-model /path/to/reordered_bf16 \
  --trust-remote-code
```

常用参数：`--layers 0:3` / `--layers 0,15,31` / `--layers all`。

需在 `HiFloat4/` 为 cwd，或保证 `HiFloat4` 在 `PYTHONPATH` 上。

## 输出

```text
output_dir/
├── config.json
├── permutations.pt      # { "model.layers.i.mlp": LongTensor }
├── layer_metrics.jsonl  # 每层一行，中途可断点查看
└── summary.json
```

`permutations.pt` 约定：`perm[new_position] = old_channel_index`。

单独应用：

```python
from permutation_optimization import apply_permutations_from_file
apply_permutations_from_file(model, "output_dir/permutations.pt")
```

## 指标含义

每层在验证激活上比较四种排列：

| 名称 | 含义 |
|------|------|
| identity | 不重排 |
| random | 固定种子随机排列 |
| q99_sort | 按 activation/weight log q99 均值稳定排序 |
| hierarchical | 本算法 |

`accepted` 要求 hierarchical 同时优于 identity 的真实 HiF4 tensor loss 与 down 输出 NRMSE；否则该层保存 identity。

## 测试

```bash
cd /path/to/HiF4_Sp
conda run -n hif4 --no-capture-output python -m pytest HiFloat4/permutation_optimization/tests -v
```

## 已知范围

- 仅标准 `gate_proj` / `up_proj` / `down_proj` SwiGLU 命名
- 不重排 residual / attention
- `d_ff` 必须被 64 整除，否则报错
- 浮点等价性失败时禁止保存 checkpoint
