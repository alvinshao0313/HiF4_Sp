# E0 NVFP4 Frozen-Input Operator Parity 诊断计划

> 日期：2026-09-01  
> 前置结论：TP1 / TP2 / semantic 三方定位为 Case C：`MIXED_NUMERIC_PATH_DIFFERENCE`  
> 前置报告：`results/long_trajectory_stability/trajectory_stability_smoke_e0_tp1_control/E0_TP1_TP2_SEMANTIC_TRIAGE.md`  
> 本轮目标：找到 **相同输入张量** 下，第一个出现不可忽略数值差的 NVFP4 算子/数值路径。  
> 本轮禁止：启动 formal、解释 H1/H2/H3、修改 production runtime、修改 max-scale/QDQ 定义、降低 0.99 parity gate。

---

# 1. 先纠正一个关键解释边界

TP1 与 TP2 一旦 free-run 发生首次 token 分叉，后续所有 token 都是在不同历史前缀上生成，不能继续作为算子 parity 证据。

当前 TP1↔TP2 首次分叉：

| prompt | first divergence |
|---|---:|
| `n159_c94220492` | 95 |
| `n173_c87875564` | 47 |
| `n246_c189472942` | 6 |
| `n346_c216392826` | 24 |

因此原 4 个 semantic mismatch 中：

- `n159 / decode=10`：TP1/TP2 仍在同一前缀，可用于三方同输入现象描述；
- `n346 / decode=12`：TP1/TP2 仍在同一前缀，可用；
- `n346 / decode=24`：正好是 TP1↔TP2 首次分叉预测点，预测前历史仍相同，可用；
- `n346 / decode=35`：TP1/TP2 已在 decode=24 分叉，**不能**再把三者 token 差异当成 operator mismatch 证据。

Task 4 从此不再使用“不同 free-run trajectory 的后续 token”做算子定位。所有算子比较必须满足：

```text
same checkpoint tensors
+ same exact input tensor
+ same scale
+ same routing ids/weights（MoE）
+ only implementation path differs
```

---

# 2. 代码隔离

所有新代码只放在：

```text
Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/nvfp4_operator_parity/
```

测试只放在：

```text
Native_NVFP4_HiF4_Linear_Puncture/tests/long_trajectory_stability/nvfp4_operator_parity/
```

结果放在：

```text
Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/e0_nvfp4_operator_parity/
```

禁止修改：

- `3rdparty/vllm/**`；
- `Native_NVFP4_HiF4_Linear_Puncture/src/**`；
- `experiments/e2e_diag_reconstruction/**`；
- Phase-A artifact / shared materialized checkpoint；
- 现有 long-trajectory smoke 结果。

如确实发现 production bug，只在报告中给出证据，不在本轮直接修改。

---

# 3. 本轮总体结构

严格按以下阶段执行，并采用“逐级停止/定位”原则：

```text
P0 scale / checkpoint metadata static audit
    ↓
P1 activation QDQ parity
    ↓
P2 weight dequant parity
    ↓
P3 dense packed-linear parity (q/k/v/o)
    ↓
P4 single-expert W13/W2 parity
    ↓
P5 full fused-MoE parity under frozen routing
    ↓
P6 manual TP2 partition/reduction parity
```

前一级如果已经出现明确语义差异，不要跳过直接归因后一级；先把该级差异定位清楚。

---

# 4. P0：静态 scale / checkpoint metadata 审计

先不跑模型 forward，只验证 semantic loader 与 vLLM ModelOpt loading 规则在 48 层是否数值一致。

新增：

```text
nvfp4_operator_parity/audit_scales.py
```

对每层检查：

## 4.1 QKV activation global scale

semantic 当前规则：

```python
qkv_input = max(q.input_scale, k.input_scale, v.input_scale)
input_global_scale_inv = 1 / qkv_input
```

vLLM ModelOpt dense rule：parallel logical widths 的 `input_scale.max()`，随后 `input_global_scale_inv = 1 / max`。

必须输出：

- raw q/k/v input_scale；
- semantic collapsed max；
- expected vLLM collapsed max；
- inverse；
- exact equal。

## 4.2 QKV weight global scale

同样检查：

```text
max(q.weight_scale_2, k.weight_scale_2, v.weight_scale_2)
```

## 4.3 O projection

O 不做 QKV shared collapse，检查 scalar input/weight global scale。

## 4.4 MoE activation scale

检查：

```text
a13_raw = all experts gate/up input_scale
a13_max = max(a13_raw)
a13_inv = 1 / a13_max

a2_raw = all experts down input_scale
a2_max = max(a2_raw)
a2_inv = 1 / a2_max
```

必须与 vLLM `EMULATION` prepare 逻辑完全一致。

## 4.5 MoE W13 weight global scale

vLLM 对每 expert 使用 gate 分支的 `w13_weight_scale_2[:,0]`，如果 gate/up 不一致只 warning。

检查每 expert：

- gate weight_scale_2；
- up weight_scale_2；
- 是否 equal/allclose；
- semantic 实际选择的 gate global scale；
- vLLM 期望选择值。

如果存在 gate/up global scale 不同，必须报告 expert/layer，但**不能自动修**。

### P0 输出

```text
P0_scale_audit.json
P0_scale_audit.md
```

### P0 判定

- 若 collapsed scale 规则/实际值不同：`P0_SCALE_SEMANTIC_MISMATCH`，停止后续 GPU operator parity，先汇报；
- 若一致：进入 P1。

---

# 5. Frozen-input 数据集

operator parity 不从 TP1/TP2 free-run 后续状态取 tensor，而从 **E0 semantic teacher-forced exact trajectory** 重新生成真实中间张量。

新增：

```text
nvfp4_operator_parity/capture_frozen_inputs.py
```

使用原 smoke：

```text
trajectory_stability_smoke/analysis/probe_plan.json
trajectory_stability_smoke/normalized/E0.jsonl
```

优先样本/位置：

### focus positions

- `n159_c94220492 / decode=10`
- `n346_c216392826 / decode=12`
- `n346_c216392826 / decode=24`

### controls

- `n159 / decode=224`
- `n159 / decode=416`
- `n346 / decode=224`
- `n346 / decode=416`

注意：decode=35 不作为 TP1/TP2 同输入证据，但可以作为纯 semantic frozen-input operator stress case；报告中必须标记 `post_tp_divergence_control_only=true`。

默认层集合：

```text
0, 1, 2, 8, 16, 24, 32, 40, 47
```

如果后续某阶段发现误差明显集中，可再局部扩展相邻层，但不先全 48 层做所有重型算子。

每层每 probe 至少保存：

- layer input hidden（pre input RMSNorm）；
- qkv input = input RMSNorm output；
- q/k/v outputs（semantic reference）；
- o_proj input；
- o_proj output；
- post-attention residual；
- MoE input = post-attention RMSNorm output；
- router logits；
- selected_experts；
- routing_weights；
- 对该 probe token 命中的每个 expert：
  - expert input row；
  - gate output；
  - up output；
  - `silu(gate)*up`（W2 input）；
  - down output before routing weight；
  - weighted down contribution。

保存 BF16/FP32 时同时记录 dtype/shape/checksum，避免错误复用。

结果：

```text
frozen_inputs/<prompt>/<decode>/layer_<L>.pt
frozen_inputs_manifest.json
```

---

# 6. P1：Activation QDQ parity

新增：

```text
nvfp4_operator_parity/run_p1_qdq.py
```

对 frozen tensors 中以下输入分别测试：

- qkv input；
- o_proj input；
- MoE expert input；
- W2 input。

比较 A/B：

### A：semantic 实际入口

```python
qdq_native_nvfp4(x, input_global_scale_inv)
```

### B：vLLM primitive

```python
ref_nvfp4_quant_dequant(x, input_global_scale_inv, block_size=16)
```

当前 A 本身就是调用 B，因此这里是调用路径门禁。

记录：

- exact element fraction；
- max_abs；
- rel_L2；
- cosine；
- output dtype；
- 是否 Triton path；
- scale exact value。

### P1 硬预期

对于同设备、同输入、同函数实现：

```text
exact_fraction = 1.0
max_abs = 0
```

如果不成立，优先查：

- 一边进入 Triton，一边 Python fallback；
- dtype/contiguous 差异；
- scale shape/reshape；
- hidden flatten 顺序。

P1 未解释清楚前不进入 P3。

---

# 7. P2：Packed weight dequant parity

新增：

```text
nvfp4_operator_parity/run_p2_weight_dequant.py
```

同一 packed weight / block scale / global scale，比较：

### A semantic loader 使用的路径

```text
modelopt_moe_checkpoint._decode_weight(...)
```

### B vLLM primitive

```text
dequantize_to_dtype(..., swizzle=False)
```

覆盖：

- q/k/v/o；
- focus layers 中实际路由到的 experts 的 gate/up/down；
- 至少 3 个未路由 expert 作为控制。

同时比较：

- FP32 dequant；
- cast-to-BF16 后的权重。

指标：

- exact fraction；
- max_abs；
- rel_L2；
- cosine。

当前 semantic loader 本身使用 vLLM `dequantize_to_dtype`，所以理论上同参数时应一致。

如果不一致，先定位 packed layout / weight scale layout / global-scale 使用值。

---

# 8. P3：Dense packed-linear parity（Q/K/V/O）

这是第一个真正可能出现 GEMM 数值路径差异的阶段。

新增：

```text
nvfp4_operator_parity/run_p3_dense_linear.py
```

对同一 frozen input `x` 比较三个路径：

## D0：current semantic decomposition

```python
x_qdq = ref_nvfp4_quant_dequant(x, input_global_scale_inv)
w_dq = dequantize_to_dtype(..., dtype=BF16)
y = F.linear(x_qdq, w_dq)
```

必须尽量逐行复现 `native_nvfp4_linear`。

## D1：vLLM dense emulation primitive

```python
run_nvfp4_emulations(
    x=x,
    input_global_scale=input_global_scale_inv,
    weight=packed_weight,
    weight_scale_swizzled=block_scale,
    weight_global_scale=weight_global_scale,
    swizzle=False,
)
```

## D2：decomposed torch.matmul control

```python
y = torch.matmul(x_qdq, w_dq.t())
```

D0 vs D2 用于区分 `F.linear` vs `torch.matmul`；D1 vs D2 用于确认 vLLM primitive 除 QDQ/dequant 外是否有额外差异。

### 覆盖

每个 focus/control tensor × selected layer × q/k/v/o。

### 指标

- max_abs；
- mean_abs；
- rel_L2；
- cosine；
- exact element fraction；
- per-output-channel max_abs top20；
- sign flip fraction；
- BF16 ULP-like step normalized error（可用 `abs(diff)/max(abs(y)*2^-7, tiny)` 作为诊断，不作为标准浮点 ULP 严格定义）。

### P3 输出

```text
P3_dense_linear_rows.jsonl
P3_dense_linear_summary.md
```

必须明确：

```text
D0↔D1
D0↔D2
D1↔D2
```

哪个 pair 首先出现差异。

---

# 9. P4：Single-expert W13/W2 parity

P4 不先调用 fused MoE；先把一个 expert 当成普通 packed linear，排除 weight/QDQ/GEMM 本身。

新增：

```text
nvfp4_operator_parity/run_p4_single_expert.py
```

对 frozen probe token 实际命中的 expert，使用相同 expert input：

## W13 gate

semantic decomposition vs `run_nvfp4_emulations`。

## W13 up

同上。

## activation

使用两侧各自 gate/up output：

```python
hidden = silu(gate) * up
```

另外做一个 frozen-hidden control：两侧都强制使用 semantic hidden，再测 W2，以区分“W13 误差传播”和“W2 自身误差”。

## W2

比较：

- semantic W2 decomposition；
- packed primitive W2；
- 输入分别用：
  1. semantic hidden；
  2. primitive hidden。

这样可以拆成：

```text
W13 intrinsic error
activation amplification
W2 intrinsic error
W13→W2 propagated error
```

### P4 特别记录

每个 expert：

- gate/up weight global scale 是否相同；
- actual w13 global scale chosen；
- a13_inv；
- a2_inv。

只报告实际值，不修改。

---

# 10. P5：Full fused-MoE parity under frozen routing

只有 P1–P4 基本对齐后才进入 P5。

目标：判断差异是否来自 fused expert sorting/padding/GEMM/routed-weight accumulation，而不是单个 linear。

新增：

```text
nvfp4_operator_parity/run_p5_fused_moe.py
```

输入固定：

- frozen MoE input `[tokens, hidden]`；
- frozen router logits；
- frozen selected_experts；
- frozen routing_weights。

**禁止重新计算 router top-k 后再比较**，否则 routing 离散变化会污染 operator parity。

比较：

## M0 semantic routed_moe decomposition

但强制使用 frozen `selected_experts/routing_weights`，不能重新 router。

## M1 vLLM NVFP4 emulation fused experts

尽量直接调用现有 `Nvfp4QuantizationEmulationTritonExperts` / `invoke_fused_moe_nvfp4_emulation_kernel` 的现有接口，在诊断脚本中构造参数；不复制/修改 kernel。

记录中间：

- W13 fused output before activation；
- post activation hidden；
- W2 output before routed weight；
- routed weighted contribution；
- final MoE sum。

若现有 API 无法直接暴露中间结果，允许在诊断目录构建 wrapper 分阶段调用已有 primitive，但不得修改 vLLM 文件。

### P5 判断

- single-expert P4 对齐、P5 不对齐：优先定位 sorting/padding/fused accumulation；
- P4 已明显不对齐：P5 只用于确认误差是否继续放大，不能把根因归到 fused routing。

---

# 11. P6：Manual TP2 partition / reduction parity

只有在以下条件下进入：

- TP1 primitive/decomposition 与 semantic 基本一致；
- 但 TP1↔TP2 full runtime 明显不同；
- 或 P3/P5 已显示 full-tensor GEMM 与 shard/reduce 可能是主要来源。

新增：

```text
nvfp4_operator_parity/run_p6_manual_tp2.py
```

使用同一 frozen input/weight，按照 Qwen3/vLLM TP2 切法显式模拟：

## Column-parallel

Q/K/V：

```text
output-dim shard 0 GEMM
output-dim shard 1 GEMM
concat
```

比较：

```text
full GEMM
vs manual TP2 concat
```

## Row-parallel

O / down 等：

```text
input-dim shard 0 partial GEMM
input-dim shard 1 partial GEMM
BF16/实际 dtype reduction
```

至少比较：

1. FP32 partial sum；
2. BF16 partial/reduction；
3. vLLM 实际约定的 dtype/order。

目标是量化“仅改变 reduction order 能产生多大 layer-output 差”。

## MoE TP2

若专家维度/中间维度按 TP 切分，按实际 vLLM partition 方式模拟 W13/W2 shard + reduction。

不要凭经验假设切分维，必须从实际 vLLM layer parameter shapes / TP world size 推导并记录。

---

# 12. 每阶段都要有 control，不只看 mismatch probes

每阶段结果必须分：

```text
focus_low_margin
uniform_control
```

如果某 operator 在 focus 和 control 上数值误差差不多，只是最终 token 在 low-margin 点翻转，说明更可能是普通数值扰动 + decision margin，而不是该 operator 对 focus state 特异失效。

如果 focus operator error 本身显著更大，才说明输入分布/量化边界可能在这些状态上更敏感。

---

# 13. 不设置拍脑袋的单一误差阈值

本轮不要用类似：

```text
rel_L2 > 1e-3 => FAIL
```

直接判 bug。

原因：最终 semantic mismatch 的 top1-top2 margin 只有 0.5–3.75，极小层级差异经过 48 层可能足以翻 token。

本轮重点是：

1. 哪个阶段**第一次**从 exact/near-exact 变成非零；
2. 差异数量级；
3. focus vs control 是否不同；
4. 后续阶段放大倍率。

只有 P0/P1/P2 这种理论上同函数/同参数的阶段，可以用 exact equality 作为硬门禁。

---

# 14. Cursor 必须输出的总报告

最终生成：

```text
results/long_trajectory_stability/e0_nvfp4_operator_parity/E0_NVFP4_OPERATOR_PARITY_REPORT.md
```

固定结构：

## 1. P0 scale audit

- QKV collapse；
- a13/a2；
- w13 gate/up global scale mismatch count；
- verdict。

## 2. P1 QDQ

- exact fraction；
- max_abs；
- focus/control；
- verdict。

## 3. P2 weight dequant

同上。

## 4. P3 dense linear

按 q/k/v/o、layer 列：

- D0-D1 rel_L2/max_abs；
- D0-D2；
- D1-D2；
- first nonzero stage。

## 5. P4 single expert

- W13 gate；
- W13 up；
- post activation；
- W2 frozen-hidden；
- W2 propagated-hidden。

## 6. P5 fused MoE

- single-expert vs fused 差异；
- sorting/padding/accumulation 是否新增误差。

## 7. P6 TP2

- full vs shard/reduce；
- row/column parallel；
- reduction dtype/order 影响。

## 8. Root-cause classification

只能从以下标签选一个主标签，可附 secondary：

```text
SCALE_METADATA_MISMATCH
ACTIVATION_QDQ_PATH_MISMATCH
WEIGHT_DEQUANT_PATH_MISMATCH
DENSE_GEMM_NUMERIC_DIFFERENCE
SINGLE_EXPERT_GEMM_NUMERIC_DIFFERENCE
FUSED_MOE_ACCUMULATION_DIFFERENCE
TP_PARTITION_REDUCTION_DIFFERENCE
MIXED_SMALL_NUMERIC_DIFFERENCES
INCONCLUSIVE
```

## 9. Production change recommendation

- `NO_PRODUCTION_CHANGE_NEEDED`：只是可接受的数值路径差异；
- `DIAGNOSTIC_TP2_ADAPTER_NEEDED`：为了长轨迹 semantic parity，应在诊断目录模拟 TP2；
- `PRODUCTION_BUG_CANDIDATE`：有同输入、同参数下明确语义不一致的证据。

只有最后一种才建议后续修改 production runtime。

---

# 15. 执行纪律

1. 先 P0；
2. 再 capture frozen inputs；
3. P1/P2；
4. P3；
5. P4；
6. P5；
7. 需要时才 P6；
8. 每阶段保存原始 JSON/JSONL，不只写 Markdown；
9. 不修改生产代码；
10. 不启动 formal；
11. 不解释 H1/H2/H3；
12. 不把 post-divergence free-run token 差异当成 operator parity；
13. 做完 `E0_NVFP4_OPERATOR_PARITY_REPORT.md` 后停止，等待下一步。