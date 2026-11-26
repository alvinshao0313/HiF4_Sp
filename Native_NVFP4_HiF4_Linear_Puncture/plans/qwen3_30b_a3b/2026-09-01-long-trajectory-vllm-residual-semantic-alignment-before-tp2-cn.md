> 状态：已被 2026-09-01-qwen3-30b-hif4-long-trajectory-real-vllm-hook-replan-cn.md 取代。
> 仅用于历史问题复盘，不再执行。

# 长轨迹诊断：先对齐 vLLM 残差 / RMSNorm / RoPE，再进入 TP2 adapter

> **状态：已停止执行。** 后续不再继续 residual/RMSNorm/RoPE/TP2 semantic adapter 路线；唯一主计划见 `2026-09-01-qwen3-30b-hif4-long-trajectory-real-vllm-hook-replan-cn.md`。

> 日期：2026-09-01  
> 前置结果：fused-MoE semantic Gate 1 从 `24/28` 提升到 `26/28`，仍剩两个 mismatch。  
> 本轮目标：先解决 **TP1/TP2 共同存在、因此与 TP 无关** 的 semantic residual mismatch，再决定 TP2 adapter 的最短实现。  
> 本轮禁止：修改 production runtime、修改 NVFP4 QDQ/scale、启动 formal、降低 0.99 gate、跑 E1–E4、提前实现 TP2 adapter。

---

## 1. 先修正上一轮的下一步判断

当前剩余两个 mismatch 不能都归为 TP2：

| probe | TP2 | TP1 | fused semantic | 性质 |
|---|---:|---:|---:|---|
| `n159_c94220492 / 10` | 279 | 279 | 419 | **TP 无关 semantic mismatch** |
| `n346_c216392826 / 24` | 24301 | 607 | 607 | **TP2-specific mismatch** |

所以：

- `n346/24` 明确需要后续 TP2 数值语义对齐；
- `n159/10` 不可能靠 TP2 adapter 本身解决，因为 TP1 和 TP2 已经一致；
- 如果现在只实现 TP2，最乐观也可能只是 `26/28 -> 27/28`，仍然不能过 Gate 1。

因此正确顺序是：

```text
先消掉 TP-independent semantic mismatch
        ↓
确认只剩 TP2-specific mismatch
        ↓
再单独设计 TP2 adapter
```

---

## 2. 已发现的关键实现差异

当前 fused semantic replay 仍沿用：

```python
post_attn = layer_input + attention_output
moe_input = semantic_rms_norm(post_attn)
layer_output = post_attn + moe_output
```

并且每层只保存一个完整 `hidden`。

但正式 vLLM Qwen3-MoE decoder 的数值路径不是这样执行的。它跨层维护：

```text
hidden_states = 当前 branch 输出
residual      = 独立 residual buffer
```

其核心语义是：

```python
if residual is None:
    residual = hidden_states
    hidden_states = input_layernorm(hidden_states)
else:
    hidden_states, residual = input_layernorm(hidden_states, residual)

attention_output = self_attn(hidden_states)

hidden_states, residual = post_attention_layernorm(attention_output, residual)
moe_output = mlp(hidden_states)

return moe_output, residual
```

最后：

```python
hidden_states, _ = final_norm(hidden_states, residual)
```

其中 CUDA `RMSNorm(x, residual)` 实际走的是：

```text
fused_add_rms_norm
```

也就是说：

```text
BF16 residual add
+ RMSNorm reduction
+ output cast/rounding
```

发生在同一 vLLM 数值路径中。

数学上它和 semantic 的显式加法 + RMSNorm 等价，但浮点执行路径不等价。

此外当前 semantic 还有两个 TP 无关实现差异：

1. Q/K norm 使用自定义 `sem._rms_norm`，vLLM 使用 `vllm.model_executor.layers.layernorm.RMSNorm`；
2. RoPE 使用 Transformers `Qwen3MoeRotaryEmbedding + apply_rotary_pos_emb`，vLLM 使用 `get_rope(...)(positions, q, k)`。

这些都必须先于 TP2 adapter 排查。

---

## 3. 代码隔离

本轮所有新代码只放：

```text
Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/vllm_semantic_alignment/
```

测试只放：

```text
Native_NVFP4_HiF4_Linear_Puncture/tests/long_trajectory_stability/vllm_semantic_alignment/
```

结果放：

```text
Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/trajectory_stability_smoke_vllm_semantic/
```

允许复用：

- 现有 `fused_semantic_adapter/native_nvfp4_fused.py`；
- 现有 E0 fused MoE bundle；
- 现有 probe plan；
- 现有 TP1/TP2 triage artifact。

禁止修改：

- `3rdparty/vllm/**`；
- `experiments/e2e_diag_reconstruction/**`；
- 已有 `causal_replay.py`；
- 已有 `causal_replay_fused.py`；
- Phase-A artifact；
- production runtime。

---

## 4. 第一阶段：先做 RMSNorm 数值微对齐，不跑完整 48 层

新增：

```text
vllm_semantic_alignment/compare_norm_paths.py
```

只使用现有 frozen input artifact：

```text
results/long_trajectory_stability/e0_nvfp4_operator_parity/frozen_inputs/**
```

重点 probe：

- `n159 / decode=10`
- `n346 / decode=24`

控制：

- `n159 / decode=224`
- `n159 / decode=416`
- `n346 / decode=224`
- `n346 / decode=416`

重点层：

```text
0, 1, 2, 8, 16, 24, 32, 40, 47
```

### 4.1 普通 RMSNorm

同一个 BF16 input 和同一个 weight，比较：

```text
A: sem._rms_norm
B: vLLM RMSNorm CUDA op
```

覆盖：

- input layernorm；
- post-attention layernorm 的“已加完 residual”输入；
- Q norm；
- K norm。

记录：

- exact fraction；
- max_abs；
- mean_abs；
- rel_L2；
- cosine。

### 4.2 fused residual-add + RMSNorm

在 layer 0 可以严格使用真实：

```text
residual = layer_input
branch   = attention output
```

比较：

```text
A:
    summed = residual + branch
    sem._rms_norm(summed)

B:
    vLLM RMSNorm(branch, residual)
```

同时记录 vLLM 返回的 updated residual。

注意：这一项只用于证明数值路径是否非 exact，不直接用 layer>0 semantic full hidden 假装 vLLM residual split。

### 4.3 判定

本阶段不设“必须为零”的要求，因为两条实现本来就允许合法浮点差异。

只回答：

1. `sem._rms_norm` vs vLLM RMSNorm 是否 non-exact；
2. fused residual-add + RMSNorm 是否引入更明显差异；
3. focus `n159/10` 是否比 uniform control 更敏感。

输出：

```text
NORM_PATH_PARITY.jsonl
NORM_PATH_PARITY.md
```

无论差异大小，只要脚本语义正确，都进入完整 replay；这个阶段主要用于归因，不作为停止 gate。

---

## 5. 第二阶段：实现 vLLM residual-state semantic replay

新增：

```text
vllm_semantic_alignment/causal_replay_vllm_residual.py
```

只先支持 E0。

不要改现有 fused replay。

### 5.1 每个 sample 必须维护两个 cache

不能继续只有：

```text
hidden_cache
```

改为：

```text
branch_cache[prompt]
residual_cache[prompt]
```

初始化：

```text
branch = embedding(input_ids)
residual = None
```

### 5.2 每层严格模拟 vLLM decoder residual 状态机

每层：

```python
if residual is None:
    residual = branch
    attn_input = input_rmsnorm(branch)
else:
    attn_input, residual = input_rmsnorm(branch, residual)

attn_output = attention(attn_input)
moe_input, residual = post_attn_rmsnorm(attn_output, residual)
moe_output = fused_nvfp4_moe(moe_input)

branch = moe_output
```

其中：

- `input_rmsnorm`：必须用 vLLM `RMSNorm` CUDA op；
- `post_attn_rmsnorm`：必须用 vLLM `RMSNorm(x, residual)`；
- `fused_nvfp4_moe`：直接复用上一轮已经完成的 `NativeNvfp4FusedBundle`；
- router 继续由现有 native semantic router 计算；
- dense Q/K/V/O 继续用当前 `native_nvfp4_linear`，因为 frozen identical input 下已经证明与 vLLM dense emulation exact。

### 5.3 Q/K norm 也切到 vLLM RMSNorm

当前：

```text
sem._rms_norm(q)
sem._rms_norm(k)
```

必须替换为：

```text
vLLM RMSNorm(head_dim=128)
```

并加载原 checkpoint 的：

```text
q_norm_weight
k_norm_weight
```

### 5.4 第一版先保持当前 RoPE + causal SDPA

第一版只改变：

```text
residual state machine
layer RMSNorm
Q/K RMSNorm
final RMSNorm
```

先不要同时改 RoPE，避免无法判断到底是哪项修复了 `n159/10`。

attention core 第一版仍用：

```text
Transformers RoPE
+ current causal SDPA
```

### 5.5 final norm 必须按 vLLM 语义

48 层结束后不能再做：

```python
sem._rms_norm(full_hidden)
```

必须：

```python
final_hidden, _ = vllm_final_rmsnorm(branch, residual)
```

再送 lm_head。

---

## 6. 第一个完整 gate：只看 28 probe

运行 root：

```text
results/long_trajectory_stability/trajectory_stability_smoke_vllm_semantic/residual_norm/
```

对比三版：

```text
old semantic      24/28
fused semantic    26/28
residual-norm semantic ?/28
```

固定重点：

```text
n159/10
n346/24
```

### Case A：n159/10 修复，n346/24 仍不修复

例如：

```text
n159/10: 279  -> match TP1/TP2
n346/24: 607  -> still match TP1, differ TP2
```

这是当前最希望看到的结果。

判定：

```text
TP_INDEPENDENT_SEMANTICS_ALIGNED = true
TP2_ADAPTER_READY = true
```

做到这里停止。

不要在本轮继续写 TP2 adapter。

### Case B：n159/10 仍然是 419

说明 residual/RMSNorm 还不足以解释 TP-independent mismatch。

进入第 7 节 RoPE 对齐。

### Case C：产生新的 mismatch

如果原本 26 个正确 probe 中出现新的 top1 flip：

- 输出新旧 mismatch 集合；
- 不认为“总体 parity 提高就行”；
- 继续第 7 节前先检查 residual state machine 是否严格复刻 vLLM；
- 如发现实现 bug，只修诊断目录。

---

## 7. 仅当 n159/10 未修复：增加 vLLM RoPE

新增：

```text
vllm_semantic_alignment/vllm_rope_attention.py
```

使用 vLLM production 同一个：

```python
from vllm.model_executor.layers.rotary_embedding import get_rope
```

构造参数必须来自 snapshot config：

```text
head_dim=128
max_position_embeddings
rope_parameters
```

严格匹配 vLLM Qwen3MoeAttention：

```text
q_proj/k_proj/v_proj
→ q/k vLLM RMSNorm
→ vLLM get_rope(positions, q, k)
→ attention core
```

第一版 attention core仍保持 causal SDPA。

重新跑同一个 28-probe gate，结果写到：

```text
.../trajectory_stability_smoke_vllm_semantic/residual_norm_rope/
```

### 若 n159/10 修复

判定：

```text
TP_INDEPENDENT_SEMANTICS_ALIGNED = true
TP2_ADAPTER_READY = true
```

停止。

### 若 n159/10 仍未修复

判定：

```text
ATTENTION_CORE_PARITY_REQUIRED = true
TP2_ADAPTER_READY = false
```

本轮停止。

不要为了继续推进而直接做 TP2 adapter。

原因：此时仍存在一个 TP1/TP2 都一致、semantic 独有的 mismatch，说明 attention core 或其它 shared runtime 数值语义尚未对齐。

---

## 8. 本轮不实现 attention backend adapter

如果第 7 节后仍剩 `n159/10`，本轮只记录：

```text
ATTENTION_CORE_PARITY_REQUIRED = true
```

下一轮再单独定位：

```text
same Q/K/V after vLLM norm + RoPE
current causal SDPA
vs
actual vLLM Attention backend
```

届时再确定 A800 当前实际 backend 与 KV-cache 数值路径，不在本轮提前写复杂 adapter。

---

## 9. 为什么 TP2 adapter 必须排在这之后

`n346/24` 的证据非常强：

```text
TP1 = 607
fused semantic = 607
TP2 = 24301
```

所以它后续确实需要 TP2 semantics。

但 `n159/10`：

```text
TP1 = TP2 = 279
fused semantic = 419
```

说明它与 TP2 无关。

只有当 `n159/10` 修复后，我们才能把剩余 error 干净地定义成：

```text
single-rank semantic aligned
+ only TP2 residual remains
```

这样下一轮 TP2 adapter 才能只做真正必要的部分。

---

## 10. 后续 TP2 adapter 的设计边界（本轮只记录，不实现）

当且仅当：

```text
TP_INDEPENDENT_SEMANTICS_ALIGNED = true
```

下一轮才设计 TP2。

已知优先级应是：

1. row-parallel O projection reduction；
2. MoE W2 / expert intermediate TP shard + reduction；
3. Q/K/V column partition 只作为次要校验，因为此前 manual TP2 column 最大差异远小于 row；
4. attention 每个 head 独立，不应无依据引入跨-head 模拟；
5. 使用实际 vLLM TP parameter shape 推导 shard，不能只按理论均分猜。

但这些实现留到下一轮。

---

## 11. 最小测试

使用 `hif4` conda 环境。

新增测试：

```text
tests/long_trajectory_stability/vllm_semantic_alignment/test_vllm_semantic_alignment.py
```

至少覆盖：

1. vLLM residual state machine：layer0 `residual=None` 分支；
2. layer1+ `RMSNorm(branch,residual)` 分支；
3. post-attention RMSNorm 更新 residual；
4. final norm 使用 `(branch,residual)`，不是 materialized full hidden；
5. Q/K RMSNorm weight shape=128；
6. replay 内 branch/residual 不被错误提前相加后存回；
7. vLLM RoPE mode 使用 snapshot config，position 无 off-by-one；
8. E0 fused MoE bundle 仍与上一轮相同，不能退回 Python expert loop。

不需要新增 production test。

---

## 12. Cursor 最终报告

生成：

```text
Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/trajectory_stability_smoke_vllm_semantic/analysis/VLLM_SHARED_SEMANTIC_ALIGNMENT_REPORT.md
```

固定回答：

1. `sem._rms_norm` vs vLLM RMSNorm 是否 exact；
2. fused residual-add + RMSNorm 差异数量级；
3. residual+vLLM RMSNorm replay parity 是多少；
4. `n159/10` 是否修复；
5. `n346/24` 是否仍保持 TP1-like=607；
6. 是否执行 vLLM RoPE 对齐；
7. RoPE 对齐后 parity 是多少；
8. `TP_INDEPENDENT_SEMANTICS_ALIGNED = true/false`；
9. `ATTENTION_CORE_PARITY_REQUIRED = true/false`；
10. `TP2_ADAPTER_READY = true/false`；
11. production runtime modified = false；
12. formal started = false。

做到报告后停止。

---

## 13. 执行纪律

本轮的唯一目标不是把 28/28 强行做出来，而是先回答：

> `n159/10` 这个 **TP1=TP2≠semantic** 的残余差异到底是不是由 vLLM 的 residual/RMSNorm/RoPE shared semantics 造成。

只有这个问题闭环之后，才进入 TP2 adapter。
