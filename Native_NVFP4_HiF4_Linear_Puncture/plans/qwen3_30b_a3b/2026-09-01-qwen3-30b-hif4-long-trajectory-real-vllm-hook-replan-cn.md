# Qwen3-30B-A3B NVFP4→HiF4 长轨迹失稳：真实 vLLM Hook 重构与完整验证计划

> 日期：2026-09-01  
> 目标模型：`nvidia/Qwen3-30B-A3B-NVFP4`  
> 主目标：用**真实 vLLM TP2 推理路径**验证 NVFP4→HiF4 后的长轨迹失稳机制，不再依赖自建 semantic replay。  
> 正式执行路径：**真实 vLLM TP2 + E0 exact token trajectory 强制解码 + worker 内 hook + 真实 KV-cache decode**。  
> 执行者：Cursor。Cursor 只按本文实现、验证和运行，不得自行恢复 semantic replay、降低门禁或设计第二套 decoder/runtime。  
> 最高原则：**正式机制结论只能来自真实 vLLM actual path；自建 PyTorch/semantic decoder 不再进入正式证据链。**

---

# 0. 本计划取代哪些旧路线

本计划是长轨迹机制实验的新的唯一主计划。

以下旧路线的核心假设已经被实验否定，不再继续：

- 手写/半语义 Qwen3-MoE causal replay；
- semantic E0 top1 parity 门禁；
- fused semantic adapter；
- residual/RMSNorm semantic adapter；
- 手工 TP2 shard/reduction semantic adapter；
- shared-attention semantic alignment；
- 用 `full_max_decode` 一次长序列前向读取早期 probe 的方法。

原因已经明确：旧 replay 在多个地方会与真实 vLLM 数值语义漂移，包括：

1. vLLM residual 双状态传播；
2. vLLM CUDA RMSNorm；
3. fused MoE 多 expert 汇聚；
4. TP reduction；
5. attention backend / KV-cache；
6. 最关键的：**即使逻辑上都 causal，同一个绝对位置在不同完整序列长度下，SDPA 的分块/累加顺序可以改变数值甚至改变 top1。**

已确认案例：`n346/24`，同一 `abs_pos=369`：

- 正确 predictor prefix `prefix_len=370`：top1=`24301`；
- 旧的长序列 `full_len=762`：同位置 top1=`607`。

这说明继续“修 semantic replay 直到像 vLLM”会把大量时间消耗在复刻 backend，而不是研究 HiF4。

因此从本文开始：

```text
真实 vLLM 是唯一数值真值
      ↓
诊断代码只做三件事
1. 强制 token trajectory
2. hook actual-path tensor
3. 离线比较/统计
```

旧的 frozen-input operator parity 结果可以作为辅助机制证据保留，但**不能替代真实 vLLM 长轨迹证据**。

---

# 1. 最终要回答的研究问题

本实验不是为了证明“量化有误差”，而是建立完整的长轨迹失稳链条。

## RQ1：格式转换误差在真实 decode 中是否随轨迹长度持续积累？

固定所有 variant 都沿 E0 的 exact token history 前进，比较：

\[
S^{E0}_{l,t}\quad\text{vs}\quad S^{E1}_{l,t}
\]

其中：

- `l`：Transformer layer；
- `t`：decode predictor position；
- `S`：真实 vLLM actual-path 中间状态。

要判断：

- 误差是否从 early → mid → late 持续增长；
- 哪些层持续放大，哪些层抵消；
- Attention 与 MoE 哪个分支主要制造新误差。

## RQ2：连续数值误差怎样变成第一次 token rank reversal？

需要把：

```text
hidden / branch error
      ↓
Attention / MoE output error
      ↓
router boundary / decision boundary
      ↓
logit margin 缩小或翻转
      ↓
first token divergence
```

连成一条实际证据链。

## RQ3：MoE router 是否构成离散放大器？

要验证：

- router logits 是否先发生小偏移；
- top-k expert 是否在低 router margin 处改变；
- expert 切换后 hidden/logit error 是否明显跳升。

## RQ4：R64、DIAG 分别改变了哪一环？

比较：

```text
E0 ↔ E1：NVFP4 → Direct HiF4，主实验
E1 ↔ E2：R64 的作用
E1 ↔ E3：DIAG 的作用
E3 ↔ E4：DIAG + R64 的组合效应
```

但执行优先级固定：先 E0/E1，后 E2/E3，最后 E4。

---

# 2. 固定机制假设

## H1：固定 token history 下存在长程 state drift

如果成立，应观察到：

- E1/E2/E3/E4 相对 E0 的 actual-path state error 随 decode bin 增长；
- final logits KL / target-rank degradation 随位置增长；
- 即使 token history 完全固定，late stage 仍显著比 early stage 差。

这支持“状态分布随长推理轨迹持续漂移”。

## H2：主要是低 decision-margin 处的轨迹分叉

如果成立，应观察到：

- fixed-history error 不一定随长度持续增长；
- free-run 首次分叉集中在 E0 top1-top2 margin 很小的位置；
- divergence 前一两个 token 的 hidden error 可能不大，但 error direction 足以改变排名。

这支持：

> 量化误差并非简单随 token 数随机累加，而是在低 margin 决策边界处更容易触发 rank reversal。

## H3：MoE router boundary amplification

如果成立，应观察到：

- 首次 token divergence 附近 router margin 更小；
- E1/E2/E3/E4 与 E0 的 top-k expert overlap 在特定层突然下降；
- router top-k 改变后，该层或后续层的 state error 出现明显跃升。

---

# 3. 方案与 runtime 定义固定

| ID | 方案 | 正式 runtime |
|---|---|---|
| E0 | native NVFP4 | 当前 Phase-A E0 真实 vLLM NVFP4 emulation |
| E1 | Direct HiF4 | Phase-A ABI-3 optimized HiF4 runtime |
| E2 | R64-only | Phase-A ABI-3 optimized HiF4 + R64 runtime |
| E3 | adopted fusable DIAG | Phase-A adopted DIAG + ABI-3 HiF4 runtime |
| E4 | adopted fusable DIAG + R64 | Phase-A adopted DIAG + R64 + ABI-3 runtime |

固定：

- TP=2；
- KV cache=BF16；
- `enforce_eager=true`；
- `max_num_seqs=1` 用于 hook causal capture；
- 禁止 speculative decoding；
- 禁止 prefix caching；
- hook capture 时一个 engine 同时只允许一个 active request；
- E1–E4 必须检查 `runtime_abi_version == 3`；
- 不重新训练 DIAG；
- 不修改 Phase-A checkpoint / sidecar；
- 不修改 `3rdparty/vllm/**`；
- 不修改 production HiF4 kernel。

`resolve_vllm_eval_spec(...)` 继续作为 variant → 正式 checkpoint/runtime 的单一解析入口，禁止复制一套新的 materialization 规则。

---

# 4. 新架构：真实 vLLM + forced trajectory + hook

## 4.1 为什么这条路是正式方案

E0、E1、E2、E3、E4 本来就都有真实 vLLM runtime。

因此 fixed-history comparison 不需要自己实现 decoder，只需要让不同 runtime 吃同一条 E0 token history：

```text
E0 exact prompt IDs
      ↓
真实 vLLM E0/E1/E2/E3/E4
      ↓
真实 prefill
      ↓
真实 decode + KV cache
      ↓
每一步 sampler 被强制选择 E0 token
      ↓
真实下一步 KV-cache/state
```

所有这些都保持 production 数值语义：

- RMSNorm；
- RoPE；
- Attention backend；
- KV cache；
- Q/K/V/O TP partition；
- MoE fused kernel；
- TP collective/reduction；
- final norm；
- LM head。

诊断代码不重新计算这些模块。

## 4.2 强制 E0 token trajectory 的唯一允许实现

使用 vLLM 自带 custom logits processor。

当前仓库已经支持：

```python
LLM(..., logits_processors=[...])
SamplingParams(extra_args={...})
```

实现一个诊断专用 dynamic forced-token processor：

```text
forced_token_ids = E0 exact output token ids
step = len(output_ids)
target = forced_token_ids[step]
保留 target 原 logit
把其它 logit 设为 -inf
```

注意：

- 强制 processor 只改变 sampling decision；
- **原始 logits 必须在 processor 修改前由 model hook 捕获**；
- 不允许通过把整条 E0 output 拼进 prompt 做“大 prefill teacher forcing”；那会再次改变真实 decode/KV-cache 数值路径；
- 不允许每个 probe 单独重新 prefill 后把结果当成正式 fixed trajectory；正式轨迹必须一次真实 prefill 后连续 incremental decode。

## 4.3 Prompt 输入禁止重新 tokenize

hook replay 直接使用已有 E0 normalized capture 中的：

- exact `input_ids`；
- exact `output_ids`。

通过 `TokensPrompt(prompt_token_ids=...)` 进入 vLLM。

禁止：

```text
text -> tokenizer -> ids
```

重新生成 prompt token IDs。

---

# 5. worker 内 hook 的实现方式固定

## 5.1 不写 plugin，不改 worker，不改 vLLM 源码

当前 vLLM `LLM` 已提供：

```python
llm.apply_model(func)
llm.collective_rpc(...)
```

`apply_model(func)` 会在每个 TP worker 的**真实模型对象**上运行函数。

因此正式实现固定使用：

```text
LLM 初始化完成
    ↓
llm.apply_model(install_hooks)
    ↓
真实 generate
    ↓
llm.apply_model(flush_sample)
    ↓
...
    ↓
llm.apply_model(remove_hooks)
```

不新增：

- 自定义 decoder；
- custom worker；
- general plugin；
- semantic model；
- TP2 simulator。

只有当实际验证证明 `apply_model` 无法保持 hook 时，立即停止并报告；Cursor 不允许擅自切换到另一套 runtime。

## 5.2 hook 状态必须挂在真实 model 上

例如：

```text
model._hif4_longtraj_diag_state
model._hif4_longtraj_hook_handles
```

state 至少保存：

- variant；
- TP rank / world size；
- 当前 sample key；
- prompt length；
- `probe_abs_positions`；
- `abs_position -> decode_index` 映射；
- 当前 layer positions；
- CPU capture buffer；
- hook fire counters；
- output root。

禁止 module-level 全局 mutable state 作为主状态，避免 worker 多进程污染。

## 5.3 一个 active request 是硬约束

正式 hook capture：

```text
max_num_seqs = 1
一次只 generate 一个 sample
```

原因：

- 让 `positions` 与 sample 唯一对应；
- 不需要逆向解析 scheduler slot/request mapping；
- 避免为了提高吞吐又造复杂 request-routing 框架。

模型只加载一次，16 个 sample 逐个跑；variant 完成后整个进程退出。

---

# 6. probe 的位置定义必须与 predictor state 严格一致

对第 `j` 个生成 token：

```text
output token index = j
predictor absolute position = prompt_len + j - 1
```

记：

\[
p(j)=L_{prompt}+j-1
\]

- `j=0`：predictor 是 prefill 最后一个 prompt token；
- `j>0`：predictor 是上一枚已强制写入 KV cache 的生成 token。

hook 不能靠“第几次 callback”猜位置，必须使用真实 forward 的 `positions` tensor。

每个 layer 的 pre-hook：

1. 读取该层真实 `positions`；
2. 找出与 `probe_abs_positions` 相交的 row；
3. 只有命中 probe 的 row 才允许保存张量。

这样同时兼容：

- prefill；
- decode；
- 内部 batching/chunk shape 变化。

---

# 7. 正式 hook 点位

第一轮正式 capture 不做“能 hook 的全部都 hook”，只记录足以回答 H1/H2/H3 的边界。

## 7.1 每层必须记录的 actual-path 状态

对 48 层全部记录，在 probe position 保存：

### A. decoder layer 输入双状态

- `branch_in`；
- `residual_in`（layer0 为 None）。

注意：vLLM 的下一层语义不是简单把二者提前合并后再传，所以必须原样分别保存。

### B. input RMSNorm 实际输出

即 self-attention 实际消费的 `hidden_states`。

这是跨 variant 对齐时最重要的“实际下一算子输入”之一。

### C. Attention core 输出

hook：

```text
layer.self_attn.attn
```

保存进入 `o_proj` 前的真实 attention backend 输出。

该张量在 TP 下是 rank-local head shard，因此：

- 正式主分析默认不把它当 global hidden；
- 两个 rank 都保存；
- 离线按 TP rank/head shard 规则拼接后再比较；
- 如果主 H1/H2/H3 只需 branch-level，可延后读取，不影响主流程。

### D. o_proj 后输出

hook：

```text
layer.self_attn.o_proj
```

这是 row-parallel reduction 后的 Attention branch 输出，预期在 TP ranks 间 replicated。

主结果只消费 rank0，但 smoke 必须比较 rank0/rank1，证明 replicated tensor 一致。

### E. post-attention RMSNorm 输出

这是 MoE/router 真正消费的 input。

同时保存 fused RMSNorm 更新后的 residual state。

### F. router logits

hook：

```text
layer.mlp.gate
```

保存完整 128-expert router logits。

离线计算：

- top-k IDs；
- top-k overlap；
- top-k exact set；
- top-k / top-(k+1) margin；
- router KL。

不再自己实现 router forward。

### G. MoE 最终输出

hook `layer.mlp` 的最终输出，即真实 fused expert + TP reduction 后输出。

主分析消费 rank0，smoke 验证 rank 间一致。

### H. decoder layer 输出双状态

保存：

- `branch_out`；
- `residual_out`。

## 7.2 Final norm 与 logits

保存：

- final RMSNorm actual output；
- custom sampling processor 修改前的真实 full-vocab logits（只在 probe position）；
- E0 forced target token logit；
- top1/top2；
- target rank。

full-vocab logits 只保存 probe，不允许每个 decode token 都保存。

TP2 下 `LogitsProcessor` 会执行 vocab-parallel gather；当前 CUDA 路径可能只在 gather 目标 rank 返回非 `None` 的 full-vocab logits。因此：

- full-vocab logits 只从实际返回非 `None` 的 rank 保存，正常预期为 rank0；
- rank1 返回 `None` 不是 hook coverage 失败；
- smoke 必须显式记录哪个 rank 拿到 full logits，之后正式分析固定消费该 rank；
- 不允许为了让两个 rank 都有 logits 再手写 all-gather。

## 7.3 vLLM hook 返回值解析必须按真实接口写死

Cursor 不允许把所有 module hook output 都假设成单个 Tensor。当前 Qwen3-MoE/vLLM 路径至少要按以下规则解析：

| hook module | 真实输出语义 | 保存内容 |
|---|---|---|
| `input_layernorm`，layer0 residual=None | Tensor | normalized hidden |
| `input_layernorm`，layer1+ residual!=None | `(normalized_hidden, updated_residual)` | 两个都保存 |
| `self_attn.attn` | Tensor | rank-local attention core output |
| `self_attn.o_proj` | `(output, bias_or_none)` | 只取 `output` |
| `post_attention_layernorm` | `(normalized_hidden, updated_residual)` | 两个都保存 |
| `mlp.gate` | `(router_logits, bias_or_none)` | 只取 `router_logits` |
| `mlp` | Tensor | TP reduction 后 MoE final output |
| `Qwen3MoeDecoderLayer` | `(branch_out, residual_out)` | 两个都保存 |
| final `model.norm(branch,residual)` | `(final_hidden, updated_residual)` | 保存 `final_hidden`，residual 只用于一致性检查 |
| model logits processor | `Tensor or None` | 仅非 None rank 保存 raw full logits |

hook 实现必须对 tuple 长度、Tensor dtype/shape 做 assert；接口与表不一致直接失败，禁止静默取 `output[0]` 兜底。

## 7.4 默认不 hook 的东西

第一轮禁止主动加入：

- 每层完整 Q/K/V；
- RoPE 后 Q/K；
- 每个 expert W13/W2 输出；
- quantizer 内部 scale/QDQ tensor；
- NCCL collective 内部 buffer。

原因：这些属于**frontier drill-down**，不是验证长轨迹失稳的第一层证据。

只有当主边界显示第一处显著误差落在 Attention 或 MoE 某个分支后，才对少量 `(sample, token, layer)` 增开深钻 hook。

---

# 8. 深钻 hook：只在确定 frontier 后启动

正式主 capture 完成后，如果需要进一步定位第一处误差来源，才运行 drill-down。

## 8.1 Attention drill-down

仅对选中的少量 frontier：

- qkv_proj output；
- q_norm output；
- k_norm output；
- RoPE 后 Q/K；
- V；
- attention backend output；
- o_proj output。

两个 TP rank 都保存。

目标：区分误差第一次显著出现于：

```text
quantized QKV projection
→ q/k norm
→ RoPE
→ attention/KV
→ o_proj
```

## 8.2 MoE drill-down

仅对选中的 frontier：

- router logits；
- selected expert IDs / routing weights（优先从真实 runtime 可用状态读取；若模块接口不暴露，只从 router logits按当前 Qwen3 routing 定义离线重建 ID，用于统计，不重算 expert output）；
- fused MoE input；
- fused MoE output。

不重新写 Python expert loop。

如必须看单 expert，复用已有 frozen-input operator parity 工具做**局部机制辅助**，不能把它混进正式长轨迹主结果。

---

# 9. 显存、CPU 内存和文件生命周期

这是新框架的硬约束，防止 hook 本身把 30B runtime 搞崩。

## 9.1 GPU 上绝不缓存长轨迹 tensor

hook 中允许：

```text
真实 tensor
  ↓ detach
只 index probe row
  ↓ clone/copy
转 CPU
```

hook state 中禁止持有：

- 原始 GPU tensor reference；
- 完整 sequence hidden；
- 完整 KV cache 副本；
- 跨 token GPU capture list。

保存完成后 GPU 引用立即释放。

## 9.2 probe-only capture

正式 16 sample 仍采用原 probe 策略：

- bins `[0,128)`；
- `[128,512)`；
- `[512,2048)`；
- `[2048,8192)`；
- `[8192,+∞)`；
- 每个有效 bin 4 个均匀 probe；
- 各 variant first divergence 的 `[-2,-1,0,+1]`；
- 最大 decode index=12287。

禁止保存 48 层 × 每一个 token 的 hidden tensor。

## 9.3 CPU buffer 按 sample flush

每个 sample 完成后：

```text
rank-local CPU buffer
   ↓
torch.save / safetensors 落盘
   ↓
clear()
```

不允许把 16 条 sample 全部放内存后统一保存。

## 9.4 rank 独立写文件，禁止跨 worker 回传大 tensor

`apply_model()`/`collective_rpc()` 只返回：

- hook count；
- fire count；
- flush status；
- 文件路径；
- 小型统计。

大 tensor 由 worker 直接写共享文件系统：

```text
hooks/<variant>/<sample_key>/rank0.pt
hooks/<variant>/<sample_key>/rank1.pt
```

避免 RPC 序列化大 tensor。

## 9.5 每个 variant 独立 OS 子进程

固定：

```text
E0 hook capture = 一个子进程
进程退出
E1 hook capture = 新子进程
进程退出
...
```

每个 variant 内复用同一个 LLM engine 跑该 variant 的所有 sample。

variant 完成后不尝试“优雅地在 Python 进程中换模型”，直接让进程退出，保证释放：

- model weights；
- vLLM engine；
- CUDA context；
- KV cache；
- hook handles；
- NCCL process group。

这是显存清理的正式方式。

## 9.6 不在 token 循环里 `torch.cuda.empty_cache()`

禁止每 token / 每 layer 手工 `empty_cache()`。

它既不能解决真实引用泄漏，还会严重扰乱 allocator/runtime。

只有诊断测试需要确认 Python 引用释放时可以在**进程结束前**调用一次；正式释放以子进程退出为准。

---

# 10. 新代码目录

保留现有真实 free-run 基础设施，新增：

```text
Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/
  real_vllm_hooks/
    __init__.py
    forced_trajectory.py
    hook_spec.py
    hook_state.py
    worker_hooks.py
    build_llm.py
    run_variant_hook_capture.py
    run_hook_matrix.py
    validate_capture.py
    compare_hook_states.py
    build_drilldown_plan.py
    run_drilldown.py
    prefix_rescue_poison.py
    summarize_long_trajectory.py
```

职责固定：

## `forced_trajectory.py`

- dynamic E0 forced-token logits processor；
- 从 `output_ids` 长度解析下一 E0 token；
- 越界直接抛错；
- 不保存/修改原始 logits 统计。

## `hook_spec.py`

定义唯一 hook 点位和 capture level：

- `core`；
- `attention_drilldown`；
- `moe_drilldown`。

禁止 runner 自行拼 hook 名称。

## `hook_state.py`

worker model 上的诊断状态结构。

## `worker_hooks.py`

只定义可以被 `llm.apply_model(...)` 调用的顶层函数：

- `install_hooks(model, ...)`；
- `begin_sample(model, ...)`；
- `flush_sample(model)`；
- `remove_hooks(model)`；
- `inspect_hook_state(model)`。

## `build_llm.py`

统一把 `resolve_vllm_eval_spec()` 转成直接 `vllm.LLM(...)` kwargs。

必须逐字段复刻当前正式运行参数，而不是自己猜：

- model path；
- TP2；
- BF16 KV；
- `enforce_eager`；
- E0 `linear_backend=emulation`；
- E0 `moe_backend=emulation`；
- E0 NVFP4 activation-scale additional config；
- E1–E4 HIF4 runtime spec env/additional config；
- `max_model_len=40960`；
- `max_num_batched_tokens` 与当前 ABI3 runtime 一致；
- prefix caching disabled；
- `max_num_seqs=1`。

## `run_variant_hook_capture.py`

一个 variant 一个进程：

1. resolve runtime；
2. build LLM；
3. install hook；
4. 顺序执行 sample；
5. verify forced tokens；
6. flush；
7. remove hooks；
8. 写 manifest；
9. 正常退出。

## `run_hook_matrix.py`

只做 GPU pool + variant subprocess 调度。

正式顺序固定：

```text
E0 → E1 → E2 → E3 → E4
```

初次 smoke 不并行；正式验证稳定后可以同时跑两个 variant，每个 variant 占 2 GPU，但不得为了并行改请求 batching。

## `validate_capture.py`

实现所有真实性门禁。

## `compare_hook_states.py`

只做离线 tensor/stat comparison，不执行模型。

---

# 11. 现存框架清理计划

用户已经明确：旧 semantic replay 精度存在根本问题，没有必要继续留存为当前框架。

因此 Cursor 在新 hook smoke 通过后执行清理；不是“并存两套”。

## 11.1 必须删除的旧实验代码

删除：

```text
experiments/long_trajectory_stability/causal_replay.py
experiments/long_trajectory_stability/run_semantic_matrix.py
experiments/long_trajectory_stability/fused_semantic_adapter/
experiments/long_trajectory_stability/vllm_semantic_alignment/
experiments/long_trajectory_stability/tp2_semantic_alignment/
experiments/long_trajectory_stability/shared_attention_semantic_alignment/
```

`run_pipeline.py` 如果仍调用 semantic replay，不保留兼容分支：直接重写成 real-vLLM-hook pipeline 或删除后由新 runner 取代。

`summarize.py` 如果依赖 `semantic/` 输出，同样不做兼容：由 `real_vllm_hooks/summarize_long_trajectory.py` 取代。

`README.md` 必须重写，禁止继续出现“semantic replay 正式主路径”的执行命令。

## 11.2 必须删除的旧 semantic tests

删除对应：

```text
tests/long_trajectory_stability/fused_semantic_adapter/
tests/long_trajectory_stability/vllm_semantic_alignment/
tests/long_trajectory_stability/tp2_semantic_alignment/
tests/long_trajectory_stability/shared_attention_semantic_alignment/
```

新增 tests 只验证真实 hook infra，不验证 semantic parity。

## 11.3 可以保留的代码

保留：

- `capture_main.py`；
- `launch_capture.py`；
- `trajectory_io.py`；
- `normalize_capture.py`；
- `compare_free_runs.py`；
- `build_probe_plan.py`；
- `prepare_analysis.py`（若只依赖 free-run/probe，可小幅清理）；
- `gpu_pool.py`；
- 纯 metric helper；
- `nvfp4_operator_parity/` 作为单算子辅助诊断。

## 11.4 旧结果清理

真实 vLLM free-run 数据继续保留：

```text
trajectory_stability_formal/free_run/
trajectory_stability_formal/normalized/
trajectory_stability_formal/analysis/free_run_summary.json
trajectory_stability_formal/analysis/divergence_events.jsonl
trajectory_stability_formal/analysis/probe_plan.json
```

保留：

```text
e0_nvfp4_operator_parity/
trajectory_stability_smoke_e0_tp1_control/
```

因为它们是实际 runtime/frozen operator 的历史证据。

删除或移出正式结果树的旧 semantic 输出：

```text
trajectory_stability_formal/semantic/
trajectory_stability_smoke/semantic/
trajectory_stability_smoke_fused_semantic/
trajectory_stability_smoke_vllm_semantic/
trajectory_stability_smoke_tp2_semantic/
trajectory_stability_smoke_shared_attention/
```

删除前只提炼一份小型复盘：

```text
results/long_trajectory_stability/LEGACY_SEMANTIC_REPLAY_POSTMORTEM.md
```

只记录：

- 为什么旧方法不可信；
- residual/RMSNorm/fused-MoE/TP/sequence-shape 教训；
- 哪些旧结论不能再用于正式机制解释。

不复制大 tensor，不做 archive 目录。

## 11.5 旧计划处理

旧 semantic 对齐 plans 不删除历史文件，但文件头增加：

```text
状态：已被 2026-09-01-qwen3-30b-hif4-long-trajectory-real-vllm-hook-replan-cn.md 取代。
仅用于历史问题复盘，不再执行。
```

至少标记：

- `2026-08-31-qwen3-30b-hif4-long-trajectory-stability-diagnostic-plan-cn.md`；
- `2026-08-31-e0-vllm-semantic-parity-triage-cn.md`；
- `2026-09-01-long-trajectory-fused-moe-semantic-alignment-cn.md`；
- `2026-09-01-long-trajectory-vllm-residual-semantic-alignment-before-tp2-cn.md`；
- `2026-09-01-qwen3-30b-hif4-format-conversion-error-accumulation-livecodebench-causal-plan-cn.md` 中依赖 semantic replay 的部分。

---

# 12. 新框架正确性门禁

新的门禁不再是“semantic E0 对 vLLM top1 parity”，因为根本没有第二套 decoder。

必须验证的是：**hook 与 forced-token 机制本身不会改变真实 vLLM。**

## 门禁 A：无 hook 的 forced trajectory 必须精确

对 E0 smoke sample：

1. 使用 exact E0 prompt IDs；
2. forced token ids 使用 E0 output IDs；
3. `LLM.generate` 输出 token IDs 必须与 forced IDs **逐 token exact**；
4. 实际长度必须等于目标长度；
5. 不允许提前 EOS/stop；
6. 不允许 decode→retokenize。

失败立即停止。

## 门禁 B：hook neutrality

中立性拆成两个互补测试，不能只在 forced trajectory 下看 token。

### B1：真实 greedy no-hook vs core-hook

同一个 E0 prompt，**不强制 token**，各跑一次 deterministic greedy：

```text
Run A：真实 E0 greedy，不安装 hook
Run B：真实 E0 greedy，安装 core hook
```

固定相同 TP2 / KV BF16 / eager / sampling 参数，比较至少前 256 generated tokens（若样本提前自然结束则比较完整输出）。

要求：

- exact generated token IDs 完全一致；
- first divergence 必须为 None。

这一项证明 core hook 的 tensor 读取/CPU copy 没有改变真实自由生成轨迹。

### B2：forced minimal-logits-hook vs forced core-hook

因为 no-hook 无法读取 raw logits，所以 raw-logit neutrality 用两种 hook 强度比较：

```text
Run C：forced E0 trajectory，只安装最小 logits-capture hook
Run D：forced E0 trajectory，安装完整 core hook + 同一个 logits-capture hook
```

要求：

- forced output IDs 都逐 token exact；
- 每个 probe 的 raw full-vocab logits、top1/top2、target logit、target rank 在 C/D 间一致；
- 同环境 deterministic 时 full logits 优先要求 exact；若存在可重复的底层非 exact，只能以 Task 5 的 E0 repeatability noise floor 为依据，不得自行放宽。

如果 B1 或 B2 失败，禁止正式运行。

## 门禁 C：hook coverage

对 48 层：

- 每个 configured module 都必须成功注册；
- probe position 每层都必须触发预期记录；
- layer index/module name 不得缺失或重复；
- hook handle count 与 spec 完全一致；
- 第二次 install 必须直接报错，禁止重复挂 hook。

## 门禁 D：TP rank 一致性

对预期 replicated tensor：

- `o_proj` output；
- post-attention norm output；
- router logits；
- MoE final output；
- branch/residual layer output；

smoke 中比较 rank0/rank1。

若不一致：

- 先确认该 tensor 实际是否应该 replicated；
- 不允许擅自只信 rank0 掩盖问题。

对明确 rank-local 的 attention core output，不要求一致。

## 门禁 E：位置语义

至少显式验证旧问题点：

`n346/24`：

- `decode_index=24`；
- predictor `abs_pos=369`；
- 正式 hook 必须在真实 incremental decode 处理 position 369 时捕获；
- 该点 E0 原始 logits top1 必须是 `24301`。

同时验证：

- `n159/10`；
- `n346/12`。

不能再出现“用更长 sequence 的同 abs_pos 替代真实 decode”这种路径。

## 门禁 F：E0 repeatability

同配置、同 exact forced trajectory，E0 独立运行两次。

记录所有 probe：

- layer state exact fraction；
- max_abs；
- rel-L2；
- logits top1/target rank。

这给后续 E0↔E1 差异建立真实 runtime noise floor。

如果 E0 自身不可重复到足以支持机制分析，先报告，不允许直接把运行噪声解释成量化误差。

---

# 13. 测试文件

新增：

```text
Native_NVFP4_HiF4_Linear_Puncture/tests/long_trajectory_stability/real_vllm_hooks/
  test_forced_trajectory.py
  test_hook_spec.py
  test_probe_position_mapping.py
  test_capture_schema.py
  test_worker_hook_lifecycle.py
```

CPU/static tests 只验证：

- forced token index；
- 越界 fail-fast；
- decode_index ↔ abs_pos 映射；
- probe mask；
- schema；
- 重复 install 拒绝；
- flush 后 buffer 清空。

真实 CUDA/vLLM smoke 单独运行，不做伪 module fallback。

---

# 14. 数据规模与 probe 计划

## 14.1 旧 64-sample free-run 的角色：保留现象证据，但不再直接作为 hook canonical trajectory

当前已有 `trajectory_stability_formal` 的 64-sample E0–E4 free-run：

- E1 median first divergence=14；
- E2=19；
- E3=19；
- E4=17；
- 四个 variant 相对 E0 均最终分叉。

这些结果来自**真实 vLLM**，因此保留为“端到端确实存在轨迹分叉”的历史现象证据。

但旧 free-run 由 lighteval/vLLM scheduler 批量提交多个 request，而新 hook 为了 request/position 唯一映射固定：

```text
max_num_seqs=1
一次只跑一个 active request
```

低 margin 位置可能受到 batch/GEMM shape、调度 batch 组成等数值路径影响。因此禁止默认假设：

```text
旧 batched E0 output token
==
新 single-request hook runtime 的自然 greedy token
```

这和前面已经发现的 sequence-shape SDPA 数值敏感是同一类实验控制问题。

## 14.2 新框架必须先生成 shape-matched isolated free-run

正式 hook 机制分析前，使用**与 hook capture 完全相同的直接 vLLM LLM 构造**生成一套新 isolated free-run：

- exact prompt IDs 仍复用旧 E0 normalized 64 prompts，避免换数据；
- E0–E4 每个 variant 独立进程；
- TP2；
- KV BF16；
- eager；
- `max_num_seqs=1`；
- 一次只提交一个 request；
- greedy；
- `max_new_tokens=16384`；
- 不安装 core hook；
- 不强制 token；
- 每个 sample 保存 exact generated IDs。

生成顺序：

```text
isolated E0
→ isolated E1
→ isolated E2
→ isolated E3
→ isolated E4
```

新的 first-divergence / survival / probe neighborhood **一律从这套 isolated free-run 重新计算**。

旧 batched free-run 与新 isolated free-run 还要额外比较一次：

- 每个 variant 的 first batch-shape divergence；
- E0 旧/new exact trajectory fraction；
- n159/n346 等低 margin probe 是否受 execution shape 影响。

这项比较本身不是 HiF4 机制结论，只用于量化 scheduler/batch-shape 数值敏感度。

## 14.3 Hook fixed-history canonical trajectory

正式 fixed-history hook 的 canonical E0 token 必须来自**新 isolated E0 free-run**，不能直接使用旧 batched E0 output IDs。

再从 isolated E0 的 64 条轨迹选 16 条：

- 8 条最长；
- 8 条覆盖剩余长度分布。

probe plan 重新由：

- isolated E0 trajectory；
- isolated E1–E4 first divergence；

构造。

旧 `probe_plan.json` 只允许在最初 API smoke 中临时复用 `n159/n346` 的 exact prompt IDs，正式机制结果禁止消费旧 plan 的 canonical output/probe 定义。

## 14.4 Smoke

先固定：

- 2 samples；
- 重点包含 `n159`、`n346`；
- probe 包含 `n159/10`、`n346/12`、`n346/24`、`n346/35`；
- 再加 4 个 uniform controls；
- 先只 E0；
- 门禁全过后跑 E1。

Smoke 不用于机制结论。

---

# 15. 正式输出 schema

结果根目录新建，不再叫 `semantic/`：

```text
results/long_trajectory_stability/trajectory_stability_real_vllm_hook/
  manifests/
    E0.json
    E1.json
    E2.json
    E3.json
    E4.json
  hooks/
    E0/
      <sample_key>/
        rank0.pt
        rank1.pt
        sample_manifest.json
    E1/...
  validation/
    forced_trajectory.json
    hook_neutrality.json
    hook_coverage.json
    tp_rank_consistency.json
    e0_repeatability.json
  metrics/
    E0_E1_probe_metrics.jsonl
    E1_E2_probe_metrics.jsonl
    E1_E3_probe_metrics.jsonl
    E3_E4_probe_metrics.jsonl
    router_events.jsonl
    logit_events.jsonl
  drilldown/
  analysis/
    LONG_TRAJECTORY_REAL_VLLM_REPORT.md
    LONG_TRAJECTORY_REAL_VLLM_SUMMARY.json
```

每个 tensor artifact 必须包含：

- variant；
- sample key；
- TP rank；
- layer；
- module boundary；
- decode index；
- abs position；
- dtype；
- original shape；
- capture schema version。

---

# 16. 指标定义

## 16.1 State error

对同一实际 hook boundary：

\[
\mathrm{RelL2}(x_0,x_v)=
\frac{\|x_v-x_0\|_2}{\|x_0\|_2+\epsilon}
\]

同时记录：

- cosine；
- max-abs；
- mean-abs；
- norm ratio。

禁止只看 raw NMSE。

## 16.2 Error growth

对每个 decode bin：

- median；
- mean；
- P25/P75；
- bootstrap CI（离线分析即可）。

定义 late/early ratio，但不把单一 ratio 当结论。

## 16.3 Router

从真实 router logits：

- KL(E0||variant)；
- top-k set exact；
- top-k overlap；
- top-k Jaccard；
- E0 top-k / top-(k+1) boundary margin；
- variant boundary margin；
- first router top-k change layer。

## 16.4 Logits

probe 保存 full vocab 后计算：

- KL(E0||variant)；
- JS；
- centered cosine；
- E0 target NLL；
- E0 target rank in variant；
- top1 agreement；
- top1-top2 margin；
- target logit delta。

## 16.5 Free-run divergence

继续使用 exact token IDs：

- first divergence；
- survival@128/512/2048/8192；
- divergence token 前后 `[-2,-1,0,+1]` actual-path metrics。

---

# 17. 正式执行任务顺序

Cursor 必须按以下顺序，不允许跳步。

## Task 1：先写旧 replay 失败复盘，不删代码

生成：

```text
results/long_trajectory_stability/LEGACY_SEMANTIC_REPLAY_POSTMORTEM.md
```

至少写清：

- old semantic 24/28；
- fused 26/28；
- residual/RMSNorm 的阶段性误判；
- per-prefix 发现推翻 `n346/24` TP2 attribution；
- 当前 25/28 的 semantic mismatch 不再继续修；
- 为什么 real vLLM hook 更可靠。

完成后才允许删旧框架。

## Task 2：实现最小 forced trajectory runner，不挂 hook

只做 E0 两个 smoke sample。

验收：

```text
forced output ids == E0 output ids
```

exact。

同时验证真实 incremental decode：

- prefill 只发生一次；
- decode 连续运行；
- 不通过“大 prefix prefill”替代。

## Task 3：实现 `apply_model` hook lifecycle

先只挂：

- decoder layer pre/post；
- final norm；
- logits processor output。

验证：

- 48 layer hook coverage；
- rank0/rank1 文件；
- probe abs position；
- sample flush；
- remove hooks。

## Task 4：扩展到 core hook 点

再加入：

- input RMSNorm；
- attention core；
- o_proj；
- post-attention RMSNorm；
- router gate；
- MLP/MoE final。

此时不跑 E1。

## Task 5：E0 hook neutrality + repeatability

严格执行五组：

1. E0 greedy no-hook，至少生成前 256 token；
2. E0 greedy core-hook，同样至少 256 token，与 1 做 exact trajectory comparison；
3. E0 forced + minimal logits-only hook；
4. E0 forced + full core hook + 同一 logits hook，和 3 比 raw probe logits；
5. 再独立启动一次 E0 forced full core hook，和 4 建立 repeatability noise floor。

注意：第 1/2 项用于验证 hook 不改变自由生成；第 3/4 项用于验证完整 core hook 不改变 forced-path 原始 logits；第 4/5 项用于测量真实 runtime 自身重复性。

必须生成 validation artifact。

任何一项失败：停止。

## Task 6：先建立 single-request E0 smoke canonical，不再把旧 batched E0 token 当真值

使用和 hook runner 完全相同的 `build_real_vllm()`，只是不挂 hook、不强制 token，对 `n159` / `n346` 单请求 greedy 运行。

保存新的 exact E0 output IDs，并和旧 batched E0 输出比较：

- first execution-shape divergence；
- n159/10；
- n346/12；
- n346/24；
- n346/35；
- 至少 4 个 uniform control。

此时**不要求**新 single-request raw top1 等于旧 token；如果不同，明确记为 `EXECUTION_SHAPE_NUMERIC_DIFFERENCE`，不能当 hook 错误。

然后用新的 isolated E0 output IDs 作为 smoke canonical，再跑：

```text
isolated E0 tokens
→ forced logits-only
→ forced core-hook
```

这一次才要求：

- forced output exact；
- E0 raw top1 == isolated E0 target token；
- minimal logits-only vs full core-hook raw logits 一致；
- hook-on/off neutrality 通过。

这样验证的是**同 execution shape 下观察机制的中立性**，而不是强迫新 runtime 复现旧 batched 数值路径。

## Task 7：E0/E1 smoke actual-path comparison

使用 Task 6 新生成的 single-request E0 exact tokens，不再使用旧 batched E0 output IDs。

输出逐 layer：

- input norm state error；
- attention output error；
- post-attention state error；
- router delta/top-k；
- MoE output error；
- layer out state error；
- logits error。

检查误差来自实际 E0/E1 runtime，不存在 replay parity 问题；同时要求 E0 baseline 本身在相同 execution shape 下闭环。

Smoke 通过后执行旧框架清理。

## Task 8：删除旧 semantic replay 框架

严格按第 11 节清单删除。

同时：

- 重写 `experiments/long_trajectory_stability/README.md`；
- 更新 plan README；
- 标记旧 plan 已取代；
- 全仓搜索 `run_semantic_matrix`、`causal_replay_*semantic*` 等正式入口，确保新 README/pipeline 不再引用。

## Task 9：先重建正式 shape-matched isolated free-run 与 probe plan

对旧 E0 normalized 64 个 exact prompt IDs，在新 direct-vLLM single-request runner 中重新跑：

```text
E0 → E1 → E2 → E3 → E4
```

每个 variant 独立进程，全部 `max_num_seqs=1`，一次只提交一个 request。

然后重新生成：

- isolated `free_run_summary.json`；
- isolated `divergence_events.jsonl`；
- isolated `probe_plan.json`。

同时输出一份 old-batched vs new-isolated execution-shape audit。旧 free-run 仍保留，但后续 H1/H2/H3 的 divergence neighborhood 一律使用 isolated 结果。

## Task 10：正式 E0/E1 16-sample hook capture

只做主问题：格式转换。

执行：

```text
E0 16 samples
进程退出
E1 16 samples
进程退出
```

所有 sample 使用 Task 9 isolated E0 的 exact tokens 和新 probe plan。

## Task 11：E0/E1 H1/H2/H3 分析

先形成主结论，不急着跑 E2/E3/E4。

### H1 判定

看 state/logit error 的 decode-bin 曲线。

### H2 判定

把 free-run first divergence 与：

- E0 margin；
- fixed-history state error；
- target-rank delta；

对齐。

### H3 判定

看 first router top-k change 与 error jump / first token divergence 的关系。

## Task 11：frontier drill-down

只选择实际数据中最有信息的：

- 4 个 earliest-divergence sample；
- 4 个 longest-stable/control sample；
- 每 sample 最多 2 个 token frontier；
- 每 token 最多 3 个候选 layer。

按主结果决定 Attention 或 MoE drill-down。

不得全模型重新抓 Q/K/V/expert tensor。

## Task 12：E2/E3 mechanism controls

主链明确后：

- E2 用同一 16 sample / probes；
- E3 用同一 16 sample / probes。

比较：

```text
E1 ↔ E2：R64 是否降低局部 state error、但改变 decision-sensitive direction/router？
E1 ↔ E3：DIAG 是否更稳定地减小 actual-path drift？
```

禁止换 cohort。

## Task 13：E4 组合验证

最后跑 E4，同一 cohort。

回答：

- DIAG+R64 是否在某些层降低 reconstruction/state error；
- 是否同时增加 router/decision-sensitive mismatch；
- 为什么局部收益没有线性变成端到端收益。

## Task 14：真实 vLLM prefix rescue / poison

不做 hidden-state simulator intervention。

用真实 token-level causal intervention：

### Rescue

对 E1 free-run divergence sample：

- 强制 E0 token 到 divergence 后 `K` 步；
- 然后释放 E1 自由生成；
- K 固定测试 `{1,4,16,64}`。

看能否回到 E0/正确轨迹附近。

### Poison

在 E0 中注入 E1 首次 divergence token，再释放 E0 自由生成。

看单个错误 token 是否足以导致长期不可恢复分叉。

这是验证 H2 的真实 vLLM 因果实验，不需要自建 decoder。

## Task 15：LiveCodeBench 外部验证

MMLU-Pro 建立机制后，再迁移到 LiveCodeBench。

优先 cohort：

- E0 pass；
- E1 fail；
- reasoning/code trajectory 足够长。

先 8–16 个样本，不全量 hook。

仍然：

- exact E0 prompt/output ids；
- actual vLLM forced trajectory；
- 同一 core hook；
- 同一 metrics。

目标：验证 MMLU-Pro 发现的 mechanism 是否与真实代码任务 pass→fail 对应。

## Task 16：最终报告

输出：

```text
LONG_TRAJECTORY_REAL_VLLM_REPORT.md
```

必须把“事实”“统计关联”“因果证据”分开写。

---

# 18. H1/H2/H3 的固定判定规则

## H1 支持

需要同时满足：

1. fixed E0 history 下，至少一种核心 actual state metric 随 decode bin 明显增长；
2. late vs early 差异跨多个 sample 一致，不由单样本驱动；
3. logits/target-rank 也有对应恶化趋势。

否则不能称“持续累计失稳”。

## H2 支持

需要：

1. divergence probe 的 E0 decision margin 显著低于 uniform control；
2. divergence 前 state error 可以较小但足以改变 rank；
3. prefix rescue / poison 至少一项显示 token bifurcation 具有明显路径依赖。

## H3 支持

需要：

1. router top-k change 在 divergence neighborhood 有 enrichment；
2. 存在明确 layer：router 改变后 state error / logits error 跳升；
3. 不能仅凭 router KL 非零就下结论。

---

# 19. 重要的负面结果也要保留

如果最终发现：

- fixed-history error 不随长度增长；
- router 大多数时间 exact；
- divergence 几乎完全由小 logit margin 决定；

那么正确结论应是：

> 主要机制不是“长轨迹数值误差持续累计”，而是量化扰动在少数低 margin decision point 触发离散 token bifurcation，之后由 autoregressive 条件变化导致轨迹快速分离。

不能为了符合“长轨迹失稳”标题而强行解释 H1。

---

# 20. Cursor 禁止事项

本计划执行过程中禁止：

- 修旧 semantic replay；
- 再追 28/28 semantic parity；
- 手写 TP2 shard/reduction 作为正式真值；
- 用 PyTorch SDPA 替代真实 vLLM attention；
- 用 full-prefix 大前向替代 incremental decode；
- 修改 `3rdparty/vllm/**`；
- 修改 production HiF4 kernel；
- 改 Phase-A artifact；
- 为了 hook 方便改模型数学语义；
- 多 request batching hook capture；
- 保存每 token 全层 hidden；
- hook 内长期持有 GPU tensor；
- 在 RPC 中回传大 tensor；
- 通过降低阈值/删 mismatch 让结果好看；
- 对单独 token 硬编码补丁；
- 在没有实际 frontier 证据时全量 hook Q/K/V/每 expert。

出现不能用真实 vLLM hook 解决的 API 限制时：

> 停止并报告具体限制，不允许自动退回 semantic simulator。

---

# 21. 最小执行命令形态

最终实现后，README 应只保留类似以下主流程：

```bash
conda run -n hif4 pytest -q \
  Native_NVFP4_HiF4_Linear_Puncture/tests/long_trajectory_stability/real_vllm_hooks

conda run -n hif4 python \
  Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/real_vllm_hooks/run_hook_matrix.py \
  --run_root Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/trajectory_stability_real_vllm_hook_smoke \
  --variants E0 \
  --smoke

conda run -n hif4 python \
  Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/real_vllm_hooks/validate_capture.py \
  --run_root Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/trajectory_stability_real_vllm_hook_smoke
```

只有 validation 全部 PASS 后，README 才给正式 E0/E1 命令。

不要在 plan 阶段提前虚构最终 CLI 的所有 optional 参数；实现时主 CLI 必须保持短，复杂配置写死在 `hook_spec.py/config.py` 的正式协议中。

---

# 22. 最终验收清单

新框架完成的定义不是“脚本能跑”，而是以下全部成立：

- [ ] 正式长轨迹 fixed-history 不再调用任何 semantic decoder；
- [ ] E0/E1/E2/E3/E4 都是真实 Phase-A vLLM runtime；
- [ ] exact E0 input/output IDs 直接复用；
- [ ] forced tokens 逐 token exact；
- [ ] 真实 incremental KV-cache decode；
- [ ] hook 用 `LLM.apply_model()` 安装在 TP workers 的真实模型对象；
- [ ] hook-on/off 不改变 E0 结果；
- [ ] `n346/24` 实际 predictor step top1=24301；
- [ ] 48 层 hook coverage 完整；
- [ ] rank replicated/local tensor 语义验证清楚；
- [ ] GPU 不缓存长轨迹 capture；
- [ ] 每 sample CPU flush；
- [ ] 每 variant 独立进程退出释放显存；
- [ ] 旧 semantic replay 代码和测试已经清理；
- [ ] 旧 semantic 结果不再混入正式结果树；
- [ ] E0/E1 主实验完成后才跑 E2/E3/E4；
- [ ] 最终报告区分 H1/H2/H3，并允许负面结论。

满足以上条件后，这套框架才可以作为后续 NVFP4→HiF4 长轨迹机制研究的正式基础设施。
