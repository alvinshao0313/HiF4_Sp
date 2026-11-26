> 状态：已被 2026-09-01-qwen3-30b-hif4-long-trajectory-real-vllm-hook-replan-cn.md 取代。
> 仅用于历史问题复盘，不再执行。

# 长轨迹诊断：Fused-MoE Semantic 对齐计划

> **状态：已停止执行。** 不再继续修 fused semantic replay；正式长轨迹机制实验改为真实 vLLM TP2 + forced E0 trajectory + worker hook。唯一主计划见 `2026-09-01-qwen3-30b-hif4-long-trajectory-real-vllm-hook-replan-cn.md`。

> 日期：2026-09-01  
> 前置结论：E0 operator parity 已定位到多 expert 汇聚/累加数值语义差异；QDQ、weight dequant、dense Q/K/V/O、single-expert W13/W2 在 frozen identical input 下均 exact。  
> 前置报告：`Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/e0_nvfp4_operator_parity/E0_NVFP4_OPERATOR_PARITY_REPORT.md`  
> 本轮目标：只修改长轨迹诊断脚手架，使 E0–E4 的 MoE 计算都使用与正式 vLLM runtime 同类的 fused expert 汇聚语义；先重新挑战 E0 parity，再决定是否需要 TP2 adapter。  
> 本轮禁止：修改 production runtime、修改 QDQ/scale、启动 formal、降低 0.99 parity gate、解释 H1/H2/H3。

---

## 1. 已确认事实

当前 operator parity 结果：

- scale metadata：48 层一致；
- activation QDQ：exact；
- packed weight dequant：exact；
- dense Q/K/V/O：exact；
- single-expert gate/up/down：exact；
- 第一个稳定非零来自多 expert 汇聚阶段；
- `semantic ↔ fused` 最大 `max_abs=0.1875`；
- `p4 weighted sum ↔ semantic` 本身也非零，最大 `max_abs=0.125`；
- manual TP2 row-parallel reduction 还能额外产生最大 `max_abs=0.5`。

因此根因更精确地表述为：

> **multi-expert accumulation / reduction numerical semantics difference**

不是“某个 fused GEMM 算错”，也不是 max-scale/QDQ bug。

本轮先只消掉主要的 MoE 汇聚语义差异。TP2 reduction 是否还需要补，必须由修完后的 E0 parity 数据决定。

---

## 2. 代码隔离

所有新增代码放在：

```text
Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/fused_semantic_adapter/
```

测试放在：

```text
Native_NVFP4_HiF4_Linear_Puncture/tests/long_trajectory_stability/fused_semantic_adapter/
```

新的 smoke 结果放在：

```text
Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/trajectory_stability_smoke_fused_semantic/
```

不修改：

- `3rdparty/vllm/**`；
- `experiments/e2e_diag_reconstruction/**`；
- 已有 `causal_replay.py`；
- 已有 smoke / operator-parity 结果；
- Phase-A artifact。

原 `causal_replay.py` 保留作为旧 semantic reference，便于做 before/after 对比。

---

## 3. 总体实现

新增一条独立 replay：

```text
causal_replay_fused.py
```

整体 attention、RoPE、RMSNorm、router 逻辑继续复用当前 semantic runtime；**只替换 routed MoE 的 expert execution + top-k contribution 汇聚实现**。

必须保证：

```text
router logits unchanged
selected_experts unchanged
routing_weights unchanged
only expert execution / accumulation backend changes
```

每层仍然 lazy load，不把完整 30B BF16 模型常驻显存。

---

## 4. E0 fused adapter

新增：

```text
fused_semantic_adapter/native_nvfp4_fused.py
```

直接复用 operator parity 中已经验证过的构造方式：

```text
Nvfp4QuantizationEmulationTritonExperts
```

要求：

1. 每层从原 NVFP4 checkpoint 加载 packed `w13/w2`、block scale、global scale；
2. `a13/a2` collapse 规则严格复用已验证的 vLLM emulation 语义；
3. router 仍由 `NativeQwen3MoELayerRuntime.router` 计算；
4. 得到 `selected_experts/routing_weights` 后，直接送入 fused experts；
5. 最终 MoE output 必须来自 production vLLM 的 `moe_sum` 路径，而不是 Python `index_add_`；
6. 返回结构必须与当前 `NativeMoEForward` 对齐，保留 router logits / routing weights / selected experts。

不要复制或改写 fused kernel。

---

## 5. E1–E4 fused adapter

新增：

```text
fused_semantic_adapter/hif4_fused.py
```

必须复用现有 production 函数：

```text
vllm.model_executor.layers.fused_moe.experts.hif4_emulation_moe.apply_hif4_fused_moe
```

不要自行实现第二套 HiF4 fused MoE。

### E1 direct HiF4

每 expert：

```text
master gate/up/down
→ HiF4 weight QDQ
→ w13 = cat(gate, up)
→ w2 = down
→ apply_hif4_fused_moe(use_r64=False)
```

### E2 R64-only

权重必须使用当前 semantic 定义相同的 R64 transform，再做 HiF4 QDQ；activation R64 由：

```text
apply_hif4_fused_moe(..., use_r64=True)
```

完成。

不得对 activation 重复做 R64。

### E3 / E4 fusable DIAG

从当前 adopted artifact 读取每层 DIAG。

先增加硬门禁：

```text
D_GU must be identity for all 48 layers
```

这是当前 Phase-A artifact 已知语义，但代码必须实际检查。

若任意层 `D_GU != 1`，立即停止并报告，不自行扩展 adapter。

当前 D_GU identity 条件满足时：

- gate weight：按现有 fusable transform 构造；
- up weight：包含 per-expert `D_UD` 输出变换；
- down weight：包含对应 `D_UD` 输入补偿；
- transformed weight 做 HiF4 QDQ；
- gate/up 拼接成 production `w13` 排列；
- E3 `use_r64=False`；
- E4 `use_r64=True`；
- activation QDQ、SwiGLU、W2、routing-weight multiply、moe_sum 全部由现有 HiF4 fused path 完成。

必须写一个最小测试验证 `w13` 顺序是 `[gate, up]`，不能反。

---

## 6. Router 不能被 adapter 改变

E0：继续用原 native router。

E1–E4：继续使用当前 student semantic 的 router 计算逻辑，包括：

- fusable router weight transform；
- `D_GU` router input 语义；
- softmax；
- top-k；
- top-k renormalization。

本轮只在 router 已经输出：

```text
router_logits
selected_experts
routing_weights
```

之后切换到 fused MoE。

每个 replay 必须额外记录：

```text
router_backend = semantic_existing
moe_backend = native_nvfp4_fused | hif4_fused
```

---

## 7. 第一关：原 28 probe E0 parity

先只跑 E0，不启动 E1–E4。

输入仍使用原 smoke：

```text
trajectory_stability_smoke/analysis/probe_plan.json
trajectory_stability_smoke/normalized/E0.jsonl
```

输出到新的：

```text
trajectory_stability_smoke_fused_semantic/semantic/E0/
```

要求：

- 原 28 个 probe 必须 **28/28 top1 match**；
- 也就是实际 parity=1.0；
- 不因为原 gate 是 0.99 就接受 27/28，27/28 实际本来就低于 0.99。

同时输出 before/after 表：

| probe | old semantic top1 | fused semantic top1 | TP2 vLLM token | fixed? |
|---|---|---|---|---|

重点必须明确原 4 个 mismatch：

- n159 / 10
- n346 / 12
- n346 / 24
- n346 / 35

其中 n346/35 只作为 semantic parity 点，不再作为 TP1/TP2 operator 因果证据。

若 28/28 不通过：

- 停止；
- 不跑 E1–E4；
- 进入第 10 节的 TP2 判定；
- 不修改 production。

---

## 8. 第二关：扩展 E0 parity stress test

28 个 probe 通过后，不立刻恢复正式 smoke。

新增：

```text
fused_semantic_adapter/build_e0_stress_probe_plan.py
```

只使用现有 4 条 E0 × 1024 greedy 轨迹，不新增 free-run。

总 probe 目标约 256 个，规则固定：

每条样本 64 个：

1. decode `[0,128)`：每 4 token 一个，共 32；
2. decode `[128,512)`：均匀 16 个；
3. decode `[512,1024)`：均匀 16 个；
4. 如果与上述重复，去重后用同 bin 邻近未选位置补齐到 64。

另外必须包含：

- 原 28 probe；
- TP1↔TP2 first divergence 前后 `[-2,-1,0,+1]`，若范围合法。

去重后可以多于 256，不允许为了凑固定数字删除重点 probe。

重新 teacher-forced replay E0。

要求输出：

```text
E0_FUSED_STRESS_PARITY.json
E0_FUSED_STRESS_PARITY.md
```

至少统计：

- overall top1 parity；
- 每个 decode bin parity；
- low-margin probe parity；
- mismatch 位置；
- mismatch 的 TP2 top1-top2 margin；
- 是否靠近 TP1↔TP2 first divergence。

通过标准：

```text
overall >= 0.99
且原 28 probe = 1.0
且没有某个 decode bin 出现系统性 parity 崩溃
```

“系统性崩溃”定义为某个 bin parity <0.95。

若失败，停止，不启动 E1–E4。

---

## 9. 通过两关后，恢复 E1–E4 fused-semantic smoke

新增独立 launcher：

```text
fused_semantic_adapter/run_fused_semantic_matrix.py
```

执行顺序：

1. E0 28-probe gate；
2. E0 stress gate；
3. E1–E4；
4. 新的 summarize。

E1–E4 必须全部使用 fused MoE adapter，不能出现：

```text
E0 fused
E1-E4 Python index_add
```

这种 backend 不对称。

结果写入新的 run root，不覆盖旧 smoke。

### 必须额外输出 old-vs-new semantic 差异

对 E1–E4，在相同 probe 上比较：

```text
old semantic hidden/router/logit
vs
fused semantic hidden/router/logit
```

用于判断原先 Python expert loop 对不同 variant 的影响是否一致。

注意：router logits 理论上在同一层输入完全相同时不受 MoE backend 直接影响，但前层 hidden 改变后会间接受影响，因此仍需记录。

---

## 10. TP2 adapter 的触发条件

本轮不要提前实现 TP2 adapter。

只有以下任一条件出现才触发下一轮设计：

### 条件 A

原 28 probe fused 后仍 <1.0。

### 条件 B

扩展 stress parity <0.99。

### 条件 C

剩余 mismatch 明显满足：

```text
TP1 token == fused-semantic top1
TP2 token != fused-semantic top1
```

或集中在已知 TP1↔TP2 divergence neighborhood。

满足后，本轮只输出：

```text
TP2_ADAPTER_REQUIRED = true
```

并列出剩余 mismatch。

**不要在同一轮继续写 TP2 adapter。**

原因：届时可以根据剩余差异究竟来自 attention row-parallel、MoE TP shard/reduce，还是二者共同作用，再设计最短路径，不提前过度实现。

---

## 11. 如果 fused adapter 已经通过 parity

若：

```text
28-probe = 1.0
stress >= 0.99
```

则说明当前长轨迹 semantic replay 已经在决策层面足够贴近 TP2 E0，至少 smoke 范围内没有证据要求立即实现 TP2 adapter。

这时允许完成 E1–E4 semantic smoke 和 summarize。

但仍然：

- 不启动 64×16k formal；
- 先把新 smoke 报告给用户；
- 由用户决定是否进入 formal。

---

## 12. 测试要求

使用 `hif4` conda 环境。

只写最小必要测试：

1. E0 fused builder 的 `a13/a2` 与 operator parity 已验证值一致；
2. HiF4 `w13=[gate,up]` 顺序；
3. E1 identity-D weight materialization 与现有 direct HiF4 semantic 单 expert 输出一致；
4. E2 R64 不发生 activation double-rotation；
5. E3/E4 D_GU identity gate；
6. frozen routing 下 fused adapter 返回 shape/dtype/router metadata 正确；
7. stress probe builder deterministic、无越界、包含所有 mandatory positions。

不新增生产依赖。

---

## 13. 最终报告

生成：

```text
Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/trajectory_stability_smoke_fused_semantic/analysis/FUSED_SEMANTIC_ALIGNMENT_REPORT.md
```

固定回答：

1. 原 28 probe：old parity / new parity；
2. 原 4 个 mismatch 是否全部修复；
3. stress probe 数量与 overall/bin parity；
4. 是否仍存在 mismatch；
5. 剩余 mismatch 是否更接近 TP1 还是 TP2；
6. E1–E4 fused semantic smoke 是否完成；
7. old semantic vs fused semantic 对 E1–E4 的 hidden/router/logit 改变量；
8. `TP2_ADAPTER_REQUIRED = true/false`；
9. production runtime modified = false；
10. formal started = false。

完成报告后停止。
