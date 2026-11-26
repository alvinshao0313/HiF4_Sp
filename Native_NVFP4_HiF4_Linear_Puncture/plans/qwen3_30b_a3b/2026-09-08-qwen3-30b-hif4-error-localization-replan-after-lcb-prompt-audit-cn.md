# Qwen3-30B-A3B NVFP4→HiF4 误差定位后续实验重计划

> 日期：2026-09-08  
> 模型：`nvidia/Qwen3-30B-A3B-NVFP4`  
> 状态：LiveCodeBench formal prompt 协议错位审计后的最新计划  
> 正式数值路径：真实 vLLM TP2 actual path  
> 原则：不恢复 semantic replay，不新建第二套 decoder/runtime，不继续验证已被现有结果否定或显著削弱的旧假设。

---

# 0. 为什么必须重计划

2026-09-07 版本计划建立在一个后来被仓库真实证据推翻的前提上：旧 Phase-A LiveCodeBench 的 E0/E1/E2/E3 formal run 可以作为同一 prompt 协议下的格式转换精度对照。

最新审计已经明确：

```text
FORMAL_PROMPT_PROTOCOL_ALIGNED = false
FORMAL_LCB_E0_E3_REUSABLE(format-only) = false
```

证据：

`results/long_trajectory_stability/format_conversion_lcb_mechanism_go_20260907/00_audit/FORMAL_PROTOCOL_MISMATCH.md`

其中：

```text
E0 formal input：Qwen chat wrapper + problem
E1/E2/E3 formal input：problem only
```

因此旧的：

```text
E0 95/175
E1 34/175
E0 Pass / E1 Fail = 61
```

只能保留为历史 observed facts，不能再解释成 NVFP4→HiF4 单独造成的 34.9pp 精度下降。

同时，新的 matched-prompt isolated greedy 第一轮 32 题得到：

```text
mechanism_regression = 1
mechanism_robust     = 25
mechanism_e0_fail    = 6
```

而这批样本本来大量来自旧 formal `E0 Pass / E1 Fail` interest seed。这强烈说明旧 formal regression 中 prompt 协议变化至少是重大混杂因素。

因此，后续实验必须先重新回答：

> **在完全相同 prompt token IDs、相同 execution shape、相同 sampled generation 协议下，E1 Direct HiF4 到底还会不会造成实质 LiveCodeBench 精度损失？**

只有这个问题得到肯定答案，才值得继续昂贵的“精度损失因果机制”实验。

---

# 1. 本计划取代关系

本计划取代：

`2026-09-07-qwen3-30b-hif4-lcb-format-conversion-mechanism-go-plan-cn.md`

作为当前后续实验唯一执行计划。旧计划和旧结果保留作历史记录，不删除。

继续复用：

```text
real_vllm_hooks/
真实 vLLM TP2
single-request incremental KV-cache decode
forced E0 trajectory
worker 内 hook
hook neutrality / coverage / TP artifact
official LCB judge helper
```

禁止恢复：

```text
causal_replay.py
run_semantic_matrix.py
semantic decoder
手写 attention replay
Python expert loop
TP2 simulator
```

---

# 2. 已经成立的结论：后续直接复用，不再重复验证

## 2.1 固定 history 下误差随 decode 长度持续单调积累 —— 不支持

真实 vLLM MMLU-Pro forced-history 已经显示：final hidden rel-L2、logit KL、target rank 都没有稳定随 decode 长度恶化。

因此取消：

- early/mid/late 长度趋势大扫描；
- “误差随 token 数单调累积”作为主假设；
- 为验证长度效应而额外增加超长 forced trajectory。

后续只研究：局部数值扰动在哪些层/分支产生，以及何时触发离散 trajectory bifurcation。

## 2.2 first divergence 必然对应低 top1/top2 margin —— 不作为 LCB 主假设

MMLU 上存在一定低裕度签名，但旧 LCB 12 条没有复现。因此取消：

- 大规模低-margin token 扫描；
- 把 margin 小直接当危险 token 判据；
- 以低 margin 为主线的统计模型。

只在真正 clean mechanism-regression 的 first divergence 上计算实际 E0 token `a` vs E1 token `b` 的 signed decision margin，以及 final-norm hidden difference 对 `w_a-w_b` 的投影。

## 2.3 router top-k change 在 first divergence 富集 —— 不支持作为主机制

已有真实 vLLM 结果显示：first-divergence router top-k change 没有高于均匀 control；并且部分明显 MoE output jump 在 top-k 不变时仍发生。

因此取消：

- router change enrichment 大扫描；
- 把所有 MoE error 都归因到 routing change；
- 独立的 router-first 假设链。

Router 只作为 MoE 局部误差分解的一部分：`q_M = q_expert + q_router`。仅在 same-input MoE puncture 已显示该层重要后才做 frozen-router。

## 2.4 token-level exact rejoin rescue/poison —— 已完成，不再重复

已有 MMLU/旧 LCB 结果已说明强路径依赖，但“精确重新贴回 E0 token 序列”不是任务正确性指标。

因此取消重复的 `K={1,4,16,64}` exact token rejoin rescue/poison。若后面需要因果干预，直接做 hidden/state reset，并先看 next-token decision，再按需看最终官方 judge。

## 2.5 MoE 是当前最优先的真实分支 —— 保留

旧真实 vLLM frontier 中，44/48 case 的主要增长落在 fused MoE 路径，且其中部分 top-k 完全稳定。

因此新的局部因果实验优先级固定：

```text
MoE same-input puncture
    > unresolved Attention/QKV/O drilldown
```

不再一开始同时抓所有 Q/K/V/O/Gate/Up/Down 全量 tensor。

---

# 3. 新的核心研究问题

## RQ0：同 prompt 下是否真的存在任务级格式转换损失？

必须用完全相同 input token IDs、execution shape、sampled generation 参数、官方 LCB judge，重新比较 E0 Native NVFP4 与 E1 Direct HiF4。

在 RQ0 没有闭环前，禁止继续声称“HiF4 让 LCB 大面积 thinking 不收束”。

## RQ1：如果确实存在 clean task loss，局部格式误差主要在哪里产生？

重点回答：

1. actual path 上哪个 layer/boundary 出现最大增长；
2. MoE 是否仍是主要增长位置；
3. same-input 下 MoE 的纯格式新注入 `q` 有多大；
4. total error `e` 中多少来自 local injection `q`，多少来自 upstream propagation `p`；
5. `q_M` 中 expert compute 与 router change 各是什么角色。

对同一 E0 MoE 输入 `x0`：

```text
y0      = M_E0(x0)
y1_same = M_E1(x0)
y1_act  = M_E1(x1)
q = y1_same - y0
p = y1_act  - y1_same
e = y1_act  - y0
```

必须验证 `e ≈ q + p`，并记录 `||e||、||q||、||p||、cos(q,p)、closure error`。

若 frozen-router identity 通过，再拆 `q_M = q_expert + q_router`。

## RQ2：局部数值误差如何变成真正的 token decision / task failure？

只对 clean mechanism-regression 做。不再做全局 H1/H2/H3 竞争模型。

主链：

```text
局部 MoE q/p
→ downstream hidden direction
→ E0 token a / E1 token b 的 decision margin 翻转
→ 必要时少量 full-prefix state reset/injection
```

## RQ3：R64 / DIAG 是否修复已经确认的那条机制？

只有 RQ0 确认存在 clean task loss，且 RQ1/RQ2 找到稳定机制后才解锁。默认不启动。

---

# 4. 当前正在运行的 55 条 matched-prompt greedy 如何处理

当前 run：

`results/long_trajectory_stability/format_conversion_lcb_mechanism_go_20260907/`

截至本计划制定时：

```text
E0 = 55/55
E1 = 41/55
E1 remaining = 14
```

55 条由 30 个 old formal_regression interest seeds + 25 个 old formal_robust interest seeds 组成。由于 old formal prompt mismatch，这 55 条是有偏的 protocol-audit cohort，不能估计全 175 的 clean regression rate。

本轮只允许它完成：

1. 补完 E1 55/55；
2. 用官方 LCB judge 重新分类 matched-prompt greedy E0/E1。

必须输出：

```text
MATCHED_PROMPT_GREEDY_AUDIT.md
MATCHED_PROMPT_GREEDY_AUDIT.json
```

并报告 mechanism_regression / mechanism_robust / mechanism_e0_fail，以及 old formal_regression seeds 在 prompt 对齐后分别变成哪一类。

**到这里设置 Stop Gate。** 旧 detached orchestrator 中 `run_go.py --through report` 后面的 feature/core/puncture/state/E2/E3 不得再无条件自动执行。

---

# 5. Task 0：截断旧自动流程，保留当前 E1 补跑

执行新计划时先检查 `run_continue_e1_then_report.sh`、E1 generation 和 `run_go.py --through report`。

- 若 E1 仍在补跑：允许补齐 55/55，但禁止随后进入旧 heavy tasks。
- 若 E1 已补齐而旧 heavy chain 尚未启动：只运行官方 greedy judge，然后进入新 Task 1。
- 若旧 heavy pipeline 已启动：停止尚未完成的 feature/core/puncture/state/E2/E3 downstream job；已经写出的 artifact 不删除，标记 `SUPERSEDED_BY_20260908_REPLAN`，除非新计划后面重新解锁，否则不纳入正式结论。

---

# 6. Task 1：完成 matched-prompt greedy protocol audit

复用 `03_isolated/E0.jsonl`、`03_isolated/E1.jsonl`、`02_cohort/benchmark_cohort.jsonl`。官方 judge 必须使用 lighteval 原实现。

除总数外，必须给 old-formal → matched-greedy 转移矩阵：

| old formal label | greedy E0 Pass/E1 Fail | greedy both Pass | greedy E0 Fail |
|---|---:|---:|---:|
| old formal regression | ... | ... | ... |
| old formal robust | ... | ... | ... |

这一步只用于定量说明旧 formal regression 现象在 prompt 对齐后有多少仍存在。55 条不是随机抽样，不得外推完整 LCB accuracy。

完成后直接进入 Task 2，不启动旧 heavy mechanism chain。

---

# 7. Task 2：重新建立统一端到端基线：复用 E0，重跑 LCB E1 + MMLU-Pro300 E1–E7

这是新计划最重要的一步。与其继续几十个假设实验，先把端到端基线彻底重置：

```text
LiveCodeBench：复用已审计通过的正式 E0，全量重跑 E1 175
MMLU-Pro：复用已审计通过的正式 E0，统一协议重跑 E1 / E2 / E3 / E4 / E5 / E6 / E7 各 300 samples
```

LCB 用于确认纯 NVFP4→HiF4 是否真的产生显著长生成任务损失；MMLU-Pro300 用于重新建立所有 Phase-A 正式方案在统一协议下的短程推理基线与算法横向对照。

**MMLU-Pro300 的 E1–E7 统一重跑是无条件必做项，不受后面的 LCB Gate A/B/C 影响；E0 直接复用旧正式结果。**

## 7.1 LCB：E0 复用，只重跑 E1

旧正式 E0 已经通过审计：175/175 使用正确 Qwen chat-wrapper prompt，raw cache 与 formal details 对齐，official judge 可复现，因此 E0 不需要重新跑。

新的 E1 必须直接复用旧 E0 的 exact input IDs，并完整复刻旧 E0 benchmark 协议与 execution shape。旧 E1/E2/E3 的 prompt-only 结果继续只作为历史 observed facts。

## 7.2 exact input IDs

优先从 00_audit 已确认的 E0 raw cache 提取 175 题 exact input IDs，构建唯一 `clean_lcb_prompt_manifest.jsonl`。

禁止：E1 自己重新走 tokenizer chat-template dispatch；从 problem text 给 E0/E1 分别 retokenize；让 E0/E1 分别决定是否套 chat template。

E1 必须直接读取旧 E0 raw cache 中已经确认的 exact `input_ids`；不得重新决定 chat-template 或重新 tokenization。

## 7.3 controlled sampled generation

固定：

```text
dataset = lcb:codegeneration_v6|0
thinking = true
temperature = 0.6
top_p = 0.95
top_k = 20
min_p = 0.0
max_new_tokens = 38912
TP = 2
KV = BF16
max_model_length = 40960
enforce_eager = true
max_num_seqs = 128
prefix cache = false
speculative decoding = false
```

`max_num_seqs=128` 直接复刻旧正式 E0 的 benchmark execution shape。`max_num_seqs=1` 只用于后续 isolated greedy / real-vLLM hook / puncture / state intervention，不得用于和旧 E0 benchmark 直接比较。

E1 使用与旧 E0 一致的 seed 配置和相同 doc 顺序。

## 7.4 全 175 都跑

不再从旧 61 regression 采样，因为旧标签已被 prompt mismatch 污染。

必须完成 E1 clean controlled sampled 175；E0 直接复用旧正式 175 题结果和 raw cache。

## 7.5 官方 judge 与 paired statistics

输出：

```text
reused_formal_lcb_e0_manifest.json
clean_lcb_e1.jsonl
clean_lcb_pair_matrix.jsonl
CLEAN_LCB_E0_E1_REPORT.md
CLEAN_LCB_E0_E1_REPORT.json
```

必须报告 E0/E1 pass、n10(E0 Pass/E1 Fail)、n01(E0 Fail/E1 Pass)、both pass/fail、`delta accuracy = E1-E0`。

统计使用 paired task：exact McNemar + paired bootstrap 95% CI。不得把 175 题当成 350 个独立样本。

## 7.6 总门禁

定义：

```text
STATISTICALLY_SUPPORTED:
  paired 95% CI upper < 0 或 exact McNemar p < 0.05

MATERIAL:
  E1 - E0 <= -5 percentage points
```

### Gate A：MATERIAL + STATISTICALLY_SUPPORTED

`CLEAN_FORMAT_LOSS_CONFIRMED = true`。允许进入完整但已缩减的 RQ1/RQ2。

### Gate B：显著但 <5pp，或 >=5pp 但统计证据不足

`CLEAN_FORMAT_LOSS = SMALL_OR_UNCERTAIN`。只允许小规模 RQ1 actual-path + MoE puncture，最多 4 个 case study；不做大规模 state intervention；不启动 E2/E3 task recovery。

### Gate C：没有支持 clean loss

`CLEAN_FORMAT_LOSS_CONFIRMED = false`。立即停止“LCB 精度损失因果链”。此时后续只允许极小的 MoE 数值特性分析，不再做 task-causal reset/injection/R64/DIAG。

注意：Gate A/B/C 只控制后续 **LCB task-causal mechanism** 是否继续，**不取消本 Task 中 MMLU-Pro300 E1–E7 的统一重跑；E0 始终复用旧正式结果。**

## 7.7 MMLU-Pro300：E0 复用，E1–E7 全部重新跑

旧 MMLU-Pro E0 已经通过当前审计并可复用，因此不再重跑 E0。E1–E7 必须全部重新评测，并严格复刻旧 E0 的输入、runner 与 execution-shape 协议，使新横向表只改变 variant 本身。

正式 variant 固定：

```text
E0 = Native NVFP4
E1 = Direct HiF4
E2 = R64-only
E3 = adopted Fusable DIAG
E4 = adopted Fusable DIAG + R64
E5 = Online DIAG
E6 = Online DIAG -> R64
E7 = Online R64 -> DIAG
```

这里的 E3/E4 必须使用正式 adopted artifact；`E3_fusable_candidate` 只允许后续局部 DIAG 机制诊断，禁止进入 MMLU-Pro 正式 E0–E7 排名表。

### 7.7.1 统一输入与样本集合

固定同一套 `mmlu_pro|0` 300 sample IDs、相同顺序、相同 prompt 构造和相同 tokenizer/chat-template 语义。

从旧正式 E0 中提取并冻结 canonical MMLU-Pro prompt/sample manifest；如果可以恢复 exact input IDs，则 E1–E7 直接复用这份 IDs。禁止每个 variant 各自重新决定 chat-template/prompt 路径。

输出至少包含：

```text
clean_mmlu_pro_300_prompt_manifest.jsonl
reused_formal_mmlu_pro_300_E0_manifest.json
clean_mmlu_pro_300_E1.jsonl
...
clean_mmlu_pro_300_E7.jsonl
clean_mmlu_pro_300_matrix.jsonl
CLEAN_MMLU_PRO_E0_E7_REPORT.md
CLEAN_MMLU_PRO_E0_E7_REPORT.json
```

### 7.7.2 统一运行协议

沿用当前 Qwen3 官方 thinking 评测语义并锁死所有 variant：

```text
task = mmlu_pro|0
max_samples = 300
thinking = true
temperature = 0.6
top_p = 0.95
top_k = 20
min_p = 0.0
max_new_tokens = 32768
TP = 2
KV = BF16
max_model_length = 40960
enforce_eager = true
```

其余会影响输入或生成的 runner 参数也必须在 manifest 中显式锁死；E1–E7 必须复刻旧 E0 的正式评测入口和协议。

旧正式 E0 的 `max_num_seqs=128`，因此 E1–E7 benchmark 也固定 `max_num_seqs=128`。禁止为了机制分析方便把 benchmark 改成 `max_num_seqs=1`，也禁止某些 variant 单独降级 TP 或更换 backend。

### 7.7.3 结果与统计

必须报告：

- E0–E7 各自 `extractive_match` / accuracy；
- 每个 variant 相对 E0 的绝对 pp 变化；
- E1 相对 E0 的 paired flip matrix；
- E2–E7 相对 E1 与 E0 的 paired per-item flip 数；
- E0/E1 至少给 paired bootstrap 95% CI + exact McNemar；
- 其余 variant 的统计以 paired per-item descriptive 为主，避免因为 7 个比较机械放大显著性解释。

最终报告必须把 MMLU-Pro300 新结果与 clean LCB E0/E1 放在同一个“统一端到端基线”章节中，用于回答：

> 相同的 NVFP4→HiF4 转换在短程选择/推理与长程代码生成上的真实影响是否不同，以及 R64/DIAG/组合方案的端到端收益是否与局部 reconstruction 指标一致。

---

# 8. Task 3：只有 Gate A/B 才构建新的 clean cohort

cohort 只能来自 Task 2 的 clean pair matrix：

```text
clean_regression = clean E0 Pass / clean E1 Fail
clean_robust     = clean E0 Pass / clean E1 Pass
```

不再使用 old formal 61 作为正式 cohort。

最多选择 8 clean_regression + 8 clean_robust；robust 尽量匹配 prompt length、E0 generation length、problem type/fn_name。

然后用 exact same prompt IDs 重跑 isolated greedy E0/E1，并用官方 judge 重判。

若 `mechanism_regression < 2`：不做 full-program state-reset causal chain，不做 regression-vs-robust 统计模型，只做 fixed-E0-history operator/local-error 分析，并在报告标记 `DETERMINISTIC_TASK_BRIDGE_INSUFFICIENT`。

若 `mechanism_regression >= 2`，允许 case-level RQ1/RQ2。只有 `mechanism_regression >= 8` 才允许 cohort-level bootstrap/statistical comparison。禁止重复随机 seed 伪造独立样本。

---

# 9. Task 4：缩减版 RQ1——actual path + MoE-first same-input puncture

## 9.1 probe

对 mechanism-regression：

```text
t* = first greedy token divergence
probe = {t*-8, t*-2, t*-1, t*}
```

越界跳过。robust control 使用匹配的相对 decode depth，不做全 token 扫描。

## 9.2 core hook

复用现有：layer_in、attention_core、post_attn_norm、router_logits、moe_out、layer_out、final_norm、raw_logits(probe-only)。

不要默认新增 all Q/K/V tensors 或 all expert intermediates。

## 9.3 frontier layer

每个 probe 自动选 Top 3 MoE-growth layers + Top 1 attention-growth layer（仅 unresolved control）。后续 puncture 只跑这些层。

## 9.4 MoE same-input puncture

对 top MoE layer 真实 production puncture，计算 `e/q/p` 和 closure。必须通过 no-op/identity 门禁。

## 9.5 Router 条件式分解

只有该层 `q_M` 明显时才运行 frozen-router，比较 `q_expert` 与 `q_router`。禁止恢复 router enrichment 全扫描。

## 9.6 quantizer-input feature 只在 puncture 点计算

只记录 RMS、amax、amax/RMS、G64 amax、HiF4 QDQ rel-L2/NMSE，以及 helper 已支持的 zero transition/clipping。

重点验证已有 8B 发现：`amax64` / local HiF4 susceptibility 是否在 30B 真正危险 MoE 点上仍有解释力。

## 9.7 Attention/QKV/O 不是默认任务

只有 actual-path 明确 Attention dominant，或 MoE `q` 很小无法解释时，才对对应少数层做 QKV/O same-input puncture。禁止全 48 层默认跑 QKV/O puncture。

---

# 10. Task 5：缩减版 RQ2——decision flip + 少量 state intervention

只有 `mechanism_regression >= 2` 才执行。

## 10.1 decision-space

在预测 first-divergence token 的 state：

```text
a = E0 greedy chosen token
b = E1 greedy chosen token
z0/z1 = final RMSNorm 后进入 lm_head 的 hidden
m0_ab = logit0[a]-logit0[b]
m1_ab = logit1[a]-logit1[b]
delta_m = m1_ab-m0_ab
```

若 lm_head 权重相同，验证 `delta_m ≈ (z1-z0)^T(w_a-w_b)`。它用于解释实际 rank reversal，不再预设“小 margin”。

## 10.2 state reset 不再 48 层全扫

每个 case 只选择 Task 4 的 Top 1~3 frontier layers，优先 patch post-MoE/layer_out full-prefix `(branch,residual)` state，必要时补 post-Attention。必须 no-op identity PASS。

## 10.3 先只测 one-step rescue

reset 后先看当前 predictor 的 E0 token `a` 是否重新成为 top1，或 signed margin 是否明显恢复。若 one-step 都不 rescue，不继续生成 38912 token。

只有 one-step rescue 成功的少量 layer/case，才运行完整 continuation + 官方 LCB judge。

## 10.4 injection 只针对 reset-rescue layer

默认只做 `alpha=0` 与 `alpha=1`，必要时加 `alpha=0.5`。取消旧计划的大剂量曲线和默认随机方向 control。

## 10.5 删除 semantic-frontier 大扫描

旧计划全套 token-prefix task-correctness frontier 暂停。只有未来 clean task loss 大、且 state intervention 已证明局部因果链后，才另立计划。

---

# 11. Task 6：RQ3 条件式机制探针——默认不启动

只有同时满足 Gate A 且 RQ1/RQ2 至少在 >=2 个 mechanism-regression case 上找到相同主要机制才解锁。

## 11.1 R64：E2

只跑同一批 case/token/frontier layer 和同一批 MoE puncture/quantizer feature/decision 指标。

主要问：R64 是否降低危险 MoE local `q`；是否降低 G64 amax/HiF4 susceptibility；是否改变 q/p alignment 或 final decision direction；是否 rescue 已确认的 next-token decision。

不默认跑全 175 E2 clean LCB。只有局部机制稳定改善后，再单独决定是否值得做 task-level E2 benchmark。

## 11.2 DIAG：禁止用 adopted E3 直接解释机制

当前 `shared_vllm_qwen3_30b/E3_fusable/artifact` 是 adopted artifact，不是干净的“训练 DIAG 做了什么”探针。

因此 DIAG 机制 probe 优先用：

```text
E3_fusable_candidate/artifact_candidate
```

用途仅限局部 diagnostic，禁止把 candidate 分数混入正式排名。

若 candidate 的 router pathology 导致机制无法干净解释，则使用 `E5 Online DIAG` 作为 expert-specific upper-bound / accepted-DIAG diagnostic control。E5 不是 Fusable 的正式替代，只用于回答 DIAG 重分配本身的作用方向。

## 11.3 本轮明确不做的，是 E4/E6/E7 **机制深钻**，不是 MMLU-Pro 基线评测

```text
不做：E4 DIAG+R64 的机制 puncture / state intervention
不做：E6 DIAG→R64 的 task-causal recovery chain
不做：E7 R64→DIAG 的 task-causal recovery chain
```

但 Task 2 中的 **MMLU-Pro300 E0–E7 统一端到端重跑全部必须完成**。这里不要把“机制实验不做”误解成“benchmark 不跑”。

---

# 12. Task 7：统计与报告

最终报告：

`analysis/FORMAT_CONVERSION_ERROR_LOCALIZATION_REPLAN_REPORT.md`

第一页必须同时给出三组信息：

1. 旧 Phase-A LCB observed scores（prompt mismatch，不可归因）；
2. 新 clean controlled LCB E0/E1 scores（真正格式对照）；
3. 新统一 MMLU-Pro300 E0–E7 全量结果表。

禁止继续把 `95/175→34/175` 写成纯格式转换掉点，也禁止继续引用旧 MMLU-Pro 横向表作为最终算法排名。

若 Gate C，主结论应是 negative result：修正 prompt/execution protocol 后没有确认足够大的 clean LCB 格式转换任务损失，因此取消基于旧 35pp gap 的大规模 causal mechanism 实验。此时只汇报局部数值误差，不声称造成 LCB accuracy collapse。

若 Gate A/B，则按“clean task loss → deterministic bridge → actual-path frontier → MoE e=q+p → expert/router split → decision flip → 少量 state reset/injection”组织证据。

统计规则：clean E0/E1 accuracy 用 paired task statistics；regression/robust n<8 不跑 logistic；小样本只报 case table + bootstrap/descriptive；不把 token 或多个 probe 当独立题目。

---

# 13. Task 8：新的 Go 执行顺序

```text
0. 保留当前 E1 55 条补跑，但截断旧 heavy auto-chain
1. matched-prompt greedy official judge audit
2a. 复用正式 LCB E0，只重跑 matched-protocol E1 全 175
2b. 复用正式 MMLU-Pro300 E0，只重跑 E1/E2/E3/E4/E5/E6/E7
3. 完成统一端到端基线报告
4. 根据 clean LCB 判 Gate A/B/C

Gate C:
    → 停止 LCB task-causal chain
    → MMLU-Pro E0 + 新 E1–E7 结果组成正式新基线
    → 可选极小 MoE numerical characterization
    → final report

Gate B:
    → MMLU-Pro E0 + 新 E1–E7 基线保留
    → <=4 case reduced RQ1
    → no broad state intervention
    → no E2/E3 task recovery
    → final report

Gate A:
    → MMLU-Pro E0 + 新 E1–E7 基线保留
    → clean cohort
    → deterministic greedy bridge
    → MoE-first actual-path + e=q+p puncture
    → decision flip
    → selected-layer one-step reset
    → only successful cases full continuation/injection
    → if mechanism stable: E2 / DIAG diagnostic probe
    → final report
```

---

# 14. 明确删除的旧实验清单

本轮不要继续跑：

1. decode-length monotonic accumulation scan；
2. LCB low-margin enrichment 大扫描；
3. router top-k enrichment 大扫描；
4. 重复 token exact-rejoin rescue/poison；
5. 旧 formal 61 regression 的 semantic-frontier；
6. 全 token × 全 layer quantizer feature scan；
7. 全 48 层 QKV/O same-input puncture；
8. 全 48 层 × Attention/MoE boundary state-reset matrix；
9. 大 alpha dose injection curve；
10. E2/E3 无条件完整 mechanism rerun；
11. adopted E3 作为“DIAG 本身”的机制解释；
12. E4/E6/E7 的机制深钻；
13. E6/E7 组合算法的 LCB task-causal 消融。

注意：上述第 12/13 条**不包含** MMLU-Pro300 正式 benchmark；MMLU-Pro E0 复用，E1–E7 仍全部重跑。

---

# 15. 最终成功标准

本计划成功不要求一定证明 HiF4 会显著损害 LCB。

必须完成：

1. 旧 formal prompt mismatch 从格式因果结论中剔除；
2. matched-prompt 55 条 greedy audit 完成；
3. LCB 正式 E0 复用 + matched-protocol E1 全 175 完成；
4. LCB paired accuracy delta + CI + McNemar；
5. MMLU-Pro300 正式 E0 复用，E1–E7 在完全复刻 E0 协议下全部重新跑完；
6. 生成新的 MMLU-Pro E0 + 新 E1–E7 横向表和 paired flip 统计；
7. 明确 Gate A/B/C；
8. 根据 gate 自动停止不再必要的 LCB 机制实验。

只有 Gate A/B 才要求：clean cohort、deterministic bridge、MoE-first RQ1、`e=q+p` closure、必要的 expert/router split。

只有 Gate A 且 deterministic bridge 足够才要求：selected-layer state reset/injection、R64/DIAG diagnostic probe。

最终目标不是“把计划全部跑完”，而是：

> **用最少的新增计算把当前真正还不确定的问题回答清楚，并在证据否定旧假设时主动停止对应实验。**
