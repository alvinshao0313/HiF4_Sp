> 状态：已被 2026-09-01-qwen3-30b-hif4-long-trajectory-real-vllm-hook-replan-cn.md 取代。
> 仅用于历史问题复盘，不再执行。

# Qwen3-30B-A3B NVFP4→HiF4 格式转换误差累计与 LiveCodeBench 推理轨迹因果诊断计划

> **状态：被新架构取代。** 本文依赖半语义 replay 的能力门禁与 intervention 设计不再执行。格式转换长轨迹与 LiveCodeBench 外部验证统一迁移到 `2026-09-01-qwen3-30b-hif4-long-trajectory-real-vllm-hook-replan-cn.md` 的真实 vLLM Hook 路线。本文仅保留历史研究问题和旧实验设计参考。

> 日期：2026-09-01  
> 目标模型：`nvidia/Qwen3-30B-A3B-NVFP4`  
> 主目标：解释 **NVFP4→HiF4 格式转换如何破坏推理精度**。  
> 次目标：在主机制明确后，解释 R64 与 DIAG 各自改变了损失链条中的哪一环。  
> 主实验环境：复用 `Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/` 中正在建设的可插桩、可干预半语义级 replay runtime。  
> 重要现状：该 replay runtime **尚未完全搭建完成**。本计划不能假定其当前已可用于正式因果实验，必须按本文定义的能力门禁逐级解锁。  
> 执行者：Cursor。Cursor 负责按本文实现增量实验、运行、监控、整理结果；不得另起第二套 replay/runtime，不得自行改变研究问题或阈值。

---

# 0. 研究主线与最高优先级

本计划只有一个主问题：

\[
\boxed{\text{NVFP4}\rightarrow\text{HiF4 格式转换，究竟怎样一步步破坏模型推理精度？}}
\]

需要建立的完整因果链是：

```text
NVFP4 → HiF4
    ↓
局部量化/算子输出扰动
    ↓
Attention / MoE 分支产生新的 hidden error
    ↓
已有 error 被后续层保持、放大或抵消
    ↓
MoE router / final token decision 的边界受到影响
    ↓
相同 token history 下 logits 排名改变
    ↓
首次 deterministic token trajectory bifurcation
    ↓
轨迹差异从“可恢复”演化为“真正破坏程序正确性”
    ↓
LiveCodeBench E0 Pass → E1 Fail
```

固定研究问题：

- **RQ1：格式转换误差在哪里产生，怎样在 48 层中累计？**
- **RQ2：这些连续数值误差怎样变成 token 轨迹分叉，并最终变成真实代码精度损失？**
- **RQ3：R64 与 DIAG 分别改变了 RQ1/RQ2 链条中的哪一环？**

主次关系必须固定：

```text
E0 ↔ E1：主实验，解释格式转换为什么损失推理精度
E1 ↔ E2：次实验，解释 R64 改变什么
E1 ↔ E3：次实验，解释 DIAG 改变什么
E4：本计划不做
```

禁止把 E1/E2/E3 平行扩成三套同等规模的分析。

---

# 1. 固定方案定义

| ID | 方案 | 本计划角色 |
|---|---|---|
| E0 | Native NVFP4 | 数值、轨迹与任务正确性参考 |
| E1 | Direct HiF4 | **主研究对象：纯格式转换损失** |
| E2 | R64-only | 旋转机制探针 |
| E3 | adopted fusable DIAG | DIAG 机制探针 |

明确排除：

- E4 adopted fusable DIAG + R64；
- E5–E7；
- 新训练 DIAG；
- 新 rotation 方案；
- 修改现有 Phase-A artifact；
- 修改 production vLLM / HiF4 kernel。

E1/E2/E3 必须沿用当前 Phase-A / ABI-3 的正式定义，不允许为了诊断方便改变模型语义。

---

# 2. 不重复构造脚手架：本计划与现有 `long_trajectory_stability` 的关系

## 2.1 现有基础设施必须直接复用

现有目录：

```text
Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/
```

已经或正在提供：

1. E0–E4 vLLM free-run capture；
2. exact prompt token IDs / exact generated token IDs；
3. trajectory normalization；
4. first-divergence 比较；
5. fixed E0-token teacher-forcing replay；
6. 48 层 lazy materialization；
7. hidden/router/logit 观测；
8. fused NVFP4 MoE adapter；
9. HiF4 fused MoE adapter；
10. vLLM residual/RMSNorm semantic alignment；
11. TP2 semantic alignment；
12. operator parity / frozen-input artifact。

本计划禁止重新实现：

- checkpoint loader；
- NVFP4 QDQ；
- HiF4 QDQ；
- Qwen3 MoE decoder；
- causal replay 主循环；
- RMSNorm；
- RoPE；
- attention backend；
- fused NVFP4 MoE；
- fused HiF4 MoE；
- TP2 shard/reduction；
- router 主实现；
- vLLM materialization；
- GPU pool。

如本计划需要调用上述能力，只允许：

1. 直接 import 已有实现；或
2. 新增一个**薄 adapter/bridge**，只负责参数映射和结果格式统一。

严禁复制粘贴形成第二套 runtime。

## 2.2 当前脚手架状态：明确视为“未完成”

当前已经知道：

- operator parity 已确认 activation QDQ / weight dequant / dense QKV/O / single-expert W13/W2 在 frozen identical input 下基本闭环；
- multi-expert accumulation / reduction 曾是重要 semantic 差异来源；
- fused MoE adapter 已推进；
- vLLM residual/RMSNorm 路径已修掉一个 TP-independent mismatch；
- `TP_INDEPENDENT_SEMANTICS_ALIGNED=true`；
- TP2 O-proj / MoE 分片与 reduction 已做局部对照；
- 但 TP2 full replay 尚未形成正式闭环 readiness；
- 当前 `trajectory_stability_smoke_tp2_semantic/phase2_first_divergence/run.log` 尚未形成最终完成报告。

因此：

> **在 TP2 E0 replay 没有正式通过本文 Gate G1 前，禁止解释 hidden/router/quantization replay 结果；在 G2A 前禁止 frozen-input/frozen-router 与 q/p/r/G 局部反事实；在 G2B 前禁止 tail reset/injection。**

本计划不负责重新设计 TP2 semantic alignment；其完成应继续沿现有 `long_trajectory_stability/tp2_semantic_alignment/` 路线推进。本计划只消费其最终通过结果。

---

# 3. Replay 能力分三级解锁，禁止跨级运行

这是本计划最重要的工程门禁。

## Gate G0：Trajectory-only 能力

G0 不依赖半语义 replay 完成。

允许做：

- 正式 LiveCodeBench 结果审计；
- E0/E1/E2/E3 pass/fail 样本整理；
- deterministic greedy vLLM free-run；
- exact token trajectory capture；
- first lexical divergence；
- trajectory survival；
- prefix rescue / poison 的 vLLM continuation；
- 代码 execution pass/fail；
- reasoning/code 位置标注。

G0 验收：

- E0/E1/E2/E3 prompt IDs 完全一致；
- exact output token IDs 来自真实 vLLM 输出，禁止 decode→retokenize；
- E1/E2/E3 runtime ABI 检查通过；
- TP=2；
- KV=BF16；
- `enforce_eager=true`。

G0 可以在 replay runtime 尚未完成时先做。

## Gate G1：Observational replay 能力

G1 通过后才允许做：

- fixed E0-token 的 48 层 hidden 比较；
- Attention/MoE 分支输出比较；
- router logits/top-k 比较；
- logits 比较；
- quantizer-input / QDQ 观测；
- cheap token scan。

G1 只允许 **actual-path observation**。此阶段可以回答“E0/E1 在哪里开始出现状态差异”，但不能计算需要替换输入状态的 \(q_A/q_M\)、不能 frozen-router，也不能做 reset/injection。

G1 必须同时满足：

1. replay 使用与正式路径一致的 vLLM residual state machine；
2. layer RMSNorm / Q/K norm / final norm 已切到已验证的 vLLM 数值语义；
3. MoE 使用 production 同类 fused accumulation 语义，而不是 Python expert `index_add_` 替代；
4. TP2 O-proj / MoE reduction 已集成到最终 replay 路径；
5. E0 smoke 固定 stress probes 对真实 TP2 vLLM token 的 top1 parity = **1.0**；
6. 正式 replay E0 top1 parity >= **0.99**；
7. 不存在未解释的、发生在共同 token prefix 内的 TP2-vs-replay mismatch；
8. parity 不能靠降低阈值通过。

建议由现有 scaffold 最终生成一个机器可读文件：

```text
results/long_trajectory_stability/.../analysis/REPLAY_READINESS.json
```

至少包含：

```text
shared_semantics_aligned
fused_moe_aligned
tp2_semantics_aligned
smoke_top1_parity
formal_top1_parity
observational_replay_ready
```

如果现有 scaffold 已有等价 final report，则本计划只解析已有 report，不要求重复生成新格式。

## Gate G2：Intervention replay 能力

G2 分成两个子门禁，不要求一次把所有干预能力搭完。

### Gate G2A：局部反事实能力

G2A 通过后才允许做：

- frozen-E0-input counterfactual；
- RQ1 \(q/p/r/G\) error ledger；
- frozen-router；
- q_expert / q_router decomposition；
- 只在单个 layer/submodule 边界替换输入状态的局部反事实。

G2A 除满足 G1 外，必须通过：

#### G2A.1 no intervention

同一 E0/E1 replay：

```text
intervention_enabled=false
```

必须与 G1 observational replay 输出一致。

#### G2A.2 same-state local patch

在任意合法 layer/token/submodule boundary：

```text
patch E0 local input state with the same E0 state
```

该 submodule 输出以及最终 logits 必须与未 patch baseline 保持数值一致；若同进程同 kernel 路径本身 deterministic，则优先要求 exact，至少不得改变 top1/rank。

#### G2A.3 frozen-router identity

E0 输入 + E0 expert IDs + E0 routing weights + E0 experts：

必须复现正常 E0 MoE 输出。

G2A 未通过时：

> 只能继续 G0/G1 actual-path 统计，禁止计算/解释 RQ1 的 q/p/r/G 与 q_expert/q_router。

### Gate G2B：持续状态干预能力

G2B 在 G2A 基础上增加：

- block/Attention/MoE state reset；
- E0/E1 canonical state-pair 替换后继续运行 tail；
- error injection；
- alpha-dose intervention；
- RQ2 layer-wise causal attribution。

G2B 必须额外通过：

#### G2B.1 alpha=0 injection

\[
S_\alpha=S_0+\alpha(S_1-S_0),\quad \alpha=0
\]

必须复现 E0 baseline。

#### G2B.2 same-state tail reset

把某层 E0 canonical state 替换为完全相同的 E0 canonical state 后继续跑 tail：

- final logits 必须与 E0 baseline 数值一致；
- top1/target rank 必须一致。

#### G2B.3 hook locality

指定：

```text
sample A / token t / layer l
```

干预只能影响该点及其后续，不得修改其它 sample/token 的 cache，也不得污染其它 probe。

G2B 未通过时：

> 可以做 G2A 的 RQ1 局部反事实，但禁止解释 reset/injection 的 RQ2 因果结果。

---

# 4. 新增代码的隔离方式：只写“实验逻辑”，不写 runtime

建议新增：

```text
Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/
  format_conversion_mechanism/
```

该目录只允许放：

```text
audit_lcb_results.py
build_lcb_cohort.py
run_lcb_greedy_matrix.py
build_lcb_probe_plan.py
replay_bridge.py
cheap_token_scan.py
rq1_error_ledger.py
rq1_router_decompose.py
rq1_drilldown.py
rq2_semantic_frontier.py
rq2_reset.py
rq2_injection.py
rq3_transform_probe.py
summarize_mechanism.py
```

其中 `replay_bridge.py` 必须是**薄桥接层**：

- import 已完成的 replay runtime；
- 暴露统一的 `observe(...)` / `intervene(...)` 调用接口；
- 不包含 Qwen3 decoder 的独立实现；
- 不复制 RMSNorm/RoPE/QDQ/MoE；
- scaffold API 后续变化时，只修改这一处 bridge。

测试放：

```text
Native_NVFP4_HiF4_Linear_Puncture/tests/long_trajectory_stability/format_conversion_mechanism/
```

结果放：

```text
Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/format_conversion_mechanism/
```

禁止修改：

```text
3rdparty/vllm/**
experiments/e2e_diag_reconstruction/**
Phase-A artifacts
production kernels
```

如果实验需求暴露出 scaffold 缺能力：

- 本计划代码只报告 `MISSING_REPLAY_CAPABILITY`；
- 不在 `format_conversion_mechanism/` 里补一套替代实现；
- 回到原 `long_trajectory_stability` scaffold 对应任务补齐后再继续。

---

# 5. 数据主线：LiveCodeBench 既做真实精度分组，也做机制轨迹

## 5.1 正式 LiveCodeBench：只负责真实任务精度与样本分组

正式协议沿用当前项目官方配置：

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
```

该协议是**真实任务精度协议**，不能用它定义 deterministic first divergence。

## 5.2 优先复用已有正式结果

先审计：

```text
results/e2e_diag_reconstruction/phaseA_refactor_20260825T035730Z/
```

当前可见：

- E0 有 LiveCodeBench；
- E1 有 LiveCodeBench；
- E2 有 LiveCodeBench；
- E3 当前主目录未见正式 LiveCodeBench 结果。

`audit_lcb_results.py` 必须检查：

- dataset/version；
- sample set；
- thinking；
- temperature；
- top_p；
- top_k；
- min_p；
- max_new_tokens；
- TP；
- KV；
- runtime ABI；
- 每样本 details 是否存在；
- 可否恢复 problem/sample ID 与 pass/fail。

规则固定：

- 通过审计的 E0/E1/E2 **直接复用，不重跑**；
- 缺失的 E3 单独补跑；
- 如果某已有 variant 审计失败，只重跑该 variant；
- 不因为 E3 缺失而重跑 E0/E1/E2。

输出：

```text
formal_lcb_audit.json
formal_lcb_sample_table.jsonl
```

---

# 6. 样本分组：主分析只围绕 E0→E1 的真实精度损失

定义：

## Core A：Format-regression

\[
E0=Pass,\qquad E1=Fail
\]

这是 RQ1/RQ2 的最高优先级样本。

## Control B：Format-robust

\[
E0=Pass,\qquad E1=Pass
\]

用于回答：

> 为什么同样经过 NVFP4→HiF4，有些样本可以承受误差？

## Auxiliary C

```text
E0=Fail
```

只保留统计，不进入主机制实验，除非后续专门需要。

## 6.1 样本数量

目标：

```text
Format-regression: 最多 16
Format-robust:     与 regression 数量相同，最多 16
```

若 regression 少于 16：全部使用。

robust 匹配规则固定：

1. 以 E0 正式输出 token 长度为主匹配变量；
2. 对每个 regression，从尚未选择的 robust 中找 E0 输出长度最接近者；
3. 若距离相同，以 `sample_id` 字典序作为 tie-break；
4. 一个 robust 只能匹配一次。

这样避免 robust 全是明显更短的简单题。

输出：

```text
cohort_manifest.json
```

必须固定保存：

- sample_id；
- prompt hash；
- E0/E1 formal pass/fail；
- E0/E1 official output length；
- cohort label；
- matching partner。

---

# 7. G0 阶段：对选中 cohort 跑 deterministic mechanism trajectory

正式 accuracy 使用 sampling；机制轨迹单独使用 greedy：

```text
thinking = true
temperature = 0.0
top_p = 1.0
top_k = -1
max_new_tokens = 38912
TP = 2
KV = BF16
enforce_eager = true
```

对 cohort 中 E0/E1/E2/E3 全部 capture。

但分析优先级仍是：

```text
先 E0/E1
后 E2/E3
```

`run_lcb_greedy_matrix.py` 不得复制 `capture_main.py` / `launch_capture.py`；它只能作为 thin launcher 调用现有 capture 基础设施，并增加：

- LCB dataset；
- cohort sample filter；
- E0/E1/E2/E3 variant list；
- mechanism greedy generation config。

输出必须沿用 exact-token 规范：

```text
greedy/E0.jsonl
greedy/E1.jsonl
greedy/E2.jsonl
greedy/E3.jsonl
```

## 7.1 首次轨迹分叉定义

对 E1：

\[
t_{lex}^{E1}
=
\min\{t:y_t^{E1}\neq y_t^{E0}\}
\]

E2/E3 同理。

保存：

```text
first_divergence_output_index
predictor_state_index
```

明确约定：

```text
predictor_state_index = 预测第一个不同 token 时所用的共同-prefix 状态
```

不要在实现中发生 off-by-one。

在 `t_lex` 之前必须验证：

```text
E0 token prefix == E1 token prefix
```

只有共同 prefix 内的 hidden 差异才能解释成纯数值格式差异。

---

# 8. G0 阶段：区分“首次 token 不同”和“真正致害的语义分叉”

首次 token 不同不等于代码失败的直接原因。

本计划定义两个轨迹级反事实。

## 8.1 E1 Rescue

给 E1 强制 E0 正确 prefix：

\[
R(k)=Pass(y^0_{1:k}+Generate_{E1}(y^0_{1:k}))
\]

回答：

> E0 正确轨迹至少需要把 E1 带到什么位置，E1 才能够自己继续生成正确程序？

## 8.2 E0 Poison

给 E0 喂 E1 失败 prefix：

\[
P(k)=Pass(y^{E1}_{1:k}+Generate_{E0}(y^{E1}_{1:k}))
\]

回答：

> 到什么位置以后，E1 轨迹已经包含足以让 E0 也进入失败结局的文本状态？

## 8.3 不允许假设 pass/fail 随 k 单调

因此禁止直接二分搜索唯一 `t_harm`。

固定扫描方式：

第一轮 coarse checkpoints：

```text
t_lex
+t 8
+t 16
+t 32
+t 64
+t 128
+t 256
+t 512
+t 1024
之后每翻倍，直到轨迹结束
```

并额外加入：

- reasoning 结束点；
- code 开始点；
- final answer/code block 开始点（若能稳定识别）。

如果相邻 coarse checkpoint 出现稳定 Pass↔Fail 状态变化，则在该区间用 stride=8 补扫。

若 outcome 非单调：

```text
SEMANTIC_FRONTIER = MULTI_FRONTIER
```

保留完整 outcome vector，不强行压成单点。

若存在稳定区间，则定义：

```text
semantic_critical_region = [poison_frontier, rescue_frontier]
```

该区域用于描述“从可恢复轨迹漂移进入真正影响正确性的阶段”。

## 8.4 execution 必须复用 LiveCodeBench evaluator

禁止自己写新的 Python sandbox / 自定义判题规则。

只允许复用当前项目 LiveCodeBench/lighteval 的 execution metric，薄封装 hybrid completion 输入。

---

# 9. G1 通过后：cheap token scan 用于选 heavy probes，不直接下因果结论

对 cohort 中固定 E0 token trajectory teacher forcing。

cheap scan 目标不是完整逐 Linear 穿刺，而是低成本筛选值得做重型 48 层反事实的位置。

## 9.1 每个 token 必须记录的基础信息

```text
sample_id
decode_index
normalized_position = t / T
phase = thinking | code | other
```

如能稳定识别 code，再辅助标注：

```text
identifier
keyword
literal
operator
newline
control_flow
function_def
return
api_call
other
```

代码类别只用于后续解释，不作为主机制定义。

## 9.2 状态/量化风险特征

主 E0/E1 至少记录：

### 实际量化器输入统计

按模块类型在线聚合：

```text
Attention QKV input
Attention O input
MoE gate/up input
MoE down input
```

每类保存：

- RMS；
- amax；
- amax/RMS；
- HiF4 QDQ rel-L2；
- NVFP4 QDQ rel-L2（若当前 E0 path 能无歧义获得）；
- 最大值及 argmax layer；
- 平均值；
- p95。

不要求 cheap scan 将所有 48×Linear 原始张量落盘。

### final hidden

保存：

- E1 vs E0 rel-L2；
- cosine；
- norm ratio。

### final logits

保存：

- KL(E0||E1)；
- top1 agreement；
- E0 top1-top2 margin；
- E0 target rank。

### router

跨 48 层聚合：

- minimum top8/top9 boundary margin；
- argmin layer；
- minimum top-k overlap；
- first layer with top-k set change。

cheap scan 只用于 probe selection 与描述性统计；RQ1 的“本层新注入误差”必须由后面的 frozen-input counterfactual 计算，不能拿 actual-path QDQ 直接替代。

---

# 10. Heavy probe 选择规则

## 10.1 Format-regression 样本

每条最多约 15 个 probe，去重后保存。

### A. first-divergence neighborhood

若 E1 存在 `t_lex`：

```text
t_lex - 8
t_lex - 4
t_lex - 2
t_lex - 1
predictor state for t_lex
```

只保留合法非负位置。

### B. high quantization-risk controls

从 cheap scan 中选 E1 HiF4 QDQ risk 最高的 3 个 token：

- 必须距离 `t_lex` 至少 16 token；
- 相邻候选之间至少间隔 8 token；
- 用于验证“大量化误差是否必然导致分叉”。

### C. low-margin controls

选 E0 logit margin 最小的 2 个稳定 token：

- 距离 `t_lex` 至少 16；
- 与 high-Qerror 控制不重复；
- 用于验证“小 decision margin 是否单独足以导致分叉”。

### D. uniform controls

按 E0 greedy 输出长度取：

```text
10%, 30%, 50%, 70%, 90%
```

距离 `t_lex` <16 的位置跳过。

## 10.2 Format-robust 样本

无 E1 first divergence 时固定：

- high-Qerror 3 个；
- low-margin 3 个；
- uniform 5 个。

这些 robust probes 是 RQ1/RQ2 的必要控制，不能只分析失败样本。

输出：

```text
heavy_probe_plan.json
```

每个 probe 必须写明 `reason`，例如：

```text
divergence_minus_1
high_qerror
low_margin
uniform_50pct
```

---

# 11. RQ1：48 层 Attention / MoE 误差账本

RQ1 以 E0↔E1 为主。

对于 variant \(v\in\{E1,E2,E3\}\)，先定义 E0 与 variant 的真实累计状态。

由于正式 vLLM Qwen3-MoE runtime 跨层维护 `branch + residual` 双状态，本实验实现时不能偷换回旧的单 `hidden` 语义。

所有 state reset / frozen-input 必须基于最终 scaffold 暴露的 canonical state pair；用于统计时可由 bridge 暴露统一的 logical/materialized state view，但不得改变运行语义。

## 11.1 Attention 三项分解

记：

- \(H_{l,in}^0\)：E0 第 l 层 Attention 前逻辑状态；
- \(H_{l,in}^v\)：variant 实际累计状态；
- \(H_{l,A}^0\)：E0 Attention residual 后状态；
- \(H_{l,A}^v\)：variant 实际 Attention residual 后状态；
- \(\widetilde H_{l,A}^v\)：把 **E0 的 canonical pre-Attention state** 喂给 variant 第 l 层 Attention 后得到的 counterfactual 状态。

定义总误差：

\[
e_{l,in}^v=H_{l,in}^v-H_{l,in}^0
\]

本层 Attention 新注入的格式误差：

\[
\boxed{q_{l,A}^v=\widetilde H_{l,A}^v-H_{l,A}^0}
\]

历史误差经过 variant Attention 后的整体传播项：

\[
\boxed{p_{l,A}^v=H_{l,A}^v-\widetilde H_{l,A}^v}
\]

于是严格有：

\[
e_{l,A}^v=q_{l,A}^v+p_{l,A}^v
\]

为了把 residual 直通与真正分支响应分开，再定义：

\[
\boxed{r_{l,A}^v=p_{l,A}^v-e_{l,in}^v}
\]

因此：

\[
\boxed{e_{l,A}^v=e_{l,in}^v+r_{l,A}^v+q_{l,A}^v}
\]

实现必须把等式 closure error 写入 debug metric；理论上按上述差值定义应只剩计算/落盘误差。

Attention 历史误差传播增益：

\[
\boxed{G_{l,A}^v=\frac{\|p_{l,A}^v\|_2}{\|e_{l,in}^v\|_2+\epsilon}}
\]

## 11.2 MoE 三项分解

类似定义：

- \(H_{l,A}^0\)：E0 post-Attention state；
- \(H_{l,A}^v\)：variant actual post-Attention state；
- \(H_{l,out}^0\)：E0 block output state；
- \(H_{l,out}^v\)：variant actual block output；
- \(\widetilde H_{l,out}^v\)：把 E0 canonical post-Attention state 喂给 variant MoE 后得到的 frozen-input counterfactual。

MoE 新注入误差：

\[
\boxed{q_{l,M}^v=\widetilde H_{l,out}^v-H_{l,out}^0}
\]

MoE 历史传播：

\[
\boxed{p_{l,M}^v=H_{l,out}^v-\widetilde H_{l,out}^v}
\]

MoE 分支对旧误差的额外响应：

\[
\boxed{r_{l,M}^v=p_{l,M}^v-e_{l,A}^v}
\]

MoE propagation gain：

\[
\boxed{G_{l,M}^v=\frac{\|p_{l,M}^v\|_2}{\|e_{l,A}^v\|_2+\epsilon}}
\]

最终：

\[
e_{l+1,in}^v=e_{l,out}^v
\]

48 层可以完整串成误差流水账。

## 11.3 每个量不只存 norm

对状态 \(H\) 至少保存：

- 相对 E0 的 rel-L2；
- cosine；
- norm ratio；
- max-abs。

对误差向量 \(e,q,p,r\) 至少保存：

- L2 norm；
- 以对应 E0 state norm 归一化的 relative magnitude；
- max-abs。

额外保存：

\[
\cos(q,p)
\]

并在范数非零时可保存：

\[
\cos(q,e_{history}),\qquad \cos(r,e_{history})
\]

用于判断本层新注入误差与历史传播误差是：

- 同向建设性累计；
- 近似正交；
- 反向抵消。

RQ1 不能只用 hidden NMSE 排“敏感层”。

---

# 12. RQ1：MoE 必须进一步拆成 Expert error 与 Router-induced error

Qwen3-30B-A3B 是 top-8 MoE，router 可能把连续 hidden perturbation 离散放大。

在固定 E0 post-Attention input 下：

1. E0 normal MoE：

\[
M^0(x^0)
\]

2. variant normal MoE：

\[
M^v(x^0)
\]

3. variant frozen-router MoE：

- expert computation 使用 variant；
- 强制 selected expert IDs = E0；
- 强制 routing weights = E0。

记：

\[
M^{v,route0}(x^0)
\]

定义：

### Expert computation error

\[
\boxed{q_{expert}^v=M^{v,route0}(x^0)-M^0(x^0)}
\]

### Routing-induced error

\[
\boxed{q_{router}^v=M^v(x^0)-M^{v,route0}(x^0)}
\]

要求验证：

\[
q_M^v\approx q_{expert}^v+q_{router}^v
\]

并记录 decomposition closure。

主要回答：

> 格式转换的 MoE 误差主要来自 expert 内连续数值误差，还是 router top-k / routing weight 改变带来的离散放大？

---

# 13. RQ1 的三级观测粒度，禁止一开始全量抓所有 Linear

## Level 1：48 层主干——必跑

每个 heavy probe × E0/E1，所有 48 层记录：

- pre-Attention state；
- post-Attention state；
- pre-MoE/router-input state；
- post-MoE/block-output state；
- \(q_A,p_A,r_A,G_A\)；
- \(q_M,p_M,r_M,G_M\)；
- final accumulated error。

E2/E3 后续只在相同 heavy probes 上复测，不重新选另一套主 probes。

## Level 2：Router decomposition——必跑

所有 heavy probes × 48 层记录：

- router KL；
- top-k overlap；
- exact top-k；
- top8/top9 margin；
- q_expert；
- q_router。

## Level 3：内部 Linear drill-down——条件触发

只有当 Level 1/2 显示某层贡献显著时才进入。

### Attention drill-down

只保存：

```text
Q output
K output
V output
attention probability / equivalent attention distribution metric
O input
O output
```

指标：

- rel-L2；
- cosine；
- attention probability KL；
- O-output local error。

目的：区分：

```text
Q/K/V 已产生误差
→ attention 内积/softmax 放大
→ O projection 再放大
```

### Expert drill-down

只保存 routed experts：

```text
gate output
up output
SiLU(gate) * up
down output
```

目的：定位：

```text
W13 intrinsic error
nonlinearity amplification
W2/down intrinsic error
W13→W2 propagated error
```

优先复用现有 `nvfp4_operator_parity` P4 的分解代码/概念，不重新实现。

---

# 14. RQ1 样本级核心比较：Format-regression vs Format-robust

主报告必须首先比较 E1：

```text
Format-regression
vs
Format-robust
```

重点不是“哪组平均 NMSE 更大”一个问题，而是同时比较：

1. \(q_A\)：Attention 新误差；
2. \(q_M\)：MoE 新误差；
3. \(G_A\)：Attention 对历史误差的传播；
4. \(G_M\)：MoE 对历史误差的传播；
5. q_expert vs q_router；
6. \(\cos(q,p)\)：误差是否建设性累计；
7. final hidden angle/norm drift。

如果 regression/robust 的 hidden NMSE 接近，不得判定“没有机制”。这将自然支持 RQ2：

> 相似大小的误差是否因为 decision sensitivity / error direction 不同而导致不同结果？

---

# 15. RQ2：先研究 first lexical divergence，因为这里 token history 仍完全相同

对于每个 Format-regression：

\[
t_{lex}=\text{E0/E1 deterministic first token difference}
\]

RQ2 的数值主分析点是**预测这个 token 之前的共同-prefix 状态**。

禁止拿 free-run 分叉后的：

\[
h_t^{E1}-h_t^{E0}
\]

直接解释成量化误差，因为此时输入 token history 已经不同。

分叉之后如果还需要比较，必须回到 fixed E0-token replay。

---

# 16. RQ2：检验三个竞争假设

## H-Q：量化误差 spike 主导

分叉前：

- HiF4 QDQ error 明显升高；
- local \(q_A/q_M\) 明显升高；
- hidden error 同步增大。

支持结论：

> 某些 token/state 的激活分布本身特别不适合 HiF4，形成局部高误差事件。

## H-N：activation norm / distribution 主导

分叉前：

- RMS；
- amax；
- amax/RMS；
- group/subgroup dispersion；

出现异常，并进一步解释 QDQ error。

注意：不能只看 residual hidden norm。

至少区分：

```text
residual stream norm
actual quantizer input RMS
actual quantizer input amax
```

## H-D：decision sensitivity 主导

量化误差不一定 spike，但 E0 top1-top2 margin 很小。

则机制可能是：

\[
\boxed{\text{普通大小的 perturbation} + \text{小 decision margin} \rightarrow rank reversal}
\]

正式统计必须让 H-Q/H-N/H-D 同时竞争，而不是预设其中一个成立。

---

# 17. RQ2：final hidden 的“误差大小 vs 方向”必须单独分析

对预测 `t_lex` 的最终 pre-lm-head hidden：

\[
h^0,\quad h^1
\]

误差：

\[
\epsilon=h^1-h^0
\]

E0 top1/top2 token 为 a,b，lm_head 权重为 \(w_a,w_b\)。

在**最终 pre-lm-head hidden** 上，decision direction 可以精确定义：

\[
\boxed{d_{decision}=w_a-w_b}
\]

真实 margin perturbation：

\[
\boxed{\Delta m=\epsilon^T d_{decision}}
\]

必须记录：

- \(\|\epsilon\|\)；
- hidden cosine；
- norm ratio；
- \(\epsilon^T d_{decision}\)；
- E0 margin；
- 是否发生 top1 flip。

重要纪律：

> **禁止把 `lm_head w_a-w_b` 直接当作任意中间层 hidden 的 decision direction。**

对于中间层，由于后面还有非线性 Transformer tail，不能用该公式直接解释。中间层必须靠 reset/injection 的因果效果衡量。

## 17.1 norm-only / direction-only final-hidden counterfactual

构造：

### norm-only

\[
h_{norm}=\frac{\|h^1\|}{\|h^0\|}h^0
\]

保持 E0 方向，使用 E1 norm。

### direction-only

\[
h_{dir}=\frac{\|h^0\|}{\|h^1\|}h^1
\]

保持 E1 方向，恢复 E0 norm。

直接送入 lm_head。

比较：

- top1；
- margin；
- KL。

用于回答用户关心的问题：

> 轨迹变化时，主要是 hidden norm 变化，还是 hidden 方向发生了更关键的改变？

---

# 18. RQ2：Router boundary 的方向分析只在 router input 处做

router 是线性映射，因此在 router input 处可以精确定义 top8/top9 boundary direction。

E0 的第 8/9 expert 为 \(e_8,e_9\)：

\[
\boxed{d_{router}=W_{e_8}-W_{e_9}}
\]

router-input hidden perturbation：

\[
\epsilon_r=x_r^{E1}-x_r^{E0}
\]

记录：

\[
\epsilon_r^T d_{router}
\]

与 E0 top8/top9 margin 比较。

回答：

> E1 是否不是产生更大 hidden error，而是恰好把 error 推向 router-sensitive direction，导致 expert set 改变？

如果 RQ1 的 q_router 很小，则该分析降级为辅助，不扩大实验。

---

# 19. G2：Layer-wise Error Reset，建立“哪一段误差真正导致 token flip”的因果证据

只在优先级最高的 Format-regression 典型样本上做，目标 8 条。

筛选要求：

- E0 formal Pass / E1 formal Fail；
- greedy E0/E1 存在确定 first divergence；
- G1 replay 在 predictor probe 通过 E0 parity；
- 覆盖 early/mid/late divergence。

## 19.1 coarse reset

先在 block output canonical state pair 做：

```text
layer = 0, 8, 16, 24, 32, 40, 47
```

操作：

```text
在预测 t_lex 的共同 E0 token prefix 下
正常跑 E1 到 layer l
将该点 E1 canonical state pair 替换为 E0 对应 state pair
后续 layer l+1..47 继续跑 E1
```

记录：

- E0 target token rank；
- E0 target vs E1 competing token margin；
- final KL to E0；
- top1 是否恢复为 E0 token。

定义：

```text
rescue = patched top1 == E0 top1
```

## 19.2 refine critical layer interval

若 coarse reset 存在：

```text
较早层 reset 可以 rescue
较晚层 reset 无法 rescue
```

就在这两个 coarse layer 之间逐层补测。

不要默认 rescue 对 layer 单调；若非单调，保存完整 rescue map。

## 19.3 Attention vs MoE 分支定位

只在已定位的 critical layer(s) 做两个 reset：

### A-reset

把 E1 post-Attention canonical state 替换为 E0 post-Attention state，再继续 E1 MoE + tail。

### M-reset

把 E1 post-MoE/block-output state pair 替换为 E0，再继续 E1 tail。

比较两者是否 rescue。

目的：

> 致命状态变化主要在该层 Attention 形成，还是 MoE 又进一步把它推过了决策边界？

---

# 20. G2：Error Injection，反方向验证“E1 error 是否足以毒化 E0”

对 reset 找出的关键层做。

由于 vLLM runtime 使用 `branch + residual` state pair，不允许把中间状态粗暴压成单 tensor 后再注入。

设：

```text
S0 = (branch0, residual0)
S1 = (branch1, residual1)
```

构造 dose：

\[
branch_\alpha=branch_0+\alpha(branch_1-branch_0)
\]

\[
residual_\alpha=residual_0+\alpha(residual_1-residual_0)
\]

统一 cast/rounding 必须沿 scaffold 定义。

固定：

\[
\alpha\in\{0,0.25,0.5,0.75,1.0,1.5\}
\]

把 \(S_\alpha\) 注入 E0，然后后续全部跑 E0。

记录：

- target token rank；
- margin；
- KL；
- top1 flip。

如果随着 alpha 增大出现稳定 margin crossing，则形成：

```text
E1-format-error dose
        ↓
E0 decision margin crossing
```

的强因果证据。

## 20.1 matched random-direction control

只在能够定义单 tensor canonical observation point时做；若 scaffold 不能保证 state pair 与 materialized hidden 的可逆等价，则跳过，不自行制造新语义。

优先级低于真实 error dose。

---

# 21. RQ2：事件位置统计——代码任务“什么位置”容易发生轨迹变化

对每个样本记录：

- absolute `t_lex`；
- normalized `t_lex/T`；
- reasoning 内相对位置；
- code 内相对位置；
- semantic critical region；
- trajectory length。

至少区分：

```text
thinking/reasoning
transition to code
code body
final answer
```

若代码可稳定解析，再增加：

- function definition；
- control flow；
- condition；
- loop；
- identifier；
- literal；
- operator；
- API/method call；
- return；
- indentation/newline。

统计不能直接数 divergence 次数，必须做 exposure normalization：

\[
P(\text{divergence}|\text{category})
=\frac{\text{category divergence events}}{\text{eligible positions of category}}
\]

主结论优先报告 reasoning vs code 与相对位置；细 token category 只做辅助。

---

# 22. RQ2：统计上区分 qerror、norm、margin 哪个更能解释 first divergence

对每个 Format-regression 的 first-divergence predictor 建 event。

同一样本选 matched control positions：

- 与 event 位于同一 `thinking/code` phase；
- normalized decode position 最接近；
- 距离 event 至少 16 token；
- 本身不是 high-Qerror/control 重叠点。

每个 event 目标 4 个 controls；不足则使用实际可用数量。

候选特征：

```text
HiF4 QDQ error
actual quantizer-input RMS
actual quantizer-input amax
amax/RMS
final hidden rel-L2
final hidden angle/cosine
E0 logit margin
minimum router margin
router top-k mismatch
```

正式比较：

1. 单变量 event-vs-control effect；
2. sample-block bootstrap 95% CI；
3. leave-one-sample-out logistic prediction；
4. 分别比较：

```text
norm-only features
qerror-only features
margin-only features
qerror + norm
qerror + margin
all features
```

重点看增量解释能力，而不是只报 Spearman。

注意：

- 统计模型只说明关联/预测；
- reset/injection 才用于因果归因；
- 不允许把 logistic coefficient 单独写成因果结论。

---

# 23. RQ3：E2 R64 只沿 RQ1/RQ2 已找到的关键链路复测

E2 不重新跑一套完整机制发现流程。

固定使用：

- 与 E1 完全相同 cohort；
- 相同 heavy probes；
- 相同 critical layers；
- 相同 divergence cases。

## 23.1 R64 无量化等价控制——必须有

在 RQ1 drill-down 找出的关键 Linear 上：

- 复用当前 E2 transform/fold 语义；
- 关闭 HiF4 QDQ；
- 比较 R64 BF16 路径与未旋转 BF16 路径输出。

必须确认：

\[
\|xR\|_2\approx\|x\|_2
\]

且无量化整体计算保持数值等价到合理浮点容差。

该控制回答：

> E2 的差异来自 `R64 × HiF4 quantization`，而不是把 R64 本身误描述为信息损失。

## 23.2 R64 前后分布变化

只在关键 quantizer inputs 上记录：

- RMS；
- amax；
- amax/RMS；
- 64-element group dynamic range；
- 16-element subgroup imbalance；
- max-element energy share；
- channel energy concentration；
- kurtosis；
- nearest HiF4 grid distance；
- threshold-near mass（若已有量化器 helper 可直接计算）。

不要求为所有 token/Linear 全量生成这些高级统计。

## 23.3 Rotation gain

在相同 E0 frozen input 上：

\[
e_{E1}=\frac{\|y_{E1}-y_{E0}\|}{\|y_{E0}\|}
\]

\[
e_{E2}=\frac{\|y_{E2}-y_{E0}\|}{\|y_{E0}\|}
\]

定义：

\[
\boxed{G_R=e_{E1}-e_{E2}}
\]

并研究 \(G_R\) 与：

- Δ(amax/RMS)；
- Δgroup dispersion；
- Δsub16 dispersion；
- Δgrid distance；
- error direction change；

的关系。

## 23.4 E2 机制重点

对关键 case 比较：

- \(q_A^{E2}\) vs \(q_A^{E1}\)；
- \(q_M^{E2}\) vs \(q_M^{E1}\)；
- \(G_A^{E2}\) vs E1；
- \(G_M^{E2}\) vs E1；
- q_router / q_expert；
- final hidden error norm；
- final decision-direction projection；
- first divergence 是否后移/消失。

特别寻找：

```text
local error norm 降低
但 downstream margin / trajectory 更差
```

若出现，则进一步分析 error direction，而不是得出“旋转没用”的表面结论。

---

# 24. RQ3：E3 DIAG 同样只沿关键链路复测

使用与 E1/E2 相同 cohort/heavy probes/critical layers。

重点看 DIAG 是否改变：

- actual quantizer-input channel amplitude；
- RMS / amax / amax-RMS；
- channel imbalance；
- group/subgroup dynamic range；
- \(q_A/q_M\)；
- propagation gain；
- q_router/q_expert；
- final hidden direction drift；
- final token decision margin；
- first divergence position。

DIAG 的机制假设只能作为待验证假设：

> DIAG 更可能通过通道尺度/幅值重分配减少局部格式转换误差；R64 更可能通过局部能量混合改变 group 分布与 error direction。

若结果不支持，不允许强行这样总结。

---

# 25. E2/E3 在真实 LiveCodeBench regression cases 上的“修复效果”

对 E0 Pass / E1 Fail 主 cohort，记录正式 E2/E3 pass/fail：

```text
E1 Fail → E2 Pass : rotation recovery
E1 Fail → E2 Fail : not recovered
E1 Fail → E3 Pass : diag recovery
E1 Fail → E3 Fail : not recovered
```

并在 greedy mechanism trajectory 上记录：

\[
\Delta t_{lex}^{E2}=t_{lex}^{E2}-t_{lex}^{E1}
\]

\[
\Delta t_{lex}^{E3}=t_{lex}^{E3}-t_{lex}^{E1}
\]

若某 variant 不再发生 divergence，标记：

```text
FULLY_RESCUED_WITHIN_CAPTURE
```

不能用一个人为大整数代替无 divergence。

---

# 26. 必须生成的核心图，不做无边界图表堆积

## Figure 1：格式转换 48 层误差流水账——最核心

E1 主图。

横轴：layer 0–47。

至少画：

- \(\|q_A\|\)；
- \(\|p_A\|\) / \(G_A\)；
- \(\|q_M\|\)；
- \(\|p_M\|\) / \(G_M\)；
- final accumulated error。

Regression vs Robust 分开统计。

## Figure 2：MoE error source

48 层：

```text
q_expert
q_router
router top8/top9 margin
```

回答 MoE 是否存在离散放大。

## Figure 3：first divergence 前的 token-level risk

以 `t_lex=0` 对齐横轴，例如：

```text
[-32, ..., -1, 0]
```

画：

- qerror；
- actual input RMS/amax；
- final hidden angle；
- E0 logit margin；
- router margin。

Regression 与 Robust/matched controls 对比。

## Figure 4：Layer reset rescue map

纵轴 case，横轴 layer，颜色表示：

- reset 是否恢复 E0 token；
- 或 margin recovery。

关键层再标 Attention/MoE reset。

## Figure 5：E2/E3 相对 E1 修复了哪一环

只画关键指标差值：

```text
Δ local injection
Δ propagation gain
Δ router contribution
Δ final dangerous-direction effect
Δ first-divergence position
```

不要再做一整套 E2/E3 duplicate figures。

---

# 27. Task 执行顺序

## Task 0：只做 plan/runtime dependency audit，不写新 replay

检查：

- `long_trajectory_stability` 当前各 alignment 子目录；
- G0/G1/G2A/G2B 需要的能力现在哪些已具备；
- 哪些仍由现有 scaffold 任务负责。

输出：

```text
format_conversion_mechanism/analysis/CAPABILITY_AUDIT.md
```

必须明确：

```text
G0_READY = true/false
G1_READY = true/false
G2A_READY = true/false
G2B_READY = true/false
```

如果 G1/G2A/G2B 任一未就绪，只记录缺失能力；不在本计划目录里补第二套 runtime。

## Task 1：正式 LCB 结果审计与 cohort

执行 `audit_lcb_results.py` + `build_lcb_cohort.py`。

优先复用 E0/E1/E2，补 E3 缺失正式 LCB。

验收：

- cohort manifest 存在；
- regression/robust 定义正确；
- sample IDs 可追溯到正式 LCB details。

## Task 2：G0 greedy mechanism capture

只跑 cohort E0/E1/E2/E3。

验收：

- exact prompt token IDs 一致；
- normalized trajectory 完整；
- first divergence 可重复；
- E4 未启动。

## Task 3：G0 semantic frontier

先对 Format-regression 做 E1 Rescue / E0 Poison。

验收：

- 使用正式 LCB evaluator；
- outcome vector 完整；
- 不假设单调；
- 输出 semantic critical region 或 MULTI_FRONTIER。

## Task 4：等待/复核 G1

如果 G1=false：

```text
STOP_OBSERVATIONAL_REPLAY = true
```

此时只允许整理 G0 结果，不执行 replay 统计。

如果 G1=true，继续。

## Task 5：cheap token scan + heavy probe plan

只做 actual-path observation，并生成 heavy probes。

验收：

- regression/robust 都有 probes；
- divergence/high-Qerror/low-margin/uniform 来源明确；
- probe 数量受控；
- 没有偷偷执行 frozen-input/frozen-router。

## Task 6：RQ2 observational token-risk statistics

在不需要反事实 intervention 的范围内，先完成：

- qerror / norm / margin / hidden direction / router margin 的 event-vs-control 描述与预测分析；
- first-divergence 前 risk curve；
- regression vs robust actual-path 对照。

该 Task 只能形成相关/预测结论，不写因果语言。

## Task 7：等待/复核 G2A

若 G2A=false：

```text
STOP_RQ1_COUNTERFACTUAL = true
```

此时保留 G0/G1 结果，禁止 Task 8–10 的 q/p/r/G、frozen-router 与局部反事实。

## Task 8：RQ1 E0/E1 48-layer error ledger

G2A 通过后先只跑 E0/E1。

不得一开始启动 E2/E3。

验收：

- 48 层完整；
- Attention/MoE q,p,r,G 可计算；
- same-state local patch identity 已通过；
- decomposition closure 正常；
- regression/robust 都有结果。

## Task 9：RQ1 router decomposition

E0/E1 heavy probes × 48 层。

验收：

- frozen-router identity 测试通过；
- q_expert/q_router decomposition closure 正常。

## Task 10：条件式 Level-3 drill-down

由 Task 8/9 自动选关键层：

建议 selection rule：

- 在 regression 中，某层 median local injection 位于全层 top 20%；或
- propagation gain 位于 top 20%；或
- q_router 位于 top 20%；
- 并优先与 robust 有明显差异的层。

最多选 8 层做 drill-down，避免失控。

## Task 11：等待/复核 G2B

若 G2B=false：

```text
STOP_TAIL_INTERVENTION = true
```

可以继续 G2A 的 RQ1 结果与 G1 observational statistics，但不能 reset/injection。

## Task 12：RQ2 layer reset

仅典型 regression cases，先 coarse 后 refine，再 Attention/MoE split。

## Task 13：RQ2 error injection

仅 reset 找出的 critical layers。

执行真实 E1 error dose。

## Task 14：RQ3 E2

只对相同 probes/critical layers：

- R64 BF16 no-quant control；
- distribution change；
- q/G/router/final direction；
- trajectory repair。

其中涉及 q/G/frozen-input 的部分要求 G2A；涉及 reset/injection 的部分要求 G2B。

## Task 15：RQ3 E3

同样只沿已发现关键链路复测；门禁规则与 E2 相同。

## Task 16：最终汇总

生成：

```text
FORMAT_CONVERSION_ERROR_ACCUMULATION_REPORT.md
FORMAT_CONVERSION_ERROR_ACCUMULATION_SUMMARY.json
```

---

# 28. 每个 Task 的断点续跑与结果隔离

所有正式脚本必须：

- 支持现有结果跳过；
- `--force` 才覆盖；
- 记录 input artifact/path/hash；
- 记录 variant；
- 记录 sample cohort manifest；
- 记录 replay readiness artifact；
- 记录 git commit 若可用，但不因为工作区有未提交修改而失败；
- 不使用重量级哈希门禁阻塞实验。

建议结果结构：

```text
results/long_trajectory_stability/format_conversion_mechanism/
  audit/
  cohort/
  greedy/
  semantic_frontier/
  cheap_scan/
  probes/
  rq1/
    error_ledger/
    router_decompose/
    drilldown/
  rq2/
    reset/
    injection/
    statistics/
  rq3/
    e2_rotation/
    e3_diag/
  analysis/
```

---

# 29. 测试要求：只测试新增实验逻辑，不重复 production 测试

新增 tests 只覆盖：

1. cohort regression/robust 分类；
2. robust length matching；
3. first divergence / predictor index 无 off-by-one；
4. semantic frontier 对非单调 outcome 不误用二分；
5. heavy probe 去重与距离约束；
6. q/p/r algebra decomposition；
7. propagation gain 0-denominator guard；
8. frozen-router decomposition closure；
9. state-pair interpolation alpha=0 identity；
10. result resume/skip；
11. E4 永远不进入 variant list；
12. `replay_bridge` 不包含独立 decoder/QDQ/MoE 实现。

不需要重新测试：

- production HiF4 kernel；
- vLLM 全量单测；
- Phase-A training。

---

# 30. 结果解释纪律

## 30.1 不得从相关性直接写因果

例如：

```text
first divergence 处 amax 较大
```

只能说 enrichment/correlation。

只有 reset/injection/frozen-router 等干预才用于“导致/放大”表述。

## 30.2 不得把 R64 描述为本身丢信息

R64 是正交变换。必须说：

```text
R64 改变了 HiF4 QDQ 前的局部能量排布，从而改变量化误差大小和/或方向。
```

## 30.3 不得把 hidden NMSE 当充分指标

若：

```text
E2 hidden NMSE 更小
但 token 更早 flip
```

必须继续检查：

- error direction；
- final decision projection；
- router boundary；
- propagation gain。

## 30.4 不得把 first lexical divergence 等同于程序失败原因

必须结合 semantic frontier / execution outcome。

## 30.5 不得跨越 G1/G2A/G2B 能力门禁解释机制

- G1 前：不解释 replay hidden/router/quantization 结果；
- G2A 前：不解释 frozen-input / frozen-router / q-p-r-G 局部反事实；
- G2B 前：不解释 reset/injection 的 tail-level 因果结果。

脚手架未完成不是“可以先看趋势”的理由。

---

# 31. 最终报告必须回答的固定问题

最终报告不允许只堆图，必须按顺序回答：

## RQ1

1. Direct HiF4 的新误差主要产生在 Attention、Expert 还是 Router？
2. 误差是持续均匀注入，还是集中在少数层/少数 token state？
3. 48 层累计增长主要来自新误差不断注入，还是历史误差被某些层放大？
4. residual 大多只是保存误差，还是存在明显抵消？
5. Format-regression 与 Format-robust 最早在哪个机制量上拉开差异？

## RQ2

6. first divergence 前是否存在明显 quantization-error spike？
7. actual quantizer-input RMS/amax 是否比 residual hidden norm 更有解释力？
8. first divergence 是否主要出现在 E0 小 logit-margin 状态？
9. final hidden 的危险变化主要是 norm 还是 direction？
10. router top-k boundary 是否构成重要离散放大器？
11. layer reset 能否找到真正影响 token flip 的关键深度/分支？
12. error injection 是否能把 E0 token decision 推过边界？
13. first lexical divergence 与 semantic-critical divergence 相隔多远？
14. harmful divergence 更常发生在 reasoning 还是 code 阶段？

## RQ3

15. R64 主要改变局部误差大小、分布、方向还是传播增益？
16. 为什么 R64 对不同 case 可能表现不同？
17. DIAG 主要修复的是幅值/尺度分布、局部 injection 还是下游 propagation？
18. E2/E3 的 task-level recovery 是否能被前述机制指标解释？

最后才能形成算法指导结论，例如：

```text
如果主损失来自特定 group distribution → 改格式前数据重排/scale
如果主损失来自 router boundary → router-aware calibration / protection
如果主损失来自 dangerous-direction projection → 不应只优化 NMSE
如果主损失来自少数层 propagation amplification → layer-aware precision/protection
```

这些是结果驱动的可能方向，不允许在实验前预设为结论。

---

# 32. Cursor 的停止条件

以下任何一个发生都必须停止对应阶段并写报告，不得自行绕开：

1. G1 未通过却准备解释 observational replay；
2. G2A 未通过却准备跑 RQ1 q/p/r/G 或 frozen-router；
3. G2B 未通过却准备做 tail reset/injection；
4. replay 需要修改 production runtime 才能继续；
5. TP2 replay 与真实 E0 仍存在共同-prefix unexplained mismatch；
6. E0/E1 prompt exact token IDs 不一致；
7. variant runtime ABI 不符合当前正式定义；
8. frozen-router identity 不通过；
9. state reset no-op / alpha=0 identity 不通过；
10. formal LCB results 协议不一致但脚本试图混用；
11. E4 被误加入本计划；
12. 发现已有 scaffold 已经提供某能力但新代码又复制实现了一遍。

遇到这些情况输出：

```text
STOP_REASON.md
```

说明：

- 当前 Task；
- 违反的 gate；
- 已有证据；
- 最小下一步；

然后停止，不自行扩大修改范围。
