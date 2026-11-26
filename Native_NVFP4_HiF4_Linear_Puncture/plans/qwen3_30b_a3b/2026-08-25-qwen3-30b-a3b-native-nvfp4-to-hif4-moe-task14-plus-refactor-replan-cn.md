# Qwen3-30B-A3B Native NVFP4 → HiF4：Task 14+ 重构后正式实验重规划

日期：2026-08-25

适用模型：`nvidia/Qwen3-30B-A3B-NVFP4`

适用目录：`Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction`

本计划**仅替换原 2026-08-21 计划中的 Task 14–19**。Task 0–13 的模型契约、Native semantic、ModelOpt loader、HiF4/R64 数学、vLLM runtime 和统一评测框架仍然作为前置基础，不重新设计；但由于 Task 13 之后对 Router、rollback、artifact 和 MoE resume 做了重构，正式实验前必须执行一次新的受影响路径回归门禁。

---

# 1. 为什么必须重做 Task 14 之后的计划

Task 13 以后发生了以下事实性改动：

1. Fusable Router gate 改为使用真实的 `post_attention_layernorm` 后 Router input，而不是错误的 decoder-layer entry hidden。
2. Router 对齐新增 KL auxiliary loss：
   - `router_align_type = kl`
   - `temperature = 1.0`
   - `router_align_loss_weight` 默认 `0.0`
   - 正式比较 `0.0` 与 `0.5`
   - Router KL 梯度只允许更新 `D_GU`。
3. rollback 被拆成两个独立原因：
   - `loss_rollback`
   - `router_rollback`
4. Router rollback 由原先“整层 DIAG 全清零”改为只回退 `D_GU`；`D_QKV / D_VO / D_UD` 保留。
5. `router_rollback=off` 时 Router mismatch 只记录，不报错、不回退。
6. artifact 升级为 schema v3，同时保存：
   - `candidate_z`
   - `adopted_z`
   - Router KL / rollback policy / R64 / rot order 等可重放语义。
7. E2E eval 支持 `--artifact_diag_variant adopted|candidate`。
8. Qwen3-MoE `start_layer>0` resume 已修正：
   - 不再覆盖旧 `config.json`
   - replay `[0,start_layer)` adopted/native 前缀
   - 新旧 rollback 语义兼容
   - 最终 artifact 合并旧 prefix + 新后缀。
9. 当前 `evaluation/summarize.py` 仍主要按旧 metrics 字段统计，必须在正式实验前同步升级，否则新的 `loss_would_rollback / router_would_rollback` 等信息会被错误汇总。

因此，原 Task 14–19 的实验结果不能机械延续；必须重新划分“可复用、仅作历史参考、必须重跑”。

---

# 2. 当前结果复用审计

旧 Phase A 目录：

`Native_NVFP4_HiF4_Linear_Puncture/results/e2e_diag_reconstruction/phase1_20260824T052122Z/`

## 2.1 可直接复用

### A. Shared calibration cache

`s1k_original / train128 / val32 / seed42` 的 shared cache 可以直接复用。

只要以下字段一致，不重做数据准备：

```text
calib_source = s1k_original
calib_nsamples = 128
calib_val_nsamples = 32
calib_seed = 42
```

### B. Task 0–13 的未受影响设计结论

以下结论继续复用：

```text
48 layers
128 experts/layer
top-8
Router BF16
KV BF16
TP=2 formal eval
no native online Hadamard
Fusable = shared D_QKV / D_VO / D_GU + per-expert D_UD
Online = expert-specific upper bound
R64 = fixed R4⊗R4⊗R4
```

Native TP1/TP2 parity 的 Case-A 诊断也继续作为历史 correctness 证据，不重复完整 puncture 调查。

### C. 旧 Online E5/E6 也不进入正式复用

虽然 Online 数学本身没有因为 Router/Fusable 重构发生改变，但进一步代码审计发现：当前 MoE trainer 没有实际使用 `cfg.diag_scheduler`。旧 E5/E6 的 `config.json` 写的是 `diag_scheduler=cosine`，实际训练却一直使用恒定学习率。

因此旧 E5/E6 即使某些前缀层是在坏 resume 之前连续训练得到，也仍然不满足原正式实验定义：

```text
formal recipe 声明 = cosine
old MoE 实际执行 = constant LR
```

再叠加后续坏 resume 的 hidden-cache 错位，旧 E5/E6 只能保留为探索性历史记录，**正式 corrected Phase-A 必须从 layer0 全部重训**。

这也避免在同一个 48-layer 模型中混合“旧 constant-LR prefix + 新 cosine-LR suffix”。

## 2.2 只能作历史参考，禁止进入正式排名

### E3 Fusable DIAG

旧 E3 48 层全部发生 `router_gate_rollback=true`；而旧实现：

```text
wrong Router input
+ strict BF16 top-k gate
+ Router failure => whole-layer DIAG zero
+ candidate DIAG 未保存
```

因此旧 E3 最终 artifact 本质接近 Direct HiF4，不能用于评价 Fusable DIAG。

### E4 Fusable DIAG → R64

同 E3，旧 E4 48 层全部 Router rollback，不能用于评价“DIAG 是否在 R64 下联合优化有效”。

旧 E3/E4：

```text
保留目录
保留旧 summary
保留为 bug/history evidence
禁止进入新的 run_map
禁止作为 B_all / Phase-C baseline
```

## 2.3 必须新做

```text
E0 Native NVFP4 formal fast eval
E1 Direct HiF4 formal fast eval
E2 Direct HiF4 + R64 formal fast eval
E3 corrected Fusable DIAG full retrain
E4 corrected Fusable DIAG → R64 full retrain
E5 Online DIAG full retrain
E6 Online DIAG → R64 full retrain
E7 Online R64 → DIAG full retrain
```

当前旧 Phase-A 目录没有完成统一的 E0–E7 formal fast eval，因此 E0/E1/E2 不训练，但必须在 corrected Phase-A 下重新评测。

---

# 3. 全局正式实验规则

除某个 Task 明确改变某一因素外，默认全部锁死：

```text
model = nvidia/Qwen3-30B-A3B-NVFP4
calib = s1k_original
train = 128
val = 32
seed = 42
batch = 4
recon_loss = block_delta_nmse
optimizer = AdamW
weight_decay = 0.0
lr = 5e-3
epochs = 20
scheduler = cosine，按 optimizer step 调度
clamp = [-4,4]
calib_input_mode = progressive_student
loss_rollback = on
KV = BF16
TP = 2 formal eval
eval_seed = 42
```

Fusable 默认 Phase-A 基线显式写：

```text
router_align_loss_weight = 0.0
router_rollback = on
```

不要依赖 `inherit` 作为新 Fusable 正式结果的实验描述；manifest 中必须能直接看出正式 policy。

Online 沿用旧行为；Router rollback 对 Online 不产生算法效果，不单独做 Router 消融。

正式 winner 排序始终：

```text
1. MMLU-Pro 300
2. ARC-Challenge
3. ARC-Easy
```

不使用 layer reconstruction proxy 决定最终 winner。

Candidate artifact 的 E2E 结果只作为机制诊断；**正式 winner 默认只从真实训练-policy 的 adopted run 中选择**。

## 正式 generation / reasoning 协议（Qwen3 官方推荐）

本轮 E0/E1/E2/E3/E4/E5/E6/E7 以及后续所有同任务对照必须共享完全一致的生成协议；禁止某个 variant 单独改变 thinking 或 sampling 参数。

Qwen3 默认启用 thinking。正式 MMLU-Pro、AIME 与 LiveCodeBench 都采用 thinking 模式，并使用官方推荐：

```text
enable_thinking = true
temperature = 0.6
top_p = 0.95
top_k = 20
min_p = 0
do_sample = true
```

输出长度：

```text
MMLU-Pro300:
  max_new_tokens = 32768

AIME25 avg@5:
  max_new_tokens = 38912

LiveCodeBench codegeneration_v6:
  max_new_tokens = 38912
```

Qwen3 官方建议多数问题使用 32768；复杂数学/编程 benchmark 建议 38912。因此 AIME 与 LiveCodeBench 使用 38912，MMLU-Pro 使用 32768。禁止 greedy decoding。

共同 runtime：

```text
TP = 2
KV = BF16
max_model_length = 40960
enforce_eager = true
```

旧的 `MMLU-Pro enable_thinking=false + temperature=0.7 / top_p=0.8 / top_k=20` 不再作为 corrected Task14+ 的正式协议。

---

# Task 14：重构后 correctness gate + corrected Phase A E0–E7

> **2026-08-28 runtime 适配备注**：E2 旧评测是在优化后的 `R64 -> HiF4 QDQ` Triton runtime 接入前启动，现有 E2 结果仅保留作历史诊断，不进入正式 E0–E7 对比。当前默认从 E3 继续；E3–E7 以及未来补跑 E2 必须使用 HiF4 runtime ABI v3。启动层会隔离无 ABI/旧 ABI 的 materialized checkpoint 与 metrics；vLLM runtime 也会拒绝旧 `r64/fusable/online` sidecar。未来正式补 E2 时显式设置 `RUN_E2_WITH_ABI3=1`。

## Task 14.0：先修正式实验基础设施

### 14.0.1 更新 summarizer

修改：

`evaluation/summarize.py`

当前 `layer_stats()` 仍查旧的 `would_rollback`，并依赖 summary 顶层 `accepted/rollback`；新 trainer 的 summary 以 per-layer metrics 为准。

必须改成逐层汇总：

```text
accepted_count
rollback_count
loss_would_rollback_count
loss_rollback_applied_count
router_would_rollback_count
router_rollback_applied_count
candidate_best_val_loss mean/median
adopted_val_loss mean/median
candidate_best_router_kl mean/median
adopted_router_kl mean/median
candidate_router_topk_mismatch_tokens sum
candidate_router_topk_mismatch_ratio mean
router_topk_mismatch_tokens sum
router_topk_mismatch_ratio mean
```

`collect_run_row()` 增加：

```text
loss_rollback
router_rollback
router_align_type
router_align_temperature
router_align_loss_weight
artifact_diag_variant
```

Candidate diagnostic 行必须有明确标记，禁止被 `best_fusable / best_online / best_overall` 自动选中。

### 14.0.2 增加 adopted/candidate eval helper

当前 `fast_eval_run()` 不能传 `--artifact_diag_variant`。

修改 `scripts/common.sh`，新增单独 helper，例如：

```text
fast_eval_artifact_variant_run GPU OUT_DIR ARTIFACT adopted|candidate
```

不要改变旧 `fast_eval_run()` 默认 adopted 行为。

Candidate eval 必须使用独立 output dir，不能覆盖 adopted 的：

```text
eval/arc/metrics.json
eval/mmlu_pro/metrics.json
```

建议 companion dir：

```text
<phase>/diagnostics/E3_fusable_candidate/
<phase>/diagnostics/E4_fusable_r64_candidate/
```

companion dir 只保存：

```text
config.json（注明 source_run + artifact_diag_variant=candidate）
eval/*
```

不复制模型权重。

### 14.0.3 修正 MoE trainer 中“配置写了但没生效”的训练语义

这一步是 corrected Phase-A 的硬门禁。

#### A. `diag_scheduler`

当前 MoE trainer 只创建 optimizer，没有 scheduler；因此旧配置写 `cosine` 实际执行 constant LR。

修正为与 dense 路径一致的 per-step cosine：

```text
n_batches = 当前 train epoch 的 batch 数
T_max = diag_epochs * n_batches
CosineAnnealingLR(eta_min=0)
optimizer.step()
scheduler.step()
```

`diag_scheduler=constant` 时不创建 scheduler。

必须新增测试验证：

```text
cosine: LR 随 step 下降
constant: LR 保持不变
```

#### B. optimizer / weight decay

当前 MoE 使用 `AdamW(...)` 默认 `weight_decay=0.01`，这是一个未显式声明但会把 log2-DIAG 拉向 0（即 D=1）的正则项。

新正式口径锁死：

```text
optimizer = AdamW
weight_decay = 0.0
```

不得继续依赖 PyTorch 默认 weight decay。建议把 optimizer/weight_decay 写入 config 与 artifact manifest，确保后续 recipe clone 可追踪。

#### C. `diag_log2_clamp=none`

当前 MoE `_train_layer()` 使用：

```text
cfg.diag_log2_clamp or (-4,4)
```

导致 `none` 仍被强制 clamp。

修正语义：

```text
if clamp is not None:
    clamp
else:
    不做任何 clamp
```

并新增 unit test 锁死 `none` 真正无裁剪，否则 Task17 clamp ablation 禁止执行。

#### D. `fusable_diag_components`

当前 MoE 路径只解析/保存 `fusable_diag_components`，没有像 dense `LayerDiagState.configure_fusable_components()` 那样实际冻结未选中的参数。

在 `MoEFusableDiagState` 增加等价 component mask：

```text
qkv  -> z_qkv trainable only
vo   -> z_vo trainable only
gu   -> z_gu trainable only
ud   -> z_ud trainable only
attn -> z_qkv + z_vo
mlp  -> z_gu + z_ud
all  -> all
```

optimizer 只接收 `requires_grad=True` 的参数。

GU 不活跃时必须验证：

```text
z_gu 始终为0
D_GU 始终为1
Router KL 不更新 z_gu
Router rollback 不应被 component run 触发
```

否则 Task16 component ablation 禁止启动。

### 14.0.4 修 stale help / metadata

`cli/evaluate.py` 当前 help 仍写“MMLU-Pro temporarily suspended”，但正式 `FAST_EVAL_GROUPS` 已是 `arc,mmlu_pro_300`。

同步修正，避免 Cursor/操作者误判。

## Task 14.1：重构后测试门禁

在任何正式 train 前执行：

```bash
conda run -n hif4 pytest \
  Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction -q
```

当前预期至少：

```text
105 passed
```

再执行旧 8B 回归，确认 schema/artifact 改动没有破坏 dense 历史路径。

## Task 14.2：新增真实模型 mini smoke

### A. Fusable λ=0 smoke

真实 layer0：

```text
train=2
val=1
epoch=1
router_align_loss_weight=0
router_rollback=on
loss_rollback=on
```

必须验证：

```text
candidate_best_diag.pt exists
best_diag.pt exists
candidate_metrics.json exists
conversion artifact schema=3
candidate_z/adopted_z both exist
如果 Router rollback，则仅 z_gu 不同
```

### B. Fusable λ=0.5 smoke

同上，只改：

```text
router_align_loss_weight=0.5
```

验证：

```text
router KL finite
candidate_best_router_kl recorded
objective = recon + 0.5 * KL
D_QKV/D_VO/D_UD 不接收 Router-KL-only gradient（由 unit test + smoke metrics共同确认）
```

### C. adopted/candidate materialize + TP2 generate

同一个 schema-v3 artifact 分别：

```text
artifact_diag_variant=adopted
artifact_diag_variant=candidate
```

均能从原 packed NVFP4 checkpoint 重建、materialize、TP2/BF16-KV generate。

### D. real resume parity smoke

做一个极小的 3-layer 对照：

```text
Run-A: layer0→2 连续
Run-B1: layer0→1
Run-B2: start_layer=2 resume
```

设置：

```text
train=2
val=1
epoch=1
seed相同
```

要求 Run-A 与 Run-B 在 layer2 的：

```text
identity input / val loss
best_epoch
candidate/adopted DIAG
```

在确定性允许的数值容差内一致。

这一步不过，不得启动任何可能需要中断后续跑的正式 48-layer run；corrected E5/E6 本轮仍从 layer0 全新训练。

## Task 14.3：建立 corrected Phase-A 新目录

新目录禁止复用旧 `phase1_20260824T052122Z` 作为正式汇总根目录。

建议：

```text
results/e2e_diag_reconstruction/phaseA_refactor_20260825_<STAMP>/
```

### E3

从 layer0 完整重训：

```text
diag_mode=fusable
use_r64=false
router_align_loss_weight=0
loss_rollback=on
router_rollback=on
```

### E4

从 layer0 完整重训：

```text
diag_mode=fusable
use_r64=true
rot_order=diag_then_rot
router_align_loss_weight=0
loss_rollback=on
router_rollback=on
```

### E5

从 layer0 完整重训：

```text
online
use_r64=false
rot_order=diag_then_rot
layer0–47
```

### E6

从 layer0 完整重训：

```text
online
use_r64=true
rot_order=diag_then_rot
layer0–47
```

### E7

完整新训练：

```text
online
use_r64=true
rot_order=rot_then_diag
layer0–47
```

## Task 14.4：Phase-A artifact completeness gate

E3–E7 每个 adopted artifact 必须：

```text
schema_version = 3
n_layers = 48
layer_ids = 0..47
source_model correct
use_r64 / rot_order correct
KV=bfloat16
```

E3/E4 additionally：

```text
router_align_loss_weight=0
resolved_loss_rollback_enabled=true
resolved_router_rollback_enabled=true
```

E5/E6 本轮也必须使用全新 schema-v3 训练结果；旧 constant-LR / 坏-resume artifact 不得混入 corrected Phase-A。

## Task 14.5：正式 E0–E7 fast eval

全部：

```text
ARC-Easy
ARC-Challenge
MMLU-Pro300
TP=2
KV=BF16
eval_seed=42
```

并严格遵循本计划前述“Qwen3 官方推荐 generation / reasoning 协议”。E0 Native baseline 必须在该协议下 fresh eval，不能直接拿旧 TP1 / KV-auto / 旧 sampling 结果作为正式 E0。

矩阵：

| ID | Method | Train source |
|---|---|---|
| E0 | Native NVFP4 | fresh eval only |
| E1 | Direct HiF4 | fresh eval only |
| E2 | Direct HiF4 + R64 | fresh eval only |
| E3 | corrected Fusable | full new |
| E4 | corrected Fusable→R64 | full new |
| E5 | Online | full new |
| E6 | Online→R64 | full new |
| E7 | Online R64→DIAG | full new |

正式排名只用 adopted。

### Task 14.5b：E0/E1/E2 LiveCodeBench 基线

为尽早观察“Native NVFP4 → Direct HiF4 / Direct HiF4+R64”对代码生成能力的损失，E0/E1/E2 在 Task14 阶段额外跑完整：

```text
LiveCodeBench task = lcb:codegeneration_v6|0
max_samples = full / None
thinking = true
temperature = 0.6
top_p = 0.95
top_k = 20
min_p = 0
max_new_tokens = 38912
max_model_length = 40960
TP = 2
KV = BF16
enforce_eager = true
```

执行对象固定：

```text
E0 Native NVFP4
E1 Direct HiF4
E2 Direct HiF4 + R64
```

E3–E7 及 Task15–18 中间大规模消融不跑 LiveCodeBench，避免把高成本代码生成评测扩散到所有 run。LiveCodeBench 不参与中间 winner 排名；Task19 再对最终 finalists 统一复测/补测。

结果落盘：

```text
<run>/eval/livecodebench/metrics.json
metric = codegen_pass@1:16
```

## Task 14.6：Fusable candidate diagnostic eval

额外对 E3/E4：

```text
artifact_diag_variant=candidate
ARC-E / ARC-C / MMLU-Pro300
```

目的：在完全相同训练轨迹下，直接看“Router rollback 后 adopted”与“保留 candidate D_GU”的部署结果差异。

注意：

```text
candidate eval != router_rollback=off training policy
```

因为下一层训练 hidden 已经受 adopted policy 影响；candidate 只做部署时替换诊断，不作为正式 policy winner。

Task 14 输出：

```text
best_fusable_structure = E3 or E4
best_online_structure = E5/E6/E7
best_direct = E1 or E2
```

---

# Task 15：新增 Router 策略 2×2 正式消融

这是本轮重构后必须新增的正式实验。

先按 Task14 adopted ranking 选：

```text
F* = best(E3, E4)
```

F* 的 `use_r64 / rot_order` 固定不再变化。

只改变两个因素：

```text
A: router_align_loss_weight ∈ {0.0, 0.5}
B: router_rollback ∈ {on, off}
```

`loss_rollback` 在所有四格中固定 `on`。

## 15.1 2×2 矩阵

| ID | KL weight | Router rollback | 来源 |
|---|---:|---|---|
| R00 | 0.0 | on | reuse F* |
| R10 | 0.5 | on | new full run |
| R01 | 0.0 | off | new full run |
| R11 | 0.5 | off | new full run |

总新增完整训练：3 runs。

## 15.2 每个 run 统一记录

```text
candidate recon loss
candidate Router KL
candidate top-k mismatch count
candidate mismatch-token ratio
adopted recon loss
adopted Router KL
adopted mismatch count
D_GU rollback layer count
loss rollback layer count
D_GU z distribution
ARC-E
ARC-C
MMLU-Pro300
```

## 15.3 paired candidate diagnostic

对 R00 和 R10 的 schema-v3 artifact 再跑 candidate E2E。

R00 candidate 若已由 Task14 对 F* 跑过，直接复用。

用于分开回答两个问题：

```text
训练 policy 层面：router_rollback on vs off
部署采用层面：同一 on-policy artifact adopted vs candidate
```

## 15.4 winner

正式 Router policy winner：

```text
F_router* = best adopted run among R00/R10/R01/R11
```

禁止把 candidate diagnostic alias 直接选成 `F_router*`。

---

# Task 16：Fusable component ablation，改为围绕最终 Router policy

旧 Task15 固定 `use_r64=false`，现在不再合理，因为如果 F* 是 E4，强行去掉 R64 会同时改变结构背景和 DIAG component。

新规则：

```text
inherit F_router* 的：
  use_r64
  rot_order
  router_align_loss_weight
  router_rollback
  loss_rollback
  recon_loss
  input mode
  clamp
```

只改变 `fusable_diag_components`。

## 16.1 结构参考 B0

若 `F_router*.use_r64=false`：

```text
B0 = E1 Direct HiF4
```

若 `F_router*.use_r64=true`：

```text
B0 = E2 Direct HiF4 + R64
```

## 16.2 component 矩阵

```text
B1 qkv
B2 vo
B3 gu
B4 ud
B5 attn = qkv+vo
B6 mlp  = gu+ud
B7 all  = reuse F_router*
```

新增 6 个完整训练 run。

GU 不参与的 component：

```text
qkv / vo / ud / attn
```

Router KL/rollback 数学应为 no-op；实现中允许保留 recipe metadata，但必须验证 D_GU=identity、Router KL 不产生非零训练梯度。禁止因为无 GU 而误更新 z_gu。

B3/B6 额外跑 candidate eval，用来查看 D_GU 回退本身对 component 恢复的影响。

报告：

```text
vs B0 E2E gain
layer reconstruction recovery
Router KL/mismatch
D_GU rollback count
per-expert D_UD statistics
expert coverage
```

---

# Task 17：训练机制与稳定性

这部分替代旧 Task16。

首先固定：

```text
F_base = F_router*
O_base = Task14 best online structure
```

所有 ablation 都是**one-factor-at-a-time**。

禁止看到多个单因素各自有收益后，直接把它们拼成一个未测试组合。

最终 `F_mech* / O_mech*` 从“真实运行过的 baseline + 单因素变体”中按正式排名选。

## 17.1 Input mode

Baseline：

```text
progressive_student = reuse F_base / O_base
```

新增：

```text
F teacher-input
O teacher-input
```

2 runs。

这里的 `teacher` 仅表示 native hidden propagation，不是 Teacher-CoT；仍然不恢复 Teacher-CoT 数据生成。

## 17.2 Reconstruction loss

Baseline：

```text
block_delta_nmse = reuse
```

新增：

```text
F block_output_nmse
F mse
O block_output_nmse
O mse
```

4 runs。

Fusable 每个变体必须继承 F_base 的 Router KL/rollback policy，不能退回默认参数。

## 17.3 Loss rollback

Router rollback 已在 Task15 完整做 2×2，不在这里重复。

这里只测 reconstruction rollback：

```text
F loss_rollback=on  = reuse F_base
F loss_rollback=off = 1 new run
```

Router policy保持 F_base 不变。

1 run。

## 17.4 Clamp

```text
F clamp=-4,4 = reuse F_base
F clamp=none = 1 new run
```

1 run。

Task17 总新增：8 runs。

## 17.5 recipe cloning 基础设施

旧 `run_input_mode_ablation.sh / run_loss_ablation.sh / run_clamp_ablation.sh / run_calib_ablation.sh` 只继承 structure preset，会丢失新的 Router policy、loss policy 等配置，因此不能直接作为正式脚本继续使用。

新增统一 recipe helper，例如：

```text
scripts/utils/recipe_args.py
```

输入：

```text
--source_run <dir>
--override key=value
```

只输出训练 CLI 可接受字段，并自动排除：

```text
output_dir
start_layer
end_layer
model_type/num_layers/num_experts 等 derived readonly metadata
```

所有 Task17/18 新脚本必须从 reference run 的 `config.json` 克隆 exact recipe，只覆盖被消融的单个因素。

这样避免：

```text
F_router* = KL0.5/router_rb_off
但 loss ablation 脚本偷偷恢复 KL0/router_rb_on
```

这种无效对照。

---

# Task 18：Calibration source ablation

先选：

```text
F_mech* = best measured fusable run from Task15/17
O_mech* = best measured online run from Task14/17
```

然后 exact clone recipe，只改变 calibration source。

比较：

```text
s1k_original
s1k_question
wikitext2
c4
```

`s1k_original` 直接复用 F_mech*/O_mech*。

新增：

```text
F s1k_question
F wikitext2
F c4
O s1k_question
O wikitext2
O c4
```

6 runs。

每个 source 必须报告：

```text
train sample count
val sample count
total tokens
P50/P95/max length
128-expert train routed coverage
128-expert val routed coverage
never-routed experts
min/median/max routed tokens
```

仍然只解释为“实用 calibration recipe 比较”，不宣称等 token-budget domain study。

输出：

```text
F_cal* = best fusable calibration run
O_cal* = best online calibration run
```

---

# Task 19：Finalists AIME25 avg@5 + LiveCodeBench

不对所有 ablation 做 AIME / LiveCodeBench。

候选池：

```text
1. E0 Native NVFP4
2. best_direct = best(E1,E2)
3. F_mech*（s1k_original 最佳实际 recipe）
4. O_mech*
5. F_cal*，若与 F_mech* 不是同一个 run
6. O_cal*，若与 O_mech* 不是同一个 run
```

去重后统一跑：

```text
AIME25 avg@5
LiveCodeBench codegeneration_v6
TP=2
KV=BF16
enforce_eager=true
```

其中 AIME 与 LiveCodeBench 都严格使用前述 Qwen3 官方 thinking 协议；LiveCodeBench 指标固定为 `codegen_pass@1:16`。E0/E1/E2 若 Task14 已有完全相同协议的 LCB 结果可直接复用，不重复跑；其余 finalist 补跑。

ARC/MMLU-Pro300 已有结果直接复用，不重复跑。

Candidate diagnostic 版本默认不进入 AIME/LiveCodeBench finalist。

---

# Task 20：最终报告

新报告建议：

`Native_NVFP4_HiF4_Linear_Puncture/results/e2e_diag_reconstruction/QWEN3_30B_A3B_MOE_NVFP4_TO_HIF4_REFACTORED_ABLATION_REPORT_2026-08-25.md`

必须单独说明旧 Phase-A 为什么不能正式复用：E3/E4 受错误 Router gate 输入、整层 rollback 和 candidate 丢失影响；E5/E6 还存在“配置声明 cosine、实际 constant LR”及坏 resume 后缀问题。旧结果不得与 corrected E3–E7 混表。

## 20.1 主表

1. Native / Direct / R64 baseline：E0/E1/E2
2. corrected Phase A：E3–E7
3. Router 2×2：KL × Router rollback
4. Candidate vs adopted deployment diagnostic
5. Fusable components
6. Input / reconstruction loss / loss rollback / clamp
7. Calibration source + expert coverage
8. AIME + LiveCodeBench finalists

## 20.2 Fusable 每个 run 至少报告

```text
use_r64
rot_order
router_align_loss_weight
loss_rollback policy
router_rollback policy
candidate/adopted artifact variant
candidate reconstruction loss
candidate Router KL
candidate top-k mismatch count / token ratio
adopted reconstruction loss
adopted Router KL
adopted top-k mismatch count / token ratio
loss rollback layer count
router rollback layer count
ARC-E
ARC-C
MMLU-Pro300
AIME if finalist
LiveCodeBench if finalist
vs E0 pp drop
```

## 20.3 Online 每个 run 至少报告

```text
use_r64
rot_order
recon loss
loss rollback
accepted/rollback layers
expert coverage
ARC-E
ARC-C
MMLU-Pro300
AIME if finalist
LiveCodeBench if finalist
vs E0 pp drop
```

## 20.4 结论边界

必须明确：

```text
Online = correctness-first 高自由度上界
Fusable = 当前主要部署候选
candidate artifact = 训练候选诊断/可重放状态，不自动等于正式 deployment policy
```

---

# 4. 新增/复用训练成本

从现在开始的训练任务：

```text
Task14 corrected Phase A:
  E3/E4/E5/E6/E7 full new = 5

Task15 Router 2×2:
  baseline reuse    = 0
  new               = 3

Task16 components:
  B_all reuse       = 0
  new               = 6

Task17 mechanism:
  new               = 8

Task18 calibration:
  s1k baseline reuse
  new               = 6
```

正式资源表：

```text
Phase A full: 5
Router:       3
Components:   6
Mechanism:    8
Calibration:  6
----------------
Total full:  28 个 48-layer training jobs
```

不再复用旧 E5/E6 prefix，因为旧 MoE training 的实际 scheduler 与正式配置不一致。

E0/E1/E2 只评测，不训练。

---

# 5. GPU 与执行顺序

训练：

```text
1 GPU / run
lazy layer materialization
可按 available GPU pool 并行不同 run
同一个 run 不跨 GPU 做 layer 并行
```

评测：

```text
2 GPU / eval
TP=2
KV=BF16
```

执行顺序严格：

```text
Task14.0 MoE scheduler/optimizer/clamp/component + summarizer/eval helper 修整
  ↓
Task14.1 full regression
  ↓
Task14.2 real mini smoke + real resume parity
  ↓
Task14.3 corrected E3–E7 全部从 layer0 重训
  ↓
Task14.4 artifact completeness
  ↓
Task14.5 E0–E7 adopted fast eval
  ↓
Task14.6 E3/E4 candidate diagnostic
  ↓
Task15 Router KL × rollback 2×2
  ↓
Task16 components around F_router*
  ↓
Task17 mechanism/stability
  ↓
Task18 calibration source
  ↓
Task19 AIME finalists
  ↓
Task20 report
```

禁止 Task14 corrected Phase A 未完成就启动 component/calibration；否则 baseline 会再次漂移。

---

# 6. 需要新增/修改的脚本与代码

建议新增：

```text
scripts/ablation/run_phaseA_refactor_matrix.sh
scripts/ablation/run_router_policy_ablation.sh
scripts/ablation/run_fusable_component_ablation_v2.sh
scripts/ablation/run_input_mode_ablation_v2.sh
scripts/ablation/run_loss_ablation_v2.sh
scripts/ablation/run_loss_rollback_ablation_v2.sh
scripts/ablation/run_clamp_ablation_v2.sh
scripts/ablation/run_calib_ablation_v2.sh
scripts/utils/recipe_args.py
```

建议修改：

```text
core/config.py                         # 显式 optimizer/weight_decay metadata
core/moe_semantic_hif4.py             # MoE fusable component mask
training/moe_trainer.py               # scheduler / wd=0 / clamp-none / active params
core/artifact.py                       # optimizer metadata进入 schema-v3 manifest
evaluation/summarize.py
cli/evaluate.py
evaluation/lcb_runner.py              # LiveCodeBench companion evaluator
cli/evaluate_lcb.py                   # LCB 独立 CLI
scripts/common.sh
scripts/eval/run_final_eval.sh        # finalists 同时跑 AIME + LiveCodeBench
相关 test_config / test_moe_semantic_hif4 / test_moe_resume / trainer tests
```

旧脚本保留，不删除，以保证历史 run 可追溯；正式 Task14+ orchestrator 只调用 v2/refactor 脚本。

---

# 7. 最终 Acceptance Criteria

1. `pytest Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction -q` 全通过；当前基准至少 105 passed，新增 scheduler/component/clamp 测试后基准只增不减。
2. 旧 dense/8B 测试不因 schema-v3 改动失败。
3. MoE `diag_scheduler=cosine` 真正按 optimizer step 执行；`constant` 保持恒定 LR。
4. 正式 MoE optimizer 显式为 `AdamW(weight_decay=0.0)`，不再依赖默认 0.01。
5. `diag_log2_clamp=none` 真正不做 clamp。
6. MoE `fusable_diag_components` 真正冻结未选 component；GU 未激活时 `z_gu=0 / D_GU=1` 且 Router KL 不更新它。
7. real 3-layer continuous vs resume smoke 通过，证明未来中断后的正式 run 可以安全续跑。
8. corrected E3/E4/E5/E6/E7 全部从 layer0 新训练，不读取旧 Phase-A DIAG。
9. E3–E7 最终 conversion artifact 均为完整 48-layer schema-v3。
10. Fusable 正式 artifact 同时包含 candidate/adopted DIAG，并能从原 packed NVFP4 checkpoint 重放。
11. 正式 Phase-A Fusable 基线显式为 `KL=0 / loss_rb=on / router_rb=on`。
12. Router 2×2 完成 `KL {0,0.5} × router rollback {on,off}`，且只从 adopted full-policy run 中选正式 winner。
13. Router KL 梯度仅更新 `D_GU`。
14. Router rollback 仅回退 `D_GU`；loss rollback 才允许整层 DIAG identity。
15. Component ablation 继承 F_router* 的结构/R64/Router policy，不再强行固定 no-R64。
16. Task17/18 使用 exact recipe clone，只改变单一消融因素。
17. Summarizer 正确识别 split rollback、Router KL、candidate/adopted，不再依赖旧 `would_rollback` 字段。
18. Candidate diagnostic 不进入正式 `best_fusable/best_online/best_overall` 自动排名。
19. E0–E7 fast eval 全部统一 ARC-E/C + MMLU-Pro300，TP2、BF16 KV。
20. Calibration ablation 报告 token 统计和 expert coverage。
21. E0/E1/E2 在 Task14 额外完成完整 LiveCodeBench `codegeneration_v6`，指标为 `codegen_pass@1:16`。
22. AIME25 与 LiveCodeBench 只对去重后的 finalists 做正式终评；E0/E1/E2 相同协议结果允许直接复用。
23. LiveCodeBench 与 AIME 均使用 Qwen3 thinking 官方参数 `0.6/0.95/20/min_p=0`，LCB `max_new_tokens=38912`。
24. 临时 materialized BF16 checkpoint 默认评测后删除；长期只保存 compact artifact 和 metrics。
25. 最终报告明确区分旧 invalid Phase-A training 与 corrected refactor results。

---

# 8. 给 Cursor/Codex 的执行约束

- 不允许把旧 E3–E6 的任何 DIAG、训练 prefix 或 final artifact 混入 corrected Phase-A；E3–E7 本轮全部从 layer0 新训练。
- resume 机制只用于 corrected 新 run 未来发生中断时续跑；不得拿它作为复用旧 Phase-A constant-LR 结果的理由。
- 不允许在 Task15 后的脚本只继承 `structure preset` 而丢失 Router policy；必须 exact recipe clone。
- 不允许把 candidate E2E 当成 `router_rollback=off` 的完整训练替代品。
- 不允许把多个 one-factor winner 自动组合成一个未实测 recipe。
- 不允许修改 E0 Native NVFP4 数学来追求文本 exact-match。
- 不允许恢复 Teacher-CoT；`calib_input_mode=teacher` 仅是 native hidden propagation。
- 不使用完整 30B BF16 checkpoint 作为长期实验产物。
- 遇到事实性 runtime/checkpoint 结构变化必须停在对应 Task 记录，不凭猜测改变模型语义。
