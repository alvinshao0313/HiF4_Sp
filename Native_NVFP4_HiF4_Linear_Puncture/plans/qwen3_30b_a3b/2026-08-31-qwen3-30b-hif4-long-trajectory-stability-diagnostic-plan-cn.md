> 状态：已被 2026-09-01-qwen3-30b-hif4-long-trajectory-real-vllm-hook-replan-cn.md 取代。
> 仅用于历史问题复盘，不再执行。

# Qwen3-30B-A3B NVFP4→HiF4 长轨迹稳定性诊断实验执行计划

> **状态：已停止执行。** 本计划中的 semantic replay 正式路径已被 `2026-09-01-qwen3-30b-hif4-long-trajectory-real-vllm-hook-replan-cn.md` 取代。仅保留作历史实验设计与排错记录；后续长轨迹机制实验不得再按本文的 `causal_replay.py` / `run_semantic_matrix.py` 路线推进。

> 日期：2026-08-31  
> 目标模型：`nvidia/Qwen3-30B-A3B-NVFP4`  
> 复用实验：`results/e2e_diag_reconstruction/phaseA_refactor_20260825T035730Z`  
> 新实验代码：`Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/`  
> 新结果根目录：`Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/`  
> 执行者：Cursor。Cursor 负责运行、监控和整理结果，不自行重新设计实验。

---

# 1. 研究问题

Phase-A 当前有三个值得验证的现象：

1. E2 `R64-only` 在 ARC 类固定上下文 likelihood 任务上较接近/优于 Direct HiF4，但 MMLU-Pro reasoning 表现偏低；
2. E3 fusable DIAG 没有表现出 E2 那样明显的 reasoning drop；
3. 8B Linear 穿刺中 `DIAG→R64` 的局部输出 NMSE 明显优于单独 DIAG/R64，但 30B MoE E4 并未出现同幅度的端到端叠加收益。

本实验不能预设“R64 产生随机信息损失”。R64 是正交变换，无量化时本身不丢信息。需要区分的是：

\[
Q(XR),\;Q(WR)
\]

或

\[
Q(XDR),\;Q(WD^{-1}R)
\]

产生的**量化误差方向**，是否在长 autoregressive reasoning 中通过 decision boundary、MoE routing 或 state distribution drift 被放大。

---

# 2. 必须区分的三个机制假设

## H1：long-context / state-distribution drift

固定所有模型都沿 E0 的同一条 token trajectory 前进后，E2/E4 相对 E0 的误差仍随 decode position 增长。

支持 H1 的主要证据：

- fixed-path teacher-forcing 的 full-vocab logit KL 从 early bin 到 late bin 显著增长；
- Layer 47 hidden rel-L2 同样随 decode position 增长；
- top1 agreement 随位置持续下降；
- 这种趋势在 E2/E4 上明显强于 E1/E3。

若成立，可表述为：

> R64/DIAG+R64 改变 HiF4 量化误差后，在越来越长的 reasoning state 分布上出现持续偏移；free-run 分叉是这种 state drift 的结果之一。

不能表述为“每个 token 独立随机噪声线性累加”。

## H2：decision-margin driven trajectory bifurcation

固定 E0 trajectory 后，E2/E4 的 KL/hidden error 并没有明显随长度持续恶化；但 free-run 在 E0 本身 top1/top2 margin 很小的位置发生 token 分叉。一旦 token 发生变化，后续输入条件不同，轨迹自然分开。

支持 H2 的主要证据：

- E2/E4 free-run 首次分叉位置明显更早；
- teacher-forcing late/early KL ratio 不高；
- 首次分叉点 E0 的 top1-top2 logit margin 显著低于统一 probe position；
- 首次分叉点前的 fixed-path hidden/logit error 未必很大。

若成立，更准确的结论是：

> R64 的残余量化误差方向提高了模型在小 decision margin 位置发生 token rank reversal 的概率，导致 autoregressive trajectory bifurcation。

## H3：MoE router boundary amplification

误差先改变 hidden/router input，再经过 top-k expert selection 形成离散放大。

支持 H3 的主要证据：

- 首次 token 分叉附近，某些 Transformer layer 的 router top-k exact rate 显著低于统一位置；
- router top-k overlap 下降；
- router top-k boundary margin 更小；
- E2/E4 的这种 enrichment 强于 E1/E3。

若成立，后续算法优化应从仅优化 block reconstruction NMSE 转向加入 router-aware / margin-aware 约束。

---

# 3. 为什么采用两条证据链，而不是直接看 MMLU-Pro accuracy

当前 MMLU-Pro300 单个样本就是约 0.33 pp，而且生成式评测还受到 sampling 的影响。只看 E3 70.67 vs E4 70.33 这样的差异不能解释机制。

因此实验分为：

1. **deterministic greedy free-run**：回答“真实 autoregressive trajectory 什么时候第一次分开”；
2. **fixed E0-token causal replay**：回答“在完全相同 token history 下，误差是否本身随 reasoning position 增长，以及 router/hidden/logit 哪一层先出现异常”。

两者组合才能区分 H1 与 H2/H3。

---

# 4. 固定方案和数据，不允许 Cursor 自行改动

| ID | 方案 | 来源 |
|---|---|---|
| E0 | native NVFP4 | Phase-A E0 |
| E1 | Direct HiF4 | Phase-A E1 |
| E2 | R64-only | Phase-A E2 |
| E3 | adopted fusable DIAG | Phase-A E3 |
| E4 | adopted fusable DIAG + R64 | Phase-A E4 |

正式 free-run 固定：

- Dataset：MMLU-Pro；
- `max_samples=64`；
- `max_new_tokens=16384`；
- `temperature=0.0`；
- `top_p=1.0`；
- `top_k=-1`；
- thinking 保持开启；
- TP=2；
- KV cache=BF16；
- `enforce_eager=true`；
- E1–E4 必须是 HiF4 runtime ABI 3；
- 不使用旧 E2 pre-ABI3 metrics。

注意：greedy 是**机制诊断协议**，不是替代正式 MMLU-Pro accuracy 的评测超参数。

正式 causal replay 固定：

- 从 E0 free-run 选 16 条轨迹；
- 8 条最长轨迹 + 8 条覆盖其余长度分布；
- 每个 decode bin 4 个均匀 probe；
- bins：`[0,128)`、`[128,512)`、`[512,2048)`、`[2048,8192)`、`[8192,+∞)`；
- 额外加入各 variant 首次 free-run divergence 的 `[-2,-1,0,+1]` probe；
- 最大 probe decode index=12287；
- 48 层全部记录，不抽层；
- semantic replay 一次只 lazy materialize 一层。

---

# 5. 代码隔离与禁止修改项

本轮新增代码已经全部放在：

```text
Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/
```

测试放在：

```text
Native_NVFP4_HiF4_Linear_Puncture/tests/long_trajectory_stability/
```

Cursor **禁止**为了本实验改动：

- `main.py`；
- 现有 Phase-A E0–E7 experiment definition；
- vLLM HiF4 kernel/runtime；
- `e2e_diag_reconstruction` 的正式训练 loss、rollback、materialization 语义；
- 已有 Phase-A artifact；
- MMLU-Pro 正式评测代码。

如果 smoke 因 API/path 细节失败，Cursor 只允许修复：

```text
experiments/long_trajectory_stability/**
tests/long_trajectory_stability/**
```

而且不得改变：variant 定义、样本数、probe bins、parity threshold、metric 定义或 runtime ABI 门禁。若错误需要修改已有生产路径，立即停止并报告，不自行修改。

---

# 6. 已实现脚手架职责

## `capture_main.py`

复用根目录 `main.py` 的 lighteval/vLLM 流程，但仅在当前进程替换 result tracker，额外保存：

- raw generated text；
- exact input token IDs；
- exact output token IDs。

目的：禁止 `decode -> re-tokenize` 造成 teacher-forcing trajectory 偏差。

## `launch_capture.py`

负责一个 variant 的 free-run：

- 复用当前 Phase-A shared materialization；
- E1–E4 强制检查 `runtime_abi_version==3`；
- 写 `capture_manifest.json`；
- E0 使用 native NVFP4 emulation；
- E1–E4 使用当前 optimized HiF4 sidecar。

## `trajectory_io.py` / `normalize_capture.py`

把 lighteval details 统一为 JSONL，保留 exact prompt/output token IDs。

## `compare_free_runs.py`

对 E1–E4 与 E0：

- exact prompt token IDs 必须完全一致；
- 计算 first divergence；
- 计算 survival@128/512/2048/8192；
- 输出 `divergence_events.jsonl` 和 `free_run_summary.json`。

## `build_probe_plan.py`

自动从 free-run 结果生成 causal replay probe：

- 固定长度覆盖策略；
- uniform position probes；
- variant-specific divergence neighborhood probes。

## `causal_replay.py`

核心 mechanism diagnostic：

- 复用 Qwen3-MoE lazy native/student runtime；
- 只在本进程把 SDPA 改为 causal；
- 每次只 materialize 一个 layer；
- 逐层记录 hidden/router；
- 最终 probe 位置计算 full-vocab logits。

## `metrics.py`

固定指标：

### hidden

- rel-L2；
- cosine；
- max-abs。

### router

- KL(E0||variant)；
- top-k overlap；
- exact top-k set；
- top-k / top-(k+1) boundary margin。

### logits

- full-vocab KL(E0||variant)；
- JS divergence；
- centered-logit cosine；
- top1 agreement；
- E0/variant top1-top2 margin；
- E0 trajectory target token NLL；
- target rank。

## `run_free_run_matrix.py`

自动检测空闲 GPU，按每个 variant TP2 调度 E0–E4，并支持已有结果跳过。

## `prepare_analysis.py`

free-run 后生成：

- divergence summary；
- causal probe plan。

## `run_semantic_matrix.py`

严格顺序：

1. E0 semantic reference；
2. E0 parity gate；
3. gate 通过后才并行 E1–E4。

## `summarize.py`

自动生成：

- `long_trajectory_summary.json`；
- `LONG_TRAJECTORY_STABILITY_REPORT.md`。

---

# 7. 最重要的正确性门禁：E0 semantic parity

lazy semantic runtime 与 vLLM 正式 runtime 是两个实现路径，必须先证明 causal replay 至少在 probe 位置和真实 E0 greedy trajectory 高度一致。

E0 semantic replay 在每个 probe position 检查：

```text
argmax(semantic_E0_logits) == actual_vLLM_E0_generated_token
```

正式门禁：

```text
overall top1 parity >= 0.99
```

结果文件：

```text
semantic/E0/e0_semantic_parity.json
```

### 若 parity >=0.99

继续 E1–E4 semantic replay。

### 若 parity <0.99

Cursor 必须立即停止，不能解释任何 hidden/router/logit 机制结果。

只做以下排查并汇报：

1. mismatch 是否集中在固定 layer/position range；
2. RoPE position IDs 是否和 prefix 长度一致；
3. causal SDPA 与 vLLM causal attention 是否存在语义差异；
4. native NVFP4 semantic runtime 与 ABI runtime 是否存在 activation/QDQ 差异；
5. final RMSNorm / tied lm_head 是否一致。

不允许通过降低 0.99 threshold 来让实验通过。

---

# 8. Task 0：静态 helper tests

执行：

```bash
pytest -q Native_NVFP4_HiF4_Linear_Puncture/tests/long_trajectory_stability
```

预期：基础环境下 helper tests PASS；若当前环境没有 `transformers`，heavy module import test 可以 SKIP。正式 smoke 必须在 `hif4` 环境覆盖真实依赖。

验收：

- first divergence 定义正确；
- prefix 长度不同能识别 divergence；
- probe position 在 half-open bin 内；
- identity hidden/router/logit metrics 为零误差；
- exact input/output IDs 能从 capture detail 恢复；
- 所有新增 `.py` 能 AST parse。

---

# 9. Task 1：先跑完整 smoke 闭环

先设置：

```bash
export RUN_ROOT=Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/trajectory_stability_smoke
```

## 1.1 free-run smoke

```bash
python Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/run_free_run_matrix.py \
  --run_root "$RUN_ROOT" \
  --max_samples 4 \
  --max_new_tokens 1024 \
  --max_parallel 1
```

验收：

- E0–E4 都有 `capture_manifest.json`；
- E1–E4 manifest 的 `runtime_abi_version=3`；
- E0–E4 都生成 normalized JSONL；
- 每个 JSONL 4 条；
- prompt exact token IDs 后续可完全对齐。

## 1.2 smoke divergence + probe plan

```bash
python Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/prepare_analysis.py \
  --run_root "$RUN_ROOT" \
  --num_samples 2 \
  --probes_per_bin 2 \
  --max_decode_index 511
```

验收：

- `analysis/free_run_summary.json` 存在；
- `analysis/divergence_events.jsonl` 存在；
- `analysis/probe_plan.json` 存在；
- probe plan 中 exact input/output IDs 非空。

## 1.3 smoke semantic replay

```bash
python Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/run_semantic_matrix.py \
  --run_root "$RUN_ROOT" \
  --max_parallel 1
```

首先只关心 E0 parity。

若 E0 parity 不通过，停止 smoke；禁止继续 formal。

若通过：

- E1–E4 均生成 `semantic_metrics.jsonl`；
- 每个 variant 包含 `scope=layer` 和 `scope=logit`；
- layer id 覆盖 0–47。

## 1.4 smoke report

```bash
python Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/summarize.py \
  --run_root "$RUN_ROOT"
```

Smoke 只用于验证闭环，不记录机制结论。

---

# 10. Task 2：正式 greedy free-run

Smoke 全通过后，新建正式目录：

```bash
export RUN_ROOT=Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/trajectory_stability_formal
```

执行：

```bash
python Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/run_free_run_matrix.py \
  --run_root "$RUN_ROOT"
```

默认自动采用：

- 64 samples；
- 16k new tokens；
- E0–E4；
- 每 variant TP2；
- 空闲 GPU pair 并行；
- 已成功 variant 自动跳过。

## 正式 free-run 数据门禁

Cursor 检查：

1. 五个 normalized JSONL 行数都相同且为64；
2. 每个 variant 的 prompt set 和 exact input token IDs 与 E0 完全一致；
3. E1–E4 ABI=3；
4. `max_output_len` 至少应覆盖 2k+；
5. 理想情况下存在 8k+ trajectory。

如果 64 条中没有任何 trajectory 达到 8k：

- **不要修改 bin 定义**；
- 保留现有结果；
- 报告最长输出长度；
- 可以把 `--max_samples` 增到 128 重新补跑一个新的 formal run，不覆盖旧 run。

---

# 11. Task 3：正式 first-divergence 分析与 probe 选择

执行：

```bash
python Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/prepare_analysis.py \
  --run_root "$RUN_ROOT"
```

Cursor 从 `free_run_summary.json` 先整理一张表：

| Variant | exact trajectory fraction | median first divergence | survival@128 | @512 | @2048 | @8192 |
|---|---:|---:|---:|---:|---:|---:|

此时**只描述现象，不解释原因**。

重点看：

- E2 是否显著早于 E1/E3 分叉；
- E4 更接近 E2、E3 还是介于两者；
- 分叉是否主要发生在很早的 0–128，还是随长序列才明显增加。

---

# 12. Task 4：正式 E0 causal parity reference

`run_semantic_matrix.py` 会自动先执行 E0，因此 Cursor 不需要自行改命令。

执行：

```bash
python Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/run_semantic_matrix.py \
  --run_root "$RUN_ROOT"
```

E0 结束后立即查看：

```text
$RUN_ROOT/semantic/E0/e0_semantic_parity.json
```

### PASS

`top1_parity >= 0.99`，脚本自动继续 E1–E4。

### FAIL

脚本会报错停止。Cursor 不得通过 `--min_e0_top1_parity` 降低阈值。

---

# 13. Task 5：E1–E4 fixed-path semantic replay

E0 gate 通过后脚本自动并行 E1–E4。

每个 variant 的 `semantic_metrics.jsonl` 必须包含两类记录：

## `scope=layer`

每个：

```text
sample × probe_position × layer(0..47)
```

包含：

- hidden_rel_l2；
- hidden_cosine；
- router_kl；
- router_topk_overlap；
- router_topk_exact；
- router boundary margins。

## `scope=logit`

每个：

```text
sample × probe_position
```

包含：

- logit KL；
- JS；
- centered cosine；
- top1 agreement；
- top1/top2 margin；
- E0 trajectory target NLL/rank。

验收：

- E1–E4 都完成；
- layer 0–47 均存在；
- 每个 planned probe 都有 logit row；
- `divergence:<variant>:+0` probe 能和该 variant free-run first divergence 对应。

---

# 14. Task 6：自动生成总报告

执行：

```bash
python Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/summarize.py \
  --run_root "$RUN_ROOT"
```

输出：

```text
analysis/long_trajectory_summary.json
analysis/LONG_TRAJECTORY_STABILITY_REPORT.md
```

Cursor 需要在自动报告基础上再补一个不改原始统计的人工结论段，明确回答下面四个问题。

---

# 15. 必须回答的问题 A：R64 的误差是否真的“越生成越差”

对 E2 比较：

```text
d0000_0127
d0128_0511
d0512_2047
d2048_8191
d8192_plus
```

主要看：

- full-vocab logit KL；
- Layer47 hidden rel-L2；
- top1 agreement。

### 支持“fixed-path drift”

如果 late-bin KL/hidden rel-L2 明显高于 early bin，并表现出稳定增长。

### 不支持“fixed-path drift”

如果 KL 大致平稳，甚至不随长度上升，而 free-run 仍明显早分叉。

后一种情况应转向 H2，而不是继续说“误差不断累积”。

---

# 16. 必须回答的问题 B：free-run 分叉是否主要由小 logit margin 触发

比较：

```text
variant first-divergence +0 probe
vs
uniform position probes
```

主要指标：

```text
E0 top1-top2 margin
```

如果 E2 divergence point 的 margin 显著更小，而 fixed-path KL 并没有明显随长度扩大，优先支持：

```text
quantization perturbation
      ↓
small decision margin
      ↓
rank reversal
      ↓
first token bifurcation
      ↓
different future states
```

这会是比“随机信息损失随 token 数累积”更严谨的解释。

---

# 17. 必须回答的问题 C：MoE router 是否在分叉附近放大误差

对每个 layer 计算：

```text
router_topk_exact(divergence points)
vs
router_topk_exact(uniform probes)
```

同时看：

- top-k overlap；
- router KL；
- E0 router boundary margin；
- hidden rel-L2。

重点列出 enrichment 最明显的 5–10 层。

如果 E2/E4 在首次 token 分叉前若干层先出现 expert set change，则这是很强的 H3 证据。

如果 router 基本完全一致，而 logits 仍分叉，则 router 不是主要机制。

---

# 18. 必须回答的问题 D：为什么 E4 没出现 8B Linear-MSE 的 1+1>1

用 E2/E3/E4 的三种模式区分：

## Case D1：E4 local hidden 也比 E2/E3 更差

说明在 30B MoE fusable 参数化下，`DIAG→R64` 本身就没有复现 8B independent-linear 的局部协同；优先检查共享 `D_QKV/D_VO/D_UD` 的自由度约束，而不是归因长轨迹。

## Case D2：E4 local hidden/NMSE 更好，但 final logit KL/free-run 更差

这是最重要的结果之一，说明：

\[
\text{local reconstruction quality}
\not\Rightarrow
\text{decision-sensitive E2E quality}
\]

即 MSE proxy 与下游 long-horizon sensitivity 脱钩。

## Case D3：E4 fixed-path 也很好，但 free-run 更容易在小 margin 位置分叉

说明组合变换没有明显破坏平均表示质量，但改变了 residual error direction，使 decision boundary robustness 变差。

## Case D4：E4 router mismatch enrichment 最明显

说明组合变换主要通过 MoE routing 边界损失端到端收益，应优先引入 router-aware objective/gate，而不是进一步降低普通 hidden NMSE。

---

# 19. 预定义判断门槛只用于“提示”，不作为统计显著性结论

自动 report 当前使用几个 heuristic flag：

- late/early logit KL ratio >= 2：标记明显 teacher-forcing drift；
- divergence median E0 margin < 0.75 × uniform median：标记 small-margin enrichment；
- 某层 divergence router exact 比 uniform 低 >=0.10：标记 router mismatch enrichment。

这些门槛只用于自动筛查，不等同于统计显著性检验。

Cursor 最终汇报必须同时给出原始数值，不得只报告 flag=True/False。

---

# 20. 结果完整性清单

正式 run 完成前逐项确认：

- [ ] helper tests 通过；
- [ ] smoke free-run E0–E4 完成；
- [ ] smoke E1–E4 runtime ABI=3；
- [ ] smoke exact prompt token alignment 通过；
- [ ] smoke E0 semantic top1 parity >=0.99；
- [ ] formal E0–E4 greedy free-run 完成；
- [ ] formal normalized 五个方案样本集合一致；
- [ ] formal free-run summary 完成；
- [ ] probe plan 包含 uniform + divergence neighborhood；
- [ ] formal E0 semantic parity >=0.99；
- [ ] E1–E4 48层 semantic metrics 完整；
- [ ] final logit metrics 完整；
- [ ] 自动 Markdown/JSON 报告生成；
- [ ] Cursor 人工补充 A/B/C/D 四个问题的结论；
- [ ] 没有修改现有 E0–E7 production/experiment runtime。

---

# 21. Cursor 最终必须整理的汇报格式

最终请生成：

```text
$RUN_ROOT/analysis/CURSOR_EXECUTION_SUMMARY.md
```

固定包含：

## 1. Execution status

- smoke status；
- formal status；
- GPU/命令；
- 是否有重试；
- 所有 artifact 路径。

## 2. Runtime validity

- E1–E4 ABI version；
- E0 semantic parity；
- prompt exact alignment。

## 3. Free-run divergence table

E1/E2/E3/E4：

- exact trajectory fraction；
- median first divergence；
- survival@128/512/2048/8192。

## 4. Fixed-path position table

每个 variant × decode bin：

- logit KL；
- final hidden rel-L2；
- router top-k exact；
- top1 agreement；
- target NLL delta。

## 5. Divergence margin analysis

- divergence E0 margin median；
- uniform E0 margin median；
- ratio。

## 6. Router localization

列出最明显的 5–10 个 layer：

- uniform router exact；
- divergence router exact；
- overlap；
- hidden rel-L2。

## 7. Hypothesis verdict

分别给：

```text
H1: SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE
H2: SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE
H3: SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE
```

每项必须引用具体数值。

## 8. E4 explanation

明确归类为 D1/D2/D3/D4，可多选，并给依据。

## 9. Next action

只能从数据推导下一步；不要自行启动新的算法训练。

若 H3 强：建议下一步做 router-aware objective。  
若 H2 强、H3 弱：建议研究 logit-margin / sensitivity-aware reconstruction。  
若 H1 强：建议对 late reasoning states 做 calibration/trajectory-aware objective。  
若 E4 属 D1：优先回查 fusable freedom/component ablation。

---

# 22. 给 Cursor 的执行纪律

1. 先 smoke，绝不直接 formal；
2. E0 parity 不通过就停止；
3. 不降低任何门禁；
4. 不自行修改生产代码；
5. 不改变 E0–E4 定义；
6. 不将 greedy diagnosis 分数与正式 MMLU-Pro accuracy 混为一谈；
7. 不用 raw text re-tokenize 代替 exact output token IDs；
8. 不删除/覆盖 Phase-A 结果；
9. formal 结果必须放新 `results/long_trajectory_stability` 目录；
10. 最终结论必须区分“固定轨迹误差增长”和“free-run token 分叉”，不能把二者混成一个概念。

---

# 23. 当前计划的最终研究闭环

```text
Phase-A anomaly
    ↓
E0-E4 deterministic free-run
    ↓
first-token divergence / survival curve
    ↓
exact E0 token trajectory
    ↓
causal semantic replay parity gate
    ↓
48-layer hidden + router diagnostics
    ↓
full-vocab logit diagnostics
    ↓
position trend + divergence-margin + router enrichment
    ↓
H1/H2/H3 verdict
    ↓
E4 D1/D2/D3/D4 classification
    ↓
决定下一轮算法优化目标
```

本轮到这里结束。Cursor 不在本轮自动启动新的 DIAG/R64 训练或 component ablation；先把机制证据闭环整理清楚。
