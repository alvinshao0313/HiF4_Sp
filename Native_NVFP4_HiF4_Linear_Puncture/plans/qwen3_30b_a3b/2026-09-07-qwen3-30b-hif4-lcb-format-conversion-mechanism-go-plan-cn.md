# Qwen3-30B-A3B NVFP4→HiF4 格式转换精度损失机制：LiveCodeBench 真实 vLLM 端到端 Go 执行计划

> 日期：2026-09-07  
> 目标模型：`nvidia/Qwen3-30B-A3B-NVFP4`  
> 主目标：解释 **NVFP4→HiF4 格式转换为什么会真实破坏推理精度，尤其为什么 LiveCodeBench 从正常作答变成思考不收束/生成失控**。  
> 次目标：在主机制定位后，解释 E2 R64 与 E3 DIAG 分别改变了哪一个环节。  
> 正式数值真值：**真实 vLLM TP2 actual path**。  
> 执行方式：交给 Codex Go 模式，从仓库当前状态继续端到端执行；除真正的科学/实现硬阻塞外，不在 Task 间停下来询问用户。  
> 本计划明确排除 E4；不再恢复旧 semantic replay，也不重新搭第二套 decoder/runtime。

---

# 0. 本计划取代/继承关系

本计划是当前这项研究的**最新唯一执行计划**。

它继承并复用：

```text
plans/qwen3_30b_a3b/
  2026-09-01-qwen3-30b-hif4-long-trajectory-real-vllm-hook-replan-cn.md
```

已经落地的真实 vLLM hook 基础设施和历史结果，但重新定义实验重点。

它取代以下旧计划作为当前执行入口：

```text
2026-09-01-qwen3-30b-hif4-format-conversion-error-accumulation-livecodebench-causal-plan-cn.md
2026-09-01-qwen3-30b-hif4-long-trajectory-real-vllm-hook-replan-cn.md 中后续“继续扩展机制实验”的部分
```

旧文件保留作历史设计与排错记录，不删除。

本计划不再把 MMLU-Pro 当主数据集。MMLU-Pro 已完成的真实 vLLM 结果保留为对照事实，不重复跑主链。新的机制研究以 **LiveCodeBench** 为主，因为真实任务精度损失在那里最明确。

---

# 1. 重新审核仓库后的当前真实状态

Codex 开始执行前必须先读本节，并再次用仓库真实文件验证；不得根据旧计划猜进度。

## 1.1 已经完成、禁止重复搭建的基础设施

正式长轨迹诊断已经迁移到：

```text
Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/real_vllm_hooks/
```

当前 core path 已经支持：

```text
真实 vLLM TP2
+ max_num_seqs=1
+ single-request incremental KV-cache decode
+ E0 exact isolated trajectory 强制解码
+ LLM.apply_model() 在真实 TP worker 上安装 hook
+ 每 sample CPU flush
+ 每 variant 独立进程退出释放 GPU/runtime
```

当前 core hook 已捕获：

```text
layer_in: branch / residual
input_norm
attention_core
TP-reduced o_proj
post_attn_norm: normalized / updated_residual
router_logits
TP-reduced moe_out
layer_out: branch / residual
final_norm
raw full-vocab logits（probe-only）
```

旧的：

```text
causal_replay.py
run_semantic_matrix.py
fused semantic adapter
vLLM semantic alignment
TP2 semantic simulator
```

已经不再是正式路径，不能恢复。

## 1.2 当前 real-vLLM hook 已通过的基础门禁

当前仓库已有 smoke 证据：

- forced trajectory exact；
- hook neutrality PASS；
- 48 层 coverage；
- rank0/rank1 artifact；
- execution-shape 数值敏感已被明确区分；
- `n346/24` isolated single-request predictor 语义已按真实 vLLM 处理；
- 当前 helper test：

```text
conda run -n hif4 pytest -q \
  Native_NVFP4_HiF4_Linear_Puncture/tests/long_trajectory_stability/real_vllm_hooks
```

在 2026-09-07 审核时为：

```text
5 passed
```

Codex 修改后必须重新跑，不得把这次人工审核结果当成修改后的测试结果。

## 1.3 已有真实 vLLM MMLU-Pro 机制结果：只复用，不重跑

已有 E0/E1 `forced_core` 16/16 条。

现有结论：

- 固定 E0 history 下，没有清楚证据支持 error 随 decode 长度持续单调增长；
- E1 很早发生自由轨迹分叉；
- router top-k 改变并不在 first divergence 处明显富集；
- frontier 上最大的 actual-path 跳变多数落在 fused MoE 内部；
- 但这些结果**没有**严格拆出 same-input local format injection 与历史误差传播；
- 现有 prefix rescue/poison 只测“是否重新精确贴回 E0 token 尾迹”，不是任务正确性 frontier。

因此本计划不再重复 MMLU 的 H1/H2/H3 全链。

## 1.4 LiveCodeBench 已经出现非常强的真实精度损失

正式 sampled LCB 协议固定为：

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

现有正式结果：

| 方案 | LCB pass@1 |
|---|---:|
| E0 Native NVFP4 | 0.542857 = 95/175 |
| E1 Direct HiF4 | 0.194286 = 34/175 |
| E2 R64-only | 0.240000 = 42/175 |
| E3 DIAG | 0.222857 ≈ 39/175 |

E0↔E1 逐题：

```text
E0 Pass / E1 Fail = 61
E0 Pass / E1 Pass = 34
E0 Fail / E1 Fail = 80
E0 Fail / E1 Pass = 0
```

更重要的是，61 个真实 regression 中：

```text
60：思考/起草没有收束，基本顶满 38912 token
 1：正常写完，但 WA（doc 58）
 0：正常写完整程序后 TLE
 0：正常写完整程序后编译失败
```

因此新的主问题不能再泛化为“代码任务量化后更差”。

真正要解释的是：

> **为什么 E0 能正常结束 thinking 并写出程序，而 E1 在大量 E0 本来能做对的题上进入长时间反复思考、改口、代码 fence/草稿循环，最终无法正常结束。**

## 1.5 已有 LCB mechanism run 的局限

已有：

```text
results/long_trajectory_stability/trajectory_stability_real_vllm_lcb/
```

只包含 12 个手工写死 ID 的：

```text
formal E0 Pass / E1 Fail
```

它证明：

- E1/E2/E3 都很早与 E0 greedy 轨迹分叉；
- E1 median first divergence ≈14；
- LCB 上没有复现 MMLU 的“first divergence 处 E0 margin 更小”签名；
- token-level exact rejoin rescue/poison 仍表现出很强路径依赖。

但它仍有四个关键缺口：

1. 没有 `E0 Pass / E1 Pass` robust 控制组；
2. 12 个 regression ID 是手工选择，不是自动、可复现的 cohort；
3. formal LCB 是 sampling，而机制 free-run 是 greedy，尚未用官方 judge 建立 greedy E0/E1 的任务正确性桥梁；
4. 现有分析是 actual-path boundary comparison，没有 same-input local format puncture、state reset/injection 和真正的 task-correctness semantic frontier。

## 1.6 当前代码明确还没有的能力

当前 `real_vllm_hooks/` 中没有：

- hidden/state reset；
- state injection；
- frozen-router actual-production puncture；
- same-input production MoE puncture；
- same-input QKV/O projection puncture；
- local-format-injection vs historical-propagation 的严格分解；
- regression vs robust 自动 matched cohort；
- isolated greedy LCB official judge；
- 基于最终任务 Pass/Fail 的 semantic-critical frontier；
- E2/E3 在同一机制 frontier 上的系统复测。

这些才是本计划真正需要补的内容。

---

# 2. 研究问题重新定死

## RQ1：格式转换误差在哪里真正产生，并如何向后传播？

主对照：

```text
E0 Native NVFP4
vs
E1 Direct HiF4
```

需要区分：

1. actual path 上状态什么时候开始偏；
2. Attention 还是 MoE 是主要误差增长位置；
3. 对于可以做 identical-input 的 stateless / local 算子，格式转换本身新注入多少误差；
4. 后续大误差是本层新注入，还是已有 error 被网络动力学放大。

特别注意：

> full Attention 依赖真实 KV cache，不允许把它假装成无状态函数单独 forward 后定义 `q_A`。

因此：

- MoE：允许做真实 production same-input puncture；
- QKV/O projection：允许做真实 production same-input puncture；
- full Attention core：只看 actual-path boundary + full-prefix state intervention，不手写/重放 attention。

## RQ2：数值误差为什么会变成 LCB 的真实精度损失？

重点竞争假设：

### H-Q：量化难度/格式适配性主导

first divergence 前某些 token 的真实量化器输入更难用 HiF4 表示：

- HiF4 QDQ error 更大；
- group/subgroup 动态范围更差；
- actual-path branch output 随后明显偏移。

### H-N：activation norm / distribution 主导

危险状态主要由：

- RMS；
- amax；
- amax/RMS；
- group/subgroup imbalance；

驱动，进而放大 HiF4 误差。

这里必须看**真实 quantizer input**，不能只看 residual hidden norm。

### H-D：decision sensitivity / error direction 主导

量化误差本身未必突然变大，但某个 state 的 logit decision 对 perturbation 特别敏感。

LCB 现有结果已经说明：

> 不能预设“first divergence 一定发生在更小 E0 margin”。

因为已有 12 条 LCB 上该签名没有成立。

因此 qerror / norm / hidden direction / logit margin 必须并列比较。

### H-M：MoE 内部计算/路由放大

已有 frontier 提示 MoE 很重要，但 router top-k change 不是必要条件。

需要严格区分：

```text
expert/fused-MoE compute error
vs
router decision change
```

而不是把所有 `moe_out` 增长都归因于 router。

## RQ3：R64 与 DIAG 分别改变了哪一环？

只在 RQ1/RQ2 找出的关键样本、token、layer 上复测：

```text
E1 ↔ E2：R64
E1 ↔ E3：DIAG
```

E4 完全排除。

---

# 3. 研究证据分三层，禁止混写

最终报告必须把以下三类证据分开。

## 3.1 事实

例如：

- LCB pass/fail；
- 是否顶满 38912 token；
- first divergence；
- actual-path rel-L2；
- router top-k；
- `</think>` 是否出现。

## 3.2 统计关联

例如：

- first divergence 的 qerror 是否比 matched controls 大；
- regression 是否有更大的 amax/RMS；
- router margin 是否更小。

只能写“关联/富集/预测能力”。

## 3.3 因果干预

只有下面这些才能用于“导致/放大/救回”措辞：

- production same-input puncture；
- frozen-router puncture；
- full-prefix layer/boundary state reset；
- E0↔E1 state injection；
- token-prefix rescue/poison + 官方 LCB judge。

---

# 4. 正式与机制协议必须分开

## 4.1 正式 LCB sampled 结果

用于：

- 定义 benchmark regression；
- 定义 benchmark robust；
- 计算 E2/E3 对真实题目的 recovery；
- 最终 task accuracy 结论。

现有 E0/E1/E2/E3 结果协议一致，优先直接复用，**禁止无理由重跑 175 题正式 LCB**。

## 4.2 isolated greedy mechanism trajectory

用于：

- deterministic first divergence；
- fixed E0 history；
- real-vLLM hook；
- state intervention；
- 因果机制。

固定：

```text
temperature = 0
top_p = 1
top_k = 0
min_p = 0
TP = 2
KV = BF16
enforce_eager = true
max_num_seqs = 1
prefix cache = false
speculative decoding = false
```

## 4.3 sampled regression 不能直接等于 greedy regression

这是硬门禁。

对 selected formal cohort 的 isolated greedy E0/E1 输出，必须再次使用**官方 LiveCodeBench judge**。

定义：

### Benchmark-regression

```text
formal sampled E0 Pass / E1 Fail
```

### Mechanism-regression

```text
isolated greedy E0 Pass / E1 Fail
```

### Mechanism-robust

```text
isolated greedy E0 Pass / E1 Pass
```

只有 Mechanism-regression 能用于：

> 从 deterministic 数值轨迹 → deterministic 程序失败

的最强因果链。

如果某题 formal regression，但 greedy E0 也 Fail：

- 保留 benchmark relevance；
- 可以分析数值轨迹；
- 禁止把 greedy divergence 直接写成 formal sampled failure 原因。

---

# 5. 复用边界：禁止第二套脚手架

## 5.1 必须直接复用

```text
real_vllm_hooks/build_llm.py
real_vllm_hooks/forced_trajectory.py
real_vllm_hooks/hook_state.py
real_vllm_hooks/worker_hooks.py
real_vllm_hooks/run_variant_hook_capture.py
real_vllm_hooks/run_hook_matrix.py
real_vllm_hooks/run_isolated_free_run.py
real_vllm_hooks/compare_hook_states.py
real_vllm_hooks/validate_capture.py
real_vllm_hooks/gpu_pool integration
```

以及：

```text
src/formats.py::qdq_hif4_direct
src/native_nvfp4.py
现有 R64 / DIAG transform helper
nvfp4_operator_parity 中已验证过的局部 operator/checkpoint helper
正式 lighteval LCB extract_code / check_correctness
```

## 5.2 禁止新增

- 第二个 `build_llm`；
- 第二个 Qwen3 decoder；
- semantic replay；
- PyTorch SDPA 作为正式 attention；
- 手工 TP2 simulator；
- Python expert loop 作为正式 MoE；
- 自定义 LCB judge；
- 复制一份 HiF4/NVFP4 quantizer；
- 修改 `3rdparty/vllm/**` 来方便诊断；
- 修改 production HiF4 kernel。

## 5.3 新代码位置

只新增机制逻辑：

```text
Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/
  real_vllm_hooks/
    format_conversion/
      __init__.py
      build_cohort.py
      judge_isolated_lcb.py
      build_probe_plan.py
      quant_features.py
      production_puncture.py
      state_intervention.py
      semantic_frontier.py
      analyze_mechanism.py
      run_go.py
```

如果某两个脚本实际很短，Codex 可以合并，但不得增加更多碎片文件。

所有 runtime 调用必须通过已有 `build_real_vllm()` / worker hook 基础设施。

---

# 6. 新结果目录

固定：

```text
Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/
  format_conversion_lcb_mechanism_go_20260907/
```

结构建议：

```text
00_audit/
01_formal_matrix/
02_cohort/
03_isolated/
04_greedy_judge/
05_probe_plan/
06_feature_capture/
07_core_capture/
08_rq1_puncture/
09_rq2_statistics/
10_semantic_frontier/
11_state_intervention/
12_rq3_e2/
13_rq3_e3/
analysis/
```

不要把新结果混回旧：

```text
trajectory_stability_real_vllm_hook/
trajectory_stability_real_vllm_lcb/
```

旧结果只读复用。

---

# 7. Go 模式执行纪律

Codex 必须做到：

1. 从当前工作区继续，不 reset、不 checkout、不覆盖用户 staged changes；
2. 不创建 branch/worktree；
3. 所有 Python/test 命令用 `conda run -n hif4 ...`；
4. Task 完成后自动继续下一个 Task，不逐 Task 等用户确认；
5. 已存在且通过 manifest/协议审计的结果直接复用；
6. 新 runner 默认 resume，已有完整 artifact 跳过；
7. 只在真正阻塞正确性的情况下停止；
8. 不因某个可选因果子实验 blocked 而删除/伪造结论；
9. 不允许 fallback 到 semantic simulator；
10. 最终必须生成完整报告，即使结论是否定的。

## 7.1 硬阻塞

遇到以下情况，停止依赖该能力的 Task，并写：

```text
analysis/STOP_REASON.md
```

- real-vLLM hook neutrality 失败；
- forced trajectory 不 exact；
- E0/E1/E2/E3 runtime 协议错位；
- TP2 不可用；
- production puncture 的 no-op identity 不通过；
- state intervention no-op 不通过；
- 官方 LCB judge 不能复现已有 formal label；
- 需要修改 production vLLM/kernel 才能继续。

如果 blocker 只影响一个后续因果模块，例如 state intervention，而不影响已完成的 observational RQ1/RQ2：

- 保留已完成结果；
- final report 明确 `CAUSAL_STATE_INTERVENTION_BLOCKED`；
- 不用替代实现补洞。

---

# 8. Task 1：仓库与结果 readiness 审计

Codex 首先只读审计，不改 runtime。

必须核：

## 8.1 代码

- `causal_replay.py` / `run_semantic_matrix.py` 已从正式路径删除；
- `real_vllm_hooks` core 文件存在；
- `build_real_vllm()` 仍使用 TP2/BF16 KV/eager/max_num_seqs=1；
- E1/E2/E3 ABI=3；
- current hook boundaries 与 README 一致；
- 当前不存在已有 state intervention/frozen-router 功能，避免重复实现同名能力。

## 8.2 旧结果

验证以下 artifact 可读：

```text
phaseA_refactor_20260825T035730Z/E0..E3/eval/livecodebench
trajectory_stability_real_vllm_hook
trajectory_stability_real_vllm_hook_smoke
trajectory_stability_real_vllm_lcb
```

## 8.3 正式 LCB 协议一致性

E0/E1/E2/E3 必须全部一致：

- dataset；
- max_new_tokens；
- temperature；
- top_p；
- top_k；
- min_p；
- thinking；
- TP；
- KV；
- eager；
- max_model_length。

## 8.4 重新跑当前 helper tests

```bash
conda run -n hif4 pytest -q \
  Native_NVFP4_HiF4_Linear_Puncture/tests/long_trajectory_stability/real_vllm_hooks
```

如果失败，先修本轮相关问题；不能靠跳 test 继续。

输出：

```text
00_audit/CURRENT_REPO_AUDIT.md
00_audit/CURRENT_REPO_AUDIT.json
```

必须明确：

```text
REAL_VLLM_CORE_READY = true/false
FORMAL_LCB_E0_E3_REUSABLE = true/false
OLD_MMLU_RESULTS_REUSABLE = true/false
```

---

# 9. Task 2：构建完整 formal E0/E1/E2/E3 逐题矩阵

不要只保留旧 `E0-pass/E1-fail` 分类。

从四个正式 LCB details 读取 175 个相同 doc ID。

输出每题：

```text
doc_id
E0_pass
E1_pass
E2_pass
E3_pass
E0_generation_len
E1_generation_len
E2_generation_len
E3_generation_len
E0_finished_thinking
E1_finished_thinking
E2_finished_thinking
E3_finished_thinking
```

其中 `finished_thinking` 必须来自**未剥离 thinking 的原始生成**：

- 如果 formal details 保存 raw generation token IDs，优先直接查 tokenizer 的 `</think>` token/subsequence；
- 如果只保存 raw generation text，则在 raw text 上检查 `</think>`；
- **禁止**因为 `text_post_processed` 中没有 `</think>` 就判定未结束——lighteval 可能已经把成对 thinking 段剥离；
- 如果当前 formal details 无法无歧义恢复 raw thinking completion，就把 formal `finished_thinking` 记为 `unknown`，沿用已有 `LCB_ERROR_TAXONOMY_VS_MMLU_PRO.md` 的已核对 failure taxonomy；
- isolated greedy 一定使用 `run_isolated_free_run.py` 落盘的原始 `output_ids/raw_text` 重新精确判定；
- 不改变正式 judge。

E0/E1 主分类：

```text
formal_regression = E0 Pass / E1 Fail
formal_robust     = E0 Pass / E1 Pass
formal_both_fail  = E0 Fail / E1 Fail
```

再直接计算：

```text
E2_formal_observed_recovery = E0 Pass / E1 Fail / E2 Pass
E3_formal_observed_recovery = E0 Pass / E1 Fail / E3 Pass
```

这一步回答正式 sampled LCB 中 R64/DIAG 出现了哪些**逐题 observed recovery/翻牌**。由于 formal 协议 `temperature=0.6` 且每题只有一次 generation，禁止把单题 `E1 Fail → E2/E3 Pass` 直接写成“因果救回”；因果解释必须由后续 deterministic greedy bridge、production puncture 和 state intervention 支撑。

输出：

```text
01_formal_matrix/formal_task_matrix.jsonl
01_formal_matrix/formal_task_summary.json
```

不得重跑正式 175 题，只读已有结果。

---

# 10. Task 3：自动构建 regression + robust benchmark cohort

删除“靠 DEFAULT_DOC_IDS 手工挑 12 条”作为新实验选择方式。

旧 `build_lcb_cohort.py` 保留作历史 Task15 复现，不直接改坏旧结果。

新 `format_conversion/build_cohort.py` 自动选择。

## 10.1 regression 候选

全部：

```text
formal E0 Pass / E1 Fail
```

当前预计 61。

第一轮选最多 16：

1. doc 58（唯一当前已知“写完但 WA”）若仍满足 regression，强制包含；
2. 剩余 regression 按 E0 正式生成长度排序；
3. 在排序序列上均匀取覆盖分位点，直到总数 16；
4. tie 用 doc_id 数值顺序，保证可复现。

目的：覆盖 E0 正确轨迹长度，不追求虚假的错误类型均衡，因为 60/61 本来就是 thinking 不收束。

## 10.2 robust 控制

从：

```text
formal E0 Pass / E1 Pass
```

为每个 regression 一对一匹配一个未使用 robust：

优先级：

1. E0 正式 generation length 最接近；
2. 若并列，prompt token length 最接近；
3. 再并列，doc_id 最小。

最多 16。

## 10.3 prompt 构造

可以继续从正式 E0 details 的 `model_response.input` 生成 mechanism prompt IDs，但必须：

- 只 tokenize 一次；
- 保存 prompt text hash；
- E0/E1/E2/E3 全部复用同一 `input_ids`；
- 明确标记 `prompt_source=phaseA_formal_e0_text_retokens_once`；
- 不声称这些是 formal run 当时内部 exact prompt IDs。

因为 mechanism 真值由后续 same-shape isolated E0 重新建立。

## 10.4 保存官方 judge 所需 doc 信息

cohort row 必须包含/可索引：

- `inputs`；
- `outputs`；
- `fn_name`；
- gold / doc_id；

使 isolated greedy completion 可以离线调用同一个 LCB `extract_code + check_correctness`。

输出：

```text
02_cohort/benchmark_cohort.jsonl
02_cohort/benchmark_cohort_manifest.json
```

---

# 11. Task 4：isolated greedy E0/E1/E2/E3，只补缺失结果

对 benchmark cohort 跑：

```text
E0 → E1 → E2 → E3
```

全部使用已有：

```text
run_isolated_free_run.py
build_real_vllm()
```

不要创建第二个 free-run 实现。

固定：

```text
max_num_seqs=1
greedy
TP2
KV BF16
eager
max_new_tokens=38912
```

结果：

```text
03_isolated/E0.jsonl
03_isolated/E1.jsonl
03_isolated/E2.jsonl
03_isolated/E3.jsonl
```

## 11.1 复用旧 12 条

如果旧：

```text
trajectory_stability_real_vllm_lcb/isolated/E*.jsonl
```

里存在完全相同：

- prompt_key；
- input_ids；
- runtime config；
- max_new_tokens；

则可以按行复用。

缺的 sample 才真正生成。

禁止为了方便重跑全部 32×4。

---

# 12. Task 5：用官方 LCB judge 重判 isolated greedy 输出

新增：

```text
format_conversion/judge_isolated_lcb.py
```

必须直接 import 当前 lighteval：

```text
extract_code
check_correctness
```

不写自己的 judge。

对 E0/E1/E2/E3 isolated greedy completion 全部判：

```text
pass/fail
finished_thinking
extracted_empty
syntax_ok
judge result
failure class
generation length
```

输出：

```text
04_greedy_judge/greedy_task_matrix.jsonl
04_greedy_judge/greedy_task_summary.json
```

## 12.1 构建 mechanism cohort

### Mechanism-regression

```text
greedy E0 Pass / greedy E1 Fail
```

### Mechanism-robust

```text
greedy E0 Pass / greedy E1 Pass
```

优先目标：

```text
8 mechanism-regression
8 matched mechanism-robust
```

如果第一轮 benchmark cohort 得不到 8 个 mechanism-regression：

- 从剩余 formal regression 按同样 E0-length coverage 规则每次增加 8 个；
- 只生成新增 sample；
- 直到 mechanism-regression ≥8 或 61 个 formal regression 全覆盖。

robust 不足时同理从 34 个 formal robust 扩展。

这是 Go mode 自适应步骤，不询问用户。

如果最终 mechanism-regression <4：

```text
MECHANISM_LABEL_BRIDGE_INSUFFICIENT = true
```

此时：

- 可以继续做 observational format analysis；
- 禁止把后续 deterministic causal intervention 写成 LCB accuracy regression 的一般机制；
- final report 必须说明。

---

# 13. Task 6：重新构建机制 probe，重点放在 first divergence 之前

对 mechanism cohort 单独生成 probe plan。

## 13.1 first divergence

基于 isolated greedy：

```text
E0 vs E1
```

定义：

```text
t_lex = first output token difference
```

真正做数值归因的位置是预测 `t_lex` 的 predictor state。

## 13.2 regression heavy probes

每个 mechanism-regression 固定加入：

```text
t_lex - 8
t_lex - 4
t_lex - 2
t_lex - 1
t_lex
```

去除负数和重复。

再加入稳定控制：

```text
8
16
32
64
128
```

只保留轨迹内位置，并且距离 `t_lex` <8 的控制跳过。

再加入：

- E0 `</think>` predictor 前 1 个位置；
- E0 code answer/final answer transition 前 1 个位置；

如果存在。

## 13.3 robust probes

对 matched mechanism-robust：

- 使用对应 regression 的 normalized decode positions 映射；
- 加 `[8,16,32,64,128]`；
- 加自身 E0 `</think>` transition。

这样 robust 与 regression 的位置分布尽量可比。

输出：

```text
05_probe_plan/probe_plan.json
05_probe_plan/divergence_events.jsonl
05_probe_plan/think_transition_events.jsonl
```

---

# 14. Task 7：增加一个“早期真实 quantizer-input feature capture”，但不增加新 runtime

用户最关心之一是：

> first divergence 是否和 token 的量化误差/norm 有关？

现有 core hook 只在 sparse probe 保存 full tensor，不能系统扫描 first-divergence 前的真实 quantizer input。

新增一个**轻量 capture level**，仍集成在现有 `worker_hooks.py/hook_state.py`，不是第二套框架。

建议名称：

```text
feature_scan
```

只捕获：

```text
input_norm       # QKV 实际输入
post_attn_norm   # MoE gate/up 实际输入
```

不要捕获：

- attention core；
- o_proj；
- moe_out；
- full logits；
- layer in/out。

## 14.1 正式 feature-scan 的 trajectory 语义

正式 feature 数据必须对 **E0 与 E1 都使用同一条 isolated E0 canonical output IDs 做 forced decode**，与现有 `forced_core` 的 fixed-history 语义一致：

```text
E0 runtime + forced E0 tokens
E1 runtime + forced E0 tokens
```

这样每个 `decode_index` 的离散 token history 完全相同，E0/E1 activation distribution 差异才可以解释为 runtime/格式转换引起的 state drift。

禁止把 E0/E1 各自 natural greedy 在 first divergence 之后的 feature 直接相减后称为“格式误差”；那时 token history 已经不同。

`greedy no-hook vs greedy feature-scan hook` 只用于下一节的 hook neutrality，不作为正式 feature comparison 数据。

## 14.2 capture position

regression：

```text
0 .. min(max(63, t_lex+8), 255)
```

robust：

```text
0 .. 63
```

所有 48 层。

只将真实 row tensor detach→CPU；不在 hook 中额外运行 QDQ kernel。

## 14.3 neutrality smoke

因为 dense capture 增加 CPU copy，必须先对 2 条 sample 比：

```text
greedy no-hook
greedy feature-scan hook
```

至少前 128 token exact。

失败则停止 feature_scan，不能继续解释其结果。

## 14.4 文件量控制

feature scan 每 sample 结束立即 flush。

禁止把 16/32 条样本同时放内存。

---

# 15. Task 8：离线计算量化适配性与 norm/distribution 特征

新增：

```text
format_conversion/quant_features.py
```

只读 feature capture 的真实 actual-path input。

## 15.1 主特征

对每个：

```text
sample × decode_index × layer × boundary(input_norm/post_attn_norm)
```

保存：

- RMS；
- amax；
- amax/RMS；
- L2 norm；
- max-element energy share；
- 64-element group 内 amax/RMS 分布；
- 64-group dynamic range；
- 16-element subgroup imbalance；
- kurtosis（数值稳定实现）。

## 15.2 E1 HiF4 susceptibility

对相同 E0 state `x`：

\[
E_H(x)=\frac{\|Q_{HiF4}(x)-x\|_2}{\|x\|_2+\epsilon}
\]

直接复用：

```text
src/formats.py::qdq_hif4_direct
```

这是**same-state format susceptibility**，不是 actual vLLM branch output error。

字段名必须明确：

```text
hif4_same_state_qdq_rel_l2
```

禁止简称成 `actual_quant_error`。

同时必须在分析中明确：`E_H(x)` 只度量**激活侧**对 HiF4 网格的敏感性；E0↔E1 production puncture 的 `q` 同时包含当前真实 runtime 中的**权重格式转换 + 激活量化/变换**影响。若某层出现 `E_H(x)` 很小但 production `||q||` 很大，优先标记为 `WEIGHT_OR_OPERATOR_DOMINATED_CANDIDATE`，再复用现有 frozen-input/operator-parity 与静态 weight-format helper 做少量 selected-layer 辅助拆解；禁止为了这一步另建第二套正式 runtime，也禁止把 activation QDQ susceptibility 当成完整格式转换误差。

## 15.3 NVFP4 对照

只有当该 boundary 能从已验证 checkpoint/runtime metadata 无歧义恢复：

- input global scale；
- activation grouping；
- 是否需要前置 transform；

时才计算：

\[
E_N(x)=\frac{\|Q_{NVFP4}(x)-x\|}{\|x\|}
\]

和：

\[
E_{conv}(x)=\frac{\|Q_{HiF4}(x)-Q_{NVFP4}(x)\|}{\|Q_{NVFP4}(x)\|}
\]

必须复用 `src/native_nvfp4.py` / 已验证 operator-parity metadata。

如果 MoE input 的 scale/expert 语义存在歧义：

- 不猜；
- 该 boundary 只报告 HiF4 susceptibility；
- 不新增另一套 NVFP4 模拟。

## 15.4 聚合到 token level

每个 token 保存：

```text
max_hif4_qdq_rel_l2 + argmax layer/boundary
median_hif4_qdq_rel_l2
max_amax_over_rms + argmax
median_amax_over_rms
```

用于 event-vs-control 分析。

---

# 16. Task 9：E0/E1 正式 heavy-probe `forced_core`，只补缺失 sample

使用现有：

```text
run_hook_matrix.py
run_variant_hook_capture.py
compare_hook_states.py
```

对最终 mechanism cohort：

```text
E0
E1
```

使用同一个 E0 isolated exact output trajectory 强制解码。

已有旧 12 条、相同 probe 若完全一致可以复用；probe 不一致的 sample 只补当前新 probe capture。

正式 actual-path 比较至少消费：

```text
input_norm
attention_core
o_proj
post_attn_norm
router_logits
moe_out
layer_out
final_norm
raw_logits
```

## 16.1 主要 RQ1 actual-path 图

对每个 heavy probe × 48 layer，画：

```text
input_norm relL2
attention_core relL2
o_proj relL2
post_attn_norm relL2
moe_out relL2
layer_out branch/residual relL2
```

注意：

> `moe_out_rel_l2 - post_attn_norm_rel_l2` 只能叫“边界误差增长”，不能直接叫 MoE local quantization error。

真正 local injection 由下一 Task production same-input puncture定义。

## 16.2 regression vs robust

必须把两组分开：

- first-divergence neighborhood；
- stable controls；
- thinking transition。

不能只分析失败样本。

---

# 17. Task 10：真实 production same-input puncture，补上 RQ1 最关键的局部因果量

新增：

```text
format_conversion/production_puncture.py
```

必须通过：

```text
LLM.apply_model()/worker callable
```

直接调用已经加载在真实 worker 上的 production module。

不加载第二个 semantic model。

## 17.1 MoE same-input puncture——最高优先级

MoE 是 token-wise/stateless 的，适合严格 same-input。

对选中的：

```text
sample × token × layer
```

从 E0 `post_attn_norm` capture 取相同 input `x0`。

在 E0 runtime：

\[
y_0=M_{E0}(x_0)
\]

在 E1 runtime：

\[
y_1=M_{E1}(x_0)
\]

必须直接调用真实：

```text
layer.mlp(x0)
```

包含 production：

- gate；
- fused experts；
- TP reduction。

定义：

\[
q_M^{E1}=y_1-y_0
\]

这是本实验第一个可以严格称为：

> **固定同一输入状态下，格式转换在 MoE 分支新注入的实际输出误差。**

## 17.2 MoE no-op identity gate

在跑 E1 前，E0 必须验证：

```text
captured E0 x0
→ 再调用 production E0 layer.mlp(x0)
```

与 actual-path captured E0 `moe_out` 在合理确定性条件下对齐。

优先要求 exact；若 kernel 自身重复执行存在 deterministic but non-exact noise，必须先测 E0 repeatability noise floor，不能自己放宽。

identity 不通过：

```text
MOE_PUNCTURE_BLOCKED
```

不得解释 q_M。

## 17.3 Frozen-router 分解

不要手写 expert loop。

做法固定为：

1. 对 E0 `x0` 先捕获 E0 production router logits `r0`；
2. 在 production `layer.mlp(x0)` 调用期间，临时给 `layer.mlp.gate` 装一个**局部临时 hook**，让 gate 返回 `r0`；
3. 其余 `Qwen3MoeSparseMoeBlock.forward` 完全走 production `self.experts + TP reduction`；
4. 调用结束立即 remove 临时 hook。

得到：

\[
y_{E1,route0}
\]

定义：

### Expert/fused compute error

\[
q_{expert}=y_{E1,route0}-y_0
\]

### Routing-induced component

\[
q_{router}=y_1-y_{E1,route0}
\]

验证：

\[
q_M=q_{expert}+q_{router}
\]

必须记录 closure。

### Frozen-router identity gate

E0 `x0` + E0 `r0` 注回 E0 gate：

必须复现 E0 `layer.mlp(x0)`。

失败则禁止 frozen-router 结论。

## 17.4 QKV projection same-input puncture

对 E0 `input_norm x0`：

直接 production 调：

```text
layer.self_attn.qkv_proj(x0)
```

比较 E0/E1 输出。

这是合法的 stateless projection local error。

## 17.5 O projection same-input puncture

对 E0 `attention_core` rank-local output：

直接 production 调：

```text
layer.self_attn.o_proj(x0_rank_local)
```

两个 TP rank 同时执行，让 production row-parallel reduction 正常工作。

比较 E0/E1 TP-reduced output。

## 17.6 用 actual-path + same-input puncture 做严格 `e = q + p` 分解

这是 RQ1 的核心定义，必须对 **QKV、O projection、MoE** 三类可做 same-input puncture 的算子统一计算，不能只报告 `q`。

对任一局部算子 `F`，记：

```text
x0 = E0 actual-path 输入
y0 = F_E0(x0) = E0 actual-path 输出
x1 = E1 actual-path 输入
y1 = F_E1(x1) = E1 actual-path 输出
y1_fix = F_E1(x0) = E1 production same-input puncture 输出
```

定义总 actual-path 输出误差：

\[
e = y_1-y_0
\]

定义本算子在固定 E0 输入上**新注入**的格式转换误差：

\[
q = y_{1,fix}-y_0
\]

定义由上游已经存在的输入状态误差，在 E1 算子内部继续传播/放大的部分：

\[
p = y_1-y_{1,fix}
\]

必须逐元素/逐向量验证：

\[
e=q+p
\]

并保存：

- `||e|| / ||y0||`；
- `||q|| / ||y0||`；
- `||p|| / ||y0||`；
- `cos(q,p)`；
- closure max-abs / rel-L2。

解释固定为：

- `||q||` 大：该算子本身是主要新误差源；
- `||p||` 大：主要是上游状态已经偏移，该算子继续传播/放大；
- `cos(q,p)>0`：local injection 与历史传播同向建设性叠加；
- `cos(q,p)<0`：两者存在抵消，不能只用 local NMSE 判断危险性。

对 MoE 还要继续把 `q` 拆成前一节的：

\[
q=q_{expert}+q_{router}
\]

因此 MoE 最终应同时报告：

```text
actual total e
local format injection q
upstream propagation p
expert/fused component q_expert
routing component q_router
```

E2/E3 后续机制复测必须复用同一套定义，才能比较 R64/DIAG 到底是在减少本地注入 `q`，还是改变上游传播 `p` / 两者夹角。

## 17.7 明确禁止 full Attention standalone puncture

不能把 captured hidden + 手工 KV 做：

```text
self_attn.forward(...)
```

然后称为真实 local Attention error。

完整 Attention 的因果作用交给后面的 full-prefix state intervention。

## 17.8 puncture 规模

只对：

- 8 mechanism-regression；
- 8 matched robust；
- 每 sample 最多 5 个 token；
- 每 token 最多 6 个候选 layer。

候选 layer 来自 actual-path error frontier：

- MoE 增长最大的层；
- Attention 增长最大的层；
- router first-change 层；

去重后最多 6。

禁止 48 层 × 所有 early token 全 puncture。

---

# 18. Task 11：RQ2 统计——量化误差、norm、hidden direction、margin 谁真正区分分叉？

新增/集中到：

```text
format_conversion/analyze_mechanism.py
```

## 18.1 event

每个 mechanism-regression：

```text
first-divergence predictor
```

## 18.2 matched controls

同一 sample 内选最多 4 个：

- 与 event 同属 thinking/code phase；
- 离 event 至少 8 token；
- decode position 尽量接近；
- 仍处于 E0/E1 natural greedy common-prefix 的位置优先。

如果 common-prefix 太短，使用 earlier stable positions。

## 18.3 特征

### 上游格式适配

- HiF4 same-state QDQ rel-L2；
- RMS；
- amax；
- amax/RMS；
- group/subgroup dispersion。

### actual-path 状态

- attention_core rel-L2；
- moe_out rel-L2；
- layer_out rel-L2；
- final_hidden rel-L2；
- final_hidden cosine；
- final_hidden norm ratio。

### decision

- E0 top1-top2 margin；
- E1 target rank；
- raw logit KL；
- E0 top1 与 E1 top1 是否翻转。

### router

- min top8/top9 margin across layers；
- argmin layer；
- first top-k change layer；
- top-k mismatch count。

### local puncture / propagation decomposition

- QKV `e/q/p` 及 `cos(q,p)`；
- O projection `e/q/p` 及 `cos(q,p)`；
- MoE `e/q/p` 及 `cos(q,p)`；
- MoE `q_expert`；
- MoE `q_router`。

## 18.4 统计输出

首先做 matched event-control：

- 每 feature 的 per-sample difference；
- median difference；
- sample-block bootstrap 95% CI。

如果有效 mechanism-regression ≥16，才额外做：

- 标准化 logistic；
- leave-one-sample-out；
- feature family 比较：

```text
qerror only
norm/distribution only
decision only
router only
qerror + decision
all
```

样本不足 16 时不硬做不稳定多变量模型。

## 18.5 final hidden 的 norm vs direction

用户关心“隐状态是否出现明显方向或分布改变”。

必须区分：

### 上游 norm

真实 quantizer input 的 RMS/amax 可能改变 QDQ error。

### final hidden norm

final linear head 本身是线性读出；单纯正比例缩放 final hidden 在无 bias 情况下不会改变 token 排名。

因此 final token flip 更直接依赖：

- hidden direction / relative coordinates；
- logit relative margin。

必须在报告中同时给：

```text
final_hidden norm_ratio
final_hidden cosine
raw logit margin delta
```

不得把“norm 大”直接等同于“token 会错”。

---

# 19. Task 12：定位 thinking 不收束发生在轨迹什么位置

这一步是 LCB 特有主线，不能只看 first lexical divergence。

对 E0/E1 isolated greedy 与正式 sampled output 都标：

```text
thinking_start
thinking_end (</think>)
code/final-answer start
output end
```

输出：

- first divergence 是否发生在 thinking 内；
- 距离 E0 `</think>` 还有多少 token；
- E1 是否最终发出 `</think>`；
- E1 thinking length / E0 thinking length；
- E1 是否出现 repeated reconsideration/fence pattern。

其中重复文本模式只作辅助描述，不作为因果机制。

需要重点比较：

```text
mechanism-regression
vs
mechanism-robust
```

回答：

> 格式转换是很早就改变了推理方向，还是主要在“应该结束 thinking 并提交答案”的阶段无法收束？

---

# 20. Task 13：真正的 task-correctness Semantic Frontier

现有 `prefix_rescue_poison.py` 只看是否 exact rejoin E0 256-token tail，不足以定义“什么时候开始真正破坏程序正确性”。

新增：

```text
format_conversion/semantic_frontier.py
```

仍调用真实 vLLM，不做 hidden simulator。

只对最多 8 mechanism-regression 做。

## 20.1 Rescue：正确 E0 prefix 能否救回 E1 的任务正确性

对不同 k：

```text
强制 E0 token prefix 到 k
→ 释放 E1
→ 允许 E1 正常生成
→ 用官方 LCB judge 判最终 completion
```

目标不是 exact rejoin E0，而是：

```text
Pass/Fail
是否结束 thinking
是否生成完整 code
failure class
```

## 20.2 Poison：E1 prefix 是否足以把 E0 推入失败

```text
强制 E1 trajectory prefix 到 k
→ 释放 E0
→ 官方 judge
```

注意：只能使用同一 mechanism prompt 的 deterministic isolated E1 token trajectory。

## 20.3 不假设 Pass/Fail 随 k 单调

先粗扫：

```text
k = t_lex
    t_lex+8
    +16
    +32
    +64
    +128
    +256
    +512
```

再额外加入：

- E0 thinking_end 附近；
- E0 code_start 附近。

每个粗点第一轮可以先用最大 4096 free-tail 检查：

- 是否结束 thinking；
- 是否进入 code/final answer；

这只是收敛状态筛选，不当作任务正确性。

只对出现明显状态变化的相邻 coarse points，使用完整剩余 LCB generation budget 生成 completion 并跑官方 judge。

这样避免每个 k 都跑 38912 token。

如果 Pass/Fail 非单调：

```text
SEMANTIC_FRONTIER = MULTI_FRONTIER
```

保存完整向量，不强行压单点。

## 20.4 最终定义

区分：

```text
t_lex              # first lexical divergence
thinking_frontier   # 开始影响是否结束 thinking
accuracy_frontier   # 开始改变 LCB Pass/Fail 的区域
```

这三者不能混写。

---

# 21. Task 14：真实 vLLM full-prefix layer/boundary state reset

这是本计划最强的 hidden-state 因果实验。

只在 semantic frontier 与 actual-path analysis 已经选出的最多 8 个 mechanism-regression 上做。

不重建 decoder。

## 21.1 为什么不能只 patch predictor token

Attention 后续依赖历史 KV cache。

如果只把 predictor token 的 hidden 换成 E0，而历史 token 在该 layer 仍是 E1：

- 后续层 KV cache 仍含 E1 history；
- 不能回答“此前误差是否在这层前已经形成”。

所以 state reset 必须对**从 prefill 到 predictor 的整个 prefix**在指定 boundary 使用 E0 reference state。

## 21.2 reference capture

扩展现有 hook infra，增加一个 intervention reference capture mode：

```text
selected layer
selected boundary
all positions from prompt prefill through target predictor
```

只捕获该一个 layer/boundary，不捕获全模型全 token。

保存 rank-local canonical state。

对于 replicated boundary 仍保存两个 rank，identity smoke 后可以按 rank 匹配。

## 21.3 canonical state pair

vLLM layer 使用：

```text
branch
residual
```

不能压成单 tensor。

block-level reset 必须同时替换：

```text
(branch, residual)
```

## 21.4 coarse layer reset

先测：

```text
0, 8, 16, 24, 32, 40, 47
```

流程：

```text
E1 + forced E0 token history
→ 在 layer l 指定 boundary 对 prefix 全位置替换成 E0 reference state
→ 继续真实 E1 tail
→ 在 t_lex predictor 读取 raw logits
```

主要 endpoint：

- E0 target rank；
- E0/E1 competing token margin；
- top1 是否恢复为 E0 token；
- KL to E0。

不要求自由生成整个题。

若某些 reset 能恢复 token，而晚层不能：

- 在 transition 区间逐层 refine；
- 不预设单调；
- 保存 full rescue map。

## 21.5 Attention vs MoE boundary split

关键层再做：

### post-attn reset

替换真实：

```text
post_attention_layernorm(normalized, updated_residual)
```

整个 prefix 的该 boundary state。

### layer-out reset

替换整个 block：

```text
(branch_out, residual_out)
```

比较两者对 final token decision 的恢复程度。

这给出：

> 误差在进入 MoE 前已经致命，还是经过 MoE 后才跨过决策边界。

## 21.6 no-op identity 硬门禁

任何真实 state reset 前必须先：

```text
E0 run
→ 用完全相同的 E0 reference state patch E0
```

要求：

- forced token exact；
- target predictor raw logits 与 baseline 在 repeatability noise floor 内；
- top1/target rank exact。

失败：

```text
STATE_RESET_BLOCKED
```

禁止正式 reset。

## 21.7 hook locality

patch 只能影响：

```text
指定 sample
指定 layer
指定 boundary
指定 positions
```

flush 后必须清空 intervention state。

第二个 sample 不得继承前一个 patch。

---

# 22. Task 15：Error Injection，反方向证明 E1 state error 是否足以毒化 E0

只在 Task 14 找出的 critical layer/boundary 上做。

使用 fixed E0 token history。

对同一 prefix 获取：

```text
S0 = E0 canonical state
S1 = E1 canonical state
```

若 state 是 branch/residual pair：

\[
branch_\alpha=branch_0+\alpha(branch_1-branch_0)
\]

\[
residual_\alpha=residual_0+\alpha(residual_1-residual_0)
\]

第一轮：

```text
alpha = 0, 0.5, 1.0
```

若发现 decision crossing，再补：

```text
0.25, 0.75
```

后续全部跑 E0 tail。

记录：

- target rank；
- margin；
- raw logit KL；
- top1 flip。

### alpha=0 identity

必须和 E0 baseline 一致。

### alpha=1

不是要求等于完整 E1，因为 tail 是 E0；它回答的是：

> **单独注入这一层/边界上真实 E1 state deviation，是否足以把 E0 后续决策推向错误。**

禁止把它解释成完整 E1 复现。

---

# 23. Task 16：RQ3 E2 R64，只沿主机制复测

E2 不重新做一套 discovery。

只使用：

- 同一个 benchmark/mechanism cohort；
- 同一个 probe plan；
- 同一个 critical layer/boundary；
- 同一个 semantic frontier case。

## 23.1 正式 sampled task observed recovery

从 Task 2 已有矩阵统计：

```text
E0 Pass / E1 Fail / E2 Pass
```

这里只定义为 formal sampled run 中的 **observed recovery/翻牌**，不是单题因果救回。

同时比较 thinking completion：

- E2 是否比 E1 更容易发出 `</think>`；
- E2 generation length 是否回落；
- E2 observed-recovery 题与 non-recovery 题的机制特征是否不同。

## 23.2 actual-path

只补 mechanism cohort 的 E2 `forced_core`。

比较：

```text
E0↔E1
E0↔E2
E1↔E2
```

重点只看主链已发现的 boundary/layer。

## 23.3 R64 分布机制

在 E0 same-state quantizer input 上：

- 复用现有 R64 transform helper；
- 只做 transform，不做第二套 runtime；
- 验证 L2 norm 保持：

\[
\|xR\|_2\approx\|x\|_2
\]

比较旋转前后：

- amax/RMS；
- group64 dispersion；
- subgroup16 imbalance；
- HiF4 QDQ susceptibility。

定义 activation-level Rotation Gain：

\[
G_R^{act}=E_H(x)-E_H(xR)
\]

正值表示 R64 让该 state 更适合 HiF4。

## 23.4 production puncture

在相同 E0 input 上比较：

```text
E1 vs E2
```

的：

- QKV same-input output；
- O same-input output；
- MoE same-input output。

特别寻找：

```text
HiF4 activation QDQ error ↓
但 production output / downstream decision 没改善
```

如果出现，再看：

- error direction；
- router component；
- critical reset layer 是否变化。

## 23.5 R64 no-quant 等价控制

只在 selected stateless puncture 上做辅助验证：

- 使用同一 master/unquantized weight；
- 复用当前 R64 transform/fold 语义；
- 禁用 HiF4 QDQ；
- 检查整体线性变换保持数学等价到 BF16/FP32 合理容差。

这是局部机制辅助，不修改正式 runtime。

其目的只是防止错误表述：

> R64 本身不是信息损失；变化来自 R64 改变了进入 HiF4 网格前的局部分布和随后误差方向。

---

# 24. Task 17：RQ3 E3 DIAG，只沿主机制复测

同样不重新 discovery。

## 24.1 正式 sampled task observed recovery

统计：

```text
E0 Pass / E1 Fail / E3 Pass
```

这里只定义为 formal sampled run 中的 **observed recovery/翻牌**，不是单题因果救回。

并比较：

- `</think>` completion；
- generation length；
- E3 observed-recovery vs non-recovery case。

## 24.2 actual-path / production puncture

在相同 critical states 比：

```text
E1 vs E3
```

- QKV local output；
- O local output；
- MoE q_M；
- q_expert/q_router；
- final hidden；
- raw logits。

## 24.3 DIAG 分布机制

从 adopted artifact 读取实际 DIAG，复用现有 transform helper。

在 E0 same-state input 上看：

- channel amplitude redistribution；
- RMS/amax/amax-RMS；
- group/subgroup dispersion；
- HiF4 QDQ susceptibility。

重点回答：

> DIAG 是降低了局部量化 injection，还是只是改了 error 方向/branch 分布，最终没有修复 thinking stability？

不能预设 DIAG 一定比 R64 更稳定。

---

# 25. Task 18：最终核心统计与图

最终至少输出以下图/表，不再堆无边界图表。

## Figure 1：正式 LCB 真实精度流向

```text
E0 Pass
 ├─ E1 Pass
 └─ E1 Fail
      ├─ E2 formal observed-recovery / non-recovery
      └─ E3 formal observed-recovery / non-recovery
```

同时列：

- thinking unfinished；
- WA；
- 其它。

## Figure 2：Regression vs Robust 的 early-token format-risk 曲线

以 first divergence 对齐：

```text
[-8,-4,-2,-1,0]
```

画：

- HiF4 same-state QDQ susceptibility；
- amax/RMS；
- final hidden cosine；
- E0 logit margin。

robust 使用 matched stable positions。

## Figure 3：48 层 actual-path error heatmap

边界：

```text
attention_core
o_proj
post_attn_norm
moe_out
layer_out
```

Regression vs Robust 分开。

## Figure 4：Same-input production puncture + `e=q+p` 分解

关键 layer：

```text
QKV: e / q / p / cos(q,p)
O:   e / q / p / cos(q,p)
MoE: e / q / p / cos(q,p)
       q
       ├─ q_expert
       └─ q_router
```

这是 RQ1 最重要的局部因果图：必须同时显示“本算子新注入多少”和“上游误差被继续传播多少”，不能只画 local q。

## Figure 5：State-reset rescue map

横轴 layer，纵轴 sample：

- block reset；
- post-attn reset；

颜色：

- E0 target rank recovery；
- top1 rescue。

## Figure 6：Thinking / accuracy frontier

每个 mechanism-regression：

```text
t_lex
thinking_frontier
accuracy_frontier
E0 </think>
```

显示 first lexical drift 到真正任务失败之间的距离。

## Figure 7：E2/E3 修了哪一环

只画相对 E1 的关键指标变化：

```text
formal sampled observed-recovery rate
thinking completion rate
HiF4 susceptibility
MoE q_M
q_expert/q_router
critical state reset behavior
```

E4 不出现。

---

# 26. Task 19：最终报告必须按固定问题回答

输出：

```text
analysis/FORMAT_CONVERSION_LCB_MECHANISM_REPORT.md
analysis/FORMAT_CONVERSION_LCB_MECHANISM_SUMMARY.json
```

必须逐项回答：

## 格式转换主机制

1. E1 为什么会让 61 个 E0-correct LCB case 变成错误？
2. thinking 不收束是不是绝对主 failure mode？
3. first lexical divergence 通常发生在 thinking 的什么阶段？
4. 真实 quantizer-input 的 HiF4 susceptibility 在 divergence 前是否异常？
5. RMS/amax/amax-RMS 是否比 qerror 更有解释力，还是反过来？
6. final hidden 主要表现为 norm drift 还是 direction drift？
7. E0 margin 在 LCB 是否真的是主触发因素？若不是，明确否定。
8. Attention 和 MoE 哪个更主要？
9. MoE 大误差主要是 expert/fused computation，还是 router decision？
10. same-input local puncture 与 actual-path growth 能否对上？
11. state reset 能否把 first wrong token 决策救回？关键深度在哪里？
12. E1 state injection 能否把 E0 决策推错？
13. first lexical divergence 与真正 accuracy-critical frontier 是否相距很远？
14. 什么机制使 E1 无法正常结束 thinking？

## R64

15. R64 在 formal sampled regression 中出现了哪些 observed recovery/翻牌？这些现象中哪些又被 deterministic mechanism 证据支持？
16. R64 是否降低了 HiF4 same-state QDQ error？
17. 如果局部 QDQ error 降了但 task 仍没恢复，问题出在哪一环？
18. R64 是否主要改变局部能量排布/error direction，而不是简单减小误差？

## DIAG

19. DIAG 在 formal sampled regression 中出现了哪些 observed recovery/翻牌？这些现象中哪些又被 deterministic mechanism 证据支持？
20. DIAG 主要改变 channel scale/distribution、local injection、MoE error 还是 downstream decision？
21. 为什么 DIAG 在 MMLU 能维持/恢复，但 LCB 仍只有约 0.223？

## 算法指导

最后只能根据结果写：

- 是否应该做 layer/branch-aware protection；
- 是否需要 router/expert-aware objective；
- 是否应该优化 format susceptibility 而不是 raw NMSE；
- 是否需要针对 thinking termination/trajectory stability 的校准目标；
- R64 是否值得 selective/adaptive 使用；
- DIAG 应该重点优化哪类 state。

不得在实验前把这些写成既定结论。

---

# 27. 新测试要求

新增：

```text
Native_NVFP4_HiF4_Linear_Puncture/tests/long_trajectory_stability/
  real_vllm_hooks/format_conversion/
```

只测试新增逻辑。

至少覆盖：

1. formal matrix 四 variant doc ID 对齐；
2. automatic cohort deterministic；
3. regression↔robust one-to-one matching；
4. greedy judge 复用 official checker；
5. first-divergence predictor 无 off-by-one；
6. `</think>` subsequence 检测；
7. feature scan 只记录指定 boundary；
8. quant feature group64/sub16 统计 shape 正确；
9. production puncture sample/layer/token key 映射；
10. frozen-router 临时 hook 必须 remove；
11. QKV/O/MoE 的 `e=q+p` closure helper；
12. MoE `q_M=q_expert+q_router` closure helper；
13. intervention state 只作用指定 layer/boundary/position；
14. alpha=0 state interpolation identity；
15. semantic frontier 不假设单调；
16. E4 不进入任何新 experiment variant list；
17. resume/skip 已完成 artifact。

完成代码后：

```bash
conda run -n hif4 pytest -q \
  Native_NVFP4_HiF4_Linear_Puncture/tests/long_trajectory_stability/real_vllm_hooks
```

必须全部通过。

真实 GPU smoke 只测最小 2 sample，不写 fake production fallback。

---

# 28. Task 20：实现唯一 Go 入口并真正端到端跑

新增：

```text
real_vllm_hooks/format_conversion/run_go.py
```

这是 Codex 最终执行的唯一主入口。

职责：

```text
审计已有结果
→ 构建 formal matrix
→ 构建 benchmark cohort
→ 复用/补 isolated greedy
→ official greedy judge
→ 自动扩 cohort 到 mechanism-valid 数量
→ build probes
→ feature capture
→ quant features
→ E0/E1 core capture
→ actual-path analysis
→ production puncture
→ RQ2 statistics
→ thinking/semantic frontier
→ state reset smoke
→ state reset
→ injection
→ E2 same-cohort controls
→ E3 same-cohort controls
→ final figures/report
```

## 28.1 运行命令

最终要求能从仓库根运行：

```bash
conda run -n hif4 python \
  Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/real_vllm_hooks/format_conversion/run_go.py \
  --run_root Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/format_conversion_lcb_mechanism_go_20260907
```

不要要求用户手工依次执行十几个脚本。

## 28.2 resume

默认：

- artifact 完整且 manifest 匹配 → skip；
- artifact 不完整 → 重跑该 Task；
- 不覆盖旧正式 Phase-A/real-hook 结果；
- 不靠 pid 文件判断结果是否有效，以 artifact/manifest 为准。

## 28.3 GPU 调度

继续使用现有 `gpu_pool.py`。

GPU-heavy Task 默认串行，优先保证语义一致；不要为了快把 single-request 重新改成多 request batching。

每个 variant/runtime 完成后进程退出释放显存。

如果没有 2 张满足现有 GPU 门禁的卡：

```text
HARDWARE_BLOCKED
```

直接报告，不降门槛、不切 TP1。

---

# 29. Codex Go 模式的具体执行顺序

Codex 接到本计划后直接：

1. 读 `AGENTS.md`；
2. 读本计划；
3. 读当前 `real_vllm_hooks/README/核心文件` 和 staged diff；
4. 做 Task 1 audit；
5. 实现 Task 2–20 所需的最小增量代码；
6. 跑新增/现有 CPU tests；
7. 跑最小 real-vLLM GPU smoke；
8. smoke 全过后直接执行 `run_go.py` 正式链；
9. 遇到可 resume 的现有数据自动复用；
10. 不在 E0/E1 主结果完成前启动 E2/E3 重型机制复测；
11. E2 完成后自动 E3；
12. E4 永不启动；
13. 完成最终报告、README 更新和结果索引；
14. 做一次最终代码/结果一致性审计；
15. 停止，不做额外算法优化。

除非出现：

- 必须修改 production math；
- 正式 judge 无法复现；
- intervention identity 失败且无法在诊断代码内修正；
- 硬件阻塞；

否则 Go 模式不要中途询问用户。

---

# 30. 最终验收清单

全部满足才算本计划完成：

- [ ] 没有恢复 semantic replay；
- [ ] 没有修改 `3rdparty/vllm/**`；
- [ ] 没有修改 production HiF4 kernel；
- [ ] 新机制实验正式 variant 只有 E0/E1/E2/E3；
- [ ] formal 175 题结果被复用而不是无理由重跑；
- [ ] formal E0/E1/E2/E3 per-task matrix 完成；
- [ ] benchmark regression + robust cohort 自动生成；
- [ ] isolated greedy 输出用官方 LCB judge 重判；
- [ ] mechanism-regression / mechanism-robust 定义清楚；
- [ ] feature scan neutrality 通过；
- [ ] 真实 quantizer input 的 qerror/norm 特征完成；
- [ ] E0/E1 forced-core regression vs robust 对比完成；
- [ ] production QKV/O/MoE same-input puncture完成或有明确 hard blocker；
- [ ] QKV/O/MoE 的 actual-path `e`、local injection `q`、upstream propagation `p` 完成 `e=q+p` closure；
- [ ] `cos(q,p)` 已用于区分建设性累计与误差抵消；
- [ ] frozen-router identity 通过后才解释 q_expert/q_router；
- [ ] thinking transition 分析完成；
- [ ] task-correctness semantic frontier 完成或明确 blocked；
- [ ] state reset no-op 通过后才做正式 reset；
- [ ] error injection alpha=0 通过；
- [ ] E2 只沿主机制复测；
- [ ] E3 只沿主机制复测；
- [ ] E4 未运行；
- [ ] 最终报告区分事实 / 统计关联 / 因果证据；
- [ ] 最终结论允许否定“长轨迹误差持续累计”“低 margin 主导”“router 主导”等预设假设；
- [ ] `run_go.py` 可以从当前结果断点续跑到结束。

完成这些后，才可以说我们真正回答了：

> **NVFP4→HiF4 格式转换是怎样从局部数值扰动，一步步演化为 Qwen3-30B-A3B 在 LiveCodeBench 上 thinking 不收束和真实精度崩塌的。**
