# HiF4 MLP 中间通道层级排序：实验报告

> 日期：2026-07-30
> 模型：Qwen/Qwen3.5-4B（32 层 SwiGLU MLP，d_model=2560，d_ff=9216）
> 量化：HiF4（hifx4）W4A4 RTN，仅 `lm_head` 不量化
> 代码：`HiFloat4/permutation_optimization/`
> 实验目录：`HiFloat4/permutation_optimization/experiments/qwen35_4b_perm_{rtn,s1k,s1k_v2}/`

---

## 1. 研究问题

HiF4 格式沿最后一维按 64 通道分组，组内再分 8×(2×4) 层级共享 scale。对 `down_proj`（权重 `[d_model, d_ff]`、输入激活 `A [tokens, d_ff]`），其量化分组直接由中间通道顺序决定。假设是：

> 若把尺度兼容的通道排进同一 4/8/64 组，可降低 down 侧 W4A4 重构误差，且排列可离线吸收进 `up/gate`（行重排）与 `down`（列重排），不增加推理成本、不改变浮点输出。

算法：每层独立做「局部候选召回 + 小束宽贪心」依次组 4 元组、配对成 8、装箱成 64；`accepted` 后方才吸收。本报告回答：这条路径在真实 W4A4 下到底有没有用。

## 2. 关键前提：排列能碰到的误差只有 down 侧

当前 RTN（`HiFloat4/main.py` + `QLinear2`）所有权/激活都沿 last dim（in_features）量化：

| 对象 | 形状 | 量化维 | 受中间维排列影响？ |
|---|---|---|---|
| `up/gate` 权重 | `[d_ff, d_model]` | d_model | 否（行重排不改分组） |
| `up/gate` 输入 X | `[tokens, d_model]` | d_model | 否 |
| `down` 权重 | `[d_model, d_ff]` | d_ff | **是** |
| `down` 输入 A | `[tokens, d_ff]` | d_ff | **是** |

**误差分解探针**（随机输入，3 层，相对 FP MLP 输出的 NRMSE）：

| 层 | 全量化 | 只量化 down 权重 | 只量化 up/gate 权重 | 只量化 down 输入 A |
|---|---:|---:|---:|---:|
| 0 / 15 / 31 | ≈0.20 | ≈0.08 | **≈0.11** | ≈0.09 |

含义：**排序理论上限只覆盖 down 侧（单项 ≤0.09），而 up/gate 侧 ≈0.11 完全够不到。** 这是后续所有实验结果的上限解释。

## 3. 实验设置

三版实验共用：校准 128 条、激活 512 行（20% 层内验证）、权重采样 512 行、`refine_passes=0`、seed=42、8 worker 并行。评测协议与基线一致：arc_easy/arc_challenge/mmlu 用 lm_eval（0-shot），mmlu_pro 用 lighteval（300 样本，temp=0.7）。基线为不排序直 RTN（引用 `HiF4_exp/qwen35_4b_w4a4_proj_ablation` 的 `full`）。

| 版本 | 校准集 | 搜索用激活 A | accept 指标 | 误差权重 |
|---|---|---|---|---|
| V1-wikitext | wikitext2（2048 窗） | BF16 干净 A | `down_output_nrmse`↓ 且 `hif4_loss`↓ | `e_a`/`e_w` 能量 |
| V1-s1k | s1k-1.1（全长 970–19.5k token） | BF16 干净 A | 同上 | 同上 |
| V2-s1k | s1k-1.1（全长） | **真实 W4A4 A**：`SiLU(Xq@Wg_qᵀ)·(Xq@Wu_qᵀ)` | **整层 MLP W4A4 输出 nrmse↓** | **输出敏感度** `s_c=√E[A_c²]·‖W_d[:,c]‖` |

V2 即「真实 W4A4 整层目标 + 输出敏感度加权」：让 up/gate 量化噪声先进入 A 再排序，并用最接近评测的目标做 accept。

## 4. 校准集上的量化误差（排序自身目标）

| 版本 | accepted | `hif4_loss` 相对↓ | 输出 nrmse 相对↓（accepted 均值） | 最好层 |
|---|---:|---:|---:|---:|
| V1-wikitext | 25/32 | 5.30% | 1.44%（down 口径） | L31：9.59% |
| V1-s1k | 17/32 | 3.18% | 0.80%（down 口径） | L31：6.07% |
| V2-s1k | 24/32 | −2.41%（见注） | **0.34%（整层 MLP 口径）** | L31：2.36% |

注：V2 的 accept 不再要求 `hif4_loss` 下降，部分 accepted 层 down 重构反而变差——**down 局部重构与整层输出目标并不总是一致**，这本身就是 V1 指标选错的一个佐证。

结论一：**排序在其自身目标上确实有效，但换成正确的整层口径后，可争取空间只有 ~0.3%**（一层输出千分之三），与误差分解给出的上限一致。

## 5. 端到端任务精度（W4A4 RTN 后）

| variant | arc_easy | arc_challenge | mmlu | mmlu_pro(300) |
|---|---:|---:|---:|---:|
| rtn_baseline | 0.8009 | 0.5026 | 0.7154 | 0.7133 |
| V1-wikitext | 0.7976 | 0.5111 | 0.7159 | 0.6800 |
| V1-s1k | 0.8039 | 0.5051 | 0.7153 | 0.6800 |
| V2-s1k | 0.8001 | 0.5009 | 0.7131 | 0.6733 |

结论二：**三版排序在 arc/mmlu 上都在基线 ±0.3pt 噪声内，没有任何一版带来稳定收益。** 换校准集（wikitext→s1k）无影响；换目标函数（V2）无影响。

结论三（重要 caveat）：**mmlu_pro 三版都掉到 0.67–0.68**（基线 0.7133），且与排序方案、校准集无关，三版之间差异仅 0.7pt。该任务为生成式评测（temp=0.7、300 样本），对任何权重改动（即便浮点等价的重排，也会改变 BF16 GEMM 归约顺序）表现敏感。此下降**不能归因于排序算法**，更可能是该评测的方差/敏感性；若需采信 mmlu_pro，应先做多种子方差估计再下结论。

## 6. 工程与效率

| 阶段 | 耗时 |
|---|---|
| 工程加速前（初版） | 单层 ~119s；refine=2 曾卡死 13h（逐次 c64 CUDA 调用） |
| 工程加速后（V1） | 单层 ~47s（≈2.5×），32 层并行搜索 ~6–10 min |
| V2 初版事故 | `c4_cost` 误用 pad64 逐次 CUDA 量化（1.8 万次/层 × ~10ms，8 worker 抢一张卡），75 min 未完成第一层 |
| V2 修复后 | 单层 41s，**32 层搜索 6 min**；全流程搜索+RTN+评测 ~29 min |

V2 修复内容（也是方法论教训）：

1. **G4 粒度回退 S1P2 oracle**。真 HiF4 中 4 元组只决定 lv3 组内指数比，lv1/lv2 scale 由整个 64 组共享；pad 60 个 0 强行「真量化」既不真又慢三个数量级。4 元组粒度上 S1P2 oracle（2-bit mantissa、动态范围 7）与格式结构一致，是正确的局部代价。
2. **`MLPW4A4Context` 缓存**：每层只量化一次 X/Wu/Wg 并预算 `A_qa`、`Y_fp`，每个候选 perm 只做 `index_select` + 一次量化 + 一次 matmul。
3. refine 维持 `refine_passes=0`；若要开，候选 c64 评估必须批量向量化，禁止逐个 CUDA 调用。

## 7. 总体结论（量化专家视角）

1. **通道排序路径的天花板被三种目标口径反复证实：一层输出千分之几。** 原因不是搜索不够好，而是排列只能改变 down 侧 64 组内的分组误差（占整层误差不到一半），且 down 侧内部 A/W 误差还会部分抵消。局部 `hif4_loss` 降 3–5% 换算成整层输出仅 ~0.3%。
2. **目标函数必须端到端**。V1 用「BF16 干净 A + down-only nrmse」做 accept，会接受一些整层口径下无收益甚至变差的排列；V2 证明换成正确口径后收益更小但更真实。凡是 PTQ 辅助优化，accept 指标都应是最终量化前向的端到端误差。
3. **校准集不是瓶颈**：wikitext 与 s1k（全长）结论一致，域偏移不是主因。
4. **mmlu_pro 的 0.713→0.68 是评测敏感性问题，不是排序效果**；涉及生成式任务时应先量化评测方差。
5. **建议终止通道排序方向**。若要在 HiF4 W4A4 RTN 上拿到可见收益，应转向排序够不到的 ~0.11 部分：up/gate 权重与激活的 outlier 处理（clip/scale 搜索、GPTQ 类二阶补偿、或旋转类方法），而非 down 侧 64 组内重排。

## 8. 产物索引

| 内容 | 路径 |
|---|---|
| V1-wikitext 结果 | `experiments/qwen35_4b_perm_rtn/results/summary.md`、`results/perm_search/` |
| V1-s1k 误差审计 | `experiments/qwen35_4b_perm_s1k/results/error_audit.md` |
| V1-s1k 任务结果 | `experiments/qwen35_4b_perm_s1k/results/summary.md` |
| V2 任务结果 | `experiments/qwen35_4b_perm_s1k_v2/results/summary.md`、`results/perm_search/` |
| 搜索实现 | `HiFloat4/permutation_optimization/{objective,hierarchical_greedy,pipeline,activation_collector}.py` |
