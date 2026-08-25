# HiF4 MLP 中间通道排序：修复后复验报告（Revalidation）

> 日期：2026-07-30
> 模型：Qwen/Qwen3.5-4B（32 层 SwiGLU MLP，d_model=2560，d_ff=9216）
> 量化：HiF4（hifx4）W4A4 RTN，仅 `lm_head` 不量化
> 实验目录：`experiments/qwen35_4b_perm_revalidation/`（不覆盖 V1/V2 任何产物）

本报告是对 `MLP_PERMUTATION_EXPERIMENT_REPORT.md` 的**修正与追加**，不替代旧报告；
旧报告中的实验数据与分析保持原样，本报告在其结论之后给出修复后的复验结论。

结论等级约定：**代码事实**（单元测试/parity 直接证明）、**层级实验观察**（独立 validation split 支持）、**端到端结论**（成对完整任务评测支持）。

---

## 1. 原实现问题

复验计划确认的 V2 实现缺陷（对应修复任务）：

| 问题 | 影响 |
|---|---|
| `val_x` 两次独立切分，实际取到 search 80% 数据 | 验证集泄漏，收益高估 |
| 激活误差与权重误差共用 `output_sensitivity` 权重 | 代理目标数学定义错误 |
| FP32 搜索目标与 BF16 RTN/QLinear2 部署路径不一致 | 搜索/部署口径错位 |
| 只选 hierarchical 候选，忽略 q99 等 | 候选选择缺陷 |
| accept 无最小改善阈值、单 split | 万分之一级噪声被当作收益 |
| G4 连续 oracle 与真实 G64 排名关系未知 | 代理有效性未验证 |
| 正式实验 `refine_passes=0` | 无法证明搜索接近局部最优 |
| BF16-only 重排漂移未隔离 | 端到端归因混杂 |
| MMLU-Pro 300 样本、temp 0.7 | 生成式评测不足以归因 |

## 2. 修复内容

- `split_utils.py`：search/validation 显式不重叠索引（seed 可复现），X 与激活共用同一 RowSplit。
- `objective.py`：恢复双方向交叉能量权重（激活误差×权重列能量，权重误差×激活能量）；新增 `DeploymentMLPContext`/`DeploymentDownContext`（BF16 部署路径 + 真实 HiF4 fake quant + RTN 回写 dtype）；新增 `batched_full_layout_hif4_loss`。
- `candidate_selection.py`：identity/q99_desc/q99_asc/hierarchical/hierarchical_refined 统一候选池；random 仅作负对照；三 split 稳健接受判据（wins≥2、相对改善≥0.1%、改善>2σ、BF16 漂移≤0.2%）。
- `hierarchical_greedy.py`：全路径统一 RowSplit；受限 seeded local refinement（预算封顶、只接受严格改善）；逐层 proxy 相关性审计。
- `pipeline.py`/`run_mlp_reorder.py`：完整 JSONL 字段（候选×split 指标、拒绝原因、split 审计哈希）、config 快照（版本/CUDA/校准索引）、输出目录防覆盖、16 条固定 s1k BF16 探针。

## 3. 单元测试与 parity 证据（代码事实）

- 完整单元测试：92 passed（`pytest HiFloat4/permutation_optimization/tests`），`compileall` exit 0。
- search/validation 索引：三个测试直接证明 disjoint、complete、X/A 行对齐。
- 交叉能量权重：c4/c64/full-layout/pair-cost 与手工公式逐项一致（且与错误的 output_sensitivity 加权显著不同）。
- `DeploymentMLPContext`：identity drift=0、total=residual；与「perm 吸收 + 真实 fake quant + F.linear」模块式前向 parity 通过（rtol/atol=1e-4）。
- refinement：单调不增、合法排列、评估预算受控。

## 4. Search/validation split 审计（代码事实 + 层级观察）

- Stage 1 层 [0, 15, 31]：`overlap_rows == 0` 全部满足：True。
- 每层 split 索引哈希记录在 `layer_metrics.jsonl` 的 `split_audit` 字段。

## 5. Proxy 与真实 G64 相关性（层级实验观察，Gate B）

- 三层 median Spearman = 0.263（阈值 ≥0.30），median top5_overlap = 0.000（阈值 ≥0.20）。
- random 负对照最优层数：0。
- **Gate B：未通过**。
- 各层 Spearman：['0.111', '0.438', '0.263']。

## 6. 候选选择和 refinement（层级实验观察，Gate C）

- Stage 1 接受层数：1/3；**Gate C：通过**。

| 层 | accepted | selected | rejection_reason | 最佳结构候选相对改善 |
|---|---|---|---|---|
| 0 | False | identity | no_structured_candidate_beats_identity | -0.270% |
| 15 | False | identity | improvement_not_above_split_variance | +0.343% |
| 31 | True | hierarchical | accepted | +2.841% |

## 7. BF16-only 重排漂移（Gate D）

- Stage 1 参考（仅 1 层被重排）：mean_logit_nrmse=0.00320，argmax_flip_rate=0.00244，max_abs_logit_delta=0.1250。
  注意：该值已超过 Gate D 的 0.002 阈值，说明 BF16 重排漂移在深层传播下不可忽略（见 STOP_REASON.md 附注）。
- Stage 3 未运行。

## 8. W4A4 层输出收益（层级实验观察）

- Stage 2 未运行。

## 9. 端到端成对任务结果（端到端结论）

- Stage 4 未运行。

## 10. 结论边界

- 层级指标（含 total_nrmse 分解）由 3 个独立 validation split 支持；端到端指标由同一脚本成对评测支持。
- 在当前候选空间、搜索预算和评测协议下观察到的现象，不能外推为所有排列方法的理论上界。
- MMLU-Pro 为单次确定性解码结果，未做多 seed 方差估计；其差异只作参考，不单独构成归因。
- 条件 5（第二个独立 search seed=52 的一致性）如未运行，则「有效」结论自动降级为「未观察到稳定收益」。

## 11. 是否继续排序方向的决策

**决策：B. 继续算法研究，但当前 proxy 失败：C4 代理与真实 G64 排名相关性不足（或 random 负对照胜出）。下一步应研究条件化 G8/G64 代理、直接输出误差局部搜索或更强匹配算法，而不是扩大 beam。**

## 12. 完整复现实验命令

```bash
bash run_stage1_layer_audit.sh   # Gate A/B/C
bash run_stage2_full_search.sh   # 完整 32 层搜索 + BF16 探针 + 保存 identity/permuted BF16
bash run_stage3_bf16_control.sh  # Gate D
bash run_stage4_w4a4_eval.sh     # W4A4 成对评测 + MMLU-Pro 确定性解码
python summarize_revalidation.py --test-count <N>
```

