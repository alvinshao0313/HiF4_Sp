# STOP_REASON：Gate B（代理可用性）未通过，按计划在 Stage 1 后停止

## failed gate
Gate B：C4 proxy 与真实 G64 排名相关性（Execution Order 中的 `Stage 1 Proxy Gate B`）。

## command
```bash
bash HiFloat4/permutation_optimization/experiments/qwen35_4b_perm_revalidation/run_stage1_layer_audit.sh
```

## exit code
0（实验本身成功完成；停止是计划内 gate 判定，不是运行错误）

## 关键指标（layers 0 / 15 / 31，128 个确定性候选/块）
| 层 | Spearman | Pearson | top1_match | top5_overlap |
|---|---:|---:|---|---:|
| 0 | 0.111 | 0.092 | False | 0.0 |
| 15 | 0.438 | 0.340 | False | 0.0 |
| 31 | 0.263 | 0.278 | False | 0.0 |
| **median** | **0.263** | — | — | **0.0** |

Gate 阈值：median Spearman ≥ 0.30 且 median top5_overlap ≥ 0.20。
实测 median Spearman = 0.263 < 0.30，median top5_overlap = 0.0 < 0.20 → **不通过**。
random 负对照未在任何一层系统性胜出（该子条件通过）。
search/validation `overlap_rows == 0` 三层全部满足（Gate A 条件通过）。

附注（不计入 gate 判定，仅供后续参考）：
- 尽管 proxy 相关性弱，L31 的 hierarchical 候选在严格多 split 判据下仍被接受（3/3 split 改善，mean +2.84%，漂移 ~1e-6）；L15 hierarchical +0.343% 但因「改善未显著大于 split 方差」被拒绝。
- Stage 1 全模型 BF16 探针（仅 L31 一层被重排）：mean_logit_nrmse = 0.0032，已超过 Gate D 的 0.002 阈值；argmax_flip_rate = 0.0024（低于 0.005 阈值）。这提示 BF16 重排漂移在 32 层传播下不可忽略，后续若重做该方向需同时处理累加顺序稳定性。

## 已完成任务
- Task 0–12 全部代码修复与单元测试（92 passed，compileall exit 0，Gate A 通过）
- Stage 1：layers 0/15/31 完整搜索 + proxy 审计 + BF16 探针

## 未执行任务
- Stage 2 完整 32 层搜索（Gate B 不通过，禁止消耗算力）
- Stage 3 BF16-only control
- Stage 4 W4A4 成对端到端评测

## 建议下一步
当前 G4 proxy（S1P2 oracle + 交叉能量权重）无法可靠指导真实 G64 选择（决策 B）：
1. 研究条件化 G8/G64 proxy：proxy 在评分 G4 时显式条件化当前已固定的块内容，而不是独立评分；
2. 或放弃 G4 代理，改用直接输出误差的局部搜索（在小预算下用 batched full-layout loss 直接优化）；
3. 或换更强匹配算法（如 G64 粒度的谱/匹配方法），而非继续扩大 beam；
4. 若重做端到端，需先解决 BF16 重排漂移（稳定累加顺序或只接受低漂移层）。
