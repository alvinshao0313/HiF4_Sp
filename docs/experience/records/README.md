# 实验与维护记录

记录描述特定时点和配置，不自动代表当前运行状态。原始指标、日志和生成报告留在实验目录，由这里提供入口；完整文件清单见 [自动索引](../../INDEX.md)。

## 已归类正文

| 主题 | 记录 | 相关经验 |
| --- | --- | --- |
| 2026-11-11 Qwen3 官方 BF16 GPQA | [TP8、官方权重与顺序三轮均值](qwen3_bf16_gpqa_20261111.md) | [评测协议与缓存隔离](../lessons/evaluation_evidence.md) |
| 2026-11-09 原生 NVFP4 GPQA Diamond | [Qwen 推荐采样、ModelScope 数据与 TP4 基线](nvfp4_gpqa_20261109.md) | [评测协议与缓存隔离](../lessons/evaluation_evidence.md) |
| 2026-11-09 起 Qwen3 路由专家 MXFP4 | [直接转换／旋转／UOS、CPU／GPU 证据与 GPQA×10](qwen3_mxfp4_20261109.md) | [格式与计算边界](../lessons/fake_quant_boundaries.md) |
| 2026-11-05 H-only 闭式代理 CPU诊断 | [真实q_proj、Adam对照与验证退化](matrix_only_analytic_20261105.md) | [量化代理与留出误差](../lessons/fake_quant_boundaries.md) |
| 2026-11-04 matrix_norm短程转换五任务 | [random50_e5完整结果、AIME复核与成本](matrix_norm_efficiency_20261104.md) | [评测与结论边界](../lessons/evaluation_evidence.md) |
| 2026-10-28 自生成轨迹重新校准 Ban | [三组对照、LCB 与全量 MMLU-Pro](ban_trace_calibration_20261028.md) | [评测与数据隔离](../lessons/evaluation_evidence.md) |
| 2026-10-26 235B matrix_norm 自生成轨迹转换 | [实现、数据准备与 TP8 资源阻塞](qwen3_235b_matrix_norm_20261026.md) | [训练执行验证](../lessons/model_transfer_execution.md) · [评测协议](../lessons/evaluation_evidence.md) |
| 2026-10-23 matrix_norm HiF4 的 Ban 接入 | [算法、开关与真实 TP2 短测](ban_matrix_norm_20261023.md) | [量化执行边界](../lessons/fake_quant_boundaries.md) |
| 2026-09-24 已结束训练的冗余恢复状态与缓存 | [本次清理记录](maintenance/2026-09-24-history-cleanup.md) | [目录保护与恢复状态](../lessons/workspace_protection.md) |
| 2026-09-20 源码归档、磁盘清理 | [清理原文](maintenance/2026-09-20-experiment-cleanup.md) | [活动实验与目录保护](../lessons/workspace_protection.md) |
| HiF4 dense fake activation quant | [实现记录](implementation/hif4_vllm_fake_act_quant.md) | [fake quant 边界](../lessons/fake_quant_boundaries.md) |
| NVFP4 dense fake activation quant | [实现记录](implementation/nvfp4_vllm_fake_act_quant.md) | [fake quant 边界](../lessons/fake_quant_boundaries.md) |
| Qwen3.5-4B 量化阈值 | [2026-07-30 实验报告](../../../HiFloat4/hif4_scale_threshold_optimization/experiments/EXPERIMENT_REPORT.md) | 具体结论保持在原文 |

## 原位运行证据

| 主题 | 原始目录 / 报告 |
| --- | --- |
| Native 逐层重构 | [e2e_diag_reconstruction 结果](../../../Native_NVFP4_HiF4_Linear_Puncture/results/e2e_diag_reconstruction/) |
| 不等价变换 | [non_equivalent_reconstruction 结果](../../../Native_NVFP4_HiF4_Linear_Puncture/results/non_equivalent_reconstruction/) |
| norm / DIAG / 正交旋转 / 自由矩阵 / router KL 消融 | [15 组完整结果与历史分数代入分析](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/REPORT.md) · [定义与运行记录](../../../Native_NVFP4_HiF4_Linear_Puncture/experiments/module_design_ablation/PLAN.md) · [历史组 AIME25 三轮最终核验](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/aime_repeat_final_analysis.json) · [matrix_norm 三轮追加状态](../../../Native_NVFP4_HiF4_Linear_Puncture/results/aime25_repeats/matrix_norm_20261015T081958Z/status.json) · [原生 NVFP4 三轮追加状态](../../../Native_NVFP4_HiF4_Linear_Puncture/results/aime25_repeats/native_nvfp4_20261015T103435Z/status.json) |
| AIME25 最终答案误判与四组准确率重算 | [复核报告与逐回答改判证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/aime_scoring_recomputed_20261015.md) |
| S1K 与正式测试、来源官方测试的重合审计 | [2026-10-16 原划分只读审计](../../../Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_calibration/run_20261016T132000Z/REPORT.md) · [授权清洗补足与首次生成](../../../Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_calibration/clean_s1k_20261016T030238Z/REPORT.md) · [两类校准数据与追加 Router 完整对比](../../../Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_calibration/continue_20261016T135141Z/REPORT.md) |
| B 的 BF16 Router、独立原生教师前缀 | [2026-10-20 完整结果与验证](../../../Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_router_bf16/run_20261019T031744Z/REPORT.md) |
| B 末层与输出端词表 Top-mass KL | [2026-10-21 完整结果与配对区间](../../../Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_last_head/run_20261020T031200Z/comparison.md) · [AIME 复核](../../../Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_last_head/run_20261020T031200Z/evaluation/B_last_head_vocab64/aime25/review_audit.json) · [经验边界](../lessons/evaluation_evidence.md) |
| Qwen3-235B 迁移：训练执行验证与正式接续 | [计划与运行记录](../../../Native_NVFP4_HiF4_Linear_Puncture/experiments/qwen3_235b_transfer/PLAN.md) · [最长批量验证](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_transfer/transfer_20261001T124000Z/smoke_summary.json) · [执行经验](../lessons/model_transfer_execution.md) |
| 残差 LoRA：训练结果与下游接续 | [训练结果与同输入对照](../../../Native_NVFP4_HiF4_Linear_Puncture/results/residual_lora_compensation/20260922T012547Z/TRAINING_REPORT.md) · [当前计划与运行记录](../../../Native_NVFP4_HiF4_Linear_Puncture/experiments/residual_lora_compensation/PLAN.md) |
| 五项全量精度恢复：MMLU-Pro 全量、LCB、GSM8K、ARC-E/C | [2026-10-01 验收报告](../../../Native_NVFP4_HiF4_Linear_Puncture/results/downstream_accuracy_recovery/paired_full_20260930T131000Z/REPORT.md) · [实验定义与结论](../../../Native_NVFP4_HiF4_Linear_Puncture/experiments/downstream_accuracy_recovery/PLAN.md) · [运行配置与模型来源](../../../Native_NVFP4_HiF4_Linear_Puncture/results/downstream_accuracy_recovery/paired_full_20260930T131000Z/manifest.json) |
| KL 方向复用 | [kl_direction_reuse 结果](../../../Native_NVFP4_HiF4_Linear_Puncture/results/kl_direction_reuse/) |
| 渐进式误差抵消 | [progressive_error_cancellation 结果](../../../Native_NVFP4_HiF4_Linear_Puncture/results/progressive_error_cancellation/) |
| 内部误差累计 | [internal_error_accumulation 结果](../../../Native_NVFP4_HiF4_Linear_Puncture/results/internal_error_accumulation/) |
| 长轨迹、真实 vLLM 与格式转换 | [long_trajectory_stability 结果](../../../Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/) |
| NVFP4 backport | [vLLM v0.27.0 回移报告](../../../NVFP4/reports/vllm_v027_nvfp4_backport/) |

格式转换的旧 LCB 分数受输入协议错位影响，semantic replay 也有已记录的适用限制；解释结果前读 [评测证据经验](../lessons/evaluation_evidence.md)。2026-09-24 的文档归整没有重新评估模型；后续新增实测以各报告日期和实际证据为准，handoff 标题不作为实验成功证明。

新增记录沿用已有研究主题；正文至少包含实际配置、运行标识、证据路径、结论范围与未完成事项，必要时链接对应经验。不要把同一份指标表复制到多个“最终报告”。
