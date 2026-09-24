# 实验与维护记录

记录描述特定时点和配置，不自动代表当前运行状态。原始指标、日志和生成报告留在实验目录，由这里提供入口；完整文件清单见 [自动索引](../../INDEX.md)。

## 已归类正文

| 主题 | 记录 | 相关经验 |
| --- | --- | --- |
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
| 残差 LoRA：训练结果与下游接续 | [训练结果与同输入对照](../../../Native_NVFP4_HiF4_Linear_Puncture/results/residual_lora_compensation/20260922T012547Z/TRAINING_REPORT.md) · [当前计划与运行记录](../../../Native_NVFP4_HiF4_Linear_Puncture/experiments/residual_lora_compensation/PLAN.md) |
| KL 方向复用 | [kl_direction_reuse 结果](../../../Native_NVFP4_HiF4_Linear_Puncture/results/kl_direction_reuse/) |
| 渐进式误差抵消 | [progressive_error_cancellation 结果](../../../Native_NVFP4_HiF4_Linear_Puncture/results/progressive_error_cancellation/) |
| 内部误差累计 | [internal_error_accumulation 结果](../../../Native_NVFP4_HiF4_Linear_Puncture/results/internal_error_accumulation/) |
| 长轨迹、真实 vLLM 与格式转换 | [long_trajectory_stability 结果](../../../Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/) |
| NVFP4 backport | [vLLM v0.27.0 回移报告](../../../NVFP4/reports/vllm_v027_nvfp4_backport/) |

格式转换的旧 LCB 分数受输入协议错位影响，semantic replay 也有已记录的适用限制；解释结果前读 [评测证据经验](../lessons/evaluation_evidence.md)。本次归整没有重新评估任何模型，也没有把 handoff 标题当作实验成功证明。

新增记录沿用已有研究主题；正文至少包含实际配置、运行标识、证据路径、结论范围与未完成事项，必要时链接对应经验。不要把同一份指标表复制到多个“最终报告”。
