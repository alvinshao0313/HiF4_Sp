# NVFP4 → HiF4 / MXFP4 实验导航

返回[仓库首页](../../README.md) · [目录与文件归属](../../docs/guides/repository_layout.md)。本页按用途分类现有实现；Python 包名已迁为 `fp4_convert`，实验参数和产物路径保持不变。目录存在不代表实验完成；结果以对应 manifest、日志及指标为准。

四个主模型共用的 [BRQ、GPTQ、AWQ 和 SmoothQuant 运行入口](paper_baselines/README.md) 同时覆盖 HiF4 和 MXFP4。

## HiF4 转换与重构

| 用途 | 实现入口 |
| --- | --- |
| 逐层重构、校准数据、模型导出和评测公共路径 | [e2e_diag_reconstruction](e2e_diag_reconstruction/README.md) |
| 权重侧矩阵与 norm 的非等价重构 | [non_equivalent_reconstruction](non_equivalent_reconstruction/README.md) |
| matrix_norm、旋转及目标函数消融 | [module_design_ablation](module_design_ablation/PLAN.md) |
| 短程与数据效率 | [matrix_norm_efficiency](matrix_norm_efficiency/PLAN.md) |
| R64＋scale、GPTQ、正交消融与论文补充实验 | [thesis_format_conversion_supplement](thesis_format_conversion_supplement/README.md) |
| 全专家校准覆盖、C4 校准、重复评测适配 | [thesis_format_conversion_all_experts](thesis_format_conversion_all_experts) · [thesis_format_conversion_c4](thesis_format_conversion_c4) · [thesis_matrix_norm_repeats](thesis_matrix_norm_repeats) |

## MXFP4 转换与组件对照

| 用途 | 实现入口 |
| --- | --- |
| 直接转换、E2M1/E8M0 格式、GPU runtime | [qwen3_mxfp4](qwen3_mxfp4/README.md) |
| UOS、二级尺度、旋转 | [qwen3_mxfp4_uos](qwen3_mxfp4_uos/README.md) · [qwen3_mxfp4_dualscale](qwen3_mxfp4_dualscale/PLAN.md) · [qwen3_mxfp4_r16x2](qwen3_mxfp4_r16x2) |
| UOS／旋转／二级尺度组合 | [qwen3_mxfp4_combinations](qwen3_mxfp4_combinations/README.md) |
| 旋转、组合与原生基线下游对照 | [rotation_eval](qwen3_mxfp4_rotation_eval/README.md) · [combination_eval](qwen3_mxfp4_combination_eval/README.md) · [baseline_eval](qwen3_mxfp4_baseline_eval/README.md) |
| AISBench GPQA 与 TP2 KV 容量对照 | [qwen3_aisbench_gpqa](qwen3_aisbench_gpqa/README.md) · [qwen3_gpqa_tp2_budget](qwen3_gpqa_tp2_budget/README.md) · [qwen3_mxfp4_dualscale_tp2](qwen3_mxfp4_dualscale_tp2/README.md) |

各路线的实际转换投影、权重/激活范围以具体实验为准，不能把已有 routed-expert 实验自动称为全部投影转换。

## 机制、扩展与共享依赖

| 用途 | 目录 |
| --- | --- |
| 误差累计、真实 vLLM 轨迹与目标函数 | `internal_error_accumulation/`、`long_trajectory_stability/`、`kl_direction_reuse/`、`kl_direction_reuse_downstream/`、`progressive_error_cancellation/` |
| 校准轨迹、router、末层目标 | `reasoning_trace_calibration/`、`reasoning_trace_router_bf16/`、`reasoning_trace_last_head/`、`reasoning_trace_full_kl_ce/`、`reasoning_trace_full_kl_ce_parallel/` |
| 专家剪枝及轨迹校准 | `ban/`、`ban_trace_calibration/` |
| 残差补偿、恢复目标和解析诊断 | `residual_lora_compensation/`、`downstream_accuracy_recovery/`、`matrix_only_analytic/` |
| 大模型迁移、原生 GPQA | `qwen3_235b_transfer/`、`qwen3_235b_matrix_norm/`、`nvfp4_gpqa/` |
| 被当前代码导入的早期实现 | `diag_gradient/r64_transform.py`、`long_trajectory_stability/real_vllm_hooks/` |

上述共享路径不能按实验年代归档。`e2e_diag_reconstruction/data/` 存放 Python 数据处理源码，真实数据和缓存另存结果目录；不能把该代码目录当作本地数据忽略。

旧 Qwen3-8B 与 QAD 入口见[历史归档](../../archive/legacy/README.md)。具体协议、指标和证据继续由[统一文档入口](../../docs/README.md)导航，不在本页复制结果表。
