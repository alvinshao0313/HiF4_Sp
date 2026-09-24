# Legacy archive

这些目录已退出当前“格式转换损失补偿”主线，但没有直接丢弃，便于以后查阅实验定义和历史报告。

## 当前主线仍在仓库原路径

- 可学习对角变换：`Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/`、Native `src/` 对角搜索和共享格式代码。
- 不等价变换：`Native_NVFP4_HiF4_Linear_Puncture/experiments/non_equivalent_reconstruction/`。
- 优化目标：`internal_error_accumulation/`、`kl_direction_reuse/`、`progressive_error_cancellation/`，以及 `long_trajectory_stability/real_vllm_hooks/`。

## 本次归档

- `native_qwen3_8b/activation_3d_viz/`：旧 Qwen3-8B 激活可视化。
- `native_qwen3_8b/h4_block_rotation/`：旧 Qwen3-8B H4 block rotation 实验。
- `native_qwen3_8b/diag_gradient/`：旧 DIAG/H4/R64 实验脚本和报告；当前仍被 30B 主线引用的 `r64_transform.py` 保留在原路径。
- `hifloat4/rotation/`：独立 HiFloat4 rotation 实验；Native 当前使用的是 `Native.../src/rotation.py`，两者不是同一实现。
- `qad/QAD/`、`qad/ScaleTuning/`：QAD 量化实验及其辅助代码，当前三条主线没有 import 引用。

`HiFloat4/permutation_optimization/` 的源码此前已经处于用户现有删除状态，本次没有恢复；空目录已移除。

## 尚未归档的依赖

- `HiFloat4/hif4_scale_threshold_optimization/src/` 仍被 Native 的格式和权重变体代码直接导入。
- `NVFP4/` 仍被 Native 和 vLLM NVFP4 fake-quant 路径导入。
- `long_trajectory_stability/` 的源码和结果仍有优化目标、格式转换诊断和默认入口引用，暂不移动。

QAD 原始 `.result` 目录在本次之前已经缺失，工作区也已有对应删除记录；本次没有尝试恢复或继续删除该状态。
