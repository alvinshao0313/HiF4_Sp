# NVFP4 → HiF4 / MXFP4 转换工程

文档总入口见 [docs/README.md](../docs/README.md)，研究经验见 [经验库](../docs/experience/README.md)，原始运行证据见 [记录导航](../docs/experience/records/README.md)。本页按模型和研究主题组织；旧计划、交接标题中的状态只描述记录时点。

本目录实现 **Native NVFP4 → HiF4 / MXFP4 格式转换**、GPU runtime 与下游验证。HiF4 路线包括直接转换、旋转/尺度、GPTQ 和局部重构；MXFP4 路线包括直接转换、旋转、UOS 和尺度组合。按用途查找当前入口请先读[实验导航](experiments/README.md)。

本目录是后续开发使用的维护主包，Python 导入和 `python -m` 使用 `fp4_convert`。迁移只调整包名、源码位置和引用；算法、CLI 参数、数据、checkpoint 及默认结果位置保持原有语义。旧实验的冻结源码保留在本地原路径，不能用新包覆盖旧 run 的源码哈希，见[目录与文件归属](../docs/guides/repository_layout.md)。

下文中的 `plans/`、`results/` 均指仓库根目录 `Native_NVFP4_HiF4_Linear_Puncture/` 下的原位置；它们没有随源码搬迁。计划继续随 Git 分发，结果和旧源码仅保留本地。

项目最初建立在 `ISTA-DASLab/Qwen3-8B-FPQuant-QAT-NVFP4` 上，随后主线切换到 `nvidia/Qwen3-30B-A3B-NVFP4`。因此当前目录同时保留两代实验，但二者的模型结构、checkpoint 语义和实验用途不同，不能混用结果。

## 1. 两条模型线

### Qwen3-8B QAT：历史机制实验

模型：`ISTA-DASLab/Qwen3-8B-FPQuant-QAT-NVFP4`

特点：dense Qwen3-8B，checkpoint 内含在线 block rotation。主要用于早期 Linear 穿刺、激活/权重格式转换误差分析，以及 DIAG、H4、R64 等变换机制验证。

历史入口已经集中归档到 `archive/legacy/native_qwen3_8b/`。仍保留的共享入口只有：

- 原始 Linear 穿刺实现：`src/`、`scripts/`、`configs/qwen3_8b_native_nvfp4_linear_puncture.yaml`
- R64 共享变换：`experiments/diag_gradient/r64_transform.py`
- 旧逐层重建路径：`experiments/e2e_diag_reconstruction/` 中保留的 dense 8B 逻辑
- 历史计划：`plans/qwen3_8b_qat/`
- 历史结果归档入口：`results/qwen3_8b_qat/`
- 原始根 README 已完整保留：`README_QWEN3_8B_QAT_LEGACY.md`

### Qwen3-30B-A3B：当前 MoE 主线

模型：`nvidia/Qwen3-30B-A3B-NVFP4`

特点：Qwen3 MoE / ModelOpt NVFP4 checkpoint，不沿用旧 8B checkpoint 的在线 H16 语义。当前工作围绕 MoE Native NVFP4→HiF4、逐层 reconstruction、vLLM runtime、TP2 数值语义和长生成轨迹误差累积展开。

相关入口：

- MoE reconstruction / materialize / evaluation：`experiments/e2e_diag_reconstruction/`
- 不等价变换：`experiments/non_equivalent_reconstruction/`
- 优化目标：`experiments/internal_error_accumulation/`、`experiments/kl_direction_reuse/`
- 渐进式误差抵消：`experiments/progressive_error_cancellation/`
- 真实 vLLM hook 共享实现：`experiments/long_trajectory_stability/real_vllm_hooks/`
- 当前计划：`plans/qwen3_30b_a3b/`
- 当前主要结果：`results/e2e_diag_reconstruction/`、`results/long_trajectory_stability/`

> `experiments/long_trajectory_stability/` 当前仍是正在搭建和校准的诊断 runtime。它不是已经完全复现 vLLM 的独立推理后端；RMSNorm、fused NVFP4 MoE 等部分已直接对齐 vLLM 算子，attention / RoPE / TP2 等语义仍在继续做 parity 对齐。

## 2. 当前目录职责

| 目录 | 主要职责 | 模型归属 |
|---|---|---|
| `configs/` | 原始 8B Linear puncture 配置 | Qwen3-8B QAT |
| `src/` | packed NVFP4 解析、格式、旋转及早期 capture/Linear 工具；当前转换仍复用其中实现 | 共享及 Qwen3-8B QAT |
| `scripts/` | 原始 8B Linear puncture 执行脚本 | Qwen3-8B QAT |
| `experiments/diag_gradient/r64_transform.py` | 30B 主线复用的 R64 变换实现 | 共享依赖 |
| `experiments/e2e_diag_reconstruction/` | 可学习对角变换、逐层 reconstruction 和评估 | Qwen3-30B-A3B |
| `experiments/non_equivalent_reconstruction/` | 不等价变换和最终导出对比 | Qwen3-30B-A3B |
| `experiments/internal_error_accumulation/` | 格式转换误差与保护目标研究 | Qwen3-30B-A3B |
| `experiments/kl_direction_reuse/` | KL 方向复用和优化目标实验 | Qwen3-30B-A3B |
| `experiments/progressive_error_cancellation/` | 渐进式误差抵消实验；运行中的目录受保护 | Qwen3-30B-A3B |
| `experiments/long_trajectory_stability/real_vllm_hooks/` | 优化目标依赖的真实 vLLM hook | Qwen3-30B-A3B |
| `plans/qwen3_8b_qat/` | 旧 8B 实验计划 | Qwen3-8B QAT |
| `plans/qwen3_30b_a3b/` | 当前 30B MoE 实验计划 | Qwen3-30B-A3B |
| `results/qwen3_8b_qat/` | 旧 8B 结果的独立归档入口 | Qwen3-8B QAT |
| `results/e2e_diag_reconstruction/` | 当前 reconstruction 公共缓存及 30B 结果 | 主要为 Qwen3-30B-A3B |
| `results/long_trajectory_stability/` | 30B 长轨迹和数值语义诊断结果 | Qwen3-30B-A3B |
| `tests/` | 对应各实现路径的单测/语义检查 | 混合，按子目录区分 |

## 3. 结果归属规则

不要只按 `results/` 下的时间戳判断模型来源。判断历史结果属于哪条模型线时，优先看：

1. `config.json` / `summary.json` / manifest 中的 `model_id`、`model_path` 或 `source_model`；
2. capture manifest 中记录的 source checkpoint；
3. 对旧机制实验，可继续沿 source capture run 向上追溯。

明确属于 Qwen3-8B QAT 的旧结果包括：

- `20260812T103735Z_native_nvfp4_hif4_linear_puncture`
- `20260812T103800Z_native_nvfp4_hif4_linear_puncture`
- `20260813T062121Z_theory_grid_scale_validation`
- `20260813T090200Z_diag_group_stats`
- 2026-08-15～2026-08-17 的 DIAG / H4 / R64 gradient runs
- `archive/legacy/native_qwen3_8b/activation_3d_viz/`
- `archive/legacy/native_qwen3_8b/h4_block_rotation/`
- `results/e2e_diag_reconstruction/` 中 2026-08-18 的 8B reconstruction / smoke / ablation 结果
- 旧 `shared_vllm/` 8B materialization

以下内容不要归入 8B 历史目录：

- `results/e2e_diag_reconstruction/shared_calibration/`：虽然最初由 8B 流程建立，但后续 30B 仍在复用，保持公共位置；
- `phase1_20260824*`、`smoke_refactor_20260825*`、`phaseA_refactor_20260825*`；
- `shared_vllm_qwen3_30b/`；
- `results/long_trajectory_stability/`。

## 4. 整理原则

本项目将当前研究代码与历史记录分开；以下保留早期归档边界，当前路线分类见实验导航：

- 当前转换代码维护在 `fp4_convert/`；活动实验使用的旧源码、共享依赖和产物保持原路径；
- 退出主线的代码集中放入仓库根目录 `archive/legacy/`，保留 README、报告和归档说明；
- 历史结果只在确认没有活动进程、默认入口或正式证据引用后清理；
- 不修改历史 manifest、日志和结果中的原始路径，也不通过 reset 覆盖已有工作区改动。

## 5. 环境

所有 Python、pytest、训练和 vLLM 推理命令均使用仓库规定的 `hif4` conda 环境。

旧 Qwen3-8B Linear puncture 的原始命令和约束请查看 `README_QWEN3_8B_QAT_LEGACY.md`；当前 Qwen3-30B-A3B 的具体执行方式以 `plans/qwen3_30b_a3b/` 和对应实验目录 README 为准。
