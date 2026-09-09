# Native NVFP4 → HiF4 实验工程

本目录研究 **Native NVFP4 → HiF4 格式转换** 的数值误差、变换优化、端到端精度和长生成轨迹稳定性。

项目最初建立在 `ISTA-DASLab/Qwen3-8B-FPQuant-QAT-NVFP4` 上，随后主线切换到 `nvidia/Qwen3-30B-A3B-NVFP4`。因此当前目录同时保留两代实验，但二者的模型结构、checkpoint 语义和实验用途不同，不能混用结果。

## 1. 两条模型线

### Qwen3-8B QAT：历史机制实验

模型：`ISTA-DASLab/Qwen3-8B-FPQuant-QAT-NVFP4`

特点：dense Qwen3-8B，checkpoint 内含在线 block rotation。主要用于早期 Linear 穿刺、激活/权重格式转换误差分析，以及 DIAG、H4、R64 等变换机制验证。

相关入口：

- 原始 Linear 穿刺实现：`src/`、`scripts/`、`configs/qwen3_8b_native_nvfp4_linear_puncture.yaml`
- 激活可视化：`experiments/activation_3d_viz/`
- DIAG / H4 / R64：`experiments/diag_gradient/`
- H4 block rotation：`experiments/h4_block_rotation/`
- 旧逐层重建路径：`experiments/e2e_diag_reconstruction/` 中保留的 dense 8B 逻辑
- 历史计划：`plans/qwen3_8b_qat/`
- 历史结果归档入口：`results/qwen3_8b_qat/`
- 原始根 README 已完整保留：`README_QWEN3_8B_QAT_LEGACY.md`

### Qwen3-30B-A3B：当前 MoE 主线

模型：`nvidia/Qwen3-30B-A3B-NVFP4`

特点：Qwen3 MoE / ModelOpt NVFP4 checkpoint，不沿用旧 8B checkpoint 的在线 H16 语义。当前工作围绕 MoE Native NVFP4→HiF4、逐层 reconstruction、vLLM runtime、TP2 数值语义和长生成轨迹误差累积展开。

相关入口：

- MoE reconstruction / materialize / evaluation：`experiments/e2e_diag_reconstruction/`
- 长轨迹稳定性与 semantic replay：`experiments/long_trajectory_stability/`
- 当前计划：`plans/qwen3_30b_a3b/`
- 当前主要结果：`results/e2e_diag_reconstruction/`、`results/long_trajectory_stability/`

> `experiments/long_trajectory_stability/` 当前仍是正在搭建和校准的诊断 runtime。它不是已经完全复现 vLLM 的独立推理后端；RMSNorm、fused NVFP4 MoE 等部分已直接对齐 vLLM 算子，attention / RoPE / TP2 等语义仍在继续做 parity 对齐。

## 2. 当前目录职责

| 目录 | 主要职责 | 模型归属 |
|---|---|---|
| `configs/` | 原始 8B Linear puncture 配置 | Qwen3-8B QAT |
| `src/` | 原始 packed NVFP4 解析、capture、Linear cases、格式模拟 | Qwen3-8B QAT |
| `scripts/` | 原始 8B Linear puncture 执行脚本 | Qwen3-8B QAT |
| `experiments/activation_3d_viz/` | 保存激活的可视化诊断 | Qwen3-8B QAT |
| `experiments/diag_gradient/` | DIAG / H4 / R64 局部优化和组合实验 | Qwen3-8B QAT |
| `experiments/h4_block_rotation/` | H4 四维块旋转机制实验 | Qwen3-8B QAT |
| `experiments/e2e_diag_reconstruction/` | 逐层 reconstruction 框架；先支持 8B，后增量适配 30B MoE | 共享实现，当前主线为 30B |
| `experiments/long_trajectory_stability/` | vLLM free-run、teacher-forcing replay、算子 parity、TP2 / residual / RoPE 对齐 | Qwen3-30B-A3B |
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
- `activation_3d_viz/`
- `h4_block_rotation/`
- `results/e2e_diag_reconstruction/` 中 2026-08-18 的 8B reconstruction / smoke / ablation 结果
- 旧 `shared_vllm/` 8B materialization

以下内容不要归入 8B 历史目录：

- `results/e2e_diag_reconstruction/shared_calibration/`：虽然最初由 8B 流程建立，但后续 30B 仍在复用，保持公共位置；
- `phase1_20260824*`、`smoke_refactor_20260825*`、`phaseA_refactor_20260825*`；
- `shared_vllm_qwen3_30b/`；
- `results/long_trajectory_stability/`。

## 4. 整理原则

本项目保留实验历史，不通过“整理目录”改写实验事实：

- 不删除旧代码、旧计划、旧结果或旧日志；
- 不修改历史 `config.json`、manifest、日志、`run_map.json` 中记录的原始绝对路径；
- 不为了目录统一而重写正在使用的 30B Python import 路径；
- 当前运行代码只有在确实读取被移动资源时才修改路径。

## 5. 环境

所有 Python、pytest、训练和 vLLM 推理命令均使用仓库规定的 `hif4` conda 环境。

旧 Qwen3-8B Linear puncture 的原始命令和约束请查看 `README_QWEN3_8B_QAT_LEGACY.md`；当前 Qwen3-30B-A3B 的具体执行方式以 `plans/qwen3_30b_a3b/` 和对应实验目录 README 为准。
