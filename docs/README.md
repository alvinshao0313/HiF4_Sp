# HiF4_Sp 文档入口

本目录只维护 **HiF4_Sp** 的指南、经验与研究记录。项目约束以根目录 [AGENTS.md](../AGENTS.md) 为准，运行环境为 `hif4`；其他项目的规则仅作迁移来源，不决定本项目的目录、环境或实验协议。

先按下面的主题定位，再读取所需证据。全仓库自有文档见自动生成的 [INDEX.md](INDEX.md)，维护方式见 [MANAGEMENT.md](MANAGEMENT.md)。

## 使用与安装

| 主题 | 正文 |
| --- | --- |
| 评测、量化命令与项目总览 | [仓库 README](../README.md) |
| 环境安装与排错入口 | [安装指南](guides/install.md) |
| HiFloat4 GPU fake quant 实现 | [量化流程](guides/hif4_gpu_quant_flow.md) |
| HiFloat4 RTN / GPTQ | [模块 README](../HiFloat4/README.md) |
| 量化阈值标定 | [模块 README](../HiFloat4/hif4_scale_threshold_optimization/README.md) |

## Native NVFP4 → HiF4 研究

主线模型为 `nvidia/Qwen3-30B-A3B-NVFP4`，详见 [Native 工程入口](../Native_NVFP4_HiF4_Linear_Puncture/README.md)。旧 Qwen3-8B QAT 的在线 rotation 与当前 MoE checkpoint 语义不同，结果不能直接混用。

| 研究主题 | 入口 |
| --- | --- |
| 可学习对角变换、逐层重构 | [e2e_diag_reconstruction](../Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/README.md) |
| 不等价变换、矩阵共享与 router loss | [non_equivalent_reconstruction](../Native_NVFP4_HiF4_Linear_Puncture/experiments/non_equivalent_reconstruction/README.md) |
| 优化目标、KL 方向复用 | [kl_direction_reuse](../Native_NVFP4_HiF4_Linear_Puncture/experiments/kl_direction_reuse/README.md) · [方案 v2](../Native_NVFP4_HiF4_Linear_Puncture/experiments/kl_direction_reuse/PLAN_v2.md) |
| 渐进式累计隐状态误差抵消 | [progressive_error_cancellation](../Native_NVFP4_HiF4_Linear_Puncture/experiments/progressive_error_cancellation/README.md) |
| 内部误差与真实 vLLM 机制分析 | [计划沿革](../Native_NVFP4_HiF4_Linear_Puncture/plans/qwen3_30b_a3b/README.md) · [证据入口](experience/records/README.md) |
| KL 下游方案、残差 LoRA 方案 | [KL downstream PLAN](../Native_NVFP4_HiF4_Linear_Puncture/experiments/kl_direction_reuse_downstream/PLAN.md) · [residual LoRA PLAN](../Native_NVFP4_HiF4_Linear_Puncture/experiments/residual_lora_compensation/PLAN.md) |

这些链接用于定位设计和实现，不代表实验已启动、已完成或已获准进入下一阶段。运行状态须用对应配置、日志、指标和进程核实。

## 经验、证据与历史

- [经验库](experience/README.md)：设计或调整实验前按主题检索，区分实测、解释与假设。
- [实验与维护记录](experience/records/README.md)：原始报告保留在各自实验目录，具体数值不复制成多份正文。
- [历史与方案](experience/history/README.md)：安装沿革、旧模型线、已归档源码及计划。
- 清理记录：[2026-09-24 冗余恢复状态与缓存](experience/records/maintenance/2026-09-24-history-cleanup.md) · [2026-09-20 源码归档与产物清理](experience/records/maintenance/2026-09-20-experiment-cleanup.md)。记录保留对应时点事实，不构成新的删除授权。

在仓库根目录、已激活 `hif4` 的 shell 中检索：

```bash
python tools/docs.py search 'router'
python tools/docs.py search 'PROMPT_PROTOCOL' --scope records
python tools/docs.py search 'semantic' --scope history
```

默认搜索指南与经验；`records` 包括实验目录中的生成报告，`history` 包括方案和交接旧记录。第三方入口见 [vLLM](../3rdparty/vllm/README.md)、[lighteval](../3rdparty/lighteval/README.md) 和 [llm-compressor 目录](../NVFP4/llm-compressor/)，不纳入本项目正文迁移。
