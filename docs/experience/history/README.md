# 历史与方案入口

本页统一导航既有历史与设计。方案描述意图，记录描述事实；日期新旧不能单独证明实施状态。保留在实验目录的 `PLAN*.md` 可能仍是适用方案，但不单独构成执行、变更环境或 Git 操作的授权。

## 安装沿革

- [vLLM 安装排错记录](environment/vllm_install_notes.md)：包含旧 wheel、ABI 和构建尝试；现行安装使用 [安装指南](../../guides/install.md)。
- [v0.17 源码构建记录](environment/vllm_0_17_source_build_guide.md)：保留旧迁移过程，不按原文切回旧版本。
- 可复用结论见 [构建经验](../lessons/vllm_build.md)。

## 研究方案与模型线

- [Qwen3-30B-A3B 方案沿革](../../../Native_NVFP4_HiF4_Linear_Puncture/plans/qwen3_30b_a3b/README.md)：包含格式转换、真实 vLLM、协议审计后的重规划和内部误差研究。读取当前代码和对应运行记录后再判断适用性。
- [Qwen3-8B QAT 历史方案](../../../Native_NVFP4_HiF4_Linear_Puncture/plans/qwen3_8b_qat/README.md) 与 [原工程说明](../../../Native_NVFP4_HiF4_Linear_Puncture/README_QWEN3_8B_QAT_LEGACY.md)：旧 dense / online rotation 模型线。
- [NVFP4 仿真与 backport 方案](../../../NVFP4/plans/)：2026-08-21 的设计、执行计划；相关报告在 [记录入口](../records/README.md)。
- 实验目录中的 KL 与 LoRA 方案见 [研究导航](../../README.md)。所有方案按路径列于 [自动索引](../../INDEX.md)。

## 源码归档与工具交接

[archive/legacy](../../../archive/legacy/README.md) 说明旧 8B、rotation、QAD / ScaleTuning 的源码位置。归档 README 中原命令保留当时语境，不承诺旧路径仍可直接执行；源码迁移不等于所有共享依赖一起迁移。

`.ai-bridge/` 保留原有交接状态，只在追溯具体任务时读取；不替代根目录 AGENTS、实验配置或运行证据。运行目录中的 handoff 与报告同样按记录时点解释。
