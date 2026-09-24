# 评测协议与真实执行路径

证据状态：从现有审计与复盘提炼，未重跑评测。适用于 Qwen3-30B-A3B NVFP4→HiF4 的格式转换比较和机制实验；其他模型须重新检查适用条件。

## 输入不一致时，分数差不能单独归因于格式转换

[2026-09-07 正式 LCB 协议审计](../../../Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/format_conversion_lcb_mechanism_go_20260907/00_audit/FORMAL_PROTOCOL_MISMATCH.md) 发现 E0 的 175/175 输入带 chat wrapper，而 E1/E2/E3 与 E0 完全一致的输入均为 0/175。该批分数可以保留为历史观测，不能解释成相同输入下仅改变量化格式的因果效应。

比较前应检查实际输入 token、模板、thinking 设置、样本 ID、采样参数与评分口径；配置中的 null 不能被当成显式开启。优先使用适用的已有逐样本证据，不因文档整理重复评测。正式 sampled 评测与隔离 greedy 机制诊断分开解释，不互相替代。

## 机制证据要来自所研究的执行路径

[旧 semantic replay 复盘](../../../Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/LEGACY_SEMANTIC_REPLAY_POSTMORTEM.md) 记录了同一绝对位置在增量前缀与一次完整序列计算下 top1 不同，以及手工 TP 适配改变结果的反例。逻辑 causal 相同不足以证明数值执行路径相同。

该研究线后来采用真实 vLLM TP2 的 prefill、incremental KV-cache decode 和 worker hook 作为正式机制证据。局部 frozen-input 对照仍可解释算子行为，但不能代替真实长轨迹或候选模型的下游结果。该要求属于这条实验协议，不扩展成所有 HiF4_Sp 算法任务都必须逐位复现 TP2。

## 复用权重时检查预测缓存的实际位置

2026-09-24 接续[残差 LoRA 下游计划](../../../Native_NVFP4_HiF4_Linear_Puncture/experiments/residual_lora_compensation/PLAN.md)时检查到，当前 vendored lighteval 的 [SampleCache](../../../3rdparty/lighteval/src/lighteval/utils/cache_management.py) 用 `cache_dir / model_name / model_hash` 组成路径；当 `model_name` 是绝对路径时，缓存实际落在模型目录中。仅更换结果目录不足以隔离历史生成结果，短测也可能为之后的正式运行留下预测缓存。

该实验为正式评测创建独立的模型目录视图，只链接权重和配置文件，不链接缓存子目录；旧 artifact 和已有缓存保留。此做法隔离生成缓存，不改变权重、prompt 或评分协议。适用于当前缓存实现和本地绝对模型路径，其他后端应先检查自己的缓存键与读取规则。

## 结论与状态分开

[2026-09-16 局部算术诊断](../../../Native_NVFP4_HiF4_Linear_Puncture/results/internal_error_accumulation/qwen3_30b_a3b_e0_e1_formal/runs/iea_20260909T081436Z_1076681/60_objective/corrected_objectives_v3/kernel_boundary_alignment_v1/CONCLUSIONS.md) 明确限定在固定输入、零 DIAG、单 predictor 范围，不能推广成非零 DIAG、完整反向或全序列已验证。该日的 BLOCKED / WAITING_REVIEW 同样只是当时状态。

普通 TP 归约、累加和舍入差异依 [AGENTS.md](../../../AGENTS.md) 判断是否影响算法结论；局部严格契约仍按协议验证。报告存在、进程退出或某项局部对齐通过，不自动等于全实验完成，也不授权进入后续阶段。

证据导航见 [实验记录](../records/README.md)，计划替代关系见 [方案入口](../../../Native_NVFP4_HiF4_Linear_Puncture/plans/qwen3_30b_a3b/README.md)。
