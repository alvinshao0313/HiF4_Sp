# Qwen3-30B-A3B 当前实验计划

本目录保存基于 `nvidia/Qwen3-30B-A3B-NVFP4` 的 MoE NVFP4→HiF4 当前主线计划。

## 2026-11-16 格式转换章节补充实验

[格式转换章节补充实验计划](2026-11-16-thesis-format-conversion-supplement-plan-cn.md)：排除 AIME，固定 MMLU-Pro、LCB、GSM8K、ARC-E/C；将 R64＋scale 搜索与 R64＋HiF4-GPTQ 并列为强基线，补最终方法机制、单侧正交消融与已有短程模型的精度/成本对照。

当前为计划，新增适配和 GPU 实验尚未执行；具体复用范围、启动顺序和资源安排见正文。以下计划保留各自研究问题与历史语境，不因本次补实验重新启动全部旧任务。

## 2026-09-09 模型内部误差累计机制计划

`2026-09-09-qwen3-30b-hif4-internal-error-accumulation-mechanism-plan-cn.md`

该计划不再研究 token trajectory，而是固定相同 prompt / forced history，在真实 vLLM TP2 actual path 上研究：residual stream 的结构性误差累计、Attention/MoE signed error gain、MoE `q + p_expert + p_router` 分解、Router 对最终 one-step loss 的因果贡献、层保护，以及 cumulative-residual-aware 优化目标。

## 已完成的任务级格式转换误差定位计划

`2026-09-08-qwen3-30b-hif4-error-localization-replan-after-lcb-prompt-audit-cn.md`

2026-09-08 的 formal prompt 审计确认旧 Phase-A E0 与 E1/E2/E3 的 LiveCodeBench 输入协议不一致，因此旧 `95/175 → 34/175` 只能保留为历史 observed score，不能再作为纯 NVFP4→HiF4 因果掉点。

当前主问题重新收束为：

> **先在完全相同 prompt token IDs / execution shape 下重新确认 E0→E1 是否存在实质 LCB task loss；只有确认后，才继续定位局部数值误差如何变成 token/task failure。**

正式机制证据继续固定来自：

```text
真实 vLLM TP2
+ isolated single-request incremental KV-cache decode
+ E0 exact isolated token trajectory 强制解码
+ LLM.apply_model() worker 内 hook / production puncture / state intervention
```

新计划：

- E0↔E1 为主研究对象；
- 先完成 matched-prompt greedy audit；LCB benchmark 复用已审计通过的正式 E0，只重跑完全复刻 E0 协议的 E1 全 175；
- MMLU-Pro300 复用已审计通过的正式 E0，只重跑 E1–E7，并完整复刻 E0 的 runner / prompt / TP2 / KV BF16 / generation / `max_num_seqs=128` 协议；
- MMLU-Pro300 E1–E7 重跑是无条件必做项，不受 LCB Gate A/B/C 影响；E0 不重跑；
- 依据 clean LCB task loss 设置 Gate A/B/C，自动停止不再必要的昂贵机制实验；
- 已否定/弱化的 decode-length accumulation、低-margin enrichment、router-topk enrichment、exact rejoin rescue/poison 不再重复；
- RQ1 默认 MoE-first，只在关键 layer 做 production same-input `e=q+p`；
- E2 R64、DIAG 只在 clean task loss 和主机制都确认后作为机制探针；
- DIAG 机制不再直接用 adopted E3，优先 candidate，必要时以 E5 Online DIAG 作 diagnostic upper-bound；
- E4 不进入本轮；
- 不恢复旧 semantic replay，也不重新构造第二套 decoder/runtime。

旧 `2026-09-07-qwen3-30b-hif4-lcb-format-conversion-mechanism-go-plan-cn.md` 保留为 prompt 审计前的历史计划，不再作为执行入口。

## 已完成的基础设施计划

`2026-09-01-qwen3-30b-hif4-long-trajectory-real-vllm-hook-replan-cn.md`

该计划完成了从旧 semantic replay 向真实 vLLM worker hook 主路径的迁移，并产出了 MMLU-Pro/LiveCodeBench 的现有 real-vLLM 诊断结果。它现在作为**基础设施与历史结果计划**保留，不再是后续机制研究的执行入口。

旧 semantic 对齐文档只作为历史排错记录；`nvfp4_operator_parity/` 仍可作为局部 frozen-input/operator 辅助证据，但不能替代真实 vLLM actual-path 结果。
