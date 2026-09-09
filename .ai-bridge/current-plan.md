# Qwen3-30B HiF4 error localization replan

Updated: 2026-09-08T02:20:41.555Z
Workspace: /home/shaoyuantian/program/HiF4_Sp
Target agent: Codex (codex)

## Plan

请从当前仓库真实状态继续，以 Go 模式执行最新唯一计划：

`Native_NVFP4_HiF4_Linear_Puncture/plans/qwen3_30b_a3b/2026-09-08-qwen3-30b-hif4-error-localization-replan-after-lcb-prompt-audit-cn.md`

必须先读取该计划和当前运行状态，不要继续按 2026-09-07 旧计划无条件跑 heavy mechanism chain。

关键执行约束：
- 当前正在进行的 matched-prompt E1 55 条补跑允许完成；
- E1 补完后只做 official greedy judge audit，先设置 Stop Gate；
- 禁止旧 `run_go.py --through report` 自动继续 feature/core/puncture/state/E2/E3；
- LCB benchmark 复用旧正式 E0；只重跑 E1 全 175，并直接复用 E0 exact input IDs、完整复刻旧 E0 协议与 `max_num_seqs=128`；当前任何新的 LCB E0 benchmark 重跑都应停止；
- MMLU-Pro300 复用旧正式 E0，不重新测 baseline；只重跑 E1–E7，并完整复刻旧 E0 的 runner / prompt / TP2 / KV BF16 / generation / `max_num_seqs=128` 协议；E1–E7 重跑不受 LCB Gate A/B/C 影响；
- 根据 clean LCB 的 Gate A/B/C 决定后续是否解锁 MoE-first `e=q+p`、少量 state intervention、E2/DIAG 机制探针；
- 已否定/弱化的 decode-length monotonic accumulation、low-margin enrichment、router-topk enrichment、exact rejoin rescue/poison 不再重复；
- DIAG 机制禁止直接用 adopted E3，优先 E3 candidate，必要时 E5 Online DIAG 作为 diagnostic control；
- E4/E6/E7 不做机制深钻，但它们的 MMLU-Pro300 benchmark 仍需重跑；E0 不重跑；
- 正式数值证据仍只允许真实 vLLM TP2 actual path，不恢复 semantic replay，不搭第二套 runtime。

除真正 hard blocker 外按计划门禁自动执行，但门禁要求停止的实验必须停止，不得为了“跑完整计划”强行继续。

## Implementation contract

- Work from this plan in small, reviewable steps.
- Keep edits scoped to the requested task and existing project conventions.
- Run focused verification before handing work back.
- Update .ai-bridge/agent-status.md with files touched, checks run, results, blockers, and review notes.
- Save the final review diff to .ai-bridge/implementation-diff.patch when practical.
- Append notable execution events to .ai-bridge/execution-log.jsonl when the implementation agent supports logging.
