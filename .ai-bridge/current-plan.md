# Backport vLLM NVFP4 emulation

Updated: 2026-08-21T02:22:15.250Z
Workspace: /home/shaoyuantian/program/HiF4_Sp
Target agent: Cursor (custom)

## Plan

实施方案已经由用户批准，但当前步骤只完成计划交付，不自动修改源码。后续用户要求 Cursor 开始实现时，严格依次执行以下两个文件：

1. 设计约束：`NVFP4/plans/2026-08-21-vllm-v027-nvfp4-emulation-backport-design-cn.md`
2. 闭环实施计划：`NVFP4/plans/2026-08-21-vllm-v027-nvfp4-emulation-backport-plan-cn.md`

实施原则：以 vLLM tag `v0.27.0` 为唯一 upstream 功能/数值基线；先执行 Task 0 获取 tag SHA 并生成逐文件依赖闭包白名单，再按 Task 1→13 顺序执行。不要整体升级 `3rdparty/vllm`，不要修改 upstream NVFP4 scale/QDQ 逻辑，不实现 per-expert scale 修正版，不静默 fallback 到 Marlin W4A16，不把 packed NVFP4 权重全量展开成 BF16 常驻，不修改 Qwen3 专用模型逻辑。ModelOpt 与 compressed-tensors、Dense 与 MoE 均使用 upstream emulation backend。KV cache 只使用 vLLM 自身 `kv_cache_dtype`，正式验收分别测 `bfloat16` 和 `auto`。当前工作区存在大量无关修改，所有改动必须限制在 Task 0 inventory 白名单内，并按测试先行的 red→green 步骤执行。

真实验收 checkpoint：`/home/shaoyuantian/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3/`。所有 Python/test/inference 命令使用 `hif4` conda 环境。

## Implementation contract

- Work from this plan in small, reviewable steps.
- Keep edits scoped to the requested task and existing project conventions.
- Run focused verification before handing work back.
- Update .ai-bridge/agent-status.md with files touched, checks run, results, blockers, and review notes.
- Save the final review diff to .ai-bridge/implementation-diff.patch when practical.
- Append notable execution events to .ai-bridge/execution-log.jsonl when the implementation agent supports logging.
