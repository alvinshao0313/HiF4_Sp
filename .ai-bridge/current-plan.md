# Resume Qwen3-30B MoE HiF4 implementation

Updated: 2026-08-21T10:01:40.397Z
Workspace: /home/shaoyuantian/program/HiF4_Sp
Target agent: Codex (codex)

## Plan

严格执行已修订计划：`Native_NVFP4_HiF4_Linear_Puncture/plans/2026-08-21-qwen3-30b-a3b-native-nvfp4-to-hif4-moe-e2e-ablation-no-teacher-cot-plan-cn.md`。

当前不是从零开始，也不是直接从 Task 5 开始。固定恢复流程：
1. 先做 Task 0～4 Definition-of-Done audit，只检查并补缺，不重写已经正确的文件。
2. Task 3 当前已知未完成：`training/moe_layer_runtime.py` 不存在；`test_moe_native_teacher.py` 只有 QDQ 单测和 CUDA smoke，没有真实 attention projection vs vLLM emulation puncture、完整 routed MoE output puncture、router logits/top-k reference gate。先补完 Task 3 并实际 PASS。
3. 重跑并验收 Task 4 的无-H16 transform equivalence tests。
4. 然后连续执行 Task 5A→5E、Task 6、7、8、9、10、11、12、13。完成每个 Task 后更新 `Native_NVFP4_HiF4_Linear_Puncture/results/e2e_diag_reconstruction/QWEN3_30B_IMPLEMENTATION_PROGRESS.md`，验收通过立即继续下一 Task。
5. 文件存在不等于 Task 完成；smoke 不能替代计划要求的 numerical/reference gate。
6. 在 Task 13 前不得因为“本轮改动较多/适合 review/先汇报”主动停止。small/reviewable steps 只表示实现方式，不表示交还边界。
7. Task 13 前唯一允许停止：真实 checkpoint/Transformers/vLLM 事实冲突会改变算法语义；数值门禁失败且需要用户选择新的算法定义；资源不足且无法在不改变固定实验口径下继续。触发时按计划写 `QWEN3_30B_IMPLEMENTATION_STOP_REASON.md`，否则继续修 bug。
8. Task 13 全部 correctness/smoke PASS 后停止并汇报；不要自动启动 Task 14～19 的 25 个正式 training runs。
9. 所有 Python/pytest/训练命令使用 `hif4` conda 环境；不允许 Marlin W4A16 fallback、silent OOM/adaptive batch；正式 KV=BF16；保持旧 8B 路径不被破坏。
10. 当前工作区可能含无关改动，禁止清理或覆盖无关文件。

## Implementation contract

- Work from this plan in small, reviewable steps.
- Keep edits scoped to the requested task and existing project conventions.
- Run focused verification before handing work back.
- Update .ai-bridge/agent-status.md with files touched, checks run, results, blockers, and review notes.
- Save the final review diff to .ai-bridge/implementation-diff.patch when practical.
- Append notable execution events to .ai-bridge/execution-log.jsonl when the implementation agent supports logging.
