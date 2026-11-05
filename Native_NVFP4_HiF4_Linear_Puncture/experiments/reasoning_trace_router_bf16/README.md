# B 的 BF16 Router 对齐

定义见 [PLAN.md](PLAN.md)。独立代码/结果，不改变旧 A/B/C 的源码绑定。`student.py` 读取 B 的已融合 QDQ 分片；`training.py` 负责独立前缀和 BF16 router 更新；导出、真实 TP2 验证、完整评测及配对报告分别实现。

由调用者先激活 `hif4`，在仓库根目录使用：

```bash
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.reasoning_trace_router_bf16.run \
  --source-run Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_calibration/continue_20261016T135141Z \
  --run Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_router_bf16/NEW_RUN \
  --stage prepare
```

随后同一入口以 `--stage qualify --train-gpu 6 --inference-gpus 6,7` 完成必要验证。`--stage experiment` 复用匹配源码的通过证据，从 B 原始参数开始正式训练并依次导出、真实 TP2 验证和完整评测。新输出拒绝覆盖，没有自动恢复或重试。正式命令必须后台脱离会话、stdout/stderr 落盘。

AIME 待审使评测退出 21，RUN 记录 `awaiting_aime_review`。在对应 `reviews.json` 记录逐条 raw_sha256/score/reason 后，用 `evaluation --run RUN --task aime25 --rescore-aime` 重算已有回答，再用 `run --run RUN --stage report` 汇总。旧 B 的指标不重算、不覆盖。

2026-10-19 实施及验证状态见[独立 RUN 报告](../../results/reasoning_trace_router_bf16/run_20261019T031744Z/REPORT.md)。
