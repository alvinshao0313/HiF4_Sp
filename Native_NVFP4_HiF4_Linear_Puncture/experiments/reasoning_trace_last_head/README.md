# B 末层与输出端词表蒸馏

固定算法见 [PLAN.md](PLAN.md)。此目录独立于旧 A/B/C 和 BF16 router 实验，只从原 B 恢复，不修改旧结果。

本轮 RUN：[固定 manifest](../../results/reasoning_trace_last_head/run_20261020T031200Z/manifest.json)、[必要验证证据](../../results/reasoning_trace_last_head/run_20261020T031200Z/qualification/complete.json)、[后台命令和 PID](../../results/reasoning_trace_last_head/run_20261020T031200Z/launch.json)。运行中不代表评测完成，状态以 RUN 的 manifest、逐命令退出记录及最终结果为准。

2026-10-21 已完成完整评测及 AIME 待审回答复核：[最终对比](../../results/reasoning_trace_last_head/run_20261020T031200Z/comparison.md)、[逐题配对区间](../../results/reasoning_trace_last_head/run_20261020T031200Z/comparison.json)、[AIME 复核记录](../../results/reasoning_trace_last_head/run_20261020T031200Z/evaluation/B_last_head_vocab64/aime25/reviews.json)。局部 KL 下降，本轮综合下游观测未提高；解释边界见[评测证据经验](../../../docs/experience/lessons/evaluation_evidence.md)。

在仓库根目录、调用者已激活 `hif4` 时：

```bash
python -u -m Native_NVFP4_HiF4_Linear_Puncture.experiments.reasoning_trace_last_head.run \
  --source-run Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_calibration/continue_20261016T135141Z \
  --run Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_last_head/run_UTC \
  --train-gpu 6 --inference-gpus 6,7 --stage qualify
```

`qualify` 包含两个独立前缀的完整缓存、最长 batch 短测和零步/更新 TP2 推理；通过后以同一 RUN 的 `--stage experiment` 后台启动正式筛选、20 epochs、导出、完整评测和报告。源码或输入变化会明确失败，不自动重试。正式任务 `commands.jsonl` 记录实际子命令、PID、日志与退出码；`manifest.json` 区分运行、阶段完成、失败和等待 AIME 复核。正式训练不继承短测参数或优化器。

`evaluation --run RUN --task aime25 --rescore-aime` 可在逐条人工复核并保存绑定 raw SHA 的 `manual_reviews.json` 后汇总；随后 `run --run RUN --stage report` 输出 `comparison.json`、`comparison.md`。原始生成缓存与原评分保留，报告含四项及去掉 GSM8K 的三项口径。BF16 QDQ 导出不是 packed 4-bit 存储证明。
