# 原 B 的完整 KL 与硬 CE 对照

两组从原 B 初始化，固定数据与可学习参数范围，独立于旧实验。算法、训练预算、token 边界及开发集决选规则见 [PLAN.md](PLAN.md)。

在仓库根目录、调用者已激活 `hif4` 时：

```bash
python -u -m Native_NVFP4_HiF4_Linear_Puncture.experiments.reasoning_trace_full_kl_ce.run \
  --source-run Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_calibration/continue_20261016T135141Z \
  --cache-run Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_last_head/run_20261020T031200Z \
  --run Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_full_kl_ce/run_UTC \
  --selection-policy development --stage qualify
```

验证通过后，对同一 RUN 使用 `--selection-policy development --stage experiment` 脱离会话启动。完整 KL 在 GPU 6、硬 CE 在 GPU 7 并行训练；两组都结束后，开发集和正式推理顺序使用 TP2 6/7。只读复用原 B 输入缓存，教师完整隐藏状态单独补算，不生成新轨迹。

`commands.jsonl` 记录命令、PID、GPU、日志与退出码；`manifest.json` 区分运行、阶段完成、等待 AIME 复核和完成。新模型为 BF16 QDQ 普通权重及 BF16 未量化 head，不是 packed 4-bit 存储。

出现 AIME 待审项时，逐条核对 raw 回答，在对应 `evaluation/CANDIDATE/aime25/reviews.json` 保存绑定原文 SHA 的决定，再用 `evaluation --run RUN --objective full_kl|hard_ce --task aime25 --rescore-aime` 重算。两组均完整后，`run --run RUN --selection-policy development --stage report` 生成 `comparison.json`、`comparison.md`。

2026-10-21 独立运行：[运行交付与验证记录](../../results/reasoning_trace_full_kl_ce/run_20261021T030031Z/HANDOFF.md)。该记录描述交付时点，实时状态见对应 manifest 和日志。
