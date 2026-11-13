# Qwen3-30B-A3B 官方 BF16 GPQA

独立评测 `Qwen/Qwen3-30B-A3B`，不修改量化或剪枝路径。使用仓库根目录
`main.py` 的结果记录器、vLLM 和 lighteval；当前按用户要求顺序执行三轮，
每轮 GPQA Diamond 全部 198 题、每题一次采样。种子依次为1234、1235、1236，
保持输入选项种子1234；报告三轮准确率及算术平均，不使用投票。

实际配置、官方权重来源、数据摘要及执行状态见
[2026-11-11 记录](../../docs/experience/records/qwen3_bf16_gpqa_20261111.md)。
该轮已完成，最终复核汇总见
[final_summary.json](../../results/qwen3_bf16_gpqa/official_tp8_rounds3_20261111T105127Z/final_summary.json)。

在已激活 `hif4`、已准备对应 manifest 的工作区根目录执行：

```bash
python -u -m tools.qwen3_bf16_gpqa.run \
  --run_dir results/qwen3_bf16_gpqa/official_tp8_rounds3_20261111T105127Z \
  --phase all_rounds
```

已存在的阶段不能重复启动或覆盖。三轮准备使用 `prepare --sequential_rounds 3`；
`run --phase qualification` 依次执行129题和1题的单采样64-token检查，覆盖换轮及seed切换。
完整答案评分复用已核验的真实回答和相同评分器版本；正式阶段在检查成功后启动。
独立 `round_main.py` 显式传入各轮vLLM seed，避免修改其他活动实验依赖的共享 `main.py`。
各轮结束后验证原始缓存和逐题分数，等待本轮worker退出，再启动下一轮。
每轮使用独立模型视图隔离预测缓存，并立即保存本轮结果及目前已完成轮次的平均值。
自动汇总保留当时的待复核标记；完成答案复核后另存 `final_summary.json`，
只有全部三轮完成且无待复核答案时，其 `accuracy_is_final` 才为真。

已停止的每题10次方案及其源码快照留在原RUN目录，不混入新三轮统计。

评分复用项目已复核的最终提交规则；逐回答分数、严格 JSON 诊断分数和原始缓存均保留。
有歧义的最终提交写入不含金标的 `blind_review_queue.json`，此时自动准确率为待复核值。
短测准确率不作为 benchmark 结果。
