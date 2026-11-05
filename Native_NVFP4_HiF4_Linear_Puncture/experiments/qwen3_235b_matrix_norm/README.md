# Qwen3-235B matrix_norm 转换与四任务对照

本入口从固定的原生 `nvidia/Qwen3-235B-A22B-NVFP4` revision
`21cfa2c9e152032eb60647ee7b46a2bbcd8d76d2` 初始化全部 94 层，独立于旧的
E4 → Group-top-mass 迁移。方案已获用户确认，实际运行状态以新 RUN 的日志和产物为准。

## 固定方法与数据

- 激活侧固定 R64；权重侧训练 G64 自由矩阵及每层 input/post-attention norm。
  router 权重和原生基础权重冻结；norm 和输入变化仍可间接改变路由。
- 每层 20 epochs、全局 batch=4、matrix/norm LR=`3e-4`、AdamW、weight decay=0、
  cosine、seed=42；仅用验证 block-delta NMSE 选 epoch，router KL 权重为 0。
- 四卡按样本并行，每卡保留原全局 padding。每个局部分子除以全局目标能量分母，
  梯度求和后每个原 batch 更新一次；全局未使用参数保持 `grad=None`。
  使用 `save_on_cpu(pin_memory=False)`，不缩短序列或降低全局 batch。
- 固定版本 MATH/CodeContests train 题目与来源 test/valid、四项正式测试先合并分簇，
  排除题目来源 ID 重合、规范化或数字掩码 5-gram Jaccard≥0.8 的题簇，再划分候选池。
  核验已有题目投影后复用；已有 30B 回答不进入校准。
- 235B TP8 重新生成训练 128/验证 32 条，数学与代码各半；按固定池顺序补足正常
  结束且有 final 的完整轨迹，不按答案正确性筛选。保存完整 prompt/输出 token IDs。
  temperature=0.6、top_p=0.95、top_k=20、min_p=0、seed=1234、最大输出 38,912、
  上下文 40,960；prompt 超过 2,048 tokens 的题簇在生成前排除。

## 完整前后对照

所有任务通过真正的 `main.py` / lighteval / vLLM TP8 执行，BF16 KV、eager、
max_num_seqs=128、max_num_batched_tokens=4096、gpu_memory_utilization=0.9。
数据绑定到本 RUN 中已核验摘要的 Arrow 文件，父进程和生成子进程使用同一入口。

| 任务 | 题数 | 回答数/题 | 最大输出 tokens | 指标 |
| --- | ---: | ---: | ---: | --- |
| GSM8K | 1,319 | 1 | 32,768 | extractive match |
| MMLU-Pro | 12,032 | 1 | 32,768 | extractive match |
| LCB v6 | 175 | 1 | 38,912 | 官方 checker pass@1 |
| AIME25 | 30 | 5 | 38,912 | Avg@5 |

评测采样与生成相同。双方各一轮，不选最好轮次。原生 packed NVFP4 使用 emulation；
HiF4 导出保存 BF16 QDQ 权重，配合运行时 W4A4 激活量化，不声称打包压缩或加速。
每个任务/模型/阶段使用独立模型视图及预测缓存，校验实际题目、prompt、输入 token IDs、
gold、回答数量及逐题均值。LCB 指标名称里的 `:16` 不代表本协议每题生成 16 次。

AIME 原评分器指标保留，另保存最终提交答案复核。冲突项写入
`corrected_responses.json`，退出码 21 表示等待复核，不重跑生成或覆盖原分数；
其他独立阶段继续。复核全文后，在相同目录的 `reviews.json` 以 `题目ID:回答序号`
为键填写原始文本的 `raw_sha256`、二元 `score` 和 `reason`，执行 `--rescore-aime`
或重新生成报告。正式回答的全部冲突解决前不将复核版比较标为完成。

## 执行与产物

由调用者激活 `hif4` 后，在仓库根目录运行。输出目录必须独立；下列 `NEW_RUN`
只是路径示例。入口检查实时 GPU 占用，资源不足则记录 `blocked_resources` 并退出，
不等待、不占卡、不停止他人进程、不改变 TP8 协议。

```bash
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_235b_matrix_norm.run \
  --run Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_matrix_norm/NEW_RUN \
  --gpus 0,1,2,3,4,5,6,7 --stage prepare

nohup python -u -m Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_235b_matrix_norm.run \
  --run Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_matrix_norm/NEW_RUN \
  --gpus 0,1,2,3,4,5,6,7 \
  > Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_matrix_norm/NEW_RUN/runner.log 2>&1 < /dev/null &
```

顺序为：原生生成短测 → 原生四任务短测 → 正式轨迹 → 完整原生基线 → 最长 batch
两层训练短测 → 正式 94 层训练 → 完整导出 → HiF4 四任务短测 → HiF4 四项全量评测。
生成短测保留正式引擎条件，20 个请求覆盖 16 在途槽位的补入，仅输出预算缩到 64；
训练短测采用实际最长完整 train/val batch、两次更新和两层衔接。逐样本串行累积
参考与并行使用相同 padding；损失/梯度相对误差上限 `1e-4`，参数相对 L2 上限
`1e-5`，用于约束 FP32 求和差异；冻结状态、保存重载及各副本参数/梯度/Adam/调度器
同步使用严格比较。未覆盖的 GPU 路径不能由 CPU 测试代替。

`--stage all` 可复用该 RUN 已完整通过且摘要未变的阶段；失败/部分完成的阶段不会
自动重试或覆盖。它不提供训练中断恢复，也不复用短测更新初始化正式训练。
GPU 资源不足时，重新执行同一入口即可从尚未启动的阶段继续。

仅重算已有 AIME 回答：

```bash
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_235b_matrix_norm.evaluate \
  --run Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_matrix_norm/NEW_RUN \
  --candidate matrix_norm --task aime25 --rescore-aime

python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_235b_matrix_norm.run \
  --run Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_matrix_norm/NEW_RUN --stage report
```

保留 `manifest.json`、`inputs.json`、`source_hashes.json`、`status.json`、`logs/`、
固定来源/数据/排重证据、原始轨迹、校准样本、逐层 selected 参数、正式模型、
逐题输出和原始预测缓存。最终 `comparison.json` / `REPORT.md` 列前后分数、百分点
变化及四项等权平均正向下降；没有沿用旧实验的达标阈值，也没有自动清理权重。

经验来源：[235B 执行与暂存边界](../../../docs/experience/lessons/model_transfer_execution.md)、
[固定数据版本、缓存及 AIME 评分边界](../../../docs/experience/lessons/evaluation_evidence.md)。
它们不代替本次新路径的实际运行验证。文档导航见 [docs/README.md](../../../docs/README.md)。

本次实现及实测状态见 [2026-10-26 记录](../../../docs/experience/records/qwen3_235b_matrix_norm_20261026.md)。
