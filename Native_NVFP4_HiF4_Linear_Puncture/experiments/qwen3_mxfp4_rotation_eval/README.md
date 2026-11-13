# H32 与 R16 扩展 R32：四任务下游对照

比较现有 `fht32` 与 `r16x2` 两个 Qwen3-30B-A3B checkpoint。只新增评测入口，
不重新转换权重、不修改两种旋转或 MXFP4 量化器。两组沿用 group32、E2M1、
E8M0 和 `floor(log2(amax))-2`；A800 执行软件 QDQ／BF16 GEMM。

用户已确认 reasoning 每题 10 个回答取平均、QA 重复 5 次；MMLU-Pro 使用全部
12,032题。H32 使用GPU0–3，R16扩展R32使用GPU4–7，各自TP4并行。
CPU准备和两组共六项TP4入口短测均已通过。北京时间2026-11-13 10:03，按用户要求
停止控制器3350681及两组尚未完成的MMLU-Pro，共15个相关进程；GPU0–7已释放。
两组LCB、ARC-E/C已完成，全部旧产物保留；MMLU-Pro没有完整评分，不报告准确率。
见[停止前结果汇总](../../results/qwen3_mxfp4/rotation_downstream_20261111T021111Z/stopped_results_20261113.md)
及运行目录的 `formal_status.json`。这些结果使用旧prompt，不能标为AISBench结果。

| 任务 | 数据范围 | 重复与指标 |
| --- | --- | --- |
| LCB | 原协议 `v6` 的 175 题 | 每题 10 个回答，共 1,750 个；单回答代码通过率均值，即 pass@1 的均值 |
| MMLU-Pro | 全部 12,032 题 | 每题 10 个回答，共120,320个；原 `extractive_match` 单回答评分的均值 |
| ARC-E | 全部 2,376 题 | 5 次实际选项似然评测；报告 `acc`、附 `acc_norm` |
| ARC-C | 全部 1,172 题 | 5 次实际选项似然评测；报告 `acc`、附 `acc_norm` |

ARC 使用 lm_eval 的零样本 multiple-choice 路径，没有随机生成；五轮重复推理，
不复制第一轮结果。不把十次均值称为 pass@10、投票准确率或十个不同种子的结果。

## 固定协议和证据

- reasoning 通过隔离 `entry.py` 调用根目录 `main.py`／lighteval，父进程和 spawn
  子进程注册原量化方法。沿用原 prompt、chat template、评分器以及 thinking。
- seed1234、temperature0.6、top_p0.95、top_k20、min_p0；TP4、batch256、
  8192 batched tokens、40960 上下文、每卡32 GiB BF16 KV、eager。
- MMLU-Pro 最多生成32768，LCB 沿用既有协议38912；保存实际输入截断摘要，
  两组必须一致。没有把 GPQA 的32768静默套到 LCB。
- MMLU-Pro 固定 `3373e0b32277875b8db2aa555a333b78a08477ea` 的实际 Arrow；
  LCB 和 ARC 同样固定本地数据摘要，避免缓存回退导致版本漂移。
- 每个模型、任务和短测／正式阶段使用独立模型视图与预测缓存。保留原始回答、
  每题每次得分、实际请求 token、结束原因、命令、退出状态与代码摘要。
- MMLU-Pro 单回答切片显式保留 `text_post_processed`，保证与原单回答评分器一致；
  LCB 复用原代码提取与测试执行，保存十个实际通过／失败结果。
- 两个候选的既有算子、48层 TP4 切片验证按源码和 checkpoint 字节摘要复用。
  新入口的实际 worker 方法和 packed 字节必须匹配既有已验证加载记录。

参考经验：[评测证据](../../../docs/experience/lessons/evaluation_evidence.md)中的
输入不一致、预测缓存位置、离线版本回退、指标名称与生成数量，以及评分输入边界。
这些经验用于固定比较条件，不将已有 GPQA 排名外推成四任务结论。

## 执行

在仓库根目录，由调用者激活 `hif4` 后执行。CPU 准备入口不使用 GPU：

```bash
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4_rotation_eval.prepare --output RUN
```

准备产物：
[rotation_downstream_20261111T021111Z](../../results/qwen3_mxfp4/rotation_downstream_20261111T021111Z)。
准备日志在该目录旁的同名 `.prepare.log`。`manifest.json` 只在全部准备检查成功后发布。

资源和题量已确认，分别执行 `workflow qualify` 与 `workflow formal`：两者均要求
显式传入 `--root RUN --h32-gpus GPU_LIST --r16x2-gpus GPU_LIST --mmlu-samples N`。
两组 GPU 列表各含四卡；不相交则并行，相同则顺序。`N` 仅允许300或12032。

资格检查保留正式 TP4／引擎资源配置：每候选 LCB 和 MMLU-Pro 各26题×10、最多64个
新 token，ARC-E/C 各65题、实际重复两轮。短测结果不用于准确率比较。
两候选全部检查通过后，正式任务每组依次 ARC五轮 → LCB×10 → MMLU-Pro×10。
失败停止后续阶段，不自动重试、改变协议或终止另一组已经运行的进程。

正式后台任务确认实际推进后交付运行标识并停止主动监控。完成产物为
`comparison.json`、`comparison.csv`，另保留两组各任务原始结果及逐次得分。

本次授权与启动记录分别为运行目录的 `authorization.json`、`formal_launch.json`；
短测汇总为 `qualification_summary.json`。每个阶段的实际命令、PID、日志路径记录于
`qualify_status.json`／`formal_status.json`。控制器日志为 `formal.log`，各模型原始日志为
`fht32/formal/{arc,lcb,mmlu_pro}/main.log` 和 `r16x2/formal/{arc,lcb,mmlu_pro}/main.log`。

本次复用已通过的量化算子证据，新增验证只针对四任务入口与重复评分。已完成部分中，
R16扩展R32相对H32的LCB、ARC-E、ARC-C均值分别高0.7429、0.0253、0.7509个百分点；
这些观测不证明统计显著性，也不补足未完成的MMLU-Pro。未发布完整四任务comparison。
后续用户将重测范围收窄为八个候选的GPQA，仅替换AISBench prompt；本队列不再接续。
