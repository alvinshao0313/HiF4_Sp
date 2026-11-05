# 校准轨迹与 Router 对齐实验

实验定义见 [PLAN.md](PLAN.md)。本目录复用现有重构、量化、router loss 和评测组件，不修改旧实验。用户已批准清洗补足 S1K 并重训 A；原样本及既有训练产物保留，新划分、验证和正式实验进入独立 RUN。

## 文件职责

| 入口 | 职责 |
| --- | --- |
| config.py / artifacts.py | 固定协议、RUN manifest、源码与输入摘要 |
| sources.py / isolation.py / clean_s1k.py / prepare.py | 固定版本数据投影、正式测试与官方 test/valid 排重、S1K 授权清洗补足、先分簇再划分 |
| generate.py / judging.py | 原生 packed NVFP4 TP2 完整轨迹；生成完成后独立记录正确性，不按正确性筛选 |
| training.py | 两组相同的 matrix_norm NMSE 训练；A 使用清洗数据从原生初始化重训 |
| development.py | 共同开发集单次生成、数学/代码等权选优并锁定起点 |
| router.py | 实际 gate 的 FP32 主权重、唯一 top-mass KL、LR 短筛与 epoch 0 回退 |
| export.py | BF16 QDQ 导出；C 仅另存 router，冻结权重使用硬链接并校验摘要 |
| pinned.py / reasoning_main.py / evaluation.py | 实际固定 Arrow、main.py/lighteval/vLLM TP2、ARC lm_eval、AIME 最终答案复核 |
| report.py | 四项等权平均正向损失、B−A/C−起点、逐题配对区间 |
| tests.py / qualify.py / performance.py / run.py | CPU 回归、最长 batch/真实 GPU 验证、持续补入/CUDA Graph 短测、A/B 并行执行及命令/PID/退出记录 |

## 执行

由调用者激活 `hif4`，在仓库根目录执行。初始化使用新的输出目录，拒绝覆盖。以下路径和 GPU 是命令示例，运行前按实时资源指定。

```bash
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.reasoning_trace_calibration.prepare \
  --run Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_calibration/NEW_RUN --initialize

nohup python -u -m Native_NVFP4_HiF4_Linear_Puncture.experiments.reasoning_trace_calibration.run \
  --run Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_calibration/NEW_RUN \
  --stage experiment --train-gpu 3 --inference-gpus 0,1 \
  > Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_calibration/NEW_RUN/runner.log 2>&1 < /dev/null &
```

先完成 `prepare`，`--stage experiment` 顺序执行 `generate → qualify → train → development → router → evaluate`；各阶段也可单独调用，已有正式产物拒绝覆盖，不自动重试。`--stage all` 适用于已初始化、尚未完整准备的 RUN，从完整审计开始执行。`--s1k-only` 仅用于原划分只读审计；新实验使用完整准备生成清洗后的 A。审计不通过时不会启动生成或正式训练；独立的短测不产生正式候选结果。

`generate` 会从预先固定、互不重叠的候选池依次补足正常结束且有 final 的 128/32，输出原始 token IDs。现使用 vLLM `enqueue/step` 持续补入已完成请求的空位，最多 16 个在途请求；完成结果按原候选顺序交付，在途及已完成结果共同预留剩余配额，不额外生成或中止题目。为了保留 38,912 的完整输出预算，生成前整簇排除超过 2,048 prompt tokens 的题，并记录剔除清单。开发集固定为 64 math+64 code，本来就一次提交全部题目，由 vLLM 连续调度；截断/未完成生成保留为失败，不按结果换题。

已完成生成、但在后续阶段失败的运行，用独立新 RUN 复用固定输入和原始轨迹，并以 `--stage continue` 从准备评分参考、正确性记录和必要验证继续。原失败运行与其源码摘要保留；不重复生成已完成轨迹。`sources.prepare_references` 只缓存所选 code 校准和共同开发集所需的固定版本测试列及摘要，之后评分不再依赖下载。为 `run` 指定 `--second-train-gpu` 后，A/B 各在一张 GPU 并行训练；两组完成后才导出、开发集选优和 router 对齐，各候选内部仍逐层传播。

`performance --mode eager|graph` 仅做原生 packed NVFP4 的真实 TP2 运行短测，覆盖 16 请求/64 步解码和 20 请求不同长度的持续补入。Graph 短测限定 decode 捕获批量 1/2/4/8/16；正式引擎继续使用固定的 `ENGINE`，不能将诊断通过当作正式 batch=128 或 HiF4 自定义路径的 Graph 证据。

每个 RUN 保存 `manifest.json`、`source_hashes.json`、`commands.jsonl`、`logs/`，以及 `isolation/`、`data_sources/`、`generation/`、`calibration/`、`judging/`、`qualification/`、`training/`、`models/`、`development/`、`router/`、`evaluation/`。正式训练前要求两组最长实际 batch 的验证、router 冻结/缓存契约、导出及真实 TP2 推理验证通过。适用的已完成短测须匹配源码和数据摘要才复用。推理短测仅将 decode 缩至 64 步，保持 TP2、40960 上下文、BF16 KV 与正式批容量；短测输出不进入校准或准确率结果。约 30B 参数的 BF16 QDQ 导出不是打包 4-bit 存储；普通 A/B 各约 60 GB，C 复用冻结分片。两项诊断导出真实 TP2 验证完成后，仅清理本 RUN 一次性诊断分片，保留配置、摘要、checkpoint 和日志以释放约 60 GB；不会删除既有或正式权重。

AIME 原始 metrics 不覆盖。最终提交答案保存到 `corrected_responses.json`，含糊/冲突项使该任务以 21 退出；其余候选/任务继续执行，全部结束后 RUN 标为 `awaiting_aime_review` 并保存 `evaluation/pending_reviews.json`，不汇总未复核分数。检查全文后在同目录 `reviews.json` 按 `"题目ID:回答序号"` 写入 `raw_sha256`、二元 `score` 和 `reason`，再用 `evaluation --rescore-aime` 重算。该入口只重算已有回答。`report` 要求完整三候选结果与已复核 avg5，主指标使用约定的旧原生基线。

首次只读审计和验证见 [原 RUN 报告](../../results/reasoning_trace_calibration/run_20261016T132000Z/REPORT.md)，清洗后的首次生成见 [清洗 RUN 报告](../../results/reasoning_trace_calibration/clean_s1k_20261016T030238Z/REPORT.md)，生成完成后的续跑见 [当前 RUN 报告](../../results/reasoning_trace_calibration/continue_20261016T135141Z/REPORT.md)。局部验证不代表两种数据的优劣或追加 router 的收益。
