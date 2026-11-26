# 四组组合：LCB／MMLU-Pro 五次平均

复用已完成 GPQA 的四个组合 checkpoint，不改权重或量化算子。2026-11-14 用户要求
LCB、MMLU-Pro 每题五次平均，并明确本轮 GPU0–7 均可使用。这里 R32 指能量均衡
R16 扩展 R32，UOS 指 c7.25 的 ceiling 内层 scale；G512 同时作用于专家权重和激活。

| GPU / TP2 | 候选 |
| --- | --- |
| 0–1 | R32＋UOS |
| 2–3 | UOS＋G512 |
| 4–5 | R32＋G512 |
| 6–7 | R32＋UOS＋G512 |

每组依次 LCB v6（175题×5，875个回答）→ MMLU-Pro（全量12,032题×5，60,160个回答）。
四组共244,140个回答。主指标为每个回答的准确率均值，保存五个采样序号各自的结果；
LCB为单份程序通过全部测试的比例，不是pass@5；MMLU-Pro沿用已有lighteval
`extractive_match`单回答评分，不改变为投票或按学科等权平均。

## Prompt 与固定条件

沿用 GPQA 已固定的 AISBench commit `aff1554354fbf19dccbdf2199cc3c22119f815a2`：

- [LCB v6 零样本配置](https://github.com/AISBench/benchmark/blob/aff1554354fbf19dccbdf2199cc3c22119f815a2/ais_bench/benchmark/configs/datasets/livecodebench/livecodebench_0_shot_chat_v6.py)
- [MMLU-Pro 零样本配置](https://github.com/AISBench/benchmark/blob/aff1554354fbf19dccbdf2199cc3c22119f815a2/ais_bench/benchmark/configs/datasets/mmlu_pro/mmlu_pro_gen_0_shot_str.py)

只移植 prompt 内容及选项/代码格式，执行框架仍为根目录 `main.py`／lighteval＋vLLM，
数据与评分器保留项目协议。LCB配置中的system常量没有接入实际模板，因此不额外添加。
MMLU-Pro保留选项字母，按上游格式跳过内容为`N/A`的项。CPU准备对全部题目核对上游
模板展开结果和实际chat输入；冻结题目、gold、完整prompt及实际截断后token摘要。

TP2、每卡56 GiB BF16 KV、thinking开启、seed1234、temperature0.6、top_p0.95、
top_k20、min_p0、上下文40960、batch256、8192 batched tokens、eager。
LCB最大生成38912，MMLU-Pro32768，沿用既有入口的左截断规则并记录实际受影响题数。
A800执行软件QDQ与BF16 GEMM，不是原生MXFP4硬件推理。

## 验证与执行

沿用四组算子及TP2验证，重新核对checkpoint摘要和实际推理源码。
必要新增检查覆盖五次实际生成、单回答评分输入、真实LCB代码执行及新prompt。
每候选、每任务各做52题×5、最多64-token执行检查，保留正式TP2/batch256/KV配置；
核对两个rank、全部48层、非专家方法、切片字节、实际输入token和全部输出。
短测分数不报告为准确率。八项通过后，四路正式队列启动；失败停止受影响队列，不自动重试。

已激活 `hif4` 的 shell 从仓库根目录执行，`RUN` 必须是新输出目录：

```bash
PYTHONPATH=3rdparty/lighteval/src:. python -m fp4_convert.experiments.qwen3_mxfp4_combination_eval.prepare --output RUN
python -m fp4_convert.experiments.qwen3_mxfp4_combination_eval.workflow qualify --root RUN
python -m fp4_convert.experiments.qwen3_mxfp4_combination_eval.workflow formal --root RUN
```

本轮目录：[combination_downstream_20261114T053205Z_v2](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/combination_downstream_20261114T053205Z_v2)。
第一次CPU准备检测到MMLU选项末尾空白与上游不同，已修正并对全部12,032题重新核对；
旧准备目录与失败日志保留，未在该目录启动GPU推理。
以 `qualify_exit.json`、`formal_lane0..3.json`、各阶段 `status.json` 和原始日志为实时证据；
最终 `comparison.json`／`comparison.csv` 只在全部正式结果完整并通过输入/评分核验后发布。
每阶段保存独立模型视图和缓存，既有GPQA产物保持原位。

参考经验：[评测协议与实际执行](../../../docs/experience/lessons/evaluation_evidence.md)中的
prompt差异、离线数据版本回退、缓存路径、实际采样数及`text_post_processed`边界。
没有新增量化算法，也不从GPQA排名预测这两个任务的结果。
