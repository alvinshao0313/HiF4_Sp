# 2026-11-04：matrix_norm短程转换五任务结果

研究模型为Qwen3-30B-A3B，沿用原生NVFP4快照、HiF4 W4A4语义与matrix_norm结构。
正式报告对应独立训练5轮、分层随机约50%训练token的开发集胜出候选random50_e5。
full_e5、random50_e5、coverage50_e5均完成48层独立训练和开发评测；三者开发等权分数
相同，按优化时间选择random50_e5。其他两个候选未正式完成五任务，不能推断其最终精度。

唯一正式运行目录为
[run_20261029T023839Z](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_norm_efficiency/run_20261029T023839Z/)。
方案与边界见[PLAN.md](../../../Native_NVFP4_HiF4_Linear_Puncture/experiments/matrix_norm_efficiency/PLAN.md)。
本次GPU0/1接续的全部生成于北京时间2026-11-03 21:20完成；11月4日按既定最终提交
评分完成余下5条AIME文本复核和CPU汇总。没有重跑推理、重训、修改评分定义或安装依赖。

正式结论：五项等权平均正向精度损失2.3047个百分点，未达到≤0.4目标；按题配对bootstrap
95%区间[0.8675, 4.4454]，不包含独立训练/生成seed差异。主要退化来自LCB。GSM8K的
提升按既定目标不能抵消其他任务下降。完整分数、覆盖、逐题证据与评分原分/复核分见
[REPORT.md](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_norm_efficiency/run_20261029T023839Z/REPORT.md)和
[正式配对JSON](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_norm_efficiency/run_20261029T023839Z/reports/random50_e5.json)。

同口径优化阶段13,740.92秒（3.82小时），历史20轮参照72,146.31秒（20.04小时），
在同型号单张A800上约5.25倍速度。超过最低2倍速度，未达到理想≤3小时；且精度不达标，
整体配方不通过目标。该比较同时改变数据与训练轮数，不是单因素消融。三候选搜索、
全部短测、导出、开发和正式评测成本在
[costs.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_norm_efficiency/run_20261029T023839Z/costs.json)单列；
已记录新进程GPU-hours合计73.5132，不含旧共享轨迹与历史原生基线生成，不代表纯优化耗时。

正式协议为TP2、eager、BF16 KV、batch128、batched tokens4096、context40960、
seed1234、temperature0.6/top_p0.95/top_k20/min_p0；固定实际数据版本与完整输入。
MMLU-Pro12032题、LCB175题、GSM8K1319题、AIME24/25各30题×5次。两年AIME主表
统一使用最终提交avg5，旧正式提取器分数与原始回答保留。原生LCB/GSM8K/AIME25经权重、
全部prompt/token/gold和实际回答数核验后复用；原生MMLU-Pro/AIME24为本轮生成。

5条余下复核记录分别位于原生AIME24、候选AIME24与候选AIME25的reviews.json，均绑定
原始回答SHA并明确最终提交理由；未挑取思考中碰巧正确的数字。原生AIME24一条思考为721、
最终回答601，仍计0。复核前awaiting_aime_review的控制器状态已保存独立快照，原始
exit_code=21的阶段记录保留，最终状态为complete，见
[state.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_norm_efficiency/run_20261029T023839Z/continuation_20261103_gpu01/state.json)。

本轮新增证据是“开发集同分和实测转换加速不能代替正式精度验收”，已补充
[评测经验](../lessons/evaluation_evidence.md)。机制原因尚未消融，不归因于单独数据筛选或
训练轮数；没有启动后续算法实验。原始权重、生成、指标与日志均保留，无删除或Git操作。
