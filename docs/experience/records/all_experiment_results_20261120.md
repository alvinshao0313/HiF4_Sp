# 全仓库实验结果总账（2026-11-20）

核验快照：**2026-11-20T02:32:51+00:00**。范围为本仓库当前保留的全部实验系列，包括正式下游评测、转换/重构训练、CPU/GPU短测、机制诊断，以及失败、中止、被替代和只剩历史报告的运行。**不是仅整理本轮补充实验，也不把历史结果全部改称五次均值。** 没有重新训练或生成回答，没有改动原始实验产物。

**后续状态更新（北京时间2026-11-20 15:49）：** 用户要求停止GPU0/1的GPTQ和GPU4/5的两版matrix_norm旧协议补测，所属进程已退出。GPTQ保留3个完整轮，A20/B20各保留4个完整轮；下表和CSV仍保留上述10:32的指标快照，不将部分轮补成五轮均值。新版论文统一协议另有seed、ARC指标、生成长度和执行设置要求，本总账中的“可用”均以各表自身协议及限制为准，不能直接混入新九任务主表。详见[停止记录与论文使用边界](thesis_format_conversion_20261116.md)。

## 阅读与统计口径

- 下文按**测试数据集 → 优化方案 → 指标**逐条展开；同模型、同输入/评分和运行口径的完整结果尽量合表。每表都写固定设置、方法变化项、分表原因和论文使用边界。表号T05已并入T04，保留编号以避免混淆来源。
- “单次结果”表示1轮、每题1次；“每题5/10次均值；1轮”表示一次运行内多采样；“5轮均值；每题1次”表示5次完整运行。**这些都不是5个训练种子。** BF16 GPQA是3个显式seed；其他固定seed重复不代表独立seed不确定性。详见每表。
- 指标默认百分数；ARC使用raw acc，LCB使用pass@1或其多采样平均，AIME/GPQA平均不是pass@k；PPL、NMSE、KL、NLL和耗时单列单位。题数列不含重复倍数：例如198题×10采样=1980回答。
- 只对完整结果给正式均值；本轮尚未凑齐5次的完整单轮放T35，**缺题的MMLU-Pro不列数值**。复用的同一回答、同一轮不同解析器重评分不当作新重复；历史重复次数无法核实的明确写“未核实”。逐层/逐投影均值、bootstrap区间、CUDA计时中位数也不冒充重复均值。
- **同名测试集不保证可比**：MMLU-Pro300与全量12032、旧JSON与AISBench题面/选项顺序/最终提交解析、TP2/TP4/TP8、KV32/56/20GiB均保留区别。不跨这些表选择最好的native基线来算提升，也不把旧S1K来源污染结果充当干净A20主结论。
- HiF4主线是逻辑W4A4、G64，导出BF16 QDQ权重并在运行时量化激活；MXFP4专家E2M1/G32+E8M0为**专家元素理论4.25bit**，不等于整模型存储。原生NVFP4 emulation同样是QDQ加浮点GEMM/累加；本文无原生FP4硬件吞吐或实际压缩率主张。
- 完整原始命令、模型/数据版本、checkpoint/config/代码哈希保留在每表来源manifest/protocol和各运行目录；本表没有用当前工作区HEAD冒充历史代码版本。737条指标是记录行数，含复用、诊断、被替代值，**不是独立实验数量**。

机器可读附件：[逐项指标 CSV](../../../results/experiment_catalog_20261120/scores.csv) · [协议及分表理由](../../../results/experiment_catalog_20261120/protocol_groups.json) · [成本 CSV](../../../results/experiment_catalog_20261120/costs.csv) · [全部运行目录清单](../../../results/experiment_catalog_20261120/run_inventory.csv) · [原始证据文件索引](../../../results/experiment_catalog_20261120/evidence_inventory.csv) · [本次采用证据 SHA256](../../../results/experiment_catalog_20261120/source_manifest.json) · [历史/机制原表逐行保留](../../../results/experiment_catalog_20261120/historical_diagnostic_tables.csv)。汇总生成入口：[build_catalog.py](../../../results/experiment_catalog_20261120/build_catalog.py)（在hif4执行；重读现有结果，不运行模型）。

## 可用于论文的表如何选择

当前无AIME补充实验的最终**五轮**主表是T01；GPTQ与两个matrix_norm版本还在T35/待完成清单，不能提前写成五次平均。同一AISBench下游协议的native/direct及四种MXFP4组合已合并在T13，每题5次；其GPQA见T10/T11（TP不同故分表）。旧S1K模块消融T04、旧4B/8B和被隔离记录均保留，但应遵守其污染/证据边界。

本次无新增算法经验；统计与比较遵循[评测证据经验](../lessons/evaluation_evidence.md)和[量化执行边界](../lessons/fake_quant_boundaries.md)。具体科学结论仍以各实验的完整对照为准。

## 表目录

| 表号 | 内容 | 记录行数 |
| --- | --- | --- |
| T01 | 无 AIME 补充实验：五轮完整重测 | 5 |
| T02 | 干净校准主方法：S1K / 自生成轨迹及 Router 追加 | 18 |
| T03 | B20 起点的后训练：Router / Top-mass / 完整 KL / 硬 CE | 22 |
| T04 | 旧 S1K 模块消融：五项非 AIME 完整结果 | 85 |
| T06 | 短程效率：random50_e5 的全量测试 | 10 |
| T07 | 不等价变换：共享粒度与 Router 目标 | 15 |
| T08 | 残差 LoRA：Attention / MoE / Both | 20 |
| T09 | MXFP4 / 旋转 / UOS / 二级 scale：旧 JSON prompt GPQA | 8 |
| T10 | AISBench prompt GPQA：TP4 / KV 32GiB | 5 |
| T11 | AISBench prompt GPQA：TP2 / KV 56GiB | 6 |
| T12 | MXFP4 两种旋转：旧 prompt LCB 与 ARC | 6 |
| T13 | AISBench prompt 下游：原生、直转与四种组合共表 | 12 |
| T14 | 官方 BF16 基线：GPQA 三个独立 seed | 1 |
| T15 | 旧 matrix_norm 上的 Ban：预算约 6.07 专家 | 8 |
| T16 | Ban 6.4 专家预算：C4 vs 自生成重新校准 | 15 |
| T17 | MiniMax-M3.1-Flash 原生基线 | 5 |
| T18 | 早期 E0–E7：修复输入后的受控评测 | 13 |
| T19 | Qwen3.5-4B 阈值实验：历史报告中的端到端结果 | 35 |
| T20 | 4B reasoning 历史值：采样数与截断证据不足 | 6 |
| T21 | 旧 AIME25：自动提取分与重复轮（不用于最终答案比较） | 27 |
| T22 | AIME25 原正式轮：四组最终提交纠正 | 4 |
| T23 | GPQA 旧严格 JSON 提取器分：评分诊断 | 8 |
| T24 | 已隔离 Phase-A 旧分数：仅供溯源 | 29 |
| T25 | 独立开发集：A/B、效率与完整 KL/CE 的选择结果 | 35 |
| T26 | KL 方向复用：真实 TP2 最终分布指标 | 45 |
| T27 | 渐进式误差抵消：六种完整训练的留出指标 | 24 |
| T28 | H-only 解析更新：CPU 单 Linear | 6 |
| T29 | 不等价重构：完整48层的局部统计 | 16 |
| T30 | 残差 LoRA：各自输入上的配对局部统计 | 36 |
| T31 | 4B 阈值研究：合成分布与权重局部 NMSE | 18 |
| T32 | 4B 阈值研究：激活校准及跨域迁移 | 14 |
| T33 | 8B QAT 早期穿刺：DIAG / H4 / R64 | 9 |
| T34 | 论文新基线实现：CPU 真实投影短测 | 6 |
| T35 | 五轮尚未齐全：已完成的逐轮结果 | 43 |
| T36 | MXFP4 二级 scale：MoE 局部性能 | 96 |
| T37 | MXFP4 单卡局部算子：R64 / H32 / R16x2 / UOS | 24 |
| T38 | Ban 自生成校准早期6专家预算：只完成两轮 | 2 |

## T01　无 AIME 补充实验：五轮完整重测

**使用范围：** 可作为当前五次均值结果；其余方法未满五轮前不计算跨方法 L5 或宣称领先。

**固定设置：** Qwen3-30B-A3B，原生 NVIDIA NVFP4 revision `2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`；reasoning：根 main.py 的 lighteval/vLLM、TP2、BF16 KV、eager、context=40960、batch/max_num_seqs=128、batched_tokens=4096、seed=1234、temperature=0.6、top_p=0.95、top_k=20、min_p=0、thinking；MMLU-Pro/GSM8K max_new_tokens=32768，LCB=38912；ARC：lm_eval、0-shot、raw acc（不是 acc_norm）、context=4096、seed=42。

**为何单独成表：** 与 T02–T08 的历史单轮结果分开；与 T13 的 AISBench prompt、每题多采样及固定 KV 容量也不同。

来源及完整配置：[scale_mean5.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/c4_20261118T024647Z/scale_mean5.json) · [protocol.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/c4_20261118T024647Z/protocol.json)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| ARC-C | R64＋HiF4 scale（full） | 51.2628 % | 1172 | 5轮均值；每题1次 | G64；S0 offset=-2,-1,0,1,2；每块8种e8/e4；固定在线R64；直接保存搜索QDQ值 | 5个独立完整运行、固定任务 seed；非5个训练种子；data-free |
| ARC-E | R64＋HiF4 scale（full） | 77.3990 % | 2376 | 5轮均值；每题1次 | G64；S0 offset=-2,-1,0,1,2；每块8种e8/e4；固定在线R64；直接保存搜索QDQ值 | 5个独立完整运行、固定任务 seed；非5个训练种子；data-free |
| GSM8K | R64＋HiF4 scale（full） | 95.5421 % | 1319 | 5轮均值；每题1次 | G64；S0 offset=-2,-1,0,1,2；每块8种e8/e4；固定在线R64；直接保存搜索QDQ值 | 5个独立完整运行、固定任务 seed；非5个训练种子；data-free |
| LiveCodeBench v6 | R64＋HiF4 scale（full） | 52.6857 % | 175 | 5轮均值；每题1次 | G64；S0 offset=-2,-1,0,1,2；每块8种e8/e4；固定在线R64；直接保存搜索QDQ值 | 5个独立完整运行、固定任务 seed；非5个训练种子；data-free |
| MMLU-Pro | R64＋HiF4 scale（full） | 75.0997 % | 12032 | 5轮均值；每题1次 | G64；S0 offset=-2,-1,0,1,2；每块8种e8/e4；固定在线R64；直接保存搜索QDQ值 | 5个独立完整运行、固定任务 seed；非5个训练种子；data-free |

## T02　干净校准主方法：S1K / 自生成轨迹及 Router 追加

**使用范围：** 可作历史单轮数据/方案比较，不替代正在补测的五轮均值；A/B不是只改优化算法。

**固定设置：** Qwen3-30B-A3B，原生 NVIDIA NVFP4 revision `2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`；reasoning：根 main.py 的 lighteval/vLLM、TP2、BF16 KV、eager、context=40960、batch/max_num_seqs=128、batched_tokens=4096、seed=1234、temperature=0.6、top_p=0.95、top_k=20、min_p=0、thinking；MMLU-Pro/GSM8K max_new_tokens=32768，LCB=38912；ARC：lm_eval、0-shot、raw acc（不是 acc_norm）、context=4096、seed=42。

**为何单独成表：** 评测与 T01 对齐，但除 AIME 每题5次外均只有历史单轮；校准来源、追加阶段为显式实验变量。与 T04 的旧 S1K 清洗边界不同。

来源及完整配置：[comparison.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_calibration/continue_20261016T135141Z/comparison.json) · [manifest.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_calibration/continue_20261016T135141Z/manifest.json)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| AIME25 | matrix_norm_native_traces | 70.0000 % | 30 | 每题5次均值；1轮 | B20：干净自生成128/32；其余同A20；独立原生轨迹 | AIME为最终提交复核分；每层validation选优；AdamW wd0/cosine（A/B） |
| AIME25 | matrix_norm_s1k | 66.6667 % | 30 | 每题5次均值；1轮 | A20：干净S1K128/32；20epoch；lr3e-4；batch4；G64；norm+自由矩阵；router冻结 | AIME为最终提交复核分；每层validation选优；AdamW wd0/cosine（A/B） |
| AIME25 | winner_router_aligned | 68.6667 % | 30 | 每题5次均值；1轮 | A20后追加router：5epoch；LR候选1e-5/3e-5/1e-4×3epoch；top8+tail KL，T=1；FP32 master | AIME为最终提交复核分；每层validation选优；AdamW wd0/cosine（A/B） |
| ARC-C | matrix_norm_native_traces | 49.9147 % | 1172 | 单次结果 | B20：干净自生成128/32；其余同A20；独立原生轨迹 | AIME为最终提交复核分；每层validation选优；AdamW wd0/cosine（A/B） |
| ARC-C | matrix_norm_s1k | 52.4744 % | 1172 | 单次结果 | A20：干净S1K128/32；20epoch；lr3e-4；batch4；G64；norm+自由矩阵；router冻结 | AIME为最终提交复核分；每层validation选优；AdamW wd0/cosine（A/B） |
| ARC-C | winner_router_aligned | 52.6451 % | 1172 | 单次结果 | A20后追加router：5epoch；LR候选1e-5/3e-5/1e-4×3epoch；top8+tail KL，T=1；FP32 master | AIME为最终提交复核分；每层validation选优；AdamW wd0/cosine（A/B） |
| ARC-E | matrix_norm_native_traces | 77.5673 % | 2376 | 单次结果 | B20：干净自生成128/32；其余同A20；独立原生轨迹 | AIME为最终提交复核分；每层validation选优；AdamW wd0/cosine（A/B） |
| ARC-E | matrix_norm_s1k | 79.1667 % | 2376 | 单次结果 | A20：干净S1K128/32；20epoch；lr3e-4；batch4；G64；norm+自由矩阵；router冻结 | AIME为最终提交复核分；每层validation选优；AdamW wd0/cosine（A/B） |
| ARC-E | winner_router_aligned | 78.9141 % | 2376 | 单次结果 | A20后追加router：5epoch；LR候选1e-5/3e-5/1e-4×3epoch；top8+tail KL，T=1；FP32 master | AIME为最终提交复核分；每层validation选优；AdamW wd0/cosine（A/B） |
| GSM8K | matrix_norm_native_traces | 95.2237 % | 1319 | 单次结果 | B20：干净自生成128/32；其余同A20；独立原生轨迹 | AIME为最终提交复核分；每层validation选优；AdamW wd0/cosine（A/B） |
| GSM8K | matrix_norm_s1k | 95.0720 % | 1319 | 单次结果 | A20：干净S1K128/32；20epoch；lr3e-4；batch4；G64；norm+自由矩阵；router冻结 | AIME为最终提交复核分；每层validation选优；AdamW wd0/cosine（A/B） |
| GSM8K | winner_router_aligned | 95.4511 % | 1319 | 单次结果 | A20后追加router：5epoch；LR候选1e-5/3e-5/1e-4×3epoch；top8+tail KL，T=1；FP32 master | AIME为最终提交复核分；每层validation选优；AdamW wd0/cosine（A/B） |
| LiveCodeBench v6 | matrix_norm_native_traces | 53.7143 % | 175 | 单次结果 | B20：干净自生成128/32；其余同A20；独立原生轨迹 | AIME为最终提交复核分；每层validation选优；AdamW wd0/cosine（A/B） |
| LiveCodeBench v6 | matrix_norm_s1k | 52.5714 % | 175 | 单次结果 | A20：干净S1K128/32；20epoch；lr3e-4；batch4；G64；norm+自由矩阵；router冻结 | AIME为最终提交复核分；每层validation选优；AdamW wd0/cosine（A/B） |
| LiveCodeBench v6 | winner_router_aligned | 52.0000 % | 175 | 单次结果 | A20后追加router：5epoch；LR候选1e-5/3e-5/1e-4×3epoch；top8+tail KL，T=1；FP32 master | AIME为最终提交复核分；每层validation选优；AdamW wd0/cosine（A/B） |
| MMLU-Pro | matrix_norm_native_traces | 74.4348 % | 12032 | 单次结果 | B20：干净自生成128/32；其余同A20；独立原生轨迹 | AIME为最终提交复核分；每层validation选优；AdamW wd0/cosine（A/B） |
| MMLU-Pro | matrix_norm_s1k | 75.0499 % | 12032 | 单次结果 | A20：干净S1K128/32；20epoch；lr3e-4；batch4；G64；norm+自由矩阵；router冻结 | AIME为最终提交复核分；每层validation选优；AdamW wd0/cosine（A/B） |
| MMLU-Pro | winner_router_aligned | 74.5928 % | 12032 | 单次结果 | A20后追加router：5epoch；LR候选1e-5/3e-5/1e-4×3epoch；top8+tail KL，T=1；FP32 master | AIME为最终提交复核分；每层validation选优；AdamW wd0/cosine（A/B） |

## T03　B20 起点的后训练：Router / Top-mass / 完整 KL / 硬 CE

**使用范围：** 五项非AIME已完成结果可用；完整KL/CE的AIME仍待复核，只在未完成清单登记，不填最终分。

**固定设置：** Qwen3-30B-A3B，原生 NVIDIA NVFP4 revision `2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`；reasoning：根 main.py 的 lighteval/vLLM、TP2、BF16 KV、eager、context=40960、batch/max_num_seqs=128、batched_tokens=4096、seed=1234、temperature=0.6、top_p=0.95、top_k=20、min_p=0、thinking；MMLU-Pro/GSM8K max_new_tokens=32768，LCB=38912；ARC：lm_eval、0-shot、raw acc（不是 acc_norm）、context=4096、seed=42。

**为何单独成表：** 与 T02 使用相同 B20 起点和测试口径，独立成表是因为训练范围、参数精度与选优策略不同；不同策略只能比较整体配方。

来源及完整配置：[comparison.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_router_bf16/run_20261019T031744Z/comparison.json) · [comparison.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_last_head/run_20261020T031200Z/comparison.json) · [PLAN.md](../../../Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_full_kl_ce/run_20261021T030031Z/PLAN.md) · [selection.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_full_kl_ce/run_20261021T030031Z/development/full_kl/selection.json) · [selection.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_full_kl_ce/run_20261021T030031Z/development/hard_ce/selection.json)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| AIME25 | B_last_head_vocab64 | 64.0000 % | 30 | 每题5次均值；1轮 | 仅末层+final norm+head matrix；FP32优化/BF16前向；20epoch；词表top64+tail KL；T1；head不量化 | LR筛选1e-5/3e-5/1e-4各3epoch；验证集选优，允许epoch0；AIME最终提交 |
| AIME25 | B_router_bf16 | 62.0000 % | 30 | 每题5次均值；1轮 | 48层router BF16参数/梯度/动量；5epoch；top8+tail KL；T1；constant LR；独立native prefix | LR筛选1e-5/3e-5/1e-4各3epoch；验证集选优，允许epoch0；AIME最终提交 |
| ARC-C | B_last_head_full_kl | 50.7679 % | 1172 | 单次结果 | B20同128/32；末层+norm+head；完整词表151936/T1或next-token CE（含prompt）；FP32优化 | 三LR×3epoch+重置后20epoch；独立64math+64code决选；两者最终均选lr3e-5/epoch3；AIME尚待审 |
| ARC-C | B_last_head_hard_ce | 51.1945 % | 1172 | 单次结果 | B20同128/32；末层+norm+head；完整词表151936/T1或next-token CE（含prompt）；FP32优化 | 三LR×3epoch+重置后20epoch；独立64math+64code决选；两者最终均选lr3e-5/epoch3；AIME尚待审 |
| ARC-C | B_last_head_vocab64 | 50.7679 % | 1172 | 单次结果 | 仅末层+final norm+head matrix；FP32优化/BF16前向；20epoch；词表top64+tail KL；T1；head不量化 | LR筛选1e-5/3e-5/1e-4各3epoch；验证集选优，允许epoch0；AIME最终提交 |
| ARC-C | B_router_bf16 | 50.5973 % | 1172 | 单次结果 | 48层router BF16参数/梯度/动量；5epoch；top8+tail KL；T1；constant LR；独立native prefix | LR筛选1e-5/3e-5/1e-4各3epoch；验证集选优，允许epoch0；AIME最终提交 |
| ARC-E | B_last_head_full_kl | 78.0303 % | 2376 | 单次结果 | B20同128/32；末层+norm+head；完整词表151936/T1或next-token CE（含prompt）；FP32优化 | 三LR×3epoch+重置后20epoch；独立64math+64code决选；两者最终均选lr3e-5/epoch3；AIME尚待审 |
| ARC-E | B_last_head_hard_ce | 77.8199 % | 2376 | 单次结果 | B20同128/32；末层+norm+head；完整词表151936/T1或next-token CE（含prompt）；FP32优化 | 三LR×3epoch+重置后20epoch；独立64math+64code决选；两者最终均选lr3e-5/epoch3；AIME尚待审 |
| ARC-E | B_last_head_vocab64 | 77.8199 % | 2376 | 单次结果 | 仅末层+final norm+head matrix；FP32优化/BF16前向；20epoch；词表top64+tail KL；T1；head不量化 | LR筛选1e-5/3e-5/1e-4各3epoch；验证集选优，允许epoch0；AIME最终提交 |
| ARC-E | B_router_bf16 | 78.0724 % | 2376 | 单次结果 | 48层router BF16参数/梯度/动量；5epoch；top8+tail KL；T1；constant LR；独立native prefix | LR筛选1e-5/3e-5/1e-4各3epoch；验证集选优，允许epoch0；AIME最终提交 |
| GSM8K | B_last_head_full_kl | 95.2995 % | 1319 | 单次结果 | B20同128/32；末层+norm+head；完整词表151936/T1或next-token CE（含prompt）；FP32优化 | 三LR×3epoch+重置后20epoch；独立64math+64code决选；两者最终均选lr3e-5/epoch3；AIME尚待审 |
| GSM8K | B_last_head_hard_ce | 96.0576 % | 1319 | 单次结果 | B20同128/32；末层+norm+head；完整词表151936/T1或next-token CE（含prompt）；FP32优化 | 三LR×3epoch+重置后20epoch；独立64math+64code决选；两者最终均选lr3e-5/epoch3；AIME尚待审 |
| GSM8K | B_last_head_vocab64 | 95.2237 % | 1319 | 单次结果 | 仅末层+final norm+head matrix；FP32优化/BF16前向；20epoch；词表top64+tail KL；T1；head不量化 | LR筛选1e-5/3e-5/1e-4各3epoch；验证集选优，允许epoch0；AIME最终提交 |
| GSM8K | B_router_bf16 | 95.7544 % | 1319 | 单次结果 | 48层router BF16参数/梯度/动量；5epoch；top8+tail KL；T1；constant LR；独立native prefix | LR筛选1e-5/3e-5/1e-4各3epoch；验证集选优，允许epoch0；AIME最终提交 |
| LiveCodeBench v6 | B_last_head_full_kl | 51.4286 % | 175 | 单次结果 | B20同128/32；末层+norm+head；完整词表151936/T1或next-token CE（含prompt）；FP32优化 | 三LR×3epoch+重置后20epoch；独立64math+64code决选；两者最终均选lr3e-5/epoch3；AIME尚待审 |
| LiveCodeBench v6 | B_last_head_hard_ce | 52.0000 % | 175 | 单次结果 | B20同128/32；末层+norm+head；完整词表151936/T1或next-token CE（含prompt）；FP32优化 | 三LR×3epoch+重置后20epoch；独立64math+64code决选；两者最终均选lr3e-5/epoch3；AIME尚待审 |
| LiveCodeBench v6 | B_last_head_vocab64 | 53.1429 % | 175 | 单次结果 | 仅末层+final norm+head matrix；FP32优化/BF16前向；20epoch；词表top64+tail KL；T1；head不量化 | LR筛选1e-5/3e-5/1e-4各3epoch；验证集选优，允许epoch0；AIME最终提交 |
| LiveCodeBench v6 | B_router_bf16 | 54.2857 % | 175 | 单次结果 | 48层router BF16参数/梯度/动量；5epoch；top8+tail KL；T1；constant LR；独立native prefix | LR筛选1e-5/3e-5/1e-4各3epoch；验证集选优，允许epoch0；AIME最终提交 |
| MMLU-Pro | B_last_head_full_kl | 75.1995 % | 12032 | 单次结果 | B20同128/32；末层+norm+head；完整词表151936/T1或next-token CE（含prompt）；FP32优化 | 三LR×3epoch+重置后20epoch；独立64math+64code决选；两者最终均选lr3e-5/epoch3；AIME尚待审 |
| MMLU-Pro | B_last_head_hard_ce | 74.4348 % | 12032 | 单次结果 | B20同128/32；末层+norm+head；完整词表151936/T1或next-token CE（含prompt）；FP32优化 | 三LR×3epoch+重置后20epoch；独立64math+64code决选；两者最终均选lr3e-5/epoch3；AIME尚待审 |
| MMLU-Pro | B_last_head_vocab64 | 75.0499 % | 12032 | 单次结果 | 仅末层+final norm+head matrix；FP32优化/BF16前向；20epoch；词表top64+tail KL；T1；head不量化 | LR筛选1e-5/3e-5/1e-4各3epoch；验证集选优，允许epoch0；AIME最终提交 |
| MMLU-Pro | B_router_bf16 | 74.2520 % | 12032 | 单次结果 | 48层router BF16参数/梯度/动量；5epoch；top8+tail KL；T1；constant LR；独立native prefix | LR筛选1e-5/3e-5/1e-4各3epoch；验证集选优，允许epoch0；AIME最终提交 |

## T04　旧 S1K 模块消融：五项非 AIME 完整结果

**使用范围：** 完整历史探索证据；旧校准已发现与测试来源重合，不作为无污染主结论。采用修复MMLU revision后的canonical comparison。

**固定设置：** Qwen3-30B-A3B，原生 NVIDIA NVFP4 revision `2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`；reasoning：根 main.py 的 lighteval/vLLM、TP2、BF16 KV、eager、context=40960、batch/max_num_seqs=128、batched_tokens=4096、seed=1234、temperature=0.6、top_p=0.95、top_k=20、min_p=0、thinking；MMLU-Pro/GSM8K max_new_tokens=32768，LCB=38912；ARC：lm_eval、0-shot、raw acc（不是 acc_norm）、context=4096、seed=42。

**为何单独成表：** 相对 T02 使用清洗前 S1K，且各参数族学习率经同预算筛选；两阶段已有模型单列初始化差异。不能充当干净 A20 的单变量消融。 9月30日同口径验收轮已并入此表，并在方案名区分；不因运行日期不同另设表。

来源及完整配置：[comparison.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/comparison.json) · [comparison_validity.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/comparison_validity.json) · [manifest.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/manifest.json) · [REPORT.md](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/REPORT.md) · [comparison.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/downstream_accuracy_recovery/paired_full_20260930T131000Z/comparison.json) · [manifest.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/downstream_accuracy_recovery/paired_full_20260930T131000Z/manifest.json) · [prompt_audit.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/downstream_accuracy_recovery/paired_full_20260930T131000Z/prompt_audit.json)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| ARC-C | current | 49.4881 % | 1172 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003, 'matrices': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-C | current_kl | 51.1945 % | 1172 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003, 'matrices': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-C | diag | 50.6826 % | 1172 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'diag': 0.015} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-C | direct_hif4 | 50.3413 % | 1172 | 单次结果 | 无重构训练；direct_hif4 | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-C | existing_two_stage | 51.6212 % | 1172 | 单次结果 | E4 DIAG重构后追加group_top_mass；非原生单阶段初始化 | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-C | fixed_r64 | 52.0478 % | 1172 | 单次结果 | 无重构训练；fixed_r64 | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-C | group_top_mass（2026-09-30独立验收轮） | 51.6212 % | 1172 | 单次结果 | 原生；或E4→group G64/top-mass router KL两阶段 | 曾复用于后续报告；不可与同一原始回答再次平均；与10月模块消融同评测口径，单列原始轮，未混算均值 |
| ARC-C | matrix | 51.7918 % | 1172 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'matrices': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-C | matrix_norm | 51.8771 % | 1172 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003, 'matrices': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-C | matrix_norm_kl | 49.4881 % | 1172 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003, 'matrices': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-C | native_nvfp4 | 52.2184 % | 1172 | 单次结果 | 无重构训练；native_nvfp4 | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-C | native_nvfp4（2026-09-30独立验收轮） | 52.2184 % | 1172 | 单次结果 | 原生；或E4→group G64/top-mass router KL两阶段 | 曾复用于后续报告；不可与同一原始回答再次平均；与10月模块消融同评测口径，单列原始轮，未混算均值 |
| ARC-C | norm | 52.1331 % | 1172 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-C | norm_kl | 51.7065 % | 1172 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-C | rotation | 49.1468 % | 1172 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'rotations': 0.003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-C | rotation_norm | 51.8771 % | 1172 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.0001, 'rotations': 0.001} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-C | rotation_norm_kl | 49.9147 % | 1172 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.0001, 'rotations': 0.001} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-E | current | 77.9040 % | 2376 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003, 'matrices': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-E | current_kl | 77.8199 % | 2376 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003, 'matrices': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-E | diag | 77.5253 % | 2376 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'diag': 0.015} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-E | direct_hif4 | 78.4091 % | 2376 | 单次结果 | 无重构训练；direct_hif4 | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-E | existing_two_stage | 78.4512 % | 2376 | 单次结果 | E4 DIAG重构后追加group_top_mass；非原生单阶段初始化 | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-E | fixed_r64 | 78.6195 % | 2376 | 单次结果 | 无重构训练；fixed_r64 | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-E | group_top_mass（2026-09-30独立验收轮） | 78.4512 % | 2376 | 单次结果 | 原生；或E4→group G64/top-mass router KL两阶段 | 曾复用于后续报告；不可与同一原始回答再次平均；与10月模块消融同评测口径，单列原始轮，未混算均值 |
| ARC-E | matrix | 77.9882 % | 2376 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'matrices': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-E | matrix_norm | 78.7879 % | 2376 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003, 'matrices': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-E | matrix_norm_kl | 78.6616 % | 2376 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003, 'matrices': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-E | native_nvfp4 | 78.5774 % | 2376 | 单次结果 | 无重构训练；native_nvfp4 | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-E | native_nvfp4（2026-09-30独立验收轮） | 78.5774 % | 2376 | 单次结果 | 原生；或E4→group G64/top-mass router KL两阶段 | 曾复用于后续报告；不可与同一原始回答再次平均；与10月模块消融同评测口径，单列原始轮，未混算均值 |
| ARC-E | norm | 78.3249 % | 2376 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-E | norm_kl | 78.6195 % | 2376 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-E | rotation | 76.7677 % | 2376 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'rotations': 0.003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-E | rotation_norm | 77.9882 % | 2376 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.0001, 'rotations': 0.001} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| ARC-E | rotation_norm_kl | 77.5673 % | 2376 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.0001, 'rotations': 0.001} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| GSM8K | current | 95.3753 % | 1319 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003, 'matrices': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| GSM8K | current_kl | 95.4511 % | 1319 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003, 'matrices': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| GSM8K | diag | 95.1478 % | 1319 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'diag': 0.015} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| GSM8K | direct_hif4 | 94.9204 % | 1319 | 单次结果 | 无重构训练；direct_hif4 | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| GSM8K | existing_two_stage | 95.2995 % | 1319 | 单次结果 | E4 DIAG重构后追加group_top_mass；非原生单阶段初始化 | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| GSM8K | fixed_r64 | 95.3753 % | 1319 | 单次结果 | 无重构训练；fixed_r64 | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| GSM8K | group_top_mass（2026-09-30独立验收轮） | 95.2237 % | 1319 | 单次结果 | 原生；或E4→group G64/top-mass router KL两阶段 | 曾复用于后续报告；不可与同一原始回答再次平均；与10月模块消融同评测口径，单列原始轮，未混算均值 |
| GSM8K | matrix | 95.3753 % | 1319 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'matrices': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| GSM8K | matrix_norm | 95.0720 % | 1319 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003, 'matrices': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| GSM8K | matrix_norm_kl | 94.7688 % | 1319 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003, 'matrices': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| GSM8K | native_nvfp4 | 94.9962 % | 1319 | 单次结果 | 无重构训练；native_nvfp4 | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| GSM8K | native_nvfp4（2026-09-30独立验收轮） | 95.4511 % | 1319 | 单次结果 | 原生；或E4→group G64/top-mass router KL两阶段 | 曾复用于后续报告；不可与同一原始回答再次平均；与10月模块消融同评测口径，单列原始轮，未混算均值 |
| GSM8K | norm | 95.0720 % | 1319 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| GSM8K | norm_kl | 94.9962 % | 1319 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| GSM8K | rotation | 94.9962 % | 1319 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'rotations': 0.003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| GSM8K | rotation_norm | 95.4511 % | 1319 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.0001, 'rotations': 0.001} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| GSM8K | rotation_norm_kl | 94.6929 % | 1319 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.0001, 'rotations': 0.001} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| LiveCodeBench v6 | current | 52.0000 % | 175 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003, 'matrices': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| LiveCodeBench v6 | current_kl | 53.1429 % | 175 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003, 'matrices': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| LiveCodeBench v6 | diag | 50.8571 % | 175 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'diag': 0.015} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| LiveCodeBench v6 | direct_hif4 | 46.8571 % | 175 | 单次结果 | 无重构训练；direct_hif4 | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| LiveCodeBench v6 | existing_two_stage | 54.8571 % | 175 | 单次结果 | E4 DIAG重构后追加group_top_mass；非原生单阶段初始化 | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| LiveCodeBench v6 | fixed_r64 | 51.4286 % | 175 | 单次结果 | 无重构训练；fixed_r64 | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| LiveCodeBench v6 | group_top_mass（2026-09-30独立验收轮） | 54.8571 % | 175 | 单次结果 | 原生；或E4→group G64/top-mass router KL两阶段 | 曾复用于后续报告；不可与同一原始回答再次平均；与10月模块消融同评测口径，单列原始轮，未混算均值 |
| LiveCodeBench v6 | matrix | 52.5714 % | 175 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'matrices': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| LiveCodeBench v6 | matrix_norm | 54.8571 % | 175 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003, 'matrices': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| LiveCodeBench v6 | matrix_norm_kl | 49.7143 % | 175 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003, 'matrices': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| LiveCodeBench v6 | native_nvfp4 | 56.0000 % | 175 | 单次结果 | 无重构训练；native_nvfp4 | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| LiveCodeBench v6 | native_nvfp4（2026-09-30独立验收轮） | 54.2857 % | 175 | 单次结果 | 原生；或E4→group G64/top-mass router KL两阶段 | 曾复用于后续报告；不可与同一原始回答再次平均；与10月模块消融同评测口径，单列原始轮，未混算均值 |
| LiveCodeBench v6 | norm | 50.2857 % | 175 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| LiveCodeBench v6 | norm_kl | 49.1429 % | 175 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| LiveCodeBench v6 | rotation | 51.4286 % | 175 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'rotations': 0.003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| LiveCodeBench v6 | rotation_norm | 49.7143 % | 175 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.0001, 'rotations': 0.001} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| LiveCodeBench v6 | rotation_norm_kl | 50.2857 % | 175 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.0001, 'rotations': 0.001} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| MMLU-Pro | current | 74.7922 % | 12032 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003, 'matrices': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| MMLU-Pro | current_kl | 74.7673 % | 12032 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003, 'matrices': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| MMLU-Pro | diag | 75.5652 % | 12032 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'diag': 0.015} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| MMLU-Pro | direct_hif4 | 74.1273 % | 12032 | 单次结果 | 无重构训练；direct_hif4 | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| MMLU-Pro | existing_two_stage | 75.3989 % | 12032 | 单次结果 | E4 DIAG重构后追加group_top_mass；非原生单阶段初始化 | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| MMLU-Pro | fixed_r64 | 74.4515 % | 12032 | 单次结果 | 无重构训练；fixed_r64 | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| MMLU-Pro | group_top_mass（2026-09-30独立验收轮） | 75.4571 % | 12032 | 单次结果 | 原生；或E4→group G64/top-mass router KL两阶段 | 曾复用于后续报告；不可与同一原始回答再次平均；与10月模块消融同评测口径，单列原始轮，未混算均值 |
| MMLU-Pro | matrix | 75.3408 % | 12032 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'matrices': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| MMLU-Pro | matrix_norm | 75.3324 % | 12032 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003, 'matrices': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| MMLU-Pro | matrix_norm_kl | 74.9834 % | 12032 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003, 'matrices': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| MMLU-Pro | native_nvfp4 | 75.7563 % | 12032 | 单次结果 | 无重构训练；native_nvfp4 | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| MMLU-Pro | native_nvfp4（2026-09-30独立验收轮） | 75.3740 % | 12032 | 单次结果 | 原生；或E4→group G64/top-mass router KL两阶段 | 曾复用于后续报告；不可与同一原始回答再次平均；与10月模块消融同评测口径，单列原始轮，未混算均值 |
| MMLU-Pro | norm | 74.7257 % | 12032 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| MMLU-Pro | norm_kl | 74.5096 % | 12032 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.00030000000000000003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| MMLU-Pro | rotation | 75.9973 % | 12032 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'rotations': 0.003} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| MMLU-Pro | rotation_norm | 74.5844 % | 12032 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.0001, 'rotations': 0.001} | 参数族和KL定义见表下注；native为本消融批次独立评测 |
| MMLU-Pro | rotation_norm_kl | 74.9501 % | 12032 | 单次结果 | 旧S1K128/32；48层×20epoch；batch4；AdamW wd0/cosine；seed42；{'norms': 0.0001, 'rotations': 0.001} | 参数族和KL定义见表下注；native为本消融批次独立评测 |

模块定义：norm=两处norm；diag=可融合成对DIAG；rotation=权重/激活两侧绑定的Cayley正交G64；matrix=仅权重侧自由G64、在线R64固定；current=matrix+norm+router块矩阵。后缀KL加λ=1、T=1、teacher top8+tail router KL；非KL组router辅助权重0。LR筛选第0层、3epoch、倍率0.3/1/3，所有正式训练重新初始化。**旧rotation不能代替T01计划中“固定在线R64、只约束权重侧正交”的消融。**

## T06　短程效率：random50_e5 的全量测试

**使用范围：** 单轮精度/成本配方对照；不能把3.82小时与旧S1K20.04小时称作仅缩轮的因果加速。

**固定设置：** Qwen3-30B-A3B，原生 NVIDIA NVFP4 revision `2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`；reasoning：根 main.py 的 lighteval/vLLM、TP2、BF16 KV、eager、context=40960、batch/max_num_seqs=128、batched_tokens=4096、seed=1234、temperature=0.6、top_p=0.95、top_k=20、min_p=0、thinking；MMLU-Pro/GSM8K max_new_tokens=32768，LCB=38912；ARC：lm_eval、0-shot、raw acc（不是 acc_norm）、context=4096、seed=42。

**为何单独成表：** 与 T02 的20epoch不同：B20同来源轨迹、独立5epoch cosine、分层约50% token完整序列筛选；原生参照按本RUN的已核验来源绑定。

来源及完整配置：[random50_e5.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_norm_efficiency/run_20261029T023839Z/reports/random50_e5.json) · [manifest.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_norm_efficiency/run_20261029T023839Z/manifest.json) · [REPORT.md](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_norm_efficiency/run_20261029T023839Z/REPORT.md)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| AIME24 | native_nvfp4（效率参照） | 76.0000 % | 30 | 每题5次均值；1轮 | 5epoch；lr3e-4/batch4/G64；同B20轨迹的分层约50% token（非50条）；无重训基线 | AIME最终提交avg5；LCB/GSM8K等复用独立来源；本批仅random50完成正式五任务 |
| AIME24 | random50_e5 | 73.3333 % | 30 | 每题5次均值；1轮 | 5epoch；lr3e-4/batch4/G64；同B20轨迹的分层约50% token（非50条）；无重训基线 | AIME最终提交avg5；LCB/GSM8K等复用独立来源；本批仅random50完成正式五任务 |
| AIME25 | native_nvfp4（效率参照） | 70.6667 % | 30 | 每题5次均值；1轮 | 5epoch；lr3e-4/batch4/G64；同B20轨迹的分层约50% token（非50条）；无重训基线 | AIME最终提交avg5；LCB/GSM8K等复用独立来源；本批仅random50完成正式五任务 |
| AIME25 | random50_e5 | 68.0000 % | 30 | 每题5次均值；1轮 | 5epoch；lr3e-4/batch4/G64；同B20轨迹的分层约50% token（非50条）；无重训基线 | AIME最终提交avg5；LCB/GSM8K等复用独立来源；本批仅random50完成正式五任务 |
| GSM8K | native_nvfp4（效率参照） | 94.9962 % | 1319 | 单次结果 | 5epoch；lr3e-4/batch4/G64；同B20轨迹的分层约50% token（非50条）；无重训基线 | AIME最终提交avg5；LCB/GSM8K等复用独立来源；本批仅random50完成正式五任务 |
| GSM8K | random50_e5 | 95.6027 % | 1319 | 单次结果 | 5epoch；lr3e-4/batch4/G64；同B20轨迹的分层约50% token（非50条）；无重训基线 | AIME最终提交avg5；LCB/GSM8K等复用独立来源；本批仅random50完成正式五任务 |
| LiveCodeBench v6 | native_nvfp4（效率参照） | 56.0000 % | 175 | 单次结果 | 5epoch；lr3e-4/batch4/G64；同B20轨迹的分层约50% token（非50条）；无重训基线 | AIME最终提交avg5；LCB/GSM8K等复用独立来源；本批仅random50完成正式五任务 |
| LiveCodeBench v6 | random50_e5 | 50.8571 % | 175 | 单次结果 | 5epoch；lr3e-4/batch4/G64；同B20轨迹的分层约50% token（非50条）；无重训基线 | AIME最终提交avg5；LCB/GSM8K等复用独立来源；本批仅random50完成正式五任务 |
| MMLU-Pro | native_nvfp4（效率参照） | 75.4737 % | 12032 | 单次结果 | 5epoch；lr3e-4/batch4/G64；同B20轨迹的分层约50% token（非50条）；无重训基线 | AIME最终提交avg5；LCB/GSM8K等复用独立来源；本批仅random50完成正式五任务 |
| MMLU-Pro | random50_e5 | 74.4265 % | 12032 | 单次结果 | 5epoch；lr3e-4/batch4/G64；同B20轨迹的分层约50% token（非50条）；无重训基线 | AIME最终提交avg5；LCB/GSM8K等复用独立来源；本批仅random50完成正式五任务 |

## T07　不等价变换：共享粒度与 Router 目标

**使用范围：** 历史单轮探索；旧S1K128/32，不能宣称独立干净测试泛化。

**固定设置：** Qwen3-30B-A3B，原生 NVIDIA NVFP4 revision `2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`；reasoning：根 main.py 的 lighteval/vLLM、TP2、BF16 KV、eager、context=40960、batch/max_num_seqs=128、batched_tokens=4096、seed=1234、temperature=0.6、top_p=0.95、top_k=20、min_p=0、thinking；MMLU-Pro/GSM8K max_new_tokens=32768，LCB=38912；ARC：lm_eval、0-shot、raw acc（不是 acc_norm）、context=4096、seed=42。

**为何单独成表：** 旧MMLU-Pro300，不能与全量12032题合并；与T08实际prompt/token版本未建立跨RUN一致性，保留各自基线。

来源及完整配置：[comparison.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/non_equivalent_reconstruction/full48_20260915T084300Z/comparison.json)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| ARC-C | E4 | 49.5734 % | 1172 | 单次结果 | 复用E4 DIAG+R64 adopted checkpoint；没有本表追加的矩阵/LoRA训练 | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| ARC-C | group_top_mass | 51.6212 % | 1172 | 单次结果 | 旧S1K128/32；48×20epoch；AdamW lr1e-4/wd0/cosine；G64；routerλ1/top8/T1；linear或group共享；top_partial或top_mass | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| ARC-C | group_top_partial | 50.6826 % | 1172 | 单次结果 | 旧S1K128/32；48×20epoch；AdamW lr1e-4/wd0/cosine；G64；routerλ1/top8/T1；linear或group共享；top_partial或top_mass | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| ARC-C | linear_top_mass | 50.0853 % | 1172 | 单次结果 | 旧S1K128/32；48×20epoch；AdamW lr1e-4/wd0/cosine；G64；routerλ1/top8/T1；linear或group共享；top_partial或top_mass | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| ARC-C | linear_top_partial | 49.9147 % | 1172 | 单次结果 | 旧S1K128/32；48×20epoch；AdamW lr1e-4/wd0/cosine；G64；routerλ1/top8/T1；linear或group共享；top_partial或top_mass | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| ARC-E | E4 | 77.9461 % | 2376 | 单次结果 | 复用E4 DIAG+R64 adopted checkpoint；没有本表追加的矩阵/LoRA训练 | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| ARC-E | group_top_mass | 78.4512 % | 2376 | 单次结果 | 旧S1K128/32；48×20epoch；AdamW lr1e-4/wd0/cosine；G64；routerλ1/top8/T1；linear或group共享；top_partial或top_mass | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| ARC-E | group_top_partial | 78.8300 % | 2376 | 单次结果 | 旧S1K128/32；48×20epoch；AdamW lr1e-4/wd0/cosine；G64；routerλ1/top8/T1；linear或group共享；top_partial或top_mass | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| ARC-E | linear_top_mass | 76.9360 % | 2376 | 单次结果 | 旧S1K128/32；48×20epoch；AdamW lr1e-4/wd0/cosine；G64；routerλ1/top8/T1；linear或group共享；top_partial或top_mass | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| ARC-E | linear_top_partial | 76.6835 % | 2376 | 单次结果 | 旧S1K128/32；48×20epoch；AdamW lr1e-4/wd0/cosine；G64；routerλ1/top8/T1；linear或group共享；top_partial或top_mass | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| MMLU-Pro | E4 | 71.6667 % | 300 | 单次结果 | 复用E4 DIAG+R64 adopted checkpoint；没有本表追加的矩阵/LoRA训练 | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| MMLU-Pro | group_top_mass | 70.0000 % | 300 | 单次结果 | 旧S1K128/32；48×20epoch；AdamW lr1e-4/wd0/cosine；G64；routerλ1/top8/T1；linear或group共享；top_partial或top_mass | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| MMLU-Pro | group_top_partial | 70.3333 % | 300 | 单次结果 | 旧S1K128/32；48×20epoch；AdamW lr1e-4/wd0/cosine；G64；routerλ1/top8/T1；linear或group共享；top_partial或top_mass | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| MMLU-Pro | linear_top_mass | 71.6667 % | 300 | 单次结果 | 旧S1K128/32；48×20epoch；AdamW lr1e-4/wd0/cosine；G64；routerλ1/top8/T1；linear或group共享；top_partial或top_mass | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| MMLU-Pro | linear_top_partial | 73.0000 % | 300 | 单次结果 | 旧S1K128/32；48×20epoch；AdamW lr1e-4/wd0/cosine；G64；routerλ1/top8/T1；linear或group共享；top_partial或top_mass | MMLU-Pro300；LCB指标名:16不代表16次生成 |

## T08　残差 LoRA：Attention / MoE / Both

**使用范围：** 历史单轮探索；旧S1K128/32，不能宣称独立干净测试泛化。

**固定设置：** Qwen3-30B-A3B，原生 NVIDIA NVFP4 revision `2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`；reasoning：根 main.py 的 lighteval/vLLM、TP2、BF16 KV、eager、context=40960、batch/max_num_seqs=128、batched_tokens=4096、seed=1234、temperature=0.6、top_p=0.95、top_k=20、min_p=0、thinking；MMLU-Pro/GSM8K max_new_tokens=32768，LCB=38912；ARC：lm_eval、0-shot、raw acc（不是 acc_norm）、context=4096、seed=42。

**为何单独成表：** MMLU-Pro仅300题；模型从E4联合训练LoRA，非同一已训练主干只加LoRA；与T07的300题实际输入/版本未建立跨RUN一致性。

来源及完整配置：[comparison.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/residual_lora_compensation/20260922T012547Z/downstream/comparison.json)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| ARC-C | E4 | 49.5734 % | 1172 | 单次结果 | 复用E4 DIAG+R64 adopted checkpoint；没有本表追加的矩阵/LoRA训练 | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| ARC-C | attention | 50.4266 % | 1172 | 单次结果 | 旧S1K128/32；48×20epoch；AdamW lr1e-4/wd0/cosine；G64；routerλ1/top8/T1；LoRA rank4/alpha8；主分支同时更新 | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| ARC-C | both | 51.1092 % | 1172 | 单次结果 | 旧S1K128/32；48×20epoch；AdamW lr1e-4/wd0/cosine；G64；routerλ1/top8/T1；LoRA rank4/alpha8；主分支同时更新 | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| ARC-C | group_top_mass | 51.6212 % | 1172 | 单次结果 | 复用E4→G64 matrix/norm/router top-mass基线；无LoRA；其训练条件见T07 | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| ARC-C | moe | 51.0239 % | 1172 | 单次结果 | 旧S1K128/32；48×20epoch；AdamW lr1e-4/wd0/cosine；G64；routerλ1/top8/T1；LoRA rank4/alpha8；主分支同时更新 | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| ARC-E | E4 | 77.9461 % | 2376 | 单次结果 | 复用E4 DIAG+R64 adopted checkpoint；没有本表追加的矩阵/LoRA训练 | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| ARC-E | attention | 78.1145 % | 2376 | 单次结果 | 旧S1K128/32；48×20epoch；AdamW lr1e-4/wd0/cosine；G64；routerλ1/top8/T1；LoRA rank4/alpha8；主分支同时更新 | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| ARC-E | both | 78.2828 % | 2376 | 单次结果 | 旧S1K128/32；48×20epoch；AdamW lr1e-4/wd0/cosine；G64；routerλ1/top8/T1；LoRA rank4/alpha8；主分支同时更新 | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| ARC-E | group_top_mass | 78.4512 % | 2376 | 单次结果 | 复用E4→G64 matrix/norm/router top-mass基线；无LoRA；其训练条件见T07 | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| ARC-E | moe | 78.3670 % | 2376 | 单次结果 | 旧S1K128/32；48×20epoch；AdamW lr1e-4/wd0/cosine；G64；routerλ1/top8/T1；LoRA rank4/alpha8；主分支同时更新 | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| LiveCodeBench v6 | E4 | 52.0000 % | 175 | 单次结果 | 复用E4 DIAG+R64 adopted checkpoint；没有本表追加的矩阵/LoRA训练 | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| LiveCodeBench v6 | attention | 52.5714 % | 175 | 单次结果 | 旧S1K128/32；48×20epoch；AdamW lr1e-4/wd0/cosine；G64；routerλ1/top8/T1；LoRA rank4/alpha8；主分支同时更新 | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| LiveCodeBench v6 | both | 54.2857 % | 175 | 单次结果 | 旧S1K128/32；48×20epoch；AdamW lr1e-4/wd0/cosine；G64；routerλ1/top8/T1；LoRA rank4/alpha8；主分支同时更新 | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| LiveCodeBench v6 | group_top_mass | 54.8571 % | 175 | 单次结果 | 复用E4→G64 matrix/norm/router top-mass基线；无LoRA；其训练条件见T07 | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| LiveCodeBench v6 | moe | 53.7143 % | 175 | 单次结果 | 旧S1K128/32；48×20epoch；AdamW lr1e-4/wd0/cosine；G64；routerλ1/top8/T1；LoRA rank4/alpha8；主分支同时更新 | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| MMLU-Pro | E4 | 71.3333 % | 300 | 单次结果 | 复用E4 DIAG+R64 adopted checkpoint；没有本表追加的矩阵/LoRA训练 | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| MMLU-Pro | attention | 72.3333 % | 300 | 单次结果 | 旧S1K128/32；48×20epoch；AdamW lr1e-4/wd0/cosine；G64；routerλ1/top8/T1；LoRA rank4/alpha8；主分支同时更新 | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| MMLU-Pro | both | 71.0000 % | 300 | 单次结果 | 旧S1K128/32；48×20epoch；AdamW lr1e-4/wd0/cosine；G64；routerλ1/top8/T1；LoRA rank4/alpha8；主分支同时更新 | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| MMLU-Pro | group_top_mass | 74.3333 % | 300 | 单次结果 | 复用E4→G64 matrix/norm/router top-mass基线；无LoRA；其训练条件见T07 | MMLU-Pro300；LCB指标名:16不代表16次生成 |
| MMLU-Pro | moe | 71.3333 % | 300 | 单次结果 | 旧S1K128/32；48×20epoch；AdamW lr1e-4/wd0/cosine；G64；routerλ1/top8/T1；LoRA rank4/alpha8；主分支同时更新 | MMLU-Pro300；LCB指标名:16不代表16次生成 |

## T09　MXFP4 / 旋转 / UOS / 二级 scale：旧 JSON prompt GPQA

**使用范围：** 最终提交复核后的完整Avg@10；不是10个训练seed、投票或pass@10。

**固定设置：** Qwen3-30B-A3B原生NVFP4来源同上；只改48×128×3=18432个路由专家矩阵；非专家原样保留。MXFP4：E2M1、K维G32、E8M0、RNE；基础指数floor(log2(amax))-2；UOS c=7.25；G256/G512增加FP32二级scale。Qwen thinking、seed1234、T0.6/p0.95/k20/min_p0，context40960/output32768、batch256/batched_tokens8192、eager、BF16 KV。 TP4，每卡32GiB KV；198题×10采样序号。

**为何单独成表：** 与 T10/T11 的 AISBench prompt及选项排列不同；与 T14 的BF16三轮模型/TP/重复定义不同。

来源及完整配置：[eight_model_summary_20261111.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/dualscale_gpu47_20261110T085025Z/eight_model_summary_20261111.json)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| GPQA Diamond | Dual-scale G256 | 60.4040 % | 198 | 每题10次均值；1轮 | dualscale_g256 | 最终提交复核；严格JSON旧分另列T23 |
| GPQA Diamond | Dual-scale G512 | 60.4545 % | 198 | 每题10次均值；1轮 | dualscale_g512 | 最终提交复核；严格JSON旧分另列T23 |
| GPQA Diamond | R16 构造 R32 + MXFP4 | 60.1515 % | 198 | 每题10次均值；1轮 | r16x2 | 最终提交复核；严格JSON旧分另列T23 |
| GPQA Diamond | R64 + MXFP4 | 59.1414 % | 198 | 每题10次均值；1轮 | r64 | 最终提交复核；严格JSON旧分另列T23 |
| GPQA Diamond | UOS c7.25 | 59.6465 % | 198 | 每题10次均值；1轮 | direct_uos_c7p25 | 最终提交复核；严格JSON旧分另列T23 |
| GPQA Diamond | 原生 NVFP4 | 62.8788 % | 198 | 每题10次均值；1轮 | native | 最终提交复核；严格JSON旧分另列T23 |
| GPQA Diamond | 标准 H32 + MXFP4 | 59.5455 % | 198 | 每题10次均值；1轮 | fht32 | 最终提交复核；严格JSON旧分另列T23 |
| GPQA Diamond | 直接 MXFP4 | 59.6970 % | 198 | 每题10次均值；1轮 | direct | 最终提交复核；严格JSON旧分另列T23 |

## T10　AISBench prompt GPQA：TP4 / KV 32GiB

**使用范围：** 同表方法可比；跨两表只作带条件差异的参考，不声称完整AISBench框架复现。

**固定设置：** Qwen3-30B-A3B原生NVFP4来源同上；只改48×128×3=18432个路由专家矩阵；非专家原样保留。MXFP4：E2M1、K维G32、E8M0、RNE；基础指数floor(log2(amax))-2；UOS c=7.25；G256/G512增加FP32二级scale。Qwen thinking、seed1234、T0.6/p0.95/k20/min_p0，context40960/output32768、batch256/batched_tokens8192、eager、BF16 KV。 TP4，每卡32GiB KV；198题×10。AISBench零样本Answer: $LETTER模板，保留本地lighteval评分/最终提交复核。

**为何单独成表：** T10与T11因TP和显式KV容量不同分表，不能把差异全归因于优化方法；与T09因prompt/选项排列不同分表。

来源及完整配置：[comparison.csv](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/aisbench_gpqa_20261113T023500Z/answer_review_20261113T144526Z/comparison.csv) · [comparison.csv](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/combinations_tp2_20261113T154617Z/answer_review_20261114/comparison.csv)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| GPQA Diamond | 原生 NVFP4 | 62.2727 % | 198 | 每题10次均值；1轮 | TP4/KV32.0GiB；native | 最终提交；未结束thinking/无唯一最终答案计零；分母保留1980 |
| GPQA Diamond | 标准 Walsh–Hadamard H32＋MXFP4 | 57.7273 % | 198 | 每题10次均值；1轮 | TP4/KV32.0GiB；fht32 | 最终提交；未结束thinking/无唯一最终答案计零；分母保留1980 |
| GPQA Diamond | 直接 MXFP4 | 58.1313 % | 198 | 每题10次均值；1轮 | TP4/KV32.0GiB；direct | 最终提交；未结束thinking/无唯一最终答案计零；分母保留1980 |
| GPQA Diamond | 能量均衡 R16 扩展 R32＋MXFP4 | 58.8889 % | 198 | 每题10次均值；1轮 | TP4/KV32.0GiB；r16x2 | 最终提交；未结束thinking/无唯一最终答案计零；分母保留1980 |
| GPQA Diamond | 能量均衡 R64＋MXFP4 | 60.3030 % | 198 | 每题10次均值；1轮 | TP4/KV32.0GiB；r64 | 最终提交；未结束thinking/无唯一最终答案计零；分母保留1980 |

## T11　AISBench prompt GPQA：TP2 / KV 56GiB

**使用范围：** 同表方法可比；跨两表只作带条件差异的参考，不声称完整AISBench框架复现。

**固定设置：** Qwen3-30B-A3B原生NVFP4来源同上；只改48×128×3=18432个路由专家矩阵；非专家原样保留。MXFP4：E2M1、K维G32、E8M0、RNE；基础指数floor(log2(amax))-2；UOS c=7.25；G256/G512增加FP32二级scale。Qwen thinking、seed1234、T0.6/p0.95/k20/min_p0，context40960/output32768、batch256/batched_tokens8192、eager、BF16 KV。 TP2，每卡56GiB KV；198题×10。AISBench零样本Answer: $LETTER模板，保留本地lighteval评分/最终提交复核。

**为何单独成表：** T10与T11因TP和显式KV容量不同分表，不能把差异全归因于优化方法；与T09因prompt/选项排列不同分表。

来源及完整配置：[comparison.csv](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/aisbench_gpqa_20261113T023500Z/answer_review_20261113T144526Z/comparison.csv) · [comparison.csv](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/combinations_tp2_20261113T154617Z/answer_review_20261114/comparison.csv)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| GPQA Diamond | UOS c7.25 | 60.4040 % | 198 | 每题10次均值；1轮 | TP2/KV56.0GiB；direct_uos_c7p25 | 最终提交；未结束thinking/无唯一最终答案计零；分母保留1980 |
| GPQA Diamond | UOS c7.25＋二级 scale G512 | 60.1515 % | 198 | 每题10次均值；1轮 | TP2/KV56.0GiB；dualscale_g512_uos_c7p25 | 最终提交；未结束thinking/无唯一最终答案计零；分母保留1980 |
| GPQA Diamond | 二级 scale G512 | 59.1414 % | 198 | 每题10次均值；1轮 | TP2/KV56.0GiB；dualscale_g512 | 最终提交；未结束thinking/无唯一最终答案计零；分母保留1980 |
| GPQA Diamond | 能量均衡 R16 扩展 R32＋UOS c7.25 | 60.8586 % | 198 | 每题10次均值；1轮 | TP2/KV56.0GiB；r16x2_uos_c7p25 | 最终提交；未结束thinking/无唯一最终答案计零；分母保留1980 |
| GPQA Diamond | 能量均衡 R16 扩展 R32＋UOS c7.25＋二级 scale G512 | 59.8990 % | 198 | 每题10次均值；1轮 | TP2/KV56.0GiB；r16x2_dualscale_g512_uos_c7p25 | 最终提交；未结束thinking/无唯一最终答案计零；分母保留1980 |
| GPQA Diamond | 能量均衡 R16 扩展 R32＋二级 scale G512 | 58.7879 % | 198 | 每题10次均值；1轮 | TP2/KV56.0GiB；r16x2_dualscale_g512 | 最终提交；未结束thinking/无唯一最终答案计零；分母保留1980 |

## T12　MXFP4 两种旋转：旧 prompt LCB 与 ARC

**使用范围：** 仅列停止前已完整的LCB/ARC；全队列停止不意味着这些已完成任务无效。

**固定设置：** Qwen3-30B-A3B原生NVFP4来源同上；只改48×128×3=18432个路由专家矩阵；非专家原样保留。MXFP4：E2M1、K维G32、E8M0、RNE；基础指数floor(log2(amax))-2；UOS c=7.25；G256/G512增加FP32二级scale。Qwen thinking、seed1234、T0.6/p0.95/k20/min_p0，context40960/output32768、batch256/batched_tokens8192、eager、BF16 KV。 TP4；LCB175题每题10次；ARC全量5轮、raw acc。

**为何单独成表：** 与T09同系列但任务/生成预算不同；与T13不同prompt、TP和采样次数。被停止的旧MMLU-Pro没有完整分数，按既定要求不列。

来源及完整配置：[stopped_results_20261113.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/rotation_downstream_20261111T021111Z/stopped_results_20261113.json)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| ARC-C | fht32 | 49.8294 % | 1172 | 5轮均值；每题1次 | TP4；lm_eval 0-shot raw acc；每轮完整任务 | 完整 |
| ARC-C | r16x2 | 50.5802 % | 1172 | 5轮均值；每题1次 | TP4；lm_eval 0-shot raw acc；每轮完整任务 | 完整 |
| ARC-E | fht32 | 77.6515 % | 2376 | 5轮均值；每题1次 | TP4；lm_eval 0-shot raw acc；每轮完整任务 | 完整 |
| ARC-E | r16x2 | 77.6768 % | 2376 | 5轮均值；每题1次 | TP4；lm_eval 0-shot raw acc；每轮完整任务 | 完整 |
| LiveCodeBench v6 | fht32 | 48.2286 % | 175 | 每题10次均值；1轮 | TP4；旧LCB模板；max_new_tokens38912 | 完整 |
| LiveCodeBench v6 | r16x2 | 48.9714 % | 175 | 每题10次均值；1轮 | TP4；旧LCB模板；max_new_tokens38912 | 完整 |

## T13　AISBench prompt 下游：原生、直转与四种组合共表

**使用范围：** 完整且可比的每题5次算术均值；不是5个独立seed，不是pass@5/多数投票。

**固定设置：** Qwen3-30B-A3B原生NVFP4来源同上；只改48×128×3=18432个路由专家矩阵；非专家原样保留。MXFP4：E2M1、K维G32、E8M0、RNE；基础指数floor(log2(amax))-2；UOS c=7.25；G256/G512增加FP32二级scale。Qwen thinking、seed1234、T0.6/p0.95/k20/min_p0，context40960/output32768、batch256/batched_tokens8192、eager、BF16 KV。 TP2/每卡56GiB BF16 KV；LCB175×5、MMLU-Pro12032×5；LCB输出38912，MMLU-Pro32768；AISBench零样本模板，经main.py/lighteval评分。

**为何单独成表：** 两批RUN已核对同输入/协议，故合并；与T01五个独立运行、旧prompt及动态显存预算不同，不能混成一张方法主表。

来源及完整配置：[comparison.csv](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/native_direct_downstream_20261116T063859Z/comparison.csv) · [comparison.csv](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/combination_downstream_20261114T053205Z_v2/comparison.csv)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| LiveCodeBench v6 | direct | 48.5714 % | 175 | 每题5次均值；1轮 | TP2/KV56GiB；同prompt/数据/gold；direct | 五采样序号分数和方差保留源CSV |
| LiveCodeBench v6 | dualscale_g512_uos_c7p25 | 52.5714 % | 175 | 每题5次均值；1轮 | TP2/KV56GiB；同prompt/数据/gold；dualscale_g512_uos_c7p25 | 五采样序号分数和方差保留源CSV |
| LiveCodeBench v6 | native | 55.8857 % | 175 | 每题5次均值；1轮 | TP2/KV56GiB；同prompt/数据/gold；native | 五采样序号分数和方差保留源CSV |
| LiveCodeBench v6 | r16x2_dualscale_g512 | 50.2857 % | 175 | 每题5次均值；1轮 | TP2/KV56GiB；同prompt/数据/gold；r16x2_dualscale_g512 | 五采样序号分数和方差保留源CSV |
| LiveCodeBench v6 | r16x2_dualscale_g512_uos_c7p25 | 51.5429 % | 175 | 每题5次均值；1轮 | TP2/KV56GiB；同prompt/数据/gold；r16x2_dualscale_g512_uos_c7p25 | 五采样序号分数和方差保留源CSV |
| LiveCodeBench v6 | r16x2_uos_c7p25 | 52.8000 % | 175 | 每题5次均值；1轮 | TP2/KV56GiB；同prompt/数据/gold；r16x2_uos_c7p25 | 五采样序号分数和方差保留源CSV |
| MMLU-Pro | direct | 74.1722 % | 12032 | 每题5次均值；1轮 | TP2/KV56GiB；同prompt/数据/gold；direct | 五采样序号分数和方差保留源CSV |
| MMLU-Pro | dualscale_g512_uos_c7p25 | 74.9053 % | 12032 | 每题5次均值；1轮 | TP2/KV56GiB；同prompt/数据/gold；dualscale_g512_uos_c7p25 | 五采样序号分数和方差保留源CSV |
| MMLU-Pro | native | 76.0888 % | 12032 | 每题5次均值；1轮 | TP2/KV56GiB；同prompt/数据/gold；native | 五采样序号分数和方差保留源CSV |
| MMLU-Pro | r16x2_dualscale_g512 | 74.6925 % | 12032 | 每题5次均值；1轮 | TP2/KV56GiB；同prompt/数据/gold；r16x2_dualscale_g512 | 五采样序号分数和方差保留源CSV |
| MMLU-Pro | r16x2_dualscale_g512_uos_c7p25 | 75.1213 % | 12032 | 每题5次均值；1轮 | TP2/KV56GiB；同prompt/数据/gold；r16x2_dualscale_g512_uos_c7p25 | 五采样序号分数和方差保留源CSV |
| MMLU-Pro | r16x2_uos_c7p25 | 75.3723 % | 12032 | 每题5次均值；1轮 | TP2/KV56GiB；同prompt/数据/gold；r16x2_uos_c7p25 | 五采样序号分数和方差保留源CSV |

## T14　官方 BF16 基线：GPQA 三个独立 seed

**使用范围：** 完整三轮均值，可作明确注明协议的BF16参考。

**固定设置：** Qwen/Qwen3-30B-A3B官方BF16；ModelScope revision8143878278a202cc0ff9ef1bbe00b2abb1adf86d；TP8/KV20GiB；batch128/batched4096；JSON answer、固定选项seed1234；推理seed1234/1235/1236顺序三轮；T0.6/p0.95/k20/min_p0；context40960/output32768。

**为何单独成表：** 与NVFP4/MXFP4表模型精度、TP、KV、prompt或重复定义不同；不能直接称AISBench基线。

来源及完整配置：[final_summary.json](../../../results/qwen3_bf16_gpqa/official_tp8_rounds3_20261111T105127Z/final_summary.json) · [completion_validation.json](../../../results/qwen3_bf16_gpqa/official_tp8_rounds3_20261111T105127Z/completion_validation.json)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| GPQA Diamond | 官方 BF16 | 64.4781 % | 198 | 3轮均值；每题1次 |  | 每轮198题×1；122/128/133正确；独立生成seed，不是3次训练 |

## T15　旧 matrix_norm 上的 Ban：预算约 6.07 专家

**使用范围：** 历史路由精度实验；AIME旧自动提取分只作原始记录；专家数减少不等于实测加速。

**固定设置：** Qwen3-30B-A3B，原生 NVIDIA NVFP4 revision `2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`；reasoning：根 main.py 的 lighteval/vLLM、TP2、BF16 KV、eager、context=40960、batch/max_num_seqs=128、batched_tokens=4096、seed=1234、temperature=0.6、top_p=0.95、top_k=20、min_p=0、thinking；MMLU-Pro/GSM8K max_new_tokens=32768，LCB=38912；ARC：lm_eval、0-shot、raw acc（不是 acc_norm）、context=4096、seed=42。 关闭prefix cache；Ban以C4 512×2048、seed0校准，k_min5；GSM8K train前8题匹配6±0.1预算；不按测试调λ。

**为何单独成表：** 与T04比较改变了稀疏路由；与T16的专家预算、校准源和重复次数不同。底座是旧S1K matrix_norm，非干净A20。

来源及完整配置：[report.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/gsm8k_8to6_20261024T022233Z/report.json) · [report.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/supplement_20261025T150926Z/report.json) · [report.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/repeats_20261027T050631Z/report.json)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| AIME25 | Ban（原补测第1轮） | 58.6667 % | 30 | 每题5次均值；1轮 | 沿用同C4 Ban | MMLU-Pro仅300题；AIME为旧自动提取器，非最终提交复核 |
| AIME25 | Ban（含首轮共4轮） | 59.6667 % | 30 | 4轮均值；每题5次 | 固定seed1234，独立缓存 | 包含上行首轮，不能当作另一批独立数据；AIME为4轮×每题5次 |
| GSM8K | Ban | 95.9060 % | 1319 | 单次结果 | top8 / 固定top6 / Ban(k_min5,实测约6.070009) | 完整 |
| GSM8K | top6 | 95.5269 % | 1319 | 单次结果 | top8 / 固定top6 / Ban(k_min5,实测约6.070009) | 完整 |
| GSM8K | top8 | 95.0720 % | 1319 | 单次结果 | top8 / 固定top6 / Ban(k_min5,实测约6.070009) | 完整 |
| LiveCodeBench v6 | Ban（原补测第1轮） | 52.0000 % | 175 | 单次结果 | 沿用同C4 Ban | MMLU-Pro仅300题；AIME为旧自动提取器，非最终提交复核 |
| LiveCodeBench v6 | Ban（含首轮共4轮） | 51.1429 % | 175 | 4轮均值；每题1次 | 固定seed1234，独立缓存 | 包含上行首轮，不能当作另一批独立数据；AIME为4轮×每题5次 |
| MMLU-Pro | Ban（原补测第1轮） | 70.3333 % | 300 | 单次结果 | 沿用同C4 Ban | MMLU-Pro仅300题；AIME为旧自动提取器，非最终提交复核 |

## T16　Ban 6.4 专家预算：C4 vs 自生成重新校准

**使用范围：** 完整五任务；AIME为最终提交复核分。底座的旧S1K污染边界仍须披露。

**固定设置：** Qwen3-30B-A3B，原生 NVIDIA NVFP4 revision `2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`；reasoning：根 main.py 的 lighteval/vLLM、TP2、BF16 KV、eager、context=40960、batch/max_num_seqs=128、batched_tokens=4096、seed=1234、temperature=0.6、top_p=0.95、top_k=20、min_p=0、thinking；MMLU-Pro/GSM8K max_new_tokens=32768，LCB=38912；ARC：lm_eval、0-shot、raw acc（不是 acc_norm）、context=4096、seed=42。 底座旧matrix_norm不重训；C4和trace均512×2048，seed0；trace源为干净B数据；k_min5，λ分别0.8875/0.7375；LCB3轮，MMLU/GSM1轮，AIME每题5次。

**为何单独成表：** 与T15预算6.07和其校准/预算匹配条件不同；与T02的更换重构训练数据不同，本表只更换Ban校准。

来源及完整配置：[all_metrics.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/remaining64_20261104T040720Z/all_metrics.json) · [REPORT.md](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/remaining64_20261104T040720Z/REPORT.md)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| AIME24 | c4 | 76.6667 % | 30 | 每题5次均值；1轮 | k_min5；实测平均专家6.395624 | top8部分基线为已核验复用；最终提交评分 |
| AIME24 | top8 | 75.3333 % | 30 | 每题5次均值；1轮 | k_min5；实测平均专家8.000000 | top8部分基线为已核验复用；最终提交评分 |
| AIME24 | trace | 76.0000 % | 30 | 每题5次均值；1轮 | k_min5；实测平均专家6.410333 | top8部分基线为已核验复用；最终提交评分 |
| AIME25 | c4 | 62.0000 % | 30 | 每题5次均值；1轮 | k_min5；实测平均专家6.398067 | top8部分基线为已核验复用；最终提交评分 |
| AIME25 | top8 | 66.0000 % | 30 | 每题5次均值；1轮 | k_min5；实测平均专家8.000000 | top8部分基线为已核验复用；最终提交评分 |
| AIME25 | trace | 67.3333 % | 30 | 每题5次均值；1轮 | k_min5；实测平均专家6.414365 | top8部分基线为已核验复用；最终提交评分 |
| GSM8K | c4 | 95.3753 % | 1319 | 单次结果 | k_min5；实测平均专家6.375020 | top8部分基线为已核验复用；最终提交评分 |
| GSM8K | top8 | 95.0720 % | 1319 | 单次结果 | k_min5；实测平均专家8.000000 | top8部分基线为已核验复用；最终提交评分 |
| GSM8K | trace | 95.6027 % | 1319 | 单次结果 | k_min5；实测平均专家6.396673 | top8部分基线为已核验复用；最终提交评分 |
| LiveCodeBench v6 | c4 | 53.3333 % | 175 | 3轮均值；每题1次 | k_min5；实测平均专家6.385061 | top8部分基线为已核验复用；最终提交评分 |
| LiveCodeBench v6 | top8 | 53.9048 % | 175 | 3轮均值；每题1次 | k_min5；实测平均专家8.000000 | top8部分基线为已核验复用；最终提交评分 |
| LiveCodeBench v6 | trace | 52.9524 % | 175 | 3轮均值；每题1次 | k_min5；实测平均专家6.400870 | top8部分基线为已核验复用；最终提交评分 |
| MMLU-Pro | c4 | 75.0582 % | 12032 | 单次结果 | k_min5；实测平均专家6.364549 | top8部分基线为已核验复用；最终提交评分 |
| MMLU-Pro | top8 | 75.3324 % | 12032 | 单次结果 | k_min5；实测平均专家8.000000 | top8部分基线为已核验复用；最终提交评分 |
| MMLU-Pro | trace | 75.5901 % | 12032 | 单次结果 | k_min5；实测平均专家6.392233 | top8部分基线为已核验复用；最终提交评分 |

## T17　MiniMax-M3.1-Flash 原生基线

**使用范围：** 已完成原生五任务；MXFP4仅转换完成，GPU评测取消，不填MXFP4精度。

**固定设置：** MiniMaxAI/MiniMax-M3.1-Flash revision2f33a8b9ae49ce7b51c994df97166a7fd059bc2e（精确SHA以manifest为准）；实际TP2×PP3、batch16/batched4096、context40960、temperature1.0/top_p0.95；LCB/AIME output38912，MMLU32768，GSM8K实测最大输出16785。

**为何单独成表：** 模型、TP/PP、采样温度及MMLU300均不同于Qwen表；不能作为同模型转换对照。

来源及完整配置：[state.json](../../../results/minimax_m3/continuation_20261103T015711Z/state.json)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| AIME24 | MiniMax 原生混合精度 | 96.0000 % | 30 | 每题5次均值；1轮 | TP2×PP3；T1.0/p0.95；其余以actual command为准 | 保留原自动评分；未宣称已用Qwen最终提交复核器复核 |
| AIME25 | MiniMax 原生混合精度 | 89.3333 % | 30 | 每题5次均值；1轮 | TP2×PP3；T1.0/p0.95；其余以actual command为准 | 保留原自动评分；未宣称已用Qwen最终提交复核器复核 |
| GSM8K | MiniMax 原生混合精度 | 90.0682 % | 1319 | 单次结果 | TP2×PP3；T1.0/p0.95；其余以actual command为准 | 保留原自动评分；未宣称已用Qwen最终提交复核器复核 |
| LiveCodeBench v6 | MiniMax 原生混合精度 | 38.8571 % | 175 | 单次结果 | TP2×PP3；T1.0/p0.95；其余以actual command为准 | 保留原自动评分；未宣称已用Qwen最终提交复核器复核 |
| MMLU-Pro | MiniMax 原生混合精度 | 81.0000 % | 300 | 单次结果 | TP2×PP3；T1.0/p0.95；其余以actual command为准 | 保留原自动评分；未宣称已用Qwen最终提交复核器复核 |

## T18　早期 E0–E7：修复输入后的受控评测

**使用范围：** 可用于早期同输入消融；不是干净A20，也不支持旧prompt错位导致的纯格式损失结论。

**固定设置：** Qwen3-30B-A3B，原生 NVIDIA NVFP4 revision `2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`；reasoning：根 main.py 的 lighteval/vLLM、TP2、BF16 KV、eager、context=40960、batch/max_num_seqs=128、batched_tokens=4096、seed=1234、temperature=0.6、top_p=0.95、top_k=20、min_p=0、thinking；MMLU-Pro/GSM8K max_new_tokens=32768，LCB=38912；ARC：lm_eval、0-shot、raw acc（不是 acc_norm）、context=4096、seed=42。

**为何单独成表：** MMLU-Pro300而非全量；使用旧adopted模型。LCB固定原生实际input_ids和controlled sampled，与后续标准runner不能只凭任务名合并。

来源及完整配置：[CLEAN_LCB_E2_E4_REPORT.md](../../../Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/format_conversion_error_localization_replan_20260908/10_clean_lcb/CLEAN_LCB_E2_E4_REPORT.md) · [CLEAN_MMLU_PRO_E0_E7_REPORT.md](../../../Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/format_conversion_error_localization_replan_20260908/20_mmlu_pro_300/CLEAN_MMLU_PRO_E0_E7_REPORT.md)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| LiveCodeBench v6 | E0 native | 54.2857 % | 175 | 单次结果 |  | exact same input_ids；1轮完整175题 |
| LiveCodeBench v6 | E1 direct | 50.8571 % | 175 | 单次结果 |  | exact same input_ids；1轮完整175题 |
| LiveCodeBench v6 | E2 R64 | 53.1429 % | 175 | 单次结果 |  | exact same input_ids；1轮完整175题 |
| LiveCodeBench v6 | E3 DIAG adopted | 52.0000 % | 175 | 单次结果 |  | exact same input_ids；1轮完整175题 |
| LiveCodeBench v6 | E4 DIAG+R64 adopted | 50.2857 % | 175 | 单次结果 |  | exact same input_ids；1轮完整175题 |
| MMLU-Pro | E0 | 70.6667 % | 300 | 单次结果 |  | 由报告300题分数还原正确数；早期固定300题协议 |
| MMLU-Pro | E1 | 70.6667 % | 300 | 单次结果 |  | 由报告300题分数还原正确数；早期固定300题协议 |
| MMLU-Pro | E2 | 69.3333 % | 300 | 单次结果 |  | 由报告300题分数还原正确数；早期固定300题协议 |
| MMLU-Pro | E3 | 70.3333 % | 300 | 单次结果 |  | 由报告300题分数还原正确数；早期固定300题协议 |
| MMLU-Pro | E4 | 72.6667 % | 300 | 单次结果 |  | 由报告300题分数还原正确数；早期固定300题协议 |
| MMLU-Pro | E5 online DIAG | 71.3333 % | 300 | 单次结果 |  | 由报告300题分数还原正确数；早期固定300题协议 |
| MMLU-Pro | E6 DIAG→R64 | 71.3333 % | 300 | 单次结果 |  | 由报告300题分数还原正确数；早期固定300题协议 |
| MMLU-Pro | E7 R64→DIAG | 69.0000 % | 300 | 单次结果 |  | 由报告300题分数还原正确数；早期固定300题协议 |

## T19　Qwen3.5-4B 阈值实验：历史报告中的端到端结果

**使用范围：** 只作历史报告转录，原始记录补齐前不放正式论文主结果；未记重复次数处明确未知。

**固定设置：** Qwen/Qwen3.5-4B、A800；标准HiF4(7,4,2)，fixed_best=(7,3.9,1.95)，weight_search=fast S0±1×8；激活网格d=5.5:0.25:7.5、t8=3.4:0.1:4.1、t4=1.70:0.05:2.05；PPL WikiText2 seqlen2048，ARC/MMLU HF fake-quant 0-shot；MMLU-Pro300用vLLM标准激活、disable_thinking、T0.7。

**为何单独成表：** 模型及HF/vLLM路径、thinking、校准/量化范围与30B主线不同；历史原始results子目录当前缺失，不能声称本轮复核过原始回答。

来源及完整配置：[EXPERIMENT_REPORT.md](../../../HiFloat4/hif4_scale_threshold_optimization/experiments/EXPERIMENT_REPORT.md)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| ARC-C | act_calib_only | 50.7700 % | 1172 | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| ARC-C | act_calib_s1k | 50.0900 % | 1172 | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| ARC-C | act_fixed_best | 50.6000 % | 1172 | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| ARC-C | baseline_standard | 50.9400 % | 1172 | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| ARC-C | joint | 50.6800 % | 1172 | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| ARC-C | weight_fixed_best | 50.2600 % | 1172 | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| ARC-C | weight_search_fast | 50.1700 % | 1172 | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| ARC-E | act_calib_only | 79.4600 % | 2376 | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| ARC-E | act_calib_s1k | 80.2600 % | 2376 | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| ARC-E | act_fixed_best | 80.0900 % | 2376 | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| ARC-E | baseline_standard | 79.8400 % | 2376 | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| ARC-E | joint | 80.0500 % | 2376 | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| ARC-E | weight_fixed_best | 80.0100 % | 2376 | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| ARC-E | weight_search_fast | 80.1800 % | 2376 | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| MMLU | act_calib_only | 72.5800 % |  | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| MMLU | act_calib_s1k | 72.7200 % |  | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| MMLU | act_fixed_best | 72.4200 % |  | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| MMLU | baseline_standard | 72.7600 % |  | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| MMLU | joint | 72.1500 % |  | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| MMLU | weight_fixed_best | 72.6000 % |  | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| MMLU | weight_search_fast | 72.4700 % |  | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| MMLU-Pro | act_calib_only | 69.6700 % | 300 | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| MMLU-Pro | act_calib_s1k | 69.6700 % | 300 | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| MMLU-Pro | act_fixed_best | 69.6700 % | 300 | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| MMLU-Pro | baseline_standard | 69.6700 % | 300 | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| MMLU-Pro | joint | 70.3300 % | 300 | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| MMLU-Pro | weight_fixed_best | 70.3300 % | 300 | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| MMLU-Pro | weight_search_fast | 68.6700 % | 300 | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| WikiText2 | act_calib_only | 10.2984 PPL |  | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| WikiText2 | act_calib_s1k | 10.2638 PPL |  | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| WikiText2 | act_fixed_best | 10.2599 PPL |  | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| WikiText2 | baseline_standard | 10.2546 PPL |  | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| WikiText2 | joint | 10.0018 PPL |  | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| WikiText2 | weight_fixed_best | 10.1796 PPL |  | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |
| WikiText2 | weight_search_fast | 9.9694 PPL |  | 重复次数未核实；非五次均值 | 历史fast预算，非T01 full | 文档只提供一个汇总值，不能推定为5次均值；MMLU-Pro实际激活始终standard |

## T20　4B reasoning 历史值：采样数与截断证据不足

**使用范围：** 仅历史记录；已知2个超长样本prompt被截到0，不能据此建立正式公平比较。

**固定设置：** 同4B checkpoint；thinking/T0.7；AIME25报告每题5次；LCB报告称16次，但同文又以29/175、32/175解释，原始缓存已缺失。

**为何单独成表：** 与T19的disable_thinking描述冲突，且LCB真实生成次数无法核实；不能因为指标名含:16就认定16次均值。

来源及完整配置：[EXPERIMENT_REPORT.md](../../../HiFloat4/hif4_scale_threshold_optimization/experiments/EXPERIMENT_REPORT.md)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| AIME25 | baseline_standard | 38.6700 % | 30 | 每题5次均值；1轮 |  | 报告称每题5次；无原始缓存复核 |
| AIME25 | weight_fixed_best | 38.0000 % | 30 | 每题5次均值；1轮 |  | 报告称每题5次；无原始缓存复核 |
| AIME25 | weight_search_fast | 39.3300 % | 30 | 每题5次均值；1轮 |  | 报告称每题5次；无原始缓存复核 |
| LiveCodeBench v6 | baseline_standard | 16.5700 % | 175 | 重复次数未核实；非五次均值 |  | 报告称16次与32/175解释并存；不认定16次均值 |
| LiveCodeBench v6 | weight_fixed_best | 18.2900 % | 175 | 重复次数未核实；非五次均值 |  | 报告称16次与32/175解释并存；不认定16次均值 |
| LiveCodeBench v6 | weight_search_fast | 18.2900 % | 175 | 重复次数未核实；非五次均值 |  | 报告称16次与32/175解释并存；不认定16次均值 |

## T21　旧 AIME25：自动提取分与重复轮（不用于最终答案比较）

**使用范围：** 完整原始数值记录；存在已证实评分错误，不用于论文最终准确率或排名。

**固定设置：** Qwen3-30B-A3B，原生 NVIDIA NVFP4 revision `2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`；reasoning：根 main.py 的 lighteval/vLLM、TP2、BF16 KV、eager、context=40960、batch/max_num_seqs=128、batched_tokens=4096、seed=1234、temperature=0.6、top_p=0.95、top_k=20、min_p=0、thinking；MMLU-Pro/GSM8K max_new_tokens=32768，LCB=38912；ARC：lm_eval、0-shot、raw acc（不是 acc_norm）、context=4096、seed=42。 AIME25每轮30题×5。

**为何单独成表：** 旧提取器会读入思考中的占位符/中间答案；与T22/T02的最终提交复核分不同。

来源及完整配置：[comparison.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/comparison.json) · [aime_scoring_recomputed_20261015.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/aime_scoring_recomputed_20261015.json)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| AIME25 | current_kl（原正式轮） | 54.6667 % | 30 | 每题5次均值；1轮 |  | 旧自动评分；已知提取问题 |
| AIME25 | current（原正式轮） | 61.3333 % | 30 | 每题5次均值；1轮 |  | 旧自动评分；已知提取问题 |
| AIME25 | diag（原正式轮） | 59.3333 % | 30 | 每题5次均值；1轮 |  | 旧自动评分；已知提取问题 |
| AIME25 | direct_hif4（原正式轮） | 58.6667 % | 30 | 每题5次均值；1轮 |  | 旧自动评分；已知提取问题 |
| AIME25 | existing_two_stage（原正式轮） | 62.0000 % | 30 | 每题5次均值；1轮 |  | 旧自动评分；已知提取问题 |
| AIME25 | fixed_r64（原正式轮） | 66.0000 % | 30 | 每题5次均值；1轮 |  | 旧自动评分；已知提取问题 |
| AIME25 | group_top_mass（追加3轮均值） | 62.4444 % | 30 | 3轮均值；每题5次 |  | 只平均这3个追加轮；不含原正式轮 |
| AIME25 | group_top_mass（追加轮1） | 62.0000 % | 30 | 每题5次均值；1轮 |  | 各追加轮单列，不挑最好轮；固定seed1234 |
| AIME25 | group_top_mass（追加轮2） | 62.0000 % | 30 | 每题5次均值；1轮 |  | 各追加轮单列，不挑最好轮；固定seed1234 |
| AIME25 | group_top_mass（追加轮3） | 63.3333 % | 30 | 每题5次均值；1轮 |  | 各追加轮单列，不挑最好轮；固定seed1234 |
| AIME25 | matrix_norm_kl（原正式轮） | 62.0000 % | 30 | 每题5次均值；1轮 |  | 旧自动评分；已知提取问题 |
| AIME25 | matrix_norm（原正式轮） | 64.0000 % | 30 | 每题5次均值；1轮 |  | 旧自动评分；已知提取问题 |
| AIME25 | matrix_norm（追加3轮均值） | 59.7778 % | 30 | 3轮均值；每题5次 |  | 只平均这3个追加轮；不含原正式轮 |
| AIME25 | matrix_norm（追加轮1） | 63.3333 % | 30 | 每题5次均值；1轮 |  | 各追加轮单列，不挑最好轮；固定seed1234 |
| AIME25 | matrix_norm（追加轮2） | 58.6667 % | 30 | 每题5次均值；1轮 |  | 各追加轮单列，不挑最好轮；固定seed1234 |
| AIME25 | matrix_norm（追加轮3） | 57.3333 % | 30 | 每题5次均值；1轮 |  | 各追加轮单列，不挑最好轮；固定seed1234 |
| AIME25 | matrix（原正式轮） | 58.0000 % | 30 | 每题5次均值；1轮 |  | 旧自动评分；已知提取问题 |
| AIME25 | native_nvfp4（原正式轮） | 67.3333 % | 30 | 每题5次均值；1轮 |  | 旧自动评分；已知提取问题 |
| AIME25 | native_nvfp4（追加3轮均值） | 66.0000 % | 30 | 3轮均值；每题5次 |  | 只平均这3个追加轮；不含原正式轮 |
| AIME25 | native_nvfp4（追加轮1） | 64.0000 % | 30 | 每题5次均值；1轮 |  | 各追加轮单列，不挑最好轮；固定seed1234 |
| AIME25 | native_nvfp4（追加轮2） | 67.3333 % | 30 | 每题5次均值；1轮 |  | 各追加轮单列，不挑最好轮；固定seed1234 |
| AIME25 | native_nvfp4（追加轮3） | 66.6667 % | 30 | 每题5次均值；1轮 |  | 各追加轮单列，不挑最好轮；固定seed1234 |
| AIME25 | norm_kl（原正式轮） | 60.0000 % | 30 | 每题5次均值；1轮 |  | 旧自动评分；已知提取问题 |
| AIME25 | norm（原正式轮） | 59.3333 % | 30 | 每题5次均值；1轮 |  | 旧自动评分；已知提取问题 |
| AIME25 | rotation_norm_kl（原正式轮） | 62.0000 % | 30 | 每题5次均值；1轮 |  | 旧自动评分；已知提取问题 |
| AIME25 | rotation_norm（原正式轮） | 60.6667 % | 30 | 每题5次均值；1轮 |  | 旧自动评分；已知提取问题 |
| AIME25 | rotation（原正式轮） | 60.6667 % | 30 | 每题5次均值；1轮 |  | 旧自动评分；已知提取问题 |

## T22　AIME25 原正式轮：四组最终提交纠正

**使用范围：** 评分纠错证据；旧S1K数据边界不因纠错消失。

**固定设置：** Qwen3-30B-A3B，原生 NVIDIA NVFP4 revision `2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`；reasoning：根 main.py 的 lighteval/vLLM、TP2、BF16 KV、eager、context=40960、batch/max_num_seqs=128、batched_tokens=4096、seed=1234、temperature=0.6、top_p=0.95、top_k=20、min_p=0、thinking；MMLU-Pro/GSM8K max_new_tokens=32768，LCB=38912；ARC：lm_eval、0-shot、raw acc（不是 acc_norm）、context=4096、seed=42。 同一批原始回答，不重新生成；最后明确最终提交，未完成回答计0。

**为何单独成表：** 与T21仅评分解释不同；修正值不是额外一次评测。其余11组未完成同口径复核，不拼接排名。

来源及完整配置：[aime_scoring_recomputed_20261015.md](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/aime_scoring_recomputed_20261015.md)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| AIME25 | existing_two_stage | 61.3333 % | 30 | 每题5次均值；1轮 |  | 原正式轮复核；与T21同回答，不重复计样本 |
| AIME25 | fixed_r64 | 70.0000 % | 30 | 每题5次均值；1轮 |  | 原正式轮复核；与T21同回答，不重复计样本 |
| AIME25 | matrix_norm | 66.0000 % | 30 | 每题5次均值；1轮 |  | 原正式轮复核；与T21同回答，不重复计样本 |
| AIME25 | native_nvfp4 | 70.6667 % | 30 | 每题5次均值；1轮 |  | 原正式轮复核；与T21同回答，不重复计样本 |

## T23　GPQA 旧严格 JSON 提取器分：评分诊断

**使用范围：** 用于说明评分误差，不能作为模型最终精度。

**固定设置：** Qwen3-30B-A3B原生NVFP4来源同上；只改48×128×3=18432个路由专家矩阵；非专家原样保留。MXFP4：E2M1、K维G32、E8M0、RNE；基础指数floor(log2(amax))-2；UOS c=7.25；G256/G512增加FP32二级scale。Qwen thinking、seed1234、T0.6/p0.95/k20/min_p0，context40960/output32768、batch256/batched_tokens8192、eager、BF16 KV。

**为何单独成表：** 与T09同一生成，差别仅答案提取；大量格式解析失败，不能与最终提交准确率混用。

来源及完整配置：[eight_model_summary_20261111.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/dualscale_gpu47_20261110T085025Z/eight_model_summary_20261111.json)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| GPQA Diamond | Dual-scale G256 | 13.0808 % | 198 | 每题10次均值；1轮 |  | 同T09原始回答；不是新重复 |
| GPQA Diamond | Dual-scale G512 | 15.8081 % | 198 | 每题10次均值；1轮 |  | 同T09原始回答；不是新重复 |
| GPQA Diamond | R16 构造 R32 + MXFP4 | 22.1212 % | 198 | 每题10次均值；1轮 |  | 同T09原始回答；不是新重复 |
| GPQA Diamond | R64 + MXFP4 | 26.9697 % | 198 | 每题10次均值；1轮 |  | 同T09原始回答；不是新重复 |
| GPQA Diamond | UOS c7.25 | 16.8687 % | 198 | 每题10次均值；1轮 |  | 同T09原始回答；不是新重复 |
| GPQA Diamond | 原生 NVFP4 | 15.5556 % | 198 | 每题10次均值；1轮 |  | 同T09原始回答；不是新重复 |
| GPQA Diamond | 标准 H32 + MXFP4 | 7.5758 % | 198 | 每题10次均值；1轮 |  | 同T09原始回答；不是新重复 |
| GPQA Diamond | 直接 MXFP4 | 28.0808 % | 198 | 每题10次均值；1轮 |  | 同T09原始回答；不是新重复 |

## T24　已隔离 Phase-A 旧分数：仅供溯源

**使用范围：** 明确不可用于论文精度因果主张；原始不利/失败结果仍保留。

**固定设置：** 早期E0–E7及candidate/adopted配置；E0用了chat wrapper，部分候选裸题面；旧摘要已QUARANTINED。

**为何单独成表：** prompt不一致，且旧MMLU结果被T18统一重测替代；不能合并进任何同口径结果表。

来源及完整配置：[summary.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/e2e_diag_reconstruction/phaseA_refactor_20260825T035730Z/quarantine_invalid_missing_chat_template.20260908T100300Z/phase_reports/summary.json)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| ARC-C | E0_native_nvfp4 | 52.2184 % | 1172 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| ARC-C | E1_direct_hif4 | 50.0000 % | 1172 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| ARC-C | E2_r64_only | 52.0478 % | 1172 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| ARC-C | E3_fusable | 50.4266 % | 1172 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| ARC-C | E4_fusable_r64 | 49.5734 % | 1172 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| ARC-C | E5_online | 50.7679 % | 1172 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| ARC-C | E6_online_diag_then_r64 | 52.8157 % | 1172 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| ARC-C | E7_online_r64_then_diag | 49.1468 % | 1172 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| ARC-E | E0_native_nvfp4 | 78.5774 % | 2376 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| ARC-E | E1_direct_hif4 | 77.4832 % | 2376 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| ARC-E | E2_r64_only | 78.6195 % | 2376 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| ARC-E | E3_fusable | 77.3990 % | 2376 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| ARC-E | E4_fusable_r64 | 77.9461 % | 2376 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| ARC-E | E5_online | 77.9882 % | 2376 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| ARC-E | E6_online_diag_then_r64 | 78.1987 % | 2376 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| ARC-E | E7_online_r64_then_diag | 77.5253 % | 2376 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| LiveCodeBench v6 | E0_native_nvfp4 | 54.2857 % | 175 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| LiveCodeBench v6 | E1_direct_hif4 | 19.4286 % | 175 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| LiveCodeBench v6 | E2_r64_only | 24.0000 % | 175 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| LiveCodeBench v6 | E3_fusable | 22.2857 % | 175 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| LiveCodeBench v6 | E4_fusable_r64 | 23.4286 % | 175 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| MMLU-Pro | E0_native_nvfp4 | 70.6667 % | 300 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| MMLU-Pro | E1_direct_hif4 | 71.3333 % | 300 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| MMLU-Pro | E2_r64_only | 68.0000 % | 300 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| MMLU-Pro | E3_fusable | 70.6667 % | 300 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| MMLU-Pro | E4_fusable_r64 | 70.3333 % | 300 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| MMLU-Pro | E5_online | 69.3333 % | 300 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| MMLU-Pro | E6_online_diag_then_r64 | 72.3333 % | 300 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |
| MMLU-Pro | E7_online_r64_then_diag | 69.3333 % | 300 | 单次结果 |  | artifact=adopted；prompt错位，不用于同条件排名 |

## T25　独立开发集：A/B、效率与完整 KL/CE 的选择结果

**使用范围：** 用于解释事先选定方法，不用于声称测试精度。

**固定设置：** 固定64 math（MATH）+64 code（CodeContests）；各域准确率及等权均值；与校准/正式测试隔离；一次开发评测。

**为何单独成表：** 选优数据不是最终测试集，不能与MMLU/LCB等正式指标合并；训练预算和候选池在各行列明。

来源及完整配置：[manifest.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_calibration/continue_20261016T135141Z/manifest.json) · [manifest.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_norm_efficiency/run_20261029T023839Z/manifest.json) · [selection.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_full_kl_ce/run_20261021T030031Z/development/full_kl/selection.json) · [selection.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_full_kl_ce/run_20261021T030031Z/development/hard_ce/selection.json)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| DEV-code | coverage50_e5 | 51.5625 % | 64 | 单次结果 |  | 用于模型选择；不是独立测试 |
| DEV-code | full_e5 | 53.1250 % | 64 | 单次结果 |  | 用于模型选择；不是独立测试 |
| DEV-code | full_kl/candidate_00 | 51.5625 % | 64 | 单次结果 | lr=1e-05；epoch=3 | 决选候选（epoch0重复引用B，不是新的训练结果） |
| DEV-code | full_kl/candidate_01 | 54.6875 % | 64 | 单次结果 | lr=3e-05；epoch=3 | 决选候选（epoch0重复引用B，不是新的训练结果） |
| DEV-code | full_kl/candidate_02 | 45.3125 % | 64 | 单次结果 | lr=0.0001；epoch=2 | 决选候选（epoch0重复引用B，不是新的训练结果） |
| DEV-code | full_kl/candidate_03 | 51.5625 % | 64 | 单次结果 | lr=0.0001；epoch=3 | 决选候选（epoch0重复引用B，不是新的训练结果） |
| DEV-code | full_kl/epoch_zero | 48.4375 % | 64 | 单次结果 | lr=0.0；epoch=0 | 决选候选（epoch0重复引用B，不是新的训练结果） |
| DEV-code | hard_ce/candidate_00 | 53.1250 % | 64 | 单次结果 | lr=1e-05；epoch=3 | 决选候选（epoch0重复引用B，不是新的训练结果） |
| DEV-code | hard_ce/candidate_01 | 56.2500 % | 64 | 单次结果 | lr=3e-05；epoch=3 | 决选候选（epoch0重复引用B，不是新的训练结果） |
| DEV-code | hard_ce/candidate_02 | 54.6875 % | 64 | 单次结果 | lr=0.0001；epoch=3 | 决选候选（epoch0重复引用B，不是新的训练结果） |
| DEV-code | hard_ce/candidate_03 | 50.0000 % | 64 | 单次结果 | lr=0.0001；epoch=4 | 决选候选（epoch0重复引用B，不是新的训练结果） |
| DEV-code | hard_ce/epoch_zero | 48.4375 % | 64 | 单次结果 | lr=0.0；epoch=0 | 决选候选（epoch0重复引用B，不是新的训练结果） |
| DEV-code | matrix_norm_native_traces | 48.4375 % | 64 | 单次结果 |  | 用于模型选择；不是独立测试 |
| DEV-code | matrix_norm_s1k | 50.0000 % | 64 | 单次结果 |  | 用于模型选择；不是独立测试 |
| DEV-code | random50_e5 | 51.5625 % | 64 | 单次结果 |  | 用于模型选择；不是独立测试 |
| DEV-math | coverage50_e5 | 95.3125 % | 64 | 单次结果 |  | 用于模型选择；不是独立测试 |
| DEV-math | full_e5 | 93.7500 % | 64 | 单次结果 |  | 用于模型选择；不是独立测试 |
| DEV-math | full_kl/candidate_00 | 95.3125 % | 64 | 单次结果 | lr=1e-05；epoch=3 | 决选候选（epoch0重复引用B，不是新的训练结果） |
| DEV-math | full_kl/candidate_01 | 95.3125 % | 64 | 单次结果 | lr=3e-05；epoch=3 | 决选候选（epoch0重复引用B，不是新的训练结果） |
| DEV-math | full_kl/candidate_02 | 95.3125 % | 64 | 单次结果 | lr=0.0001；epoch=2 | 决选候选（epoch0重复引用B，不是新的训练结果） |
| DEV-math | full_kl/candidate_03 | 95.3125 % | 64 | 单次结果 | lr=0.0001；epoch=3 | 决选候选（epoch0重复引用B，不是新的训练结果） |
| DEV-math | full_kl/epoch_zero | 95.3125 % | 64 | 单次结果 | lr=0.0；epoch=0 | 决选候选（epoch0重复引用B，不是新的训练结果） |
| DEV-math | hard_ce/candidate_00 | 93.7500 % | 64 | 单次结果 | lr=1e-05；epoch=3 | 决选候选（epoch0重复引用B，不是新的训练结果） |
| DEV-math | hard_ce/candidate_01 | 95.3125 % | 64 | 单次结果 | lr=3e-05；epoch=3 | 决选候选（epoch0重复引用B，不是新的训练结果） |
| DEV-math | hard_ce/candidate_02 | 93.7500 % | 64 | 单次结果 | lr=0.0001；epoch=3 | 决选候选（epoch0重复引用B，不是新的训练结果） |
| DEV-math | hard_ce/candidate_03 | 93.7500 % | 64 | 单次结果 | lr=0.0001；epoch=4 | 决选候选（epoch0重复引用B，不是新的训练结果） |
| DEV-math | hard_ce/epoch_zero | 95.3125 % | 64 | 单次结果 | lr=0.0；epoch=0 | 决选候选（epoch0重复引用B，不是新的训练结果） |
| DEV-math | matrix_norm_native_traces | 95.3125 % | 64 | 单次结果 |  | 用于模型选择；不是独立测试 |
| DEV-math | matrix_norm_s1k | 95.3125 % | 64 | 单次结果 |  | 用于模型选择；不是独立测试 |
| DEV-math | random50_e5 | 95.3125 % | 64 | 单次结果 |  | 用于模型选择；不是独立测试 |
| DEV-等权均值 | coverage50_e5 | 73.4375 % | 128 | 单次结果 |  | math/code各50%权重 |
| DEV-等权均值 | full_e5 | 73.4375 % | 128 | 单次结果 |  | math/code各50%权重 |
| DEV-等权均值 | matrix_norm_native_traces | 71.8750 % | 128 | 单次结果 |  | math/code各50%权重 |
| DEV-等权均值 | matrix_norm_s1k | 72.6562 % | 128 | 单次结果 |  | math/code各50%权重 |
| DEV-等权均值 | random50_e5 | 73.4375 % | 128 | 单次结果 |  | math/code各50%权重 |

## T26　KL 方向复用：真实 TP2 最终分布指标

**使用范围：** 一次训练/一次固定语料测量；token均值不是多次评测均值。

**固定设置：** 单层8/24/40，各5方案；seed42、lr1e-4、有效batch4、每配置7200秒、同router辅助损失；test有效KL token300847/NLL token300815；真实vLLM TP2。

**为何单独成表：** 不是自由生成benchmark；KL/NLL/PPL单位和分母不同于准确率，不与任务分数求平均。

来源及完整配置：[comparison.csv](../../../Native_NVFP4_HiF4_Linear_Puncture/results/kl_direction_reuse/long_v3_fourcard_20260918T164218Z/comparison.csv) · [EXPERIMENT_SUMMARY.md](../../../Native_NVFP4_HiF4_Linear_Puncture/results/kl_direction_reuse/long_v3_fourcard_20260918T164218Z/EXPERIMENT_SUMMARY.md)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 固定语料测试 | L08_cached_kl_k1 | 0.053394082 KL | 300847 | 单次结果 | 层8；cached_kl；刷新1epoch；steps=32 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L08_cached_kl_k1 | 1.1601223 NLL | 300815 | 单次结果 | 层8；cached_kl；刷新1epoch；steps=32 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L08_cached_kl_k1 | 3.1903235 PPL | 300815 | 单次结果 | 层8；cached_kl；刷新1epoch；steps=32 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L08_cached_kl_k2 | 0.057105303 KL | 300847 | 单次结果 | 层8；cached_kl；刷新2epoch；steps=48 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L08_cached_kl_k2 | 1.1472293 NLL | 300815 | 单次结果 | 层8；cached_kl；刷新2epoch；steps=48 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L08_cached_kl_k2 | 3.1494547 PPL | 300815 | 单次结果 | 层8；cached_kl；刷新2epoch；steps=48 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L08_cached_kl_k4 | 0.066584782 KL | 300847 | 单次结果 | 层8；cached_kl；刷新4epoch；steps=96 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L08_cached_kl_k4 | 1.1621591 NLL | 300815 | 单次结果 | 层8；cached_kl；刷新4epoch；steps=96 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L08_cached_kl_k4 | 3.1968281 PPL | 300815 | 单次结果 | 层8；cached_kl；刷新4epoch；steps=96 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L08_direct_kl | 0.052251637 KL | 300847 | 单次结果 | 层8；direct_kl；刷新0epoch；steps=32 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L08_direct_kl | 1.159917 NLL | 300815 | 单次结果 | 层8；direct_kl；刷新0epoch；steps=32 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L08_direct_kl | 3.1896686 PPL | 300815 | 单次结果 | 层8；direct_kl；刷新0epoch；steps=32 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L08_mse | 0.088103948 KL | 300847 | 单次结果 | 层8；mse；刷新0epoch；steps=891 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L08_mse | 1.2192609 NLL | 300815 | 单次结果 | 层8；mse；刷新0epoch；steps=891 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L08_mse | 3.3846852 PPL | 300815 | 单次结果 | 层8；mse；刷新0epoch；steps=891 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L24_cached_kl_k1 | 0.052386673 KL | 300847 | 单次结果 | 层24；cached_kl；刷新1epoch；steps=40 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L24_cached_kl_k1 | 1.1380931 NLL | 300815 | 单次结果 | 层24；cached_kl；刷新1epoch；steps=40 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L24_cached_kl_k1 | 3.1208116 PPL | 300815 | 单次结果 | 层24；cached_kl；刷新1epoch；steps=40 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L24_cached_kl_k2 | 0.080042111 KL | 300847 | 单次结果 | 层24；cached_kl；刷新2epoch；steps=80 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L24_cached_kl_k2 | 1.1707246 NLL | 300815 | 单次结果 | 层24；cached_kl；刷新2epoch；steps=80 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L24_cached_kl_k2 | 3.2243281 PPL | 300815 | 单次结果 | 层24；cached_kl；刷新2epoch；steps=80 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L24_cached_kl_k4 | 0.89028677 KL | 300847 | 单次结果 | 层24；cached_kl；刷新4epoch；steps=148 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L24_cached_kl_k4 | 1.6672107 NLL | 300815 | 单次结果 | 层24；cached_kl；刷新4epoch；steps=148 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L24_cached_kl_k4 | 5.2973713 PPL | 300815 | 单次结果 | 层24；cached_kl；刷新4epoch；steps=148 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L24_direct_kl | 0.055400506 KL | 300847 | 单次结果 | 层24；direct_kl；刷新0epoch；steps=48 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L24_direct_kl | 1.142514 NLL | 300815 | 单次结果 | 层24；direct_kl；刷新0epoch；steps=48 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L24_direct_kl | 3.1346391 PPL | 300815 | 单次结果 | 层24；direct_kl；刷新0epoch；steps=48 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L24_mse | 0.087831754 KL | 300847 | 单次结果 | 层24；mse；刷新0epoch；steps=912 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L24_mse | 1.2117932 NLL | 300815 | 单次结果 | 层24；mse；刷新0epoch；steps=912 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L24_mse | 3.3595034 PPL | 300815 | 单次结果 | 层24；mse；刷新0epoch；steps=912 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L40_cached_kl_k1 | 0.054569192 KL | 300847 | 单次结果 | 层40；cached_kl；刷新1epoch；steps=86 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L40_cached_kl_k1 | 1.1260469 NLL | 300815 | 单次结果 | 层40；cached_kl；刷新1epoch；steps=86 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L40_cached_kl_k1 | 3.0834433 PPL | 300815 | 单次结果 | 层40；cached_kl；刷新1epoch；steps=86 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L40_cached_kl_k2 | 0.059153915 KL | 300847 | 单次结果 | 层40；cached_kl；刷新2epoch；steps=144 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L40_cached_kl_k2 | 1.1133817 NLL | 300815 | 单次结果 | 层40；cached_kl；刷新2epoch；steps=144 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L40_cached_kl_k2 | 3.0446371 PPL | 300815 | 单次结果 | 层40；cached_kl；刷新2epoch；steps=144 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L40_cached_kl_k4 | 0.067459204 KL | 300847 | 单次结果 | 层40；cached_kl；刷新4epoch；steps=256 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L40_cached_kl_k4 | 1.1175926 NLL | 300815 | 单次结果 | 层40；cached_kl；刷新4epoch；steps=256 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L40_cached_kl_k4 | 3.0574846 PPL | 300815 | 单次结果 | 层40；cached_kl；刷新4epoch；steps=256 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L40_direct_kl | 0.054632861 KL | 300847 | 单次结果 | 层40；direct_kl；刷新0epoch；steps=95 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L40_direct_kl | 1.130542 NLL | 300815 | 单次结果 | 层40；direct_kl；刷新0epoch；steps=95 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L40_direct_kl | 3.0973347 PPL | 300815 | 单次结果 | 层40；direct_kl；刷新0epoch；steps=95 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L40_mse | 0.053881871 KL | 300847 | 单次结果 | 层40；mse；刷新0epoch；steps=919 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L40_mse | 1.1923041 NLL | 300815 | 单次结果 | 层40；mse；刷新0epoch；steps=919 | 1次实验；按token汇总；PPL=exp(NLL) |
| 固定语料测试 | L40_mse | 3.2946637 PPL | 300815 | 单次结果 | 层40；mse；刷新0epoch；steps=919 | 1次实验；按token汇总；PPL=exp(NLL) |

## T27　渐进式误差抵消：六种完整训练的留出指标

**使用范围：** 仅训练诊断；未定位完整下游评测，不能宣称任务提升。

**固定设置：** 同系列48层，baseline / direct / JVP / shuffled，λ=0/0.1/0.3；读取每层selected metrics。报告48层指标算术均值，不把各自student-prefix输入说成相同输入。

**为何单独成表：** 训练后局部重构/holdout指标，不是整模型最终分布或benchmark分数；与T26测量空间不同。

来源及完整配置：[protocol.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/progressive_error_cancellation/20260920T054500Z/baseline/protocol.json)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| holdout | baseline | 0.071852985 mse | 48 | 1次实验；48层/投影均值（非重复） | method=baseline；lambda=0.0 | 1次训练；48层均值；逐层原值在来源同级layers；不是48次重复 |
| holdout | baseline | 0.0044042085 router_loss | 48 | 1次实验；48层/投影均值（非重复） | method=baseline；lambda=0.0 | 1次训练；48层均值；逐层原值在来源同级layers；不是48次重复 |
| holdout | direct_l010 | 0.13058057 mse | 48 | 1次实验；48层/投影均值（非重复） | method=direct；lambda=0.1 | 1次训练；48层均值；逐层原值在来源同级layers；不是48次重复 |
| holdout | direct_l010 | 0.015981627 router_loss | 48 | 1次实验；48层/投影均值（非重复） | method=direct；lambda=0.1 | 1次训练；48层均值；逐层原值在来源同级layers；不是48次重复 |
| holdout | direct_l030 | 0.21079905 mse | 48 | 1次实验；48层/投影均值（非重复） | method=direct；lambda=0.3 | 1次训练；48层均值；逐层原值在来源同级layers；不是48次重复 |
| holdout | direct_l030 | 0.025495793 router_loss | 48 | 1次实验；48层/投影均值（非重复） | method=direct；lambda=0.3 | 1次训练；48层均值；逐层原值在来源同级layers；不是48次重复 |
| holdout | jvp_l010 | 0.11464111 mse | 48 | 1次实验；48层/投影均值（非重复） | method=jvp；lambda=0.1 | 1次训练；48层均值；逐层原值在来源同级layers；不是48次重复 |
| holdout | jvp_l010 | 0.012750183 router_loss | 48 | 1次实验；48层/投影均值（非重复） | method=jvp；lambda=0.1 | 1次训练；48层均值；逐层原值在来源同级layers；不是48次重复 |
| holdout | jvp_l030 | 0.19357372 mse | 48 | 1次实验；48层/投影均值（非重复） | method=jvp；lambda=0.3 | 1次训练；48层均值；逐层原值在来源同级layers；不是48次重复 |
| holdout | jvp_l030 | 0.020379923 router_loss | 48 | 1次实验；48层/投影均值（非重复） | method=jvp；lambda=0.3 | 1次训练；48层均值；逐层原值在来源同级layers；不是48次重复 |
| holdout | shuffled_l030 | 0.14955512 mse | 48 | 1次实验；48层/投影均值（非重复） | method=shuffled；lambda=0.3 | 1次训练；48层均值；逐层原值在来源同级layers；不是48次重复 |
| holdout | shuffled_l030 | 0.016200918 router_loss | 48 | 1次实验；48层/投影均值（非重复） | method=shuffled；lambda=0.3 | 1次训练；48层均值；逐层原值在来源同级layers；不是48次重复 |
| validation | baseline | 0.074253906 mse | 48 | 1次实验；48层/投影均值（非重复） | method=baseline；lambda=0.0 | 1次训练；48层均值；逐层原值在来源同级layers；不是48次重复 |
| validation | baseline | 0.0043256544 router_loss | 48 | 1次实验；48层/投影均值（非重复） | method=baseline；lambda=0.0 | 1次训练；48层均值；逐层原值在来源同级layers；不是48次重复 |
| validation | direct_l010 | 0.1263239 mse | 48 | 1次实验；48层/投影均值（非重复） | method=direct；lambda=0.1 | 1次训练；48层均值；逐层原值在来源同级layers；不是48次重复 |
| validation | direct_l010 | 0.014930377 router_loss | 48 | 1次实验；48层/投影均值（非重复） | method=direct；lambda=0.1 | 1次训练；48层均值；逐层原值在来源同级layers；不是48次重复 |
| validation | direct_l030 | 0.20980086 mse | 48 | 1次实验；48层/投影均值（非重复） | method=direct；lambda=0.3 | 1次训练；48层均值；逐层原值在来源同级layers；不是48次重复 |
| validation | direct_l030 | 0.023610125 router_loss | 48 | 1次实验；48层/投影均值（非重复） | method=direct；lambda=0.3 | 1次训练；48层均值；逐层原值在来源同级layers；不是48次重复 |
| validation | jvp_l010 | 0.11674408 mse | 48 | 1次实验；48层/投影均值（非重复） | method=jvp；lambda=0.1 | 1次训练；48层均值；逐层原值在来源同级layers；不是48次重复 |
| validation | jvp_l010 | 0.012184219 router_loss | 48 | 1次实验；48层/投影均值（非重复） | method=jvp；lambda=0.1 | 1次训练；48层均值；逐层原值在来源同级layers；不是48次重复 |
| validation | jvp_l030 | 0.19248673 mse | 48 | 1次实验；48层/投影均值（非重复） | method=jvp；lambda=0.3 | 1次训练；48层均值；逐层原值在来源同级layers；不是48次重复 |
| validation | jvp_l030 | 0.019277945 router_loss | 48 | 1次实验；48层/投影均值（非重复） | method=jvp；lambda=0.3 | 1次训练；48层均值；逐层原值在来源同级layers；不是48次重复 |
| validation | shuffled_l030 | 0.14743557 mse | 48 | 1次实验；48层/投影均值（非重复） | method=shuffled；lambda=0.3 | 1次训练；48层均值；逐层原值在来源同级layers；不是48次重复 |
| validation | shuffled_l030 | 0.015393095 router_loss | 48 | 1次实验；48层/投影均值（非重复） | method=shuffled；lambda=0.3 | 1次训练；48层均值；逐层原值在来源同级layers；不是48次重复 |

## T28　H-only 解析更新：CPU 单 Linear

**使用范围：** 一次诊断；解析法没有在validation上改善，保留负结果。

**固定设置：** Qwen3-30B原生第0层q_proj，4096×2048；CPU 2线程，seed42；train/val各4×64=256 token；Adam20步lr3e-4，解析法2次block sweep、damping0.01、线搜索1/0.5/0.25/0.125/0.0625。

**为何单独成表：** 样本、测量对象、预算不同于全模型正式重构；局部MSE最小化不是全模型下游。

来源及完整配置：[report.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_only_analytic/cpu_linear_20261105T085702Z/report.json)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| validation | adam/initial | 0.0010839745 NMSE | 256 | 单次结果 |  | 1次训练；迭代0；token聚合，不是256次重复 |
| validation | adam/last | 0.00094878507 NMSE | 256 | 单次结果 |  | 1次训练；迭代20；token聚合，不是256次重复 |
| validation | adam/selected | 0.00094878507 NMSE | 256 | 单次结果 |  | 1次训练；迭代20；token聚合，不是256次重复 |
| validation | analytic/initial | 0.0010839745 NMSE | 256 | 单次结果 |  | 1次训练；迭代0；token聚合，不是256次重复 |
| validation | analytic/last | 0.0011409662 NMSE | 256 | 单次结果 |  | 1次训练；迭代2；token聚合，不是256次重复 |
| validation | analytic/selected | 0.0010839745 NMSE | 256 | 单次结果 |  | 1次训练；迭代0；token聚合，不是256次重复 |

## T29　不等价重构：完整48层的局部统计

**使用范围：** 一次完整训练的48层算术均值；正式任务成绩见T07。

**固定设置：** 旧S1K128/32，seed42；E4初始化；48层×20epoch、batch4、AdamW lr1e-4、wd0、cosine；矩阵linear/group G64 × router top_partial/top_mass（k8,T1,λ1）。

**为何单独成表：** 不同student-prefix的逐层validation均值，不是同一端到端输入；top_partial与top_mass定义不同，损失值不能横比。

来源及完整配置：[summary.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/non_equivalent_reconstruction/full48_20260915T084300Z/group_top_mass/summary.json)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| validation/reconstruction_nmse | group_top_mass | 0.04246823 reconstruction_nmse | 48 | 1次实验；48层/投影均值（非重复） |  | 48层均值；一次训练，不是48次实验；top_partial可为负 |
| validation/reconstruction_nmse | group_top_partial | 0.063125867 reconstruction_nmse | 48 | 1次实验；48层/投影均值（非重复） |  | 48层均值；一次训练，不是48次实验；top_partial可为负 |
| validation/reconstruction_nmse | linear_top_mass | 0.046504761 reconstruction_nmse | 48 | 1次实验；48层/投影均值（非重复） |  | 48层均值；一次训练，不是48次实验；top_partial可为负 |
| validation/reconstruction_nmse | linear_top_partial | 0.062788302 reconstruction_nmse | 48 | 1次实验；48层/投影均值（非重复） |  | 48层均值；一次训练，不是48次实验；top_partial可为负 |
| validation/router_loss | group_top_mass | 9.3064144e-05 router_loss | 48 | 1次实验；48层/投影均值（非重复） |  | 48层均值；一次训练，不是48次实验；top_partial可为负 |
| validation/router_loss | group_top_partial | -0.10579627 router_loss | 48 | 1次实验；48层/投影均值（非重复） |  | 48层均值；一次训练，不是48次实验；top_partial可为负 |
| validation/router_loss | linear_top_mass | 0.0001015699 router_loss | 48 | 1次实验；48层/投影均值（非重复） |  | 48层均值；一次训练，不是48次实验；top_partial可为负 |
| validation/router_loss | linear_top_partial | -0.072746861 router_loss | 48 | 1次实验；48层/投影均值（非重复） |  | 48层均值；一次训练，不是48次实验；top_partial可为负 |
| validation/topk_overlap | group_top_mass | 0.97631193 topk_overlap | 48 | 1次实验；48层/投影均值（非重复） |  | 48层均值；一次训练，不是48次实验；top_partial可为负 |
| validation/topk_overlap | group_top_partial | 0.84081676 topk_overlap | 48 | 1次实验；48层/投影均值（非重复） |  | 48层均值；一次训练，不是48次实验；top_partial可为负 |
| validation/topk_overlap | linear_top_mass | 0.97474125 topk_overlap | 48 | 1次实验；48层/投影均值（非重复） |  | 48层均值；一次训练，不是48次实验；top_partial可为负 |
| validation/topk_overlap | linear_top_partial | 0.86823254 topk_overlap | 48 | 1次实验；48层/投影均值（非重复） |  | 48层均值；一次训练，不是48次实验；top_partial可为负 |
| validation/topk_set_match | group_top_mass | 0.81471631 topk_set_match | 48 | 1次实验；48层/投影均值（非重复） |  | 48层均值；一次训练，不是48次实验；top_partial可为负 |
| validation/topk_set_match | group_top_partial | 0.20419121 topk_set_match | 48 | 1次实验；48层/投影均值（非重复） |  | 48层均值；一次训练，不是48次实验；top_partial可为负 |
| validation/topk_set_match | linear_top_mass | 0.80299959 topk_set_match | 48 | 1次实验；48层/投影均值（非重复） |  | 48层均值；一次训练，不是48次实验；top_partial可为负 |
| validation/topk_set_match | linear_top_partial | 0.26616275 topk_set_match | 48 | 1次实验；48层/投影均值（非重复） |  | 48层均值；一次训练，不是48次实验；top_partial可为负 |

## T30　残差 LoRA：各自输入上的配对局部统计

**使用范围：** 一次训练、48层均值；不能由局部NMSE直接推出下游提升，正式任务见T08。

**固定设置：** Qwen3-30B；从E4初始化联合重构G64主分支与LoRA，旧group_top_mass仅作配对参照；旧S1K128/32；48×20epoch；rank4、alpha8；AdamW lr1e-4、wd0、cosine、batch4；router top8+tail KL λ1/T1。

**为何单独成表：** 每种LoRA有自己的progressive输入；只在该方法与其same-input group参照之间配对，不能把三组基线当同一输入。

来源及完整配置：[training_analysis.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/residual_lora_compensation/20260922T012547Z/training_analysis.json)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| validation/attention_residual_nmse | attention/group_same_input_mean | 0.046306336 attention_residual_nmse | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善46/48层；不是48个seed |
| validation/attention_residual_nmse | attention/selected_mean | 0.044828433 attention_residual_nmse | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善46/48层；不是48个seed |
| validation/attention_residual_nmse | both/group_same_input_mean | 0.04666938 attention_residual_nmse | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善37/48层；不是48个seed |
| validation/attention_residual_nmse | both/selected_mean | 0.046672129 attention_residual_nmse | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善37/48层；不是48个seed |
| validation/attention_residual_nmse | moe/group_same_input_mean | 0.046956691 attention_residual_nmse | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善15/48层；不是48个seed |
| validation/attention_residual_nmse | moe/selected_mean | 0.047604997 attention_residual_nmse | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善15/48层；不是48个seed |
| validation/objective | attention/group_same_input_mean | 0.043432154 objective | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善47/48层；不是48个seed |
| validation/objective | attention/selected_mean | 0.042365111 objective | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善47/48层；不是48个seed |
| validation/objective | both/group_same_input_mean | 0.043438453 objective | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善45/48层；不是48个seed |
| validation/objective | both/selected_mean | 0.042340436 objective | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善45/48层；不是48个seed |
| validation/objective | moe/group_same_input_mean | 0.043544006 objective | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善47/48层；不是48个seed |
| validation/objective | moe/selected_mean | 0.042384059 objective | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善47/48层；不是48个seed |
| validation/reconstruction_nmse | attention/group_same_input_mean | 0.043337272 reconstruction_nmse | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善47/48层；不是48个seed |
| validation/reconstruction_nmse | attention/selected_mean | 0.042271322 reconstruction_nmse | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善47/48层；不是48个seed |
| validation/reconstruction_nmse | both/group_same_input_mean | 0.043343 reconstruction_nmse | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善46/48层；不是48个seed |
| validation/reconstruction_nmse | both/selected_mean | 0.042242578 reconstruction_nmse | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善46/48层；不是48个seed |
| validation/reconstruction_nmse | moe/group_same_input_mean | 0.043447926 reconstruction_nmse | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善47/48层；不是48个seed |
| validation/reconstruction_nmse | moe/selected_mean | 0.042291089 reconstruction_nmse | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善47/48层；不是48个seed |
| validation/router_loss | attention/group_same_input_mean | 9.4881994e-05 router_loss | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善36/48层；不是48个seed |
| validation/router_loss | attention/selected_mean | 9.3789869e-05 router_loss | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善36/48层；不是48个seed |
| validation/router_loss | both/group_same_input_mean | 9.5452836e-05 router_loss | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善36/48层；不是48个seed |
| validation/router_loss | both/selected_mean | 9.7858387e-05 router_loss | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善36/48层；不是48个seed |
| validation/router_loss | moe/group_same_input_mean | 9.6080551e-05 router_loss | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善38/48层；不是48个seed |
| validation/router_loss | moe/selected_mean | 9.296957e-05 router_loss | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善38/48层；不是48个seed |
| validation/topk_overlap | attention/group_same_input_mean | 0.97602384 topk_overlap | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善39/48层；不是48个seed |
| validation/topk_overlap | attention/selected_mean | 0.97629432 topk_overlap | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善39/48层；不是48个seed |
| validation/topk_overlap | both/group_same_input_mean | 0.97593916 topk_overlap | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善36/48层；不是48个seed |
| validation/topk_overlap | both/selected_mean | 0.97599193 topk_overlap | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善36/48层；不是48个seed |
| validation/topk_overlap | moe/group_same_input_mean | 0.97588856 topk_overlap | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善43/48层；不是48个seed |
| validation/topk_overlap | moe/selected_mean | 0.97637781 topk_overlap | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善43/48层；不是48个seed |
| validation/topk_set_match | attention/group_same_input_mean | 0.81255716 topk_set_match | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善38/48层；不是48个seed |
| validation/topk_set_match | attention/selected_mean | 0.81459424 topk_set_match | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善38/48层；不是48个seed |
| validation/topk_set_match | both/group_same_input_mean | 0.81198333 topk_set_match | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善36/48层；不是48个seed |
| validation/topk_set_match | both/selected_mean | 0.81222447 topk_set_match | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善36/48层；不是48个seed |
| validation/topk_set_match | moe/group_same_input_mean | 0.81164624 topk_set_match | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善42/48层；不是48个seed |
| validation/topk_set_match | moe/selected_mean | 0.81524417 topk_set_match | 48 | 1次实验；48层/投影均值（非重复） | rank4；alpha8；attention/moe=786432参数，both=1572864 | 48层算术均值；该指标改善42/48层；不是48个seed |

## T31　4B 阈值研究：合成分布与权重局部 NMSE

**使用范围：** 历史报告级证据；样本数/重复次数未完整保留，不能宣称五次均值。

**固定设置：** Qwen3.5-4B，128目标Linear；标准e8/e4=4/2，scalar-only=3.75/1.875，noclip=3.5/1.75；权重S0-only及S0+e8/e4联合搜索；完整设置以历史报告为准。

**为何单独成表：** 合成分布和权重NMSE不是benchmark；与T19/T20同研究但统计对象不同；原始缓存已不在当前仓库。

来源及完整配置：[EXPERIMENT_REPORT.md](../../../HiFloat4/hif4_scale_threshold_optimization/experiments/EXPERIMENT_REPORT.md)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| gaussian | noclip | 0.007792 NMSE | 未留存 | 重复次数未核实；非五次均值 |  | 原报告精度；未恢复原始逐次结果 |
| gaussian | scalar_only | 0.00707 NMSE | 未留存 | 重复次数未核实；非五次均值 |  | 原报告精度；未恢复原始逐次结果 |
| gaussian | standard | 0.006905 NMSE | 未留存 | 重复次数未核实；非五次均值 |  | 原报告精度；未恢复原始逐次结果 |
| laplace | noclip | 0.009028 NMSE | 未留存 | 重复次数未核实；非五次均值 |  | 原报告精度；未恢复原始逐次结果 |
| laplace | scalar_only | 0.008176 NMSE | 未留存 | 重复次数未核实；非五次均值 |  | 原报告精度；未恢复原始逐次结果 |
| laplace | standard | 0.007984 NMSE | 未留存 | 重复次数未核实；非五次均值 |  | 原报告精度；未恢复原始逐次结果 |
| outlier | noclip | 0.008981 NMSE | 未留存 | 重复次数未核实；非五次均值 |  | 原报告精度；未恢复原始逐次结果 |
| outlier | scalar_only | 0.008463 NMSE | 未留存 | 重复次数未核实；非五次均值 |  | 原报告精度；未恢复原始逐次结果 |
| outlier | standard | 0.008348 NMSE | 未留存 | 重复次数未核实；非五次均值 |  | 原报告精度；未恢复原始逐次结果 |
| phase_boundary | noclip | 0.003095 NMSE | 未留存 | 重复次数未核实；非五次均值 |  | 原报告精度；未恢复原始逐次结果 |
| phase_boundary | scalar_only | 0.003095 NMSE | 未留存 | 重复次数未核实；非五次均值 |  | 原报告精度；未恢复原始逐次结果 |
| phase_boundary | standard | 0.003095 NMSE | 未留存 | 重复次数未核实；非五次均值 |  | 原报告精度；未恢复原始逐次结果 |
| student_t3 | noclip | 0.009707 NMSE | 未留存 | 重复次数未核实；非五次均值 |  | 原报告精度；未恢复原始逐次结果 |
| student_t3 | scalar_only | 0.009004 NMSE | 未留存 | 重复次数未核实；非五次均值 |  | 原报告精度；未恢复原始逐次结果 |
| student_t3 | standard | 0.008852 NMSE | 未留存 | 重复次数未核实；非五次均值 |  | 原报告精度；未恢复原始逐次结果 |
| 模型权重128Linear | S0_e8_e4_joint | 0.006395 NMSE | 128 | 重复次数未核实；非五次均值 |  | 128个Linear的均值；非128次重复 |
| 模型权重128Linear | S0_only | 0.006491 NMSE | 128 | 重复次数未核实；非五次均值 |  | 128个Linear的均值；非128次重复 |
| 模型权重128Linear | standard | 0.006966 NMSE | 128 | 重复次数未核实；非五次均值 |  | 128个Linear的均值；非128次重复 |

## T32　4B 阈值研究：激活校准及跨域迁移

**使用范围：** 历史局部诊断；重复次数未记录，完整token量以原报告为准。

**固定设置：** 同T31的模型/HiF4阈值；WikiText与S1K激活；standard、per-layer、per-block及离线oracle。

**为何单独成表：** activation与T31权重统计不同；oracle使用评测激活，不能列为可部署算法精度。

来源及完整配置：[EXPERIMENT_REPORT.md](../../../HiFloat4/hif4_scale_threshold_optimization/experiments/EXPERIMENT_REPORT.md)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| S1K | cross_domain_calibration | 0.008153 NMSE | 见报告 | 重复次数未核实；非五次均值 |  | 跨域校准检查；历史报告精度 |
| S1K | local_calibration | 0.008169 NMSE | 见报告 | 重复次数未核实；非五次均值 |  | 跨域校准检查；历史报告精度 |
| S1K | standard | 0.008106 NMSE | 见报告 | 重复次数未核实；非五次均值 |  | 跨域校准检查；历史报告精度 |
| WikiText | cross_domain_calibration | 0.008041 NMSE | 见报告 | 重复次数未核实；非五次均值 |  | 跨域校准检查；历史报告精度 |
| WikiText | local_calibration | 0.008024 NMSE | 见报告 | 重复次数未核实；非五次均值 |  | 跨域校准检查；历史报告精度 |
| WikiText | standard | 0.007991 NMSE | 见报告 | 重复次数未核实；非五次均值 |  | 跨域校准检查；历史报告精度 |
| WikiText激活 | oracle | 0.007303 NMSE | 见报告 | 重复次数未核实；非五次均值 |  | oracle不可部署；不是最终任务准确率 |
| WikiText激活 | oracle | 697.8 weighted_MSE | 见报告 | 重复次数未核实；非五次均值 |  | 完整 |
| WikiText激活 | per_block | 0.008081 NMSE | 见报告 | 重复次数未核实；非五次均值 |  | oracle不可部署；不是最终任务准确率 |
| WikiText激活 | per_block | 762.7 weighted_MSE | 见报告 | 重复次数未核实；非五次均值 |  | 完整 |
| WikiText激活 | per_layer | 0.008024 NMSE | 见报告 | 重复次数未核实；非五次均值 |  | oracle不可部署；不是最终任务准确率 |
| WikiText激活 | per_layer | 764.2 weighted_MSE | 见报告 | 重复次数未核实；非五次均值 |  | 完整 |
| WikiText激活 | standard | 0.007991 NMSE | 见报告 | 重复次数未核实；非五次均值 |  | oracle不可部署；不是最终任务准确率 |
| WikiText激活 | standard | 765.8 weighted_MSE | 见报告 | 重复次数未核实；非五次均值 |  | 完整 |

## T33　8B QAT 早期穿刺：DIAG / H4 / R64

**使用范围：** 历史局部实测；不能与30B终端任务表合并或称为35次重复。

**固定设置：** ISTA-DASLab/Qwen3-8B-FPQuant-QAT-NVFP4；5层(2/10/18/26/34)×7投影；train16/val16 prompts（4类各4条），每prompt每Linear最多64 token rows；Adam lr0.05、200步；d=2^z，z∈[-4,4]；HiF4 G64阈值7/4/2；保留源模型既有在线R16语义。

**为何单独成表：** 8B QAT与30B ModelOpt主线模型、原生rotation和数据完全不同；指标是35个Linear相对NMSE恢复的算术均值，非准确率。

来源及完整配置：[RESULTS_SUMMARY_2026-08-17.md](../../../archive/legacy/native_qwen3_8b/diag_gradient/RESULTS_SUMMARY_2026-08-17.md)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 35 Linear相对误差恢复 | DIAG | 4.5300 % | 35 | 1次实验；35层/投影均值（非重复） |  | 35投影均值；改善32/35；非能量加权GlobalRecovery；原始逐项见来源 |
| 35 Linear相对误差恢复 | DIAG→H4 | 4.9600 % | 35 | 1次实验；35层/投影均值（非重复） |  | 35投影均值；改善33/35；非能量加权GlobalRecovery；原始逐项见来源 |
| 35 Linear相对误差恢复 | DIAG→R64 | 8.6700 % | 35 | 1次实验；35层/投影均值（非重复） |  | 35投影均值；改善33/35；非能量加权GlobalRecovery；原始逐项见来源 |
| 35 Linear相对误差恢复 | H4 | 1.5200 % | 35 | 1次实验；35层/投影均值（非重复） |  | 35投影均值；改善28/35；非能量加权GlobalRecovery；原始逐项见来源 |
| 35 Linear相对误差恢复 | H4_G4共享DIAG | 4.1900 % | 35 | 1次实验；35层/投影均值（非重复） |  | 35投影均值；改善33/35；非能量加权GlobalRecovery；原始逐项见来源 |
| 35 Linear相对误差恢复 | H4→DIAG | 5.0800 % | 35 | 1次实验；35层/投影均值（非重复） |  | 35投影均值；改善34/35；非能量加权GlobalRecovery；原始逐项见来源 |
| 35 Linear相对误差恢复 | R64 | 1.4900 % | 35 | 1次实验；35层/投影均值（非重复） |  | 35投影均值；改善11/35；非能量加权GlobalRecovery；原始逐项见来源 |
| 35 Linear相对误差恢复 | R64→DIAG | 2.3100 % | 35 | 1次实验；35层/投影均值（非重复） |  | 35投影均值；改善15/35；非能量加权GlobalRecovery；原始逐项见来源 |
| 35 Linear相对误差恢复 | base | 0.0000 % | 35 | 1次实验；35层/投影均值（非重复） |  | 35投影均值；改善0/35；非能量加权GlobalRecovery；原始逐项见来源 |

## T34　论文新基线实现：CPU 真实投影短测

**使用范围：** 一次CPU测量；validation输出L2，非PPL/准确率；两目标格式单独标记。

**固定设置：** 原生Qwen3-30B layer0.q_proj的16输出行×2048输入；完整train1条12706 token、val1条6642 token；目标MXFP4/HiF4；GPTQ、SmoothQuant、AWQ；不是正式128条校准规模。

**为何单独成表：** 缩了输出行数/样本数的CPU实现短测，仅说明此局部路径可运行；不能代替正式TP2条件验证或正式精度。

来源及完整配置：[cpu-calibration.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/readiness_20261119T084223Z/cpu-calibration.json)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| S1K-val1 | hif4/awq | 3.8861887 output_L2 | 6642 | 单次结果 | AWQ ratio=0.35 | 1个局部投影；不是全模型评测 |
| S1K-val1 | hif4/gptq | 3.9399884 output_L2 | 6642 | 单次结果 |  | 1个局部投影；不是全模型评测 |
| S1K-val1 | hif4/smoothquant | 3.8885808 output_L2 | 6642 | 单次结果 |  | 1个局部投影；不是全模型评测 |
| S1K-val1 | mxfp4/awq | 3.6470408 output_L2 | 6642 | 单次结果 | AWQ ratio=0.45 | 1个局部投影；不是全模型评测 |
| S1K-val1 | mxfp4/gptq | 3.7548461 output_L2 | 6642 | 单次结果 |  | 1个局部投影；不是全模型评测 |
| S1K-val1 | mxfp4/smoothquant | 3.7428832 output_L2 | 6642 | 单次结果 |  | 1个局部投影；不是全模型评测 |

## T35　五轮尚未齐全：已完成的逐轮结果

**使用范围：** 可核验的中间记录；不纳入当前论文五次均值主表；未完成的单轮MMLU-Pro不填分。

**固定设置：** Qwen3-30B-A3B，原生 NVIDIA NVFP4 revision `2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`；reasoning：根 main.py 的 lighteval/vLLM、TP2、BF16 KV、eager、context=40960、batch/max_num_seqs=128、batched_tokens=4096、seed=1234、temperature=0.6、top_p=0.95、top_k=20、min_p=0、thinking；MMLU-Pro/GSM8K max_new_tokens=32768，LCB=38912；ARC：lm_eval、0-shot、raw acc（不是 acc_norm）、context=4096、seed=42。

**为何单独成表：** 算法/评测口径与T01相同，单独登记是因为五轮完整性尚不满足；这些单轮成绩不替代最终五轮均值。A/B的第1轮已在T02；这里只录新增轮，不把复用源再算一次。

来源及完整配置：[status.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/s1k_all_experts_20261118T032650Z/jobs/evaluate_r64_gptq/status.json) · [status.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/matrix_norm_repeats_20261118T034732Z/jobs/formal/status.json)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| ARC-C | b20/round_02 | 49.9147 % | 1172 | 单次结果 |  | 此轮全量完成；ARC为确定性评分，实际运行次数仍逐轮保留 |
| ARC-C | b20/round_03 | 49.9147 % | 1172 | 单次结果 |  | 此轮全量完成；ARC为确定性评分，实际运行次数仍逐轮保留 |
| ARC-C | b20/round_04 | 49.9147 % | 1172 | 单次结果 |  | 此轮全量完成；ARC为确定性评分，实际运行次数仍逐轮保留 |
| ARC-C | ours_a20/round_02 | 52.4744 % | 1172 | 单次结果 |  | 此轮全量完成；ARC为确定性评分，实际运行次数仍逐轮保留 |
| ARC-C | ours_a20/round_03 | 52.4744 % | 1172 | 单次结果 |  | 此轮全量完成；ARC为确定性评分，实际运行次数仍逐轮保留 |
| ARC-C | ours_a20/round_04 | 52.4744 % | 1172 | 单次结果 |  | 此轮全量完成；ARC为确定性评分，实际运行次数仍逐轮保留 |
| ARC-C | r64_gptq/round_01 | 49.8294 % | 1172 | 单次结果 | 干净S1K128条/1272746token；G64；block128；damp0.01；自然列序；32专家分批；所有token过所有专家；输出原top8 | 此轮全量完成；ARC为确定性评分，实际运行次数仍逐轮保留 |
| ARC-C | r64_gptq/round_02 | 49.8294 % | 1172 | 单次结果 | 干净S1K128条/1272746token；G64；block128；damp0.01；自然列序；32专家分批；所有token过所有专家；输出原top8 | 此轮全量完成；ARC为确定性评分，实际运行次数仍逐轮保留 |
| ARC-E | b20/round_02 | 77.5673 % | 2376 | 单次结果 |  | 此轮全量完成；ARC为确定性评分，实际运行次数仍逐轮保留 |
| ARC-E | b20/round_03 | 77.5673 % | 2376 | 单次结果 |  | 此轮全量完成；ARC为确定性评分，实际运行次数仍逐轮保留 |
| ARC-E | b20/round_04 | 77.5673 % | 2376 | 单次结果 |  | 此轮全量完成；ARC为确定性评分，实际运行次数仍逐轮保留 |
| ARC-E | ours_a20/round_02 | 79.1667 % | 2376 | 单次结果 |  | 此轮全量完成；ARC为确定性评分，实际运行次数仍逐轮保留 |
| ARC-E | ours_a20/round_03 | 79.1667 % | 2376 | 单次结果 |  | 此轮全量完成；ARC为确定性评分，实际运行次数仍逐轮保留 |
| ARC-E | ours_a20/round_04 | 79.1667 % | 2376 | 单次结果 |  | 此轮全量完成；ARC为确定性评分，实际运行次数仍逐轮保留 |
| ARC-E | r64_gptq/round_01 | 77.9882 % | 2376 | 单次结果 | 干净S1K128条/1272746token；G64；block128；damp0.01；自然列序；32专家分批；所有token过所有专家；输出原top8 | 此轮全量完成；ARC为确定性评分，实际运行次数仍逐轮保留 |
| ARC-E | r64_gptq/round_02 | 77.9882 % | 2376 | 单次结果 | 干净S1K128条/1272746token；G64；block128；damp0.01；自然列序；32专家分批；所有token过所有专家；输出原top8 | 此轮全量完成；ARC为确定性评分，实际运行次数仍逐轮保留 |
| GSM8K | b20/round_02 | 95.4511 % | 1319 | 单次结果 | GPTQ：干净S1K128条1272746token、所有token过所有专家；A20/B20配置见T02 | 此轮全量完成；最终五轮统计未完成 |
| GSM8K | b20/round_03 | 95.7544 % | 1319 | 单次结果 | GPTQ：干净S1K128条1272746token、所有token过所有专家；A20/B20配置见T02 | 此轮全量完成；最终五轮统计未完成 |
| GSM8K | b20/round_04 | 95.6027 % | 1319 | 单次结果 | GPTQ：干净S1K128条1272746token、所有token过所有专家；A20/B20配置见T02 | 此轮全量完成；最终五轮统计未完成 |
| GSM8K | ours_a20/round_02 | 95.4511 % | 1319 | 单次结果 | GPTQ：干净S1K128条1272746token、所有token过所有专家；A20/B20配置见T02 | 此轮全量完成；最终五轮统计未完成 |
| GSM8K | ours_a20/round_03 | 95.0720 % | 1319 | 单次结果 | GPTQ：干净S1K128条1272746token、所有token过所有专家；A20/B20配置见T02 | 此轮全量完成；最终五轮统计未完成 |
| GSM8K | ours_a20/round_04 | 95.0720 % | 1319 | 单次结果 | GPTQ：干净S1K128条1272746token、所有token过所有专家；A20/B20配置见T02 | 此轮全量完成；最终五轮统计未完成 |
| GSM8K | r64_gptq/round_01 | 94.9204 % | 1319 | 单次结果 | 干净S1K128条/1272746token；G64；block128；damp0.01；自然列序；32专家分批；所有token过所有专家；输出原top8 | 此轮全量完成；最终五轮统计未完成 |
| GSM8K | r64_gptq/round_02 | 94.9204 % | 1319 | 单次结果 | 干净S1K128条/1272746token；G64；block128；damp0.01；自然列序；32专家分批；所有token过所有专家；输出原top8 | 此轮全量完成；最终五轮统计未完成 |
| GSM8K | r64_gptq/round_03 | 94.9204 % | 1319 | 单次结果 | 干净S1K128条/1272746token；G64；block128；damp0.01；自然列序；32专家分批；所有token过所有专家；输出原top8 | 此轮全量完成；最终五轮统计未完成 |
| LiveCodeBench v6 | b20/round_02 | 53.7143 % | 175 | 单次结果 | GPTQ：干净S1K128条1272746token、所有token过所有专家；A20/B20配置见T02 | 此轮全量完成；最终五轮统计未完成 |
| LiveCodeBench v6 | b20/round_03 | 53.7143 % | 175 | 单次结果 | GPTQ：干净S1K128条1272746token、所有token过所有专家；A20/B20配置见T02 | 此轮全量完成；最终五轮统计未完成 |
| LiveCodeBench v6 | b20/round_04 | 53.7143 % | 175 | 单次结果 | GPTQ：干净S1K128条1272746token、所有token过所有专家；A20/B20配置见T02 | 此轮全量完成；最终五轮统计未完成 |
| LiveCodeBench v6 | ours_a20/round_02 | 53.1429 % | 175 | 单次结果 | GPTQ：干净S1K128条1272746token、所有token过所有专家；A20/B20配置见T02 | 此轮全量完成；最终五轮统计未完成 |
| LiveCodeBench v6 | ours_a20/round_03 | 50.2857 % | 175 | 单次结果 | GPTQ：干净S1K128条1272746token、所有token过所有专家；A20/B20配置见T02 | 此轮全量完成；最终五轮统计未完成 |
| LiveCodeBench v6 | ours_a20/round_04 | 50.2857 % | 175 | 单次结果 | GPTQ：干净S1K128条1272746token、所有token过所有专家；A20/B20配置见T02 | 此轮全量完成；最终五轮统计未完成 |
| LiveCodeBench v6 | r64_gptq/round_01 | 53.7143 % | 175 | 单次结果 | 干净S1K128条/1272746token；G64；block128；damp0.01；自然列序；32专家分批；所有token过所有专家；输出原top8 | 此轮全量完成；最终五轮统计未完成 |
| LiveCodeBench v6 | r64_gptq/round_02 | 54.8571 % | 175 | 单次结果 | 干净S1K128条/1272746token；G64；block128；damp0.01；自然列序；32专家分批；所有token过所有专家；输出原top8 | 此轮全量完成；最终五轮统计未完成 |
| LiveCodeBench v6 | r64_gptq/round_03 | 53.7143 % | 175 | 单次结果 | 干净S1K128条/1272746token；G64；block128；damp0.01；自然列序；32专家分批；所有token过所有专家；输出原top8 | 此轮全量完成；最终五轮统计未完成 |
| MMLU-Pro | b20/round_02 | 74.4348 % | 12032 | 单次结果 | GPTQ：干净S1K128条1272746token、所有token过所有专家；A20/B20配置见T02 | 此轮全量完成；最终五轮统计未完成 |
| MMLU-Pro | b20/round_03 | 74.4348 % | 12032 | 单次结果 | GPTQ：干净S1K128条1272746token、所有token过所有专家；A20/B20配置见T02 | 此轮全量完成；最终五轮统计未完成 |
| MMLU-Pro | b20/round_04 | 74.8338 % | 12032 | 单次结果 | GPTQ：干净S1K128条1272746token、所有token过所有专家；A20/B20配置见T02 | 此轮全量完成；最终五轮统计未完成 |
| MMLU-Pro | ours_a20/round_02 | 75.2660 % | 12032 | 单次结果 | GPTQ：干净S1K128条1272746token、所有token过所有专家；A20/B20配置见T02 | 此轮全量完成；最终五轮统计未完成 |
| MMLU-Pro | ours_a20/round_03 | 74.9751 % | 12032 | 单次结果 | GPTQ：干净S1K128条1272746token、所有token过所有专家；A20/B20配置见T02 | 此轮全量完成；最终五轮统计未完成 |
| MMLU-Pro | ours_a20/round_04 | 74.9834 % | 12032 | 单次结果 | GPTQ：干净S1K128条1272746token、所有token过所有专家；A20/B20配置见T02 | 此轮全量完成；最终五轮统计未完成 |
| MMLU-Pro | r64_gptq/round_01 | 75.0416 % | 12032 | 单次结果 | 干净S1K128条/1272746token；G64；block128；damp0.01；自然列序；32专家分批；所有token过所有专家；输出原top8 | 此轮全量完成；最终五轮统计未完成 |
| MMLU-Pro | r64_gptq/round_02 | 75.1496 % | 12032 | 单次结果 | 干净S1K128条/1272746token；G64；block128；damp0.01；自然列序；32专家分批；所有token过所有专家；输出原top8 | 此轮全量完成；最终五轮统计未完成 |
| MMLU-Pro | r64_gptq/round_03 | 74.8836 % | 12032 | 单次结果 | 干净S1K128条/1272746token；G64；block128；damp0.01；自然列序；32专家分批；所有token过所有专家；输出原top8 | 此轮全量完成；最终五轮统计未完成 |

## T36　MXFP4 二级 scale：MoE 局部性能

**使用范围：** 一次微基准批次的计时中位数，不是五次精度均值；迭代预算见同目录原性能JSON。

**固定设置：** TP4、A800；原报告冻结输入；G256/G512；tokens=1/256/8192；rank0/1/2/3；direct/candidate各包含local与with_output_sum；warmup5、计时20次，CUDA event统计中位数及分位数。

**为何单独成表：** 微基准不是整模型吞吐；局部与output_sum边界不同，不能和转换墙时/任务耗时合并。

来源及完整配置：[operator_performance_summary.csv](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/dualscale_gpu47_20261110T085025Z/operator_performance_summary.csv)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| MoE tokens=1 | dualscale_g256/candidate_local/rank0 | 0.25292799 ms | 1 | 20次计时中位数；不是5次精度均值 | rank0 | p10=0.25088000297546387；p90=0.2908160090446472；非端到端 |
| MoE tokens=1 | dualscale_g256/candidate_local/rank1 | 0.253952 ms | 1 | 20次计时中位数；不是5次精度均值 | rank1 | p10=0.2519040107727051；p90=0.29388800263404846；非端到端 |
| MoE tokens=1 | dualscale_g256/candidate_local/rank2 | 0.253952 ms | 1 | 20次计时中位数；不是5次精度均值 | rank2 | p10=0.25088000297546387；p90=0.2908160090446472；非端到端 |
| MoE tokens=1 | dualscale_g256/candidate_local/rank3 | 0.253952 ms | 1 | 20次计时中位数；不是5次精度均值 | rank3 | p10=0.25088000297546387；p90=0.2836480140686035；非端到端 |
| MoE tokens=1 | dualscale_g256/candidate_with_output_sum/rank0 | 0.29081599 ms | 1 | 20次计时中位数；不是5次精度均值 | rank0 | p10=0.28672000765800476；p90=0.3092480003833771；非端到端 |
| MoE tokens=1 | dualscale_g256/candidate_with_output_sum/rank1 | 0.291328 ms | 1 | 20次计时中位数；不是5次精度均值 | rank1 | p10=0.28467199206352234；p90=0.30720001459121704；非端到端 |
| MoE tokens=1 | dualscale_g256/candidate_with_output_sum/rank2 | 0.29183999 ms | 1 | 20次计时中位数；不是5次精度均值 | rank2 | p10=0.28672000765800476；p90=0.30822399258613586；非端到端 |
| MoE tokens=1 | dualscale_g256/candidate_with_output_sum/rank3 | 0.29183999 ms | 1 | 20次计时中位数；不是5次精度均值 | rank3 | p10=0.28569599986076355；p90=0.30822399258613586；非端到端 |
| MoE tokens=1 | dualscale_g256/direct_local/rank0 | 0.31744 ms | 1 | 20次计时中位数；不是5次精度均值 | rank0 | p10=0.31436800956726074；p90=0.3317759931087494；非端到端 |
| MoE tokens=1 | dualscale_g256/direct_local/rank1 | 0.31744 ms | 1 | 20次计时中位数；不是5次精度均值 | rank1 | p10=0.31436800956726074；p90=0.3317759931087494；非端到端 |
| MoE tokens=1 | dualscale_g256/direct_local/rank2 | 0.316416 ms | 1 | 20次计时中位数；不是5次精度均值 | rank2 | p10=0.31334400177001953；p90=0.32767999172210693；非端到端 |
| MoE tokens=1 | dualscale_g256/direct_local/rank3 | 0.316416 ms | 1 | 20次计时中位数；不是5次精度均值 | rank3 | p10=0.3123199939727783；p90=0.32870399951934814；非端到端 |
| MoE tokens=1 | dualscale_g256/direct_with_output_sum/rank0 | 0.35123199 ms | 1 | 20次计时中位数；不是5次精度均值 | rank0 | p10=0.3481599986553192；p90=0.35839998722076416；非端到端 |
| MoE tokens=1 | dualscale_g256/direct_with_output_sum/rank1 | 0.35020801 ms | 1 | 20次计时中位数；不是5次精度均值 | rank1 | p10=0.3481599986553192；p90=0.3614720106124878；非端到端 |
| MoE tokens=1 | dualscale_g256/direct_with_output_sum/rank2 | 0.35123199 ms | 1 | 20次计时中位数；不是5次精度均值 | rank2 | p10=0.3491840064525604；p90=0.35942399501800537；非端到端 |
| MoE tokens=1 | dualscale_g256/direct_with_output_sum/rank3 | 0.35123199 ms | 1 | 20次计时中位数；不是5次精度均值 | rank3 | p10=0.3481599986553192；p90=0.35942399501800537；非端到端 |
| MoE tokens=1 | dualscale_g512/candidate_local/rank0 | 0.242176 ms | 1 | 20次计时中位数；不是5次精度均值 | rank0 | p10=0.23654399812221527；p90=0.26419198513031006；非端到端 |
| MoE tokens=1 | dualscale_g512/candidate_local/rank1 | 0.2432 ms | 1 | 20次计时中位数；不是5次精度均值 | rank1 | p10=0.23654399812221527；p90=0.2693119943141937；非端到端 |
| MoE tokens=1 | dualscale_g512/candidate_local/rank2 | 0.243712 ms | 1 | 20次计时中位数；不是5次精度均值 | rank2 | p10=0.23654399812221527；p90=0.2662400007247925；非端到端 |
| MoE tokens=1 | dualscale_g512/candidate_local/rank3 | 0.2432 ms | 1 | 20次计时中位数；不是5次精度均值 | rank3 | p10=0.23654399812221527；p90=0.26521599292755127；非端到端 |
| MoE tokens=1 | dualscale_g512/candidate_with_output_sum/rank0 | 0.29184 ms | 1 | 20次计时中位数；不是5次精度均值 | rank0 | p10=0.28569599986076355；p90=0.30720001459121704；非端到端 |
| MoE tokens=1 | dualscale_g512/candidate_with_output_sum/rank1 | 0.29030401 ms | 1 | 20次计时中位数；不是5次精度均值 | rank1 | p10=0.28569599986076355；p90=0.30105599761009216；非端到端 |
| MoE tokens=1 | dualscale_g512/candidate_with_output_sum/rank2 | 0.291328 ms | 1 | 20次计时中位数；不是5次精度均值 | rank2 | p10=0.28569599986076355；p90=0.3020800054073334；非端到端 |
| MoE tokens=1 | dualscale_g512/candidate_with_output_sum/rank3 | 0.29183999 ms | 1 | 20次计时中位数；不是5次精度均值 | rank3 | p10=0.28672000765800476；p90=0.3020800054073334；非端到端 |
| MoE tokens=1 | dualscale_g512/direct_local/rank0 | 0.316928 ms | 1 | 20次计时中位数；不是5次精度均值 | rank0 | p10=0.31539198756217957；p90=0.32972800731658936；非端到端 |
| MoE tokens=1 | dualscale_g512/direct_local/rank1 | 0.316928 ms | 1 | 20次计时中位数；不是5次精度均值 | rank1 | p10=0.31436800956726074；p90=0.32767999172210693；非端到端 |
| MoE tokens=1 | dualscale_g512/direct_local/rank2 | 0.316928 ms | 1 | 20次计时中位数；不是5次精度均值 | rank2 | p10=0.31334400177001953；p90=0.32870399951934814；非端到端 |
| MoE tokens=1 | dualscale_g512/direct_local/rank3 | 0.31488 ms | 1 | 20次计时中位数；不是5次精度均值 | rank3 | p10=0.3123199939727783；p90=0.3266560137271881；非端到端 |
| MoE tokens=1 | dualscale_g512/direct_with_output_sum/rank0 | 0.352256 ms | 1 | 20次计时中位数；不是5次精度均值 | rank0 | p10=0.3491840064525604；p90=0.4055039882659912；非端到端 |
| MoE tokens=1 | dualscale_g512/direct_with_output_sum/rank1 | 0.352768 ms | 1 | 20次计时中位数；不是5次精度均值 | rank1 | p10=0.3491840064525604；p90=0.36454400420188904；非端到端 |
| MoE tokens=1 | dualscale_g512/direct_with_output_sum/rank2 | 0.352768 ms | 1 | 20次计时中位数；不是5次精度均值 | rank2 | p10=0.3491840064525604；p90=0.4034560024738312；非端到端 |
| MoE tokens=1 | dualscale_g512/direct_with_output_sum/rank3 | 0.352768 ms | 1 | 20次计时中位数；不是5次精度均值 | rank3 | p10=0.3481599986553192；p90=0.39526399970054626；非端到端 |
| MoE tokens=256 | dualscale_g256/candidate_local/rank0 | 1.301504 ms | 256 | 20次计时中位数；不是5次精度均值 | rank0 | p10=1.2963839769363403；p90=1.3137919902801514；非端到端 |
| MoE tokens=256 | dualscale_g256/candidate_local/rank1 | 1.302528 ms | 256 | 20次计时中位数；不是5次精度均值 | rank1 | p10=1.2974079847335815；p90=1.324031949043274；非端到端 |
| MoE tokens=256 | dualscale_g256/candidate_local/rank2 | 1.30304 ms | 256 | 20次计时中位数；不是5次精度均值 | rank2 | p10=1.2974079847335815；p90=1.3312000036239624；非端到端 |
| MoE tokens=256 | dualscale_g256/candidate_local/rank3 | 1.302528 ms | 256 | 20次计时中位数；不是5次精度均值 | rank3 | p10=1.2974079847335815；p90=1.3230079412460327；非端到端 |
| MoE tokens=256 | dualscale_g256/candidate_with_output_sum/rank0 | 1.459712 ms | 256 | 20次计时中位数；不是5次精度均值 | rank0 | p10=1.4561280012130737；p90=1.4725120067596436；非端到端 |
| MoE tokens=256 | dualscale_g256/candidate_with_output_sum/rank1 | 1.459712 ms | 256 | 20次计时中位数；不是5次精度均值 | rank1 | p10=1.457152009010315；p90=1.4663679599761963；非端到端 |
| MoE tokens=256 | dualscale_g256/candidate_with_output_sum/rank2 | 1.460224 ms | 256 | 20次计时中位数；不是5次精度均值 | rank2 | p10=1.4561280012130737；p90=1.4684159755706787；非端到端 |
| MoE tokens=256 | dualscale_g256/candidate_with_output_sum/rank3 | 1.4592 ms | 256 | 20次计时中位数；不是5次精度均值 | rank3 | p10=1.4561280012130737；p90=1.4663679599761963；非端到端 |
| MoE tokens=256 | dualscale_g256/direct_local/rank0 | 1.2759041 ms | 256 | 20次计时中位数；不是5次精度均值 | rank0 | p10=1.2718080282211304；p90=1.282047986984253；非端到端 |
| MoE tokens=256 | dualscale_g256/direct_local/rank1 | 1.2753921 ms | 256 | 20次计时中位数；不是5次精度均值 | rank1 | p10=1.269760012626648；p90=1.287168025970459；非端到端 |
| MoE tokens=256 | dualscale_g256/direct_local/rank2 | 1.270784 ms | 256 | 20次计时中位数；不是5次精度均值 | rank2 | p10=1.2666879892349243；p90=1.282047986984253；非端到端 |
| MoE tokens=256 | dualscale_g256/direct_local/rank3 | 1.273856 ms | 256 | 20次计时中位数；不是5次精度均值 | rank3 | p10=1.269760012626648；p90=1.2840960025787354；非端到端 |
| MoE tokens=256 | dualscale_g256/direct_with_output_sum/rank0 | 1.44128 ms | 256 | 20次计时中位数；不是5次精度均值 | rank0 | p10=1.4346239566802979；p90=1.4520319700241089；非端到端 |
| MoE tokens=256 | dualscale_g256/direct_with_output_sum/rank1 | 1.439232 ms | 256 | 20次计时中位数；不是5次精度均值 | rank1 | p10=1.435647964477539；p90=1.4540799856185913；非端到端 |
| MoE tokens=256 | dualscale_g256/direct_with_output_sum/rank2 | 1.440256 ms | 256 | 20次计时中位数；不是5次精度均值 | rank2 | p10=1.4335999488830566；p90=1.4551039934158325；非端到端 |
| MoE tokens=256 | dualscale_g256/direct_with_output_sum/rank3 | 1.44128 ms | 256 | 20次计时中位数；不是5次精度均值 | rank3 | p10=1.4346239566802979；p90=1.4510079622268677；非端到端 |
| MoE tokens=256 | dualscale_g512/candidate_local/rank0 | 1.192448 ms | 256 | 20次计时中位数；不是5次精度均值 | rank0 | p10=1.1868159770965576；p90=1.2318719625473022；非端到端 |
| MoE tokens=256 | dualscale_g512/candidate_local/rank1 | 1.191936 ms | 256 | 20次计时中位数；不是5次精度均值 | rank1 | p10=1.1857919692993164；p90=1.230847954750061；非端到端 |
| MoE tokens=256 | dualscale_g512/candidate_local/rank2 | 1.189888 ms | 256 | 20次计时中位数；不是5次精度均值 | rank2 | p10=1.1857919692993164；p90=1.2031999826431274；非端到端 |
| MoE tokens=256 | dualscale_g512/candidate_local/rank3 | 1.192448 ms | 256 | 20次计时中位数；不是5次精度均值 | rank3 | p10=1.1878399848937988；p90=1.230847954750061；非端到端 |
| MoE tokens=256 | dualscale_g512/candidate_with_output_sum/rank0 | 1.343488 ms | 256 | 20次计时中位数；不是5次精度均值 | rank0 | p10=1.3373440504074097；p90=1.3895679712295532；非端到端 |
| MoE tokens=256 | dualscale_g512/candidate_with_output_sum/rank1 | 1.345024 ms | 256 | 20次计时中位数；不是5次精度均值 | rank1 | p10=1.3383680582046509；p90=1.3905919790267944；非端到端 |
| MoE tokens=256 | dualscale_g512/candidate_with_output_sum/rank2 | 1.342464 ms | 256 | 20次计时中位数；不是5次精度均值 | rank2 | p10=1.3373440504074097；p90=1.3772799968719482；非端到端 |
| MoE tokens=256 | dualscale_g512/candidate_with_output_sum/rank3 | 1.344512 ms | 256 | 20次计时中位数；不是5次精度均值 | rank3 | p10=1.3383680582046509；p90=1.3916159868240356；非端到端 |
| MoE tokens=256 | dualscale_g512/direct_local/rank0 | 1.274368 ms | 256 | 20次计时中位数；不是5次精度均值 | rank0 | p10=1.2707840204238892；p90=1.2892160415649414；非端到端 |
| MoE tokens=256 | dualscale_g512/direct_local/rank1 | 1.2753921 ms | 256 | 20次计时中位数；不是5次精度均值 | rank1 | p10=1.269760012626648；p90=1.3066240549087524；非端到端 |
| MoE tokens=256 | dualscale_g512/direct_local/rank2 | 1.277952 ms | 256 | 20次计时中位数；不是5次精度均值 | rank2 | p10=1.2728320360183716；p90=1.2851200103759766；非端到端 |
| MoE tokens=256 | dualscale_g512/direct_local/rank3 | 1.2759041 ms | 256 | 20次计时中位数；不是5次精度均值 | rank3 | p10=1.2718080282211304；p90=1.2892160415649414；非端到端 |
| MoE tokens=256 | dualscale_g512/direct_with_output_sum/rank0 | 1.444352 ms | 256 | 20次计时中位数；不是5次精度均值 | rank0 | p10=1.439743995666504；p90=1.4940160512924194；非端到端 |
| MoE tokens=256 | dualscale_g512/direct_with_output_sum/rank1 | 1.44384 ms | 256 | 20次计时中位数；不是5次精度均值 | rank1 | p10=1.4366719722747803；p90=1.504256010055542；非端到端 |
| MoE tokens=256 | dualscale_g512/direct_with_output_sum/rank2 | 1.44128 ms | 256 | 20次计时中位数；不是5次精度均值 | rank2 | p10=1.4366719722747803；p90=1.4776320457458496；非端到端 |
| MoE tokens=256 | dualscale_g512/direct_with_output_sum/rank3 | 1.444864 ms | 256 | 20次计时中位数；不是5次精度均值 | rank3 | p10=1.4387199878692627；p90=1.4970879554748535；非端到端 |
| MoE tokens=8192 | dualscale_g256/candidate_local/rank0 | 9.330688 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank0 | p10=9.315327644348145；p90=9.357312202453613；非端到端 |
| MoE tokens=8192 | dualscale_g256/candidate_local/rank1 | 9.3291521 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank1 | p10=9.312255859375；p90=9.358336448669434；非端到端 |
| MoE tokens=8192 | dualscale_g256/candidate_local/rank2 | 9.336832 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank2 | p10=9.327615737915039；p90=9.362431526184082；非端到端 |
| MoE tokens=8192 | dualscale_g256/candidate_local/rank3 | 9.3306885 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank3 | p10=9.31430435180664；p90=9.369600296020508；非端到端 |
| MoE tokens=8192 | dualscale_g256/candidate_with_output_sum/rank0 | 11.864064 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank0 | p10=11.84563159942627；p90=12.004351615905762；非端到端 |
| MoE tokens=8192 | dualscale_g256/candidate_with_output_sum/rank1 | 11.8656 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank1 | p10=11.83129596710205；p90=12.005375862121582；非端到端 |
| MoE tokens=8192 | dualscale_g256/candidate_with_output_sum/rank2 | 11.872768 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank2 | p10=11.8538236618042；p90=12.020735740661621；非端到端 |
| MoE tokens=8192 | dualscale_g256/candidate_with_output_sum/rank3 | 11.866624 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank3 | p10=11.83948802947998；p90=11.911168098449707；非端到端 |
| MoE tokens=8192 | dualscale_g256/direct_local/rank0 | 57.355776 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank0 | p10=57.32454299926758；p90=57.49043273925781；非端到端 |
| MoE tokens=8192 | dualscale_g256/direct_local/rank1 | 57.631231 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank1 | p10=57.569278717041016；p90=57.76588821411133；非端到端 |
| MoE tokens=8192 | dualscale_g256/direct_local/rank2 | 57.324543 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank2 | p10=57.28665542602539；p90=57.371646881103516；非端到端 |
| MoE tokens=8192 | dualscale_g256/direct_local/rank3 | 57.344513 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank3 | p10=57.29484939575195；p90=57.436161041259766；非端到端 |
| MoE tokens=8192 | dualscale_g256/direct_with_output_sum/rank0 | 60.156416 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank0 | p10=60.06476974487305；p90=60.69862365722656；非端到端 |
| MoE tokens=8192 | dualscale_g256/direct_with_output_sum/rank1 | 60.15744 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank1 | p10=60.09241485595703；p90=60.52761459350586；非端到端 |
| MoE tokens=8192 | dualscale_g256/direct_with_output_sum/rank2 | 60.164608 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank2 | p10=60.09958267211914；p90=60.73241424560547；非端到端 |
| MoE tokens=8192 | dualscale_g256/direct_with_output_sum/rank3 | 60.155905 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank3 | p10=59.95724868774414；p90=60.693504333496094；非端到端 |
| MoE tokens=8192 | dualscale_g512/candidate_local/rank0 | 8.6533117 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank0 | p10=8.645631790161133；p90=8.70195198059082；非端到端 |
| MoE tokens=8192 | dualscale_g512/candidate_local/rank1 | 8.658432 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank1 | p10=8.651776313781738；p90=8.706048011779785；非端到端 |
| MoE tokens=8192 | dualscale_g512/candidate_local/rank2 | 8.652288 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank2 | p10=8.643584251403809；p90=8.685567855834961；非端到端 |
| MoE tokens=8192 | dualscale_g512/candidate_local/rank3 | 8.6594563 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank3 | p10=8.652799606323242；p90=8.70809555053711；非端到端 |
| MoE tokens=8192 | dualscale_g512/candidate_with_output_sum/rank0 | 11.190784 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank0 | p10=11.157504081726074；p90=11.35308837890625；非端到端 |
| MoE tokens=8192 | dualscale_g512/candidate_with_output_sum/rank1 | 11.188224 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank1 | p10=11.166720390319824；p90=11.392000198364258；非端到端 |
| MoE tokens=8192 | dualscale_g512/candidate_with_output_sum/rank2 | 11.18208 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank2 | p10=11.171839714050293；p90=11.302911758422852；非端到端 |
| MoE tokens=8192 | dualscale_g512/candidate_with_output_sum/rank3 | 11.191808 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank3 | p10=11.166720390319824；p90=11.402239799499512；非端到端 |
| MoE tokens=8192 | dualscale_g512/direct_local/rank0 | 57.347584 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank0 | p10=57.26515197753906；p90=57.51295852661133；非端到端 |
| MoE tokens=8192 | dualscale_g512/direct_local/rank1 | 57.630207 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank1 | p10=57.53855895996094；p90=57.81196975708008；非端到端 |
| MoE tokens=8192 | dualscale_g512/direct_local/rank2 | 57.381887 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank2 | p10=57.32966232299805；p90=57.53343963623047；非端到端 |
| MoE tokens=8192 | dualscale_g512/direct_local/rank3 | 57.308672 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank3 | p10=57.27743911743164；p90=57.378814697265625；非端到端 |
| MoE tokens=8192 | dualscale_g512/direct_with_output_sum/rank0 | 60.123648 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank0 | p10=60.054527282714844；p90=60.223487854003906；非端到端 |
| MoE tokens=8192 | dualscale_g512/direct_with_output_sum/rank1 | 60.125185 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank1 | p10=60.06681442260742；p90=60.2501106262207；非端到端 |
| MoE tokens=8192 | dualscale_g512/direct_with_output_sum/rank2 | 60.099072 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank2 | p10=60.042240142822266；p90=60.221439361572266；非端到端 |
| MoE tokens=8192 | dualscale_g512/direct_with_output_sum/rank3 | 60.12928 ms | 8192 | 20次计时中位数；不是5次精度均值 | rank3 | p10=60.05759811401367；p90=60.31564712524414；非端到端 |

## T37　MXFP4 单卡局部算子：R64 / H32 / R16x2 / UOS

**使用范围：** 局部性能诊断；不从单rank或不同核实现计时推算全模型生成加速。

**固定设置：** A800；真实第0层TP4 rank0切片、128专家；固定输入/路由；tokens=1/256/8192；软件QDQ、BF16 GEMM；5次预热、20次CUDA event计时。

**为何单独成表：** T36包含4个rank及TP输出SUM；此表是单rank完整MoE，不含TP输出SUM；每轮配对的native/direct计时各自保留。

来源及完整配置：[r64_gpu_validation.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/direct_gpu03_formal_20261109T100620Z/r64_gpu_validation.json) · [fht32_gpu_validation.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/fht32_tp4_20261110T032038Z/fht32_gpu_validation.json) · [r16x2_gpu_validation.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/r16x2_gpu03_20261110T040449Z/r16x2_gpu_validation.json) · [direct_uos_c7p25_gpu_validation.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/uos_c7p25_gpu03_20261110T074810Z/direct_uos_c7p25_gpu_validation.json)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| MoE tokens=1 | direct_uos_c7p25/candidate_moe | 0.316416 ms | 1 | 20次计时中位数；不是精度均值 | warmup=5；repeats=20 | 固定本轮输入；无TP输出SUM |
| MoE tokens=1 | direct_uos_c7p25/direct_moe | 0.321024 ms | 1 | 20次计时中位数；不是精度均值 | warmup=5；repeats=20 | 固定本轮输入；无TP输出SUM |
| MoE tokens=1 | fht32/candidate_moe | 0.31436801 ms | 1 | 20次计时中位数；不是精度均值 | warmup=5；repeats=20 | 固定本轮输入；无TP输出SUM |
| MoE tokens=1 | fht32/native_moe | 1.00864 ms | 1 | 20次计时中位数；不是精度均值 | warmup=5；repeats=20 | 固定本轮输入；无TP输出SUM |
| MoE tokens=1 | r16x2/candidate_moe | 0.31795201 ms | 1 | 20次计时中位数；不是精度均值 | warmup=5；repeats=20 | 固定本轮输入；无TP输出SUM |
| MoE tokens=1 | r16x2/native_moe | 1.048064 ms | 1 | 20次计时中位数；不是精度均值 | warmup=5；repeats=20 | 固定本轮输入；无TP输出SUM |
| MoE tokens=1 | r64/candidate_moe | 0.31385601 ms | 1 | 20次计时中位数；不是精度均值 | warmup=5；repeats=20 | 固定本轮输入；无TP输出SUM |
| MoE tokens=1 | r64/native_moe | 0.99737602 ms | 1 | 20次计时中位数；不是精度均值 | warmup=5；repeats=20 | 固定本轮输入；无TP输出SUM |
| MoE tokens=256 | direct_uos_c7p25/candidate_moe | 1.277952 ms | 256 | 20次计时中位数；不是精度均值 | warmup=5；repeats=20 | 固定本轮输入；无TP输出SUM |
| MoE tokens=256 | direct_uos_c7p25/direct_moe | 1.278976 ms | 256 | 20次计时中位数；不是精度均值 | warmup=5；repeats=20 | 固定本轮输入；无TP输出SUM |
| MoE tokens=256 | fht32/candidate_moe | 1.2748801 ms | 256 | 20次计时中位数；不是精度均值 | warmup=5；repeats=20 | 固定本轮输入；无TP输出SUM |
| MoE tokens=256 | fht32/native_moe | 1.912832 ms | 256 | 20次计时中位数；不是精度均值 | warmup=5；repeats=20 | 固定本轮输入；无TP输出SUM |
| MoE tokens=256 | r16x2/candidate_moe | 1.283072 ms | 256 | 20次计时中位数；不是精度均值 | warmup=5；repeats=20 | 固定本轮输入；无TP输出SUM |
| MoE tokens=256 | r16x2/native_moe | 1.943552 ms | 256 | 20次计时中位数；不是精度均值 | warmup=5；repeats=20 | 固定本轮输入；无TP输出SUM |
| MoE tokens=256 | r64/candidate_moe | 1.270272 ms | 256 | 20次计时中位数；不是精度均值 | warmup=5；repeats=20 | 固定本轮输入；无TP输出SUM |
| MoE tokens=256 | r64/native_moe | 1.903616 ms | 256 | 20次计时中位数；不是精度均值 | warmup=5；repeats=20 | 固定本轮输入；无TP输出SUM |
| MoE tokens=8192 | direct_uos_c7p25/candidate_moe | 52.170752 ms | 8192 | 20次计时中位数；不是精度均值 | warmup=5；repeats=20 | 固定本轮输入；无TP输出SUM |
| MoE tokens=8192 | direct_uos_c7p25/direct_moe | 52.150272 ms | 8192 | 20次计时中位数；不是精度均值 | warmup=5；repeats=20 | 固定本轮输入；无TP输出SUM |
| MoE tokens=8192 | fht32/candidate_moe | 52.090368 ms | 8192 | 20次计时中位数；不是精度均值 | warmup=5；repeats=20 | 固定本轮输入；无TP输出SUM |
| MoE tokens=8192 | fht32/native_moe | 56.738815 ms | 8192 | 20次计时中位数；不是精度均值 | warmup=5；repeats=20 | 固定本轮输入；无TP输出SUM |
| MoE tokens=8192 | r16x2/candidate_moe | 52.18816 ms | 8192 | 20次计时中位数；不是精度均值 | warmup=5；repeats=20 | 固定本轮输入；无TP输出SUM |
| MoE tokens=8192 | r16x2/native_moe | 56.841728 ms | 8192 | 20次计时中位数；不是精度均值 | warmup=5；repeats=20 | 固定本轮输入；无TP输出SUM |
| MoE tokens=8192 | r64/candidate_moe | 51.726336 ms | 8192 | 20次计时中位数；不是精度均值 | warmup=5；repeats=20 | 固定本轮输入；无TP输出SUM |
| MoE tokens=8192 | r64/native_moe | 56.754688 ms | 8192 | 20次计时中位数；不是精度均值 | warmup=5；repeats=20 | 固定本轮输入；无TP输出SUM |

## T38　Ban 自生成校准早期6专家预算：只完成两轮

**使用范围：** 历史中间结果，不当作完整三轮或五轮均值，不用于完整配对排名。

**固定设置：** Qwen3-30B-A3B，原生 NVIDIA NVFP4 revision `2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`；reasoning：根 main.py 的 lighteval/vLLM、TP2、BF16 KV、eager、context=40960、batch/max_num_seqs=128、batched_tokens=4096、seed=1234、temperature=0.6、top_p=0.95、top_k=20、min_p=0、thinking；MMLU-Pro/GSM8K max_new_tokens=32768，LCB=38912；ARC：lm_eval、0-shot、raw acc（不是 acc_norm）、context=4096、seed=42。底座旧matrix_norm；自生成512×2048窗口；k_min5，平均专家约6.08105；计划3轮LCB。

**为何单独成表：** 与T16目标6.4不同；T15是C4约6.07；此队列第三轮失败且其余方法未齐，单独保留已完成轮。

来源及完整配置：[report.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/continue_20261029T020854Z/report.json)。

| 测试数据／指标域 | 优化方案 | 结果 | 题数／统计单位数 | 几次均值 | 方法超参／变化项 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| LiveCodeBench v6 | trace校准/round1 | 52.5714 % | 175 | 单次结果 |  | 单轮175题完整；尚缺第3轮和配对方法，未算最终均值 |
| LiveCodeBench v6 | trace校准/round2 | 52.5714 % | 175 | 单次结果 |  | 单轮175题完整；尚缺第3轮和配对方法，未算最终均值 |

## C　转换、训练与测量成本

每行均为**1次该阶段**；未测五次成本均值。内部优化、准备、进程wall和评测耗时不是同一计时边界，包含关系已注明，不能相加重复收费，也不据此直接宣称不同算法的同条件加速比。共享自生成数据仅计一次。完整成本核算（包括失败与短测）见[效率实验costs.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_norm_efficiency/run_20261029T023839Z/costs.json)。

| 方法 | 阶段/计时字段 | 秒 | 小时 | 几次 | GPU数 | 边界 | 来源 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 旧S1K/norm | 20epoch正式重构 | 68515.253 | 19.0320 | 1次该阶段 | 1 | 原报告内部计时；筛选与正式分列；不含端到端评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/comparison.json) |
| 旧S1K/norm | 第0层LR筛选 | 834.287 | 0.2317 | 1次该阶段 | 1 | 原报告内部计时；筛选与正式分列；不含端到端评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/comparison.json) |
| 旧S1K/norm_kl | 20epoch正式重构 | 67944.149 | 18.8734 | 1次该阶段 | 1 | 原报告内部计时；筛选与正式分列；不含端到端评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/comparison.json) |
| 旧S1K/norm_kl | 第0层LR筛选 | 835.917 | 0.2322 | 1次该阶段 | 1 | 原报告内部计时；筛选与正式分列；不含端到端评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/comparison.json) |
| 旧S1K/diag | 20epoch正式重构 | 72423.271 | 20.1176 | 1次该阶段 | 1 | 原报告内部计时；筛选与正式分列；不含端到端评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/comparison.json) |
| 旧S1K/diag | 第0层LR筛选 | 878.651 | 0.2441 | 1次该阶段 | 1 | 原报告内部计时；筛选与正式分列；不含端到端评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/comparison.json) |
| 旧S1K/rotation | 20epoch正式重构 | 93018.296 | 25.8384 | 1次该阶段 | 1 | 原报告内部计时；筛选与正式分列；不含端到端评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/comparison.json) |
| 旧S1K/rotation | 第0层LR筛选 | 1106.978 | 0.3075 | 1次该阶段 | 1 | 原报告内部计时；筛选与正式分列；不含端到端评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/comparison.json) |
| 旧S1K/rotation_norm | 20epoch正式重构 | 94323.441 | 26.2010 | 1次该阶段 | 1 | 原报告内部计时；筛选与正式分列；不含端到端评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/comparison.json) |
| 旧S1K/rotation_norm | 第0层LR筛选 | 1128.878 | 0.3136 | 1次该阶段 | 1 | 原报告内部计时；筛选与正式分列；不含端到端评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/comparison.json) |
| 旧S1K/rotation_norm_kl | 20epoch正式重构 | 94151.085 | 26.1531 | 1次该阶段 | 1 | 原报告内部计时；筛选与正式分列；不含端到端评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/comparison.json) |
| 旧S1K/rotation_norm_kl | 第0层LR筛选 | 1123.078 | 0.3120 | 1次该阶段 | 1 | 原报告内部计时；筛选与正式分列；不含端到端评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/comparison.json) |
| 旧S1K/matrix | 20epoch正式重构 | 72255.419 | 20.0709 | 1次该阶段 | 1 | 原报告内部计时；筛选与正式分列；不含端到端评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/comparison.json) |
| 旧S1K/matrix | 第0层LR筛选 | 875.524 | 0.2432 | 1次该阶段 | 1 | 原报告内部计时；筛选与正式分列；不含端到端评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/comparison.json) |
| 旧S1K/matrix_norm | 20epoch正式重构 | 72146.306 | 20.0406 | 1次该阶段 | 1 | 原报告内部计时；筛选与正式分列；不含端到端评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/comparison.json) |
| 旧S1K/matrix_norm | 第0层LR筛选 | 874.946 | 0.2430 | 1次该阶段 | 1 | 原报告内部计时；筛选与正式分列；不含端到端评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/comparison.json) |
| 旧S1K/matrix_norm_kl | 20epoch正式重构 | 73014.675 | 20.2819 | 1次该阶段 | 1 | 原报告内部计时；筛选与正式分列；不含端到端评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/comparison.json) |
| 旧S1K/matrix_norm_kl | 第0层LR筛选 | 893.383 | 0.2482 | 1次该阶段 | 1 | 原报告内部计时；筛选与正式分列；不含端到端评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/comparison.json) |
| 旧S1K/current | 20epoch正式重构 | 72941.673 | 20.2616 | 1次该阶段 | 1 | 原报告内部计时；筛选与正式分列；不含端到端评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/comparison.json) |
| 旧S1K/current | 第0层LR筛选 | 883.142 | 0.2453 | 1次该阶段 | 1 | 原报告内部计时；筛选与正式分列；不含端到端评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/comparison.json) |
| 旧S1K/current_kl | 20epoch正式重构 | 72051.195 | 20.0142 | 1次该阶段 | 1 | 原报告内部计时；筛选与正式分列；不含端到端评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/comparison.json) |
| 旧S1K/current_kl | 第0层LR筛选 | 880.419 | 0.2446 | 1次该阶段 | 1 | 原报告内部计时；筛选与正式分列；不含端到端评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/comparison.json) |
| matrix_norm_s1k | 干净数据20epoch重构 | 70872.967 | 19.6869 | 1次该阶段 | 1 | comparison.calibration_compute.training_seconds；不含共享生成、额外router阶段或评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_calibration/continue_20261016T135141Z/comparison.json) |
| matrix_norm_native_traces | 干净数据20epoch重构 | 81175.302 | 22.5487 | 1次该阶段 | 1 | comparison.calibration_compute.training_seconds；不含共享生成、额外router阶段或评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_calibration/continue_20261016T135141Z/comparison.json) |
| coverage50_e5 | optimization_seconds | 13753.940 | 3.8205 | 1次该阶段 | 1 | 四种计时互有包含，不相加；不含共享生成/筛选/评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_norm_efficiency/run_20261029T023839Z/training/coverage50_e5/summary.json) |
| coverage50_e5 | preparation_seconds | 0.665 | 0.0002 | 1次该阶段 | 1 | 四种计时互有包含，不相加；不含共享生成/筛选/评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_norm_efficiency/run_20261029T023839Z/training/coverage50_e5/summary.json) |
| coverage50_e5 | legacy_comparable_seconds | 13753.945 | 3.8205 | 1次该阶段 | 1 | 四种计时互有包含，不相加；不含共享生成/筛选/评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_norm_efficiency/run_20261029T023839Z/training/coverage50_e5/summary.json) |
| coverage50_e5 | wall_seconds | 13754.647 | 3.8207 | 1次该阶段 | 1 | 四种计时互有包含，不相加；不含共享生成/筛选/评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_norm_efficiency/run_20261029T023839Z/training/coverage50_e5/summary.json) |
| full_e5 | optimization_seconds | 23123.015 | 6.4231 | 1次该阶段 | 1 | 四种计时互有包含，不相加；不含共享生成/筛选/评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_norm_efficiency/run_20261029T023839Z/training/full_e5/summary.json) |
| full_e5 | preparation_seconds | 0.718 | 0.0002 | 1次该阶段 | 1 | 四种计时互有包含，不相加；不含共享生成/筛选/评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_norm_efficiency/run_20261029T023839Z/training/full_e5/summary.json) |
| full_e5 | legacy_comparable_seconds | 23123.022 | 6.4231 | 1次该阶段 | 1 | 四种计时互有包含，不相加；不含共享生成/筛选/评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_norm_efficiency/run_20261029T023839Z/training/full_e5/summary.json) |
| full_e5 | wall_seconds | 23123.859 | 6.4233 | 1次该阶段 | 1 | 四种计时互有包含，不相加；不含共享生成/筛选/评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_norm_efficiency/run_20261029T023839Z/training/full_e5/summary.json) |
| random50_e5 | optimization_seconds | 13740.916 | 3.8169 | 1次该阶段 | 1 | 四种计时互有包含，不相加；不含共享生成/筛选/评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_norm_efficiency/run_20261029T023839Z/training/random50_e5/summary.json) |
| random50_e5 | preparation_seconds | 1.839 | 0.0005 | 1次该阶段 | 1 | 四种计时互有包含，不相加；不含共享生成/筛选/评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_norm_efficiency/run_20261029T023839Z/training/random50_e5/summary.json) |
| random50_e5 | legacy_comparable_seconds | 13740.921 | 3.8169 | 1次该阶段 | 1 | 四种计时互有包含，不相加；不含共享生成/筛选/评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_norm_efficiency/run_20261029T023839Z/training/random50_e5/summary.json) |
| random50_e5 | wall_seconds | 13742.787 | 3.8174 | 1次该阶段 | 1 | 四种计时互有包含，不相加；不含共享生成/筛选/评测 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_norm_efficiency/run_20261029T023839Z/training/random50_e5/summary.json) |
| B数据生成（A/B及效率共享） | 顶层wall | 35257.403 | 9.7937 | 1次该阶段 | 2 | 共享一次，不按每候选重复收费；GPU-hours=2×wall/3600，不是利用率 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_norm_efficiency/run_20261029T023839Z/costs.json) |
| r64_scale | 正式转换内部计时 | 433.744 | 0.1205 | 1次该阶段 | 1 | conversion.seconds；格式/校准不同；不含正式评测；非五次成本均值 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/run_20261116T103300Z/models/r64_scale/conversion.json) |
| r64_gptq | 正式转换内部计时 | 55710.054 | 15.4750 | 1次该阶段 | 1 | conversion.seconds；格式/校准不同；不含正式评测；非五次成本均值 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/s1k_all_experts_20261118T032650Z/models/r64_gptq/conversion.json) |
| r64_gptq | 转换顶层wall | 55715.351 | 15.4765 | 1次该阶段 | 1 | 包含上行内部计时，不重复相加；单次转换48层 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/s1k_all_experts_20261118T032650Z/jobs/convert/status.json) |
| CPU H-only/adam | fit_seconds_including_validation | 22.590 | 0.0063 | 1次该阶段 | 0 | CPU两线程；验证与solver时间互有包含；不与GPU全模型转换比较 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_only_analytic/cpu_linear_20261105T085702Z/report.json) |
| CPU H-only/adam | solver_seconds_excluding_measurement | 8.343 | 0.0023 | 1次该阶段 | 0 | CPU两线程；验证与solver时间互有包含；不与GPU全模型转换比较 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_only_analytic/cpu_linear_20261105T085702Z/report.json) |
| CPU H-only/analytic | fit_seconds_including_validation | 5.687 | 0.0016 | 1次该阶段 | 0 | CPU两线程；验证与solver时间互有包含；不与GPU全模型转换比较 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_only_analytic/cpu_linear_20261105T085702Z/report.json) |
| CPU H-only/analytic | solver_seconds_excluding_measurement | 3.738 | 0.0010 | 1次该阶段 | 0 | CPU两线程；验证与solver时间互有包含；不与GPU全模型转换比较 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_only_analytic/cpu_linear_20261105T085702Z/report.json) |
| 4B阈值/standard | 128 Linear搜索 | 3.310 | 0.0009 | 1次该阶段 | 历史报告 | 报告一次计时；设备/迭代配置见原报告，不是端到端模型成本 | [原始证据](../../../HiFloat4/hif4_scale_threshold_optimization/experiments/EXPERIMENT_REPORT.md) |
| 4B阈值/fast_search | 128 Linear搜索 | 14.480 | 0.0040 | 1次该阶段 | 历史报告 | 报告一次计时；设备/迭代配置见原报告，不是端到端模型成本 | [原始证据](../../../HiFloat4/hif4_scale_threshold_optimization/experiments/EXPERIMENT_REPORT.md) |
| 4B阈值/full_search | 128 Linear搜索 | 18.150 | 0.0050 | 1次该阶段 | 历史报告 | 报告一次计时；设备/迭代配置见原报告，不是端到端模型成本 | [原始证据](../../../HiFloat4/hif4_scale_threshold_optimization/experiments/EXPERIMENT_REPORT.md) |

## D　机制、算子和工程验证记录

这些是局部性质/诊断证据；表内案例数不是准确率重复次数。历史诊断的全部原表保留在附件，未通过的目标和被替代路线仍登记。

| 实验 | 已记录结果 | 条件 | 次数/聚合 | 使用边界 | 来源 |
| --- | --- | --- | --- | --- | --- |
| 内部误差目标函数筛选 | 31组candidate/参照，全部Stable=False；原报告STOP_EXPANSION | 固定16条holdout×4历史状态；KL与NMSE配对bootstrap区间 | 1次探索，CI不是重复次数 | 后续实际算子边界仍在修正；不能充当验证通过的机制收益 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/internal_error_accumulation/qwen3_30b_a3b_e0_e1_formal/runs/iea_20260909T081436Z_1076681/60_objective/OBJECTIVE_TRAINING_ABLATION_REPORT.md) |
| 真实算子边界：MoE | 固定生产路由128/128重放；自然router集合128/128相同、顺序96/128；固定集合换torch权重后输出120/128相同，平均NMSE=8.26236e-10 | 固定token/rank/专家集合的生产算子对照 | 1次审计；128个case，非128次均值 | 只是相应局部边界，不证明全链路逐位一致或精度效果 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/internal_error_accumulation/qwen3_30b_a3b_e0_e1_formal/runs/iea_20260909T081436Z_1076681/60_objective/corrected_objectives_v3/kernel_boundary_alignment_v1/CONCLUSIONS.md) |
| 真实机制重构v4 | 27个recipe准备；objective gate仍PENDING、training_started=false | 独立teacher 80/80，误差账本64×48×2=6144状态 | 准备/诊断，无训练重复 | 不把已准备recipe记作已完成训练；误差分解不是KL因果证据 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/internal_error_accumulation/qwen3_30b_a3b_e0_e1_formal/runs/iea_20260909T081436Z_1076681/60_objective/mechanism_driven_v4/objective_correctness_gate.json) |
| 长轨迹实际vLLM hook | 64提示轨迹及16 MMLU/12 LCB探针；正式结果见来源 | max_new_tokens=16384；部分探针特意筛选E0对/E1错 | 1轮定向诊断 | 不是完整测试集；不能把探针正确率当任务准确率 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/trajectory_stability_real_vllm_hook/analysis/LONG_TRAJECTORY_REAL_VLLM_REPORT.md) |
| 早期semantic replay | 历史重放边界曾不一致；旧结论有适用限制 | 离线/teacher-forcing replay与实际vLLM路线不同 | 历史诊断 | 仅保留原始证据；以统一prompt重测T18和实际算子审计为准 | [原始证据](../../../docs/experience/lessons/evaluation_evidence.md) |
| NVFP4 vLLM backport | dense max_abs=0；MoE max_abs=0.015625、rel_L2=0.00336843、NMSE=1.13463e-5 | 2专家/4token局部算子；atol0.05、rtol0.01 | 1次局部验证 | emulation是QDQ加浮点GEMM/累加，不是原生FP4硬件GEMM | [原始证据](../../../NVFP4/reports/vllm_v027_nvfp4_backport/qwen3_30b_puncture.md) |

## E　未形成完整正式结果的实验

运行中是本次核验快照；旧文件中的running/blocked只反映记录当时，不能据此推断今天仍占GPU。已完成训练、已导出和已完成评测分别登记；没有把失败短测分数写进正式表。

| 实验 | 状态 | 已知事实与缺项 | 来源 |
| --- | --- | --- | --- |
| GPTQ 五任务×五轮 | running | 快照已完成11/20个子阶段；控制PID=1221133；GPU=0,1；当前命令和日志见status。ARC-E/C合并为一个阶段。 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/s1k_all_experts_20261118T032650Z/jobs/evaluate_r64_gptq/status.json) |
| matrix_norm A20/B20 各五任务×五轮 | running | 快照已完成24/32个子阶段；控制PID=549491；GPU=4,5；当前命令和日志见status。ARC-E/C合并为一个阶段。 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/matrix_norm_repeats_20261118T034732Z/jobs/formal/status.json) |
| 固定在线R64＋权重侧正交A20 | 训练/导出已完成，正式评测缺失 | 48层训练；49分片；不能拿旧绑定rotation消融代替本实验 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/gptq_expanded_20261117T062000Z/models/orthogonal_a20/export.json) |
| full_e5 / coverage50_e5 | 训练/导出/开发集完成；缺正式下游 | T25仅开发集，不能作为LCB/MMLU成绩；random50_e5已在T06 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_norm_efficiency/run_20261029T023839Z/REPORT.md) |
| 完整KL / 硬CE AIME25 | awaiting_aime_review | 分别6条和5条待最终提交复核；其余五项非AIME完整成绩见T03；不填未审最终分 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_full_kl_ce/run_20261021T030031Z/parallel_20261021T080148Z/manifest.json) |
| 早期GPTQ自然路由校准 | 失败，已被S1K全专家方案替代 | 原始/扩增S1K自然路由不能满足所有专家输入覆盖；失败层之前的转换不能冒充最终模型 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/gptq_expanded_20261117T062000Z/jobs/convert_r64_gptq/status.json) |
| C4覆盖检查（含512×512） | 按用户要求停止/替代 | 未取得完整全层覆盖结论；最终GPTQ使用干净S1K128条全专家统计，非C4 | [原始证据](../../../docs/experience/records/thesis_format_conversion_20261116.md) |
| 235B E4→group_top_mass | 迁移训练完成；未定位完整评测 | parallel运行E4与group训练完成；downstream目录只有启动/协议证据，无完整metrics；早期shape/OOM/compile尝试详见运行清单 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_transfer/parallel_20261003T144300Z/group_top_mass/summary.json) |
| 235B matrix_norm | 启动时blocked_resources；非当前GPU判断 | 旧记录中资源不够，未正式训练/评测；不能把计划的TP8四任务表当结果 | [原始证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_matrix_norm/run_20261026T025055Z/status.json) |
| MiniMax MXFP4 | 权重转换完成；GPU验证/正式评测已取消 | 21888专家矩阵已转换；无MXFP4下游准确率；NVFP4五项见T17 | [原始证据](../../../results/minimax_m3/mxfp4_20261104T015806Z/stop_record.json) |
| 四模型新论文基线 | 权重/数据及CPU局部验证完成；正式GPU结果缺失 | 9B Nemotron、Nemotron3-30B、Qwen3-30B、Qwen3.8-27B；CPU完整导出/重载不等于全模型TP2验证或任务成绩 | [原始证据](../../../docs/experience/records/fp4_paper_baselines_20261119.md) |
| 8B/4B历史实验 | 有报告，部分原始产物不在当前文件系统 | 保留报告数值、原始精度与重复次数未知标记；不能倒推出未保存的seed/采样/方差 | [原始证据](../../../HiFloat4/hif4_scale_threshold_optimization/experiments/EXPERIMENT_REPORT.md) |

## F　全仓库覆盖清单

下面按结果系列覆盖全部运行目录。详细尝试（含smoke、失败、重试和阶段记录）逐条在运行清单CSV；同根下的每层/每epoch文件不是独立实验。共享模型视图、校准原始文本及权重二进制不复制进索引；实际采用的导出摘要仍进入证据清单。只有计划而没有运行证据的不算“跑过”。

| 结果系列 | 运行目录/记录单元数 | 正文指标表 | 补充说明 |
| --- | --- | --- | --- |
| aime25_repeats | 3 | T21 | 状态详见run_inventory；原始路径保持不动 |
| ban | 11 | T15 | 状态详见run_inventory；原始路径保持不动 |
| ban_trace_calibration | 6 | T16、T38 | 状态详见run_inventory；原始路径保持不动 |
| downstream_accuracy_recovery | 1 | T04 | 状态详见run_inventory；原始路径保持不动 |
| e2e_diag_reconstruction | 7 | T24 | 状态详见run_inventory；原始路径保持不动 |
| internal_error_accumulation | 1 | D/E及原始证据索引 | 状态详见run_inventory；原始路径保持不动 |
| kl_direction_reuse | 34 | T26 | 状态详见run_inventory；原始路径保持不动 |
| long_trajectory_stability | 10 | T18 | 状态详见run_inventory；原始路径保持不动 |
| matrix_norm_efficiency | 1 | T06、T25 | 状态详见run_inventory；原始路径保持不动 |
| matrix_only_analytic | 1 | T28 | 状态详见run_inventory；原始路径保持不动 |
| minimax_m3 | 5 | T17 | 状态详见run_inventory；原始路径保持不动 |
| model_compatibility | 1 | D/E及原始证据索引 | 状态详见run_inventory；原始路径保持不动 |
| module_design_ablation | 6 | T04、T21、T22 | 状态详见run_inventory；原始路径保持不动 |
| native_qwen3_8b | 3 | T33 | 状态详见run_inventory；原始路径保持不动 |
| non_equivalent_reconstruction | 2 | T07、T29 | 状态详见run_inventory；原始路径保持不动 |
| nvfp4_gpqa | 3 | D/E及原始证据索引 | 状态详见run_inventory；原始路径保持不动 |
| paper_baselines | 1 | T34 | 状态详见run_inventory；原始路径保持不动 |
| progressive_error_cancellation | 2 | T27 | 状态详见run_inventory；原始路径保持不动 |
| progressive_error_cancellation_smoke_20260920_v6 | 3 | D/E及原始证据索引 | 状态详见run_inventory；原始路径保持不动 |
| qwen3_235b_matrix_norm | 1 | D/E及原始证据索引 | 状态详见run_inventory；原始路径保持不动 |
| qwen3_235b_transfer | 10 | D/E及原始证据索引 | 状态详见run_inventory；原始路径保持不动 |
| qwen3_bf16_gpqa | 3 | T14 | 状态详见run_inventory；原始路径保持不动 |
| qwen3_mxfp4 | 13 | T09、T10、T11、T12、T13、T23、T36、T37 | 状态详见run_inventory；原始路径保持不动 |
| reasoning_trace_calibration | 3 | T02、T25 | 状态详见run_inventory；原始路径保持不动 |
| reasoning_trace_full_kl_ce | 1 | T03、T25 | 状态详见run_inventory；原始路径保持不动 |
| reasoning_trace_last_head | 1 | T03 | 状态详见run_inventory；原始路径保持不动 |
| reasoning_trace_router_bf16 | 1 | T03 | 状态详见run_inventory；原始路径保持不动 |
| reports | 3 | D/E及原始证据索引 | 状态详见run_inventory；原始路径保持不动 |
| residual_lora_compensation | 1 | T08、T30 | 状态详见run_inventory；原始路径保持不动 |
| thesis_format_conversion_supplement | 6 | T01、T35 | 状态详见run_inventory；原始路径保持不动 |

另含4B阈值历史总报告和8B QAT历史总报告（T19/T20/T31–T33）；全部索引共46628个文本证据文件。原始数值不可用时仅登记缺项，不从日志进度、模型存在或历史“完成”标题推算成绩。

本次仅结果整理与导航维护；未启动、停止或修改任何后台实验。既有实验依赖、输出目录及论文正文保持原位。
