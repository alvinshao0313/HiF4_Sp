# 两个小模型的单项方法准备与运行

2026-11-20。用户确认第三章先测试 Nemotron Nano 9B（N9）和 QUASAR Qwen3.5-4B（Q4Q），两个 30B MoE 的单项结果留空。后续组合、机制及规模验证的总体规划保留，不在本轮启动。原生 NVFP4 与两种直转参照由其他机器提供。

## 方法与协议

第三章 BRQ 使用完整融合版 `brq_fused`，配置 ID 为 H12/M15：norm 增益并入相邻权重，embedding、残差输出、输出头及相关输入分支配套旋转。G64/G32 分别对应 HiF4/MXFP4，随机符号 seed=17；推理残差流已经在旋转空间，不再额外在线旋转输入。旧 `brq`、H02/M02 保持在线版本身份，供第四章对照。原始实现中的 embedding 均值调整没有移植，因为它不是本次 RMSNorm 模型的等价变换。

普通 QA 使用本机 lm_eval 0.4.8，任务为 `arc_challenge,arc_easy,openbookqa,boolq`。按照用户外机命令固定 0-shot、不使用 chat template、batch=128、seed=1234、记录逐题输出；TP2、context=4096、BF16 KV、enforce_eager=False、GPU memory utilization=0.90、max_num_seqs=128、max_num_batched_tokens=8192。逐题记录保留原始指标，论文表格优先使用 acc_norm，任务未提供时使用 acc。跨机复用时核对任务版本和对应指标。五项推理仍使用 lighteval 和 1234/1235/1236；本轮未启动推理全量评测。

本地 lm_eval CLI 不含新版 `run` 子命令，通过 `paper_baselines.entry lm_eval` 启动。目标模型由 `paper_fp4` 量化后端处理全部源 NVFP4 投影；不叠加原生 NVFP4 的 emulation 开关。导出以 BF16 保存目标格式的 QDQ 操作数，文件大小不是 packed FP4 存储大小。

## 已验证与待验证

原始证据位于 [single_methods_20261120T050800Z](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/single_methods_20261120T050800Z/)；代码摘要和源/导出身份见其中 `manifest.json`。

| 项目 | 证据与范围 |
| --- | --- |
| N9 融合前后输出 | `fusion-equivalence.json`：真实第 0、1、2、14 层，包含 BF16/NVFP4 解码 Mamba、MLP、注意力；两种目标组长，关闭目标量化后比较。BF16 舍入下 NMSE 为 2.15e-5–1.44e-4；不代表任务精度。 |
| N9 MXFP4 融合导出 | `n9-brq-fused-mxfp4/conversion.json`：96 个源 NVFP4 投影全部转换，norm/embedding/分支处理清单保留。 |
| N9 正式条件 QA 短测 | `qa-qualification-v3-check.json`：四项各 40 题、560 个有限选项分数，TP2、多批请求、编译/CUDA graph，正常退出。`qa-qualification-v4-check.json` 复查原8题设置，112个选项概率均有限且正常退出；短测分数不填论文。 |
| N9 完整校准序列 | `n9-sq-hif4-long-seq/recipe.json`：使用正式 S1K 第 35 条、19,098 token，完整保留，逐层运行前三层；两项目标投影已拟合，purpose=qualification。正式128条合计1,274,376 token。 |
| Q4Q 接入 | 固定 revision `0a5cc922801e77fd53b69fa7c135ab53d084ef0c`，200个NVFP4投影、65个待融合norm的拓扑已核对。compressed-tensors 的全局尺度取倒数映射为内部反量化尺度；融合时需拆开共享 embedding/head 并单独折入末层 norm。完整权重及运行验证尚未完成。 |
| GPTQ/AWQ/SmoothQuant | 复用[既有四方法记录](fp4_paper_baselines_20261119.md)；本次未把旧短文本校准提升为正式结果。后续按完整S1K运行，逐方法及目标格式记录实际覆盖。 |

## 本次修复及无效证据

Nemotron 的本机 Transformers Mamba 路径在 19,098-token 输入上试图分配 150 GiB 的广播中间张量；短文本初始化未暴露此问题。校准改为先收缩状态维、按原 chunk 尺寸传递完整状态，保留 Linear 调用及完整文本。对真实 N9 第2层，64/129/257 token 的结果与 FP32 逐 token 递推比较，NMSE 为 0、3.74e-8、1.71e-8，详见 `mamba-parity.json`。本机原 `torch_forward` 的跨 chunk 状态收缩也与递推参照不符，不能仅要求与该路径一致；范围限于此次实际安装文件和所测路径。

QA 第一轮在启动时被 Nemotron 发布配置的远程 `auto_map` 阻塞；目标导出改用本机已验证的原生 Nemotron-H 类。第二轮虽保存了评分，但逐题对数概率为 NaN，且解释器退出时发生引擎资源清理异常，`n9-brq-mx-qa-qualification-v2` 全部无效。显式传入 Triton 量化核默认参数 FHT32=False，消除了 torch.compile 参数分析异常；CLI 结束前显式关闭引擎，避免在解释器退出时竞争清理。入口另检查选项对数概率的有限性，遇到 NaN/Inf 立即报错。后续有限输出与正常退出的证据独立保存。

## 运行入口与交付状态

正式 QA 的完整命令见实验目录 `formal_qa_command.json`，没有 `--limit`。已于北京时间13:41完成，exit_code=0；原QA launcher PID=2774678；PID、退出状态和题目覆盖以 `formal_qa_status.json` 为准，stdout/stderr 位于 `n9-brq-mx-qa-formal.log`，结果位于 `n9-brq-mx-qa-formal/`。四项 QA 完成不等于九项任务完成。

Q4Q 下载复用既有的固定版本下载工具，`quasar-source/download_manifest.json` 记录官方文件大小与摘要，`download_status.json` 记录进程及退出状态，`download.log` 记录传输进度。最初 Hugging Face 客户端因响应缺少元数据失败；现用 curl 读取同一官方固定 revision，校验大小与摘要后才发布文件，不改变模型版本。

本次经验补入[执行验证经验](../lessons/model_transfer_execution.md)和[评测证据经验](../lessons/evaluation_evidence.md)。现有原生/直转任务以及 GPU 0–5 的进程保持原状；没有暂存或提交 Git 变更。

HiF4 完整融合导出已于北京时间13:41完成，exit_code=0；原launcher PID=2774679，四个CPU线程，不占QA双卡；命令及状态见 `hif4_export_status.json`，日志 `n9-brq-fused-hif4-export.log`，输出 `n9-brq-fused-hif4/`。正式QA使用空闲GPU6/7，配置ID=N9-M15，覆盖四项共7,318题；GPU0–5未改动。GPTQ/AWQ/SmoothQuant的正式校准及Q4Q正式实验尚未启动。

## 13:45 完成复核

[完成核验](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/single_methods_20261120T050800Z/completion_review.json)重新核对了7,318条唯一题目、全部选项分数有限性，并从逐题结果重算acc/acc_norm，与汇总一致。N9-H12导出索引与两个safetensors分片字段相符，覆盖96个目标投影；尚未运行HiF4下游评测。

| N9-M15 任务 | 题数 | acc（%） | acc_norm（%） |
| --- | --- | --- | --- |
| ARC-C | 1172 | 51.54 | 54.69 |
| ARC-E | 2376 | 80.26 | 79.59 |
| OBQA | 500 | 31.20 | 42.20 |
| BoolQ | 3270 | 86.82 | 不适用 |

此时Q4Q下载约59.1%（2.49/4.22 GB），进程仍在运行。两个GPU实验进程已正常退出，GPU6/7空闲；本次仅核查和归档，没有启动新实验。尚缺HiF4评测、五项推理任务、其余三种方法的正式实验及Q4Q完整验证，不能将本次四项QA称为整套实验完成；外机原生/直转参照尚未合并，暂不判断BRQ恢复了多少精度。


## 14:22 启动16组完整队列

用户明确要求两个小模型 × 两种目标格式 × GPTQ/AWQ/SmoothQuant/完整融合BRQ，覆盖九项任务。固定队列共304个任务/种子轮次；直接复用已核验的N9-M15四项普通QA，其余300轮进入同一队列。先运行剩余普通QA和12次正式校准，再逐种子运行五项推理任务；不按短测收益筛掉方法。MoE单项仍暂缓，不启动原生或直接转换的重复评测。

正式推理统一输出32768、上下文40960；TP2、BF16 KV、FA2、编译/CUDA Graph、max_num_seqs=128、max_num_batched_tokens=8192。N9采用temperature=0.6/top_p=0.95；QUASAR采用temperature=1/top_p=0.95/top_k=20/min_p=0，LCB的presence/frequency penalty为0、repetition penalty为1，其他四项沿用presence penalty=1.5。种子1234/1235/1236同时设置引擎和采样器，prompt/选项随机性保持lighteval默认1234。四项QA保持此前外机约定的4096 context、0-shot、无聊天模板、seed=1234和batch128，并保留acc及acc_norm。

固定配置、所有实际命令及输入/输出目录见[plan.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/single_methods_20261120T050800Z/all_single_methods/plan.json)。队列PID **2824859**，GPU **6、7**；启动凭据`launch.json`，总体状态`status.json`，每阶段PID/命令/退出码和日志在`jobs/<阶段名>/status.json`与`output.log`。该队列是一次性固定批次，不轮询空闲GPU，不自动重试。正式启动后保护代码与配置，GPU0–5原任务不变。

启动前证据均为工程验证，不填正式准确率：

- 两模型、三个校准方法、两个目标格式，共12项完整长序列短测通过。N9使用19098 token并覆盖Mamba与MLP；Q4Q使用19477 token，GPTQ/AWQ覆盖前4层的GDN与普通注意力，SmoothQuant覆盖全部32层/200投影。正式校准使用已清洗的128条完整S1K训练文本，N9共1274376 token、Q4Q共1297673 token，未截断；沿用clean S1K样本ID及既有隔离审计。
- QUASAR固定版本19文件下载与校验完成。200投影各8行与compressed-tensors参考解码完全一致；融合等价性覆盖GDN、普通注意力及拆开的共享输出头。两种BRQ目标格式均已完整导出。
- N9融合BRQ-HiF4、N9 SmoothQuant-HiF4、Q4Q融合BRQ-HiF4及Q4Q SmoothQuant-MXFP4，分别完成正式QA引擎设置下四任务各40题，逐选项概率有限、覆盖完整、退出0。N9旧SmoothQuant导出的auto_map仍指向远程代码，先前一次短测因此失败；随后独立视图仅移除该失效映射，采用当前已验证的原生Nemotron类完成检查。当前正式export已采用此处理，旧产物保留。
- N9以seed=1235运行128题MMLU-Pro，Q4Q以seed=1236运行128题LCB，各仅生成8 token进行时间压缩，保留40960 context、TP2与并发128。实际采样参数、每题一份生成、缓存隔离、评分及保存均验证通过，退出0；短回答分数无精度解释。MATH/GPQA/GSM8K/MMLU-Pro的默认评分器另用题库参考答案验证正例。

汇总证据为`all_single_methods/qualification.json`、`data/protocol_audit.json`及各独立短测的日志/exit/recipe。五项数据固定在`data/manifest.json`：MATH500官方固定revision500题、ModelScope GPQA198题、GSM8K1319题、LCB v6 175题、MMLU-Pro12032题，默认prompt和评分；本机LCB指标名含`:16`但实际Doc与后端采样均为1，已核对。每个任务/种子新建模型视图，避免复用此前预测缓存。代码摘要在`code_sha256.json`及`shared_source_sha256.json`，未提交为新Git版本。

`results.py`只接收完整结果：核对题数、逐题与汇总一致、生成数量、采样参数及退出状态后更新`scores.csv`和论文`LaTeX/tables/single_methods_results.tex`。推理任务三个种子齐全才填均值，单轮分数逐条保留。正文已接入两张结果表并填入原有4项QA；原生/直转参照尚未合并，不判断收益。当前状态为**完整队列已启动，运行中**，尚未完成300个剩余轮次。

本次新增经验归入[量化执行边界](../lessons/fake_quant_boundaries.md)与[评测证据](../lessons/evaluation_evidence.md)：compressed-tensors全局尺度/共享头处理，以及不能用LCB指标名称推断采样次数。短测产物仍作为正式路径证据保留，未删除既有结果或修改活动实验。

## 表 1、表 2 三组并行

2026-11-20，按用户确认只向旧控制器 2824859 发送 SIGINT。它等待之前在 6、7 号卡运行的两份 Nemotron 9B GPTQ 校准完成并各自保存退出状态。0/1、4/5 先独立运行，6/7 的接续入口在两个校准均验证成功后执行。实际命令、PID 和日志见[切换记录](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/single_methods_20261120T050800Z/all_single_methods/parallel_0145/handoff.json)。[分工清单](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/single_methods_20261120T050800Z/all_single_methods/parallel_0145/split.json)对应 87、106、95 个剩余任务种子轮次，和已核验的 16 项普通 QA 合计 304。外机承担原生 NVFP4 和两种直转。论文汇总按各组 verified 记录填数。

SmoothQuant 的四个单项配置统一使用 α=0.5，每个通道的激活峰值来自固定的 128 条 S1K 完整序列。GPTQ 和 AWQ 分别使用输入二阶统计和权重量化后的输出误差。校准与导出是下游评测的前置步骤，校准进度不计入任务得分。

## A800 单卡评测短测

2026-11-20，以 Nemotron Nano 9B 的正式转换权重验证 TP1。HiF4 GPTQ 采用 lm_eval 四项普通 QA 各 40 题，0-shot、batch 128、上下文 4096。160 题的选项对数概率均有限，逐题重算与汇总相符。用题目和提示词哈希对齐同一权重的 TP2 结果，按论文主指标比较逐题对错，ARC-C、ARC-E、OBQA、BoolQ 分别相同 38/40、38/40、40/40、39/40。MXFP4 GPTQ 在 TP1 下完成 128 题 MMLU-Pro，各生成 64 token，上下文配置 40960，最大并发 128，BF16 KV。两次运行均正常退出。长输出时的实际吞吐仍需单独测量。短测与正式结果分开保存，见[单卡验证记录与输入输出](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/single_methods_20261120T050800Z/tp1_capacity_20261120T091805Z/summary.json)。

2026-11-20 后续评测采用可运行的 TP1。QUASAR 4B 的补测在 A800 上完成了 HiF4-GPTQ 四项普通 QA 各 40 题和 MXFP4-GPTQ 的 MMLU-Pro 128 题。短测保留 40,960 上下文、BF16 KV 和并发 128，推理输出限 64 token。真实题目覆盖、逐项得分和实际采样设置已核对，详见[两模型单卡验证](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/single_methods_20261120T050800Z/all_single_methods/tp1/qualification.json)。[六卡分工](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/single_methods_20261120T050800Z/all_single_methods/tp1/split.json)复用已验收的 32 个 TP2 普通 QA 轮次，其余 32 个普通 QA 轮次和 240 个推理轮次按 TP1 配置。正式启动状态以交接产物和任务日志为准。

2026-11-20 09:32 UTC，经用户确认，TP1 总控（PID `3075959`）向三个旧调度进程单独发送退出信号。GPU 1、6 首先启动 QUASAR 4B 的 BRQ-HiF4 和 BRQ-MXFP4，均从 MATH-500 的 seed 1234 开始。GPU 0、4、5、7 在对应旧队列的 AWQ 校准保存并通过完整性检查后接续。各卡的实际命令、日志及产物目录见 `tp1/<卡号>/jobs/<任务>/` 下的 `status.json` 和 `output.log`。等待旧队列的阶段记录与新任务的完成记录分别保存在旧、新目录中。[交接记录](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/single_methods_20261120T050800Z/all_single_methods/tp1/handoff.json)

### 六张 A800 的 TP1 评测分工

每个“模型 × 方法 × 目标格式”配置固定在一张卡上完成后续评测。同一卡上有多个配置时，按该卡的 `plan.json` 顺序执行，部分配置按任务和种子穿插。下表的轮数从 TP1 交接时计起。校准和权重导出单独记录。

| GPU | QUASAR Qwen3.5 4B | Nemotron Nano 9B v2 | 普通 QA 轮数 | 推理轮数 | 合计 |
|---:|---|---|---:|---:|---:|
| 0 | GPTQ–HiF4 | AWQ–HiF4、BRQ–HiF4 | 4 | 45 | 49 |
| 1 | BRQ–HiF4、GPTQ–MXFP4 | SmoothQuant–HiF4 | 4 | 45 | 49 |
| 4 | AWQ–HiF4、SmoothQuant–HiF4 | GPTQ–HiF4 | 8 | 45 | 53 |
| 5 | AWQ–MXFP4 | BRQ–MXFP4 | 4 | 30 | 34 |
| 6 | BRQ–MXFP4、SmoothQuant–MXFP4 | SmoothQuant–MXFP4 | 8 | 45 | 53 |
| 7 | — | AWQ–MXFP4、GPTQ–MXFP4 | 4 | 30 | 34 |
| **合计** | **8 个配置** | **8 个配置** | **32** | **240** | **272** |

两个模型、两种目标格式的 BRQ 和 GPTQ 均已有完整普通 QA 结果，共 32 轮。AWQ 和 SmoothQuant 各配置补四项普通 QA，每项全量运行一次（seed 1234）。全部 16 个配置各跑五项推理任务，每项取三个种子（1234、1235、1236），每个配置共 15 轮推理。原生 NVFP4 和两种直转参照由外机负责。[逐卡计划及首项任务](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/single_methods_20261120T050800Z/all_single_methods/tp1/split.json)

4 号卡的 QUASAR AWQ–HiF4 校准于 09:31:31 UTC 保存完成。交接代码等待旧调度进程退出，再检查当前卡的校准结果。旧调度进程同时管理 4、5 号卡，直到 5 号卡的 QUASAR AWQ–MXFP4 校准于 09:35:34 UTC 完成才退出。4 号卡于 09:35:35 UTC 接续，先导出 HiF4 权重，再运行普通 QA。两个格式的校准产物彼此独立；以后交接可按单卡校准任务的完成状态、产物和显卡占用判断就绪。[交接实现](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/single_methods_20261120T050800Z/all_single_methods/tp1/dispatch.py) · [4 号卡启动记录](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/single_methods_20261120T050800Z/all_single_methods/tp1/4/launch.json)


## 2026-11-22 已完成的单项评测

截至北京时间 21:28，两个模型、两种目标格式和四种方法共完成 56/64 轮普通 QA 和 121/240 轮推理。按论文口径，已有 20/80 项推理任务集齐三个种子，正文表 1、表 2 合计可填 76/144 个准确率。普通 QA 取 seed 1234 的一次全量结果，ARC-C、ARC-E、OBQA 用 acc_norm，BoolQ 用 acc。推理栏取三个种子 1234、1235、1236 的未舍入分数的算术均值。部分种子已跑完或尚未取得结果时，表中以“—”标示。核验时先按原始文件去重，再逐项对照评测框架输出的原始分数。

HiF4（%）

| 模型 | 方法 | MATH-500 | GPQA | GSM8K | LCB | MMLU-Pro | ARC-C | ARC-E | OBQA | BoolQ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Nemotron 9B | BRQ | — | — | — | — | — | 55.20 | 79.42 | 42.80 | 87.46 |
| Nemotron 9B | GPTQ | — | — | — | — | — | 55.03 | 79.17 | 42.20 | 87.19 |
| Nemotron 9B | AWQ | — | — | — | — | — | 54.44 | 79.38 | 42.40 | 87.00 |
| Nemotron 9B | SmoothQuant | — | — | — | — | — | — | — | — | — |
| QUASAR 4B | BRQ | — | — | — | — | — | 51.96 | 75.93 | 39.40 | 85.84 |
| QUASAR 4B | GPTQ | — | — | — | — | — | 53.84 | 76.94 | 38.60 | 84.56 |
| QUASAR 4B | AWQ | 82.07 | 70.03 | 90.32 | 27.43 | 75.89 | 52.39 | 75.00 | 39.40 | 82.72 |
| QUASAR 4B | SmoothQuant | — | — | — | — | — | 51.71 | 76.22 | 37.80 | 85.78 |

MXFP4（%）

| 模型 | 方法 | MATH-500 | GPQA | GSM8K | LCB | MMLU-Pro | ARC-C | ARC-E | OBQA | BoolQ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Nemotron 9B | BRQ | 87.27 | 48.48 | 93.76 | 48.57 | 66.96 | 54.69 | 79.59 | 42.20 | 86.82 |
| Nemotron 9B | GPTQ | 85.73 | 48.65 | 93.96 | 53.71 | 68.20 | 53.07 | 77.15 | 40.80 | 86.54 |
| Nemotron 9B | AWQ | 85.27 | 48.15 | 94.04 | 53.52 | 69.13 | 53.92 | 78.96 | 42.00 | 85.84 |
| Nemotron 9B | SmoothQuant | — | — | — | — | — | 52.05 | 77.44 | 42.20 | 85.90 |
| QUASAR 4B | BRQ | — | — | — | — | — | 48.72 | 71.68 | 38.00 | 84.77 |
| QUASAR 4B | GPTQ | — | — | — | — | — | 49.57 | 73.06 | 38.80 | 85.75 |
| QUASAR 4B | AWQ | — | — | — | — | — | 50.09 | 74.20 | 38.60 | 83.36 |
| QUASAR 4B | SmoothQuant | — | — | — | — | — | — | — | — | — |

[逐任务、逐种子分数和原始证据索引](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/single_methods_20261120T050800Z/all_single_methods/verified_rounds_20261122T212824+0800.csv)保留了上述 177 轮结果中每轮的未舍入得分（0–1）、计分题数、指标、原始结果路径及验收文件哈希，包含尚未集齐三个种子的推理轮次。分表中，HiF4 的 28 项普通 QA 与 5 项推理均值已齐，MXFP4 的 28 项普通 QA 与 15 项推理均值已齐。本次登记限于本机四种单项方法；与原生 NVFP4 和直接转换的比较待外机同口径结果到位后补充。

11 月 22 日 21:30（北京时间），7 号卡接手 QUASAR 4B 的 SmoothQuant–MXFP4，负责校准、导出和后续 19 轮下游评测。6 号卡保留当时正在运行的 BRQ–MXFP4 MMLU-Pro（seed 1234），完成后接续另外 20 个推理轮次。[交接记录](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/single_methods_20261120T050800Z/all_single_methods/tp1/reassign_67/handoff.json)

## 2026-11-23 论文表格复核

北京时间 17:51 的快照包含 184 份验收文件。逐项与框架原始输出核对后，共有 60 个普通 QA 任务轮次和 169 个推理任务种子轮次。37 个推理任务已集齐三个约定种子。正文表 1、表 2 及附录复用同一份数据，四种单项方法可填 97/144 格。计数只覆盖这四种方法。九项原生参照齐备后再计算平均正向损失。[逐轮得分、原始指标及哈希](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/single_methods_20261120T050800Z/all_single_methods/verified_rounds_20261123T175115+0800.csv)

| 目标格式 | 普通 QA 得分 | 完整三种子均值 | 已填准确率 |
| --- | ---: | ---: | ---: |
| HiF4 | 28 | 18 | 46 |
| MXFP4 | 32 | 19 | 51 |

## 2026-11-24 GPU 0 接续 HiF4 单项评测

北京时间 10:51，经用户确认，GPU 0 接手 QUASAR 4B 的 BRQ→HiF4。原先在 GPU 0 安排的 57 轮已全部验收。新队列复用同一 BRQ 权重、四项普通 QA 和 MATH-500 的种子 1234 结果，负责其余 14 轮推理。首项为全量 MMLU-Pro，种子 1234，采用 TP1、FlashAttention、并发上限 128。已核对这 12,032 题的提示词、token 和标准答案与 GPU 1 的同模型评测一致。[分工与复用来源](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/single_methods_20261120T050800Z/all_single_methods/tp1/reassign_01/split.json)

交接信号只发给 GPU 1 的原调度器（PID 3075987）。当时 QUASAR 4B 的 GPTQ→MXFP4 正在运行最后一轮 MMLU-Pro，种子为 1236，评测进程为 PID 1923979。这一轮沿用原进程生成、评分和验收。验收完成后，GPU 1 接续 Nemotron 9B 的 SmoothQuant→HiF4，按原计划完成校准、导出、四项普通 QA 和 15 轮推理。[交接记录](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/single_methods_20261120T050800Z/all_single_methods/tp1/reassign_01/handoff.json) · [两张卡的启动与进度核对](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/single_methods_20261120T050800Z/all_single_methods/tp1/reassign_01/startup_check.json)

## 2026-11-24 GPU 4 接手续跑一轮全量 MMLU-Pro

北京时间 20:05，GPU 4 在原队列 57 轮全部验收后，接手 QUASAR Qwen3.5-4B 融合 BRQ → MXFP4 的 MMLU-Pro（seed 1236）。本轮使用同一导出权重、全部 12,032 题和既定 TP1 配置。此前同权重的 seed 1234 用时 41.08 小时。交接器 PID 为 3621251，GPU 4 评测 PID 为 3621264。[正式命令与 GPU 分配](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/single_methods_20261120T050800Z/all_single_methods/tp1/reassign_46/4/plan.json) · [运行日志](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/single_methods_20261120T050800Z/all_single_methods/tp1/reassign_46/4/jobs/q4q-brq_fused-mxfp4--mmlu_pro--1236/output.log)

经用户授权，交接信号只发给 GPU 6 的队列调度器 PID 1498703。原评测 PID 1873161 继续完成 seed 1235，逐题生成和评分验收通过后，GPU 6 接续原队列其余九轮。两张卡分别保存各轮日志、退出码和验收报告；全批按 304 个唯一任务种子轮次、144 个完整论文单元格核对。[交接记录](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/single_methods_20261120T050800Z/all_single_methods/tp1/reassign_46/handoff.json) · [GPU 6 后续九轮](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/single_methods_20261120T050800Z/all_single_methods/tp1/reassign_46/6/plan.json)

## 2026-11-26 表 3 的尺度方法与直转复用

表 3 比较 Nemotron Nano 9B 与 QUASAR Qwen3.5-4B 的格式特有尺度方法。每个模型有五项新增配置，均从发布的原生 NVFP4 权重开始，分别覆盖 96 和 200 个 NVFP4 投影。HiF4 的逐组尺度搜索仅处理权重，用 full 预算搜索合法的组尺度和子尺度，导出后沿用选出的权重。逐矩阵搜索也只处理权重，在 6 到 8、步长 0.25 的候选中按整个源矩阵的重建 MSE 选一个常数。两种方法的激活均沿用标准 HiF4 G64、常数 7、阈值 4/2。MXFP4 的 UOS、G256、G512 各自作用于权重和激活。所有导出用于 W4A4 数值仿真。10 项完整导出均已重载验收，保存了逐投影耗时、含分片写盘的总耗时及投影覆盖。[正式转换成本与权重误差](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/table3_six_tasks_20261126/operations/conversion_audit.json)

GPU 检查在两个发布模型上各选一个完整投影，覆盖五种方法的保存重载与两次前向，并与独立参照核对。[转换和评测计划](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/table3_six_tasks_20261126/plan.json) · [真实投影验证](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/table3_20261126/qualification/cuda2.log)

GSM8K、GPQA Diamond、LiveCodeBench v6 的均值取三个完整种子（1234、1235、1236）；ARC-C、OpenBookQA、BoolQ 各取种子 1234 的一次全量成绩。表 3 正文仅列同模型六项任务相对原生的平均正向损失，逐项得分、重复数和实际 TP 在附录列出。按 11 月 25 日外机快照可复用原生 12/12 格、直转 18/24 格。其余六格为 Nemotron 直转 HiF4 和 QUASAR 直转 MXFP4 各三项推理；本轮本机仅安排五项新方法。[复用与缺项清单](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/table3_six_tasks_20261126/inventory.json) · [论文逐项来源](../../../LaTeX/tables/table3_sources.json)

本轮新评测使用 TP1。0、1 卡预留，4、7 卡在分配时正运行表 2；表 3 的四组作业分配在 2、3、5、6 卡。每次导出、短测和全量评测均单独保存命令、进程号、日志与退出状态，已验收的同配置同种子结果直接复用。正式轮次按逐题结果与原始输入验收，短测与正式结果各自保存。[GPU 分工](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/table3_six_tasks_20261126/operations/resource_allocation.json) · [执行记录](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/table3_six_tasks_20261126/attempts)
