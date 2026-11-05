# matrix_norm 高效、高保真转换

适用模型：`nvidia/Qwen3-30B-A3B-NVFP4@2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`。日期：2026-10-29。独立实验入口为本目录；旧实验及其活动依赖保持原位。方案登记与 CPU 准备不代表正式实验已经启动。

## 最终目标

保持 HiF4 W4A4 量化语义。在 MMLU-Pro、LCB、GSM8K、AIME24、AIME25 上，与固定原生 NVFP4 基线比较，五项等权平均正向准确率损失不超过 **0.4 个百分点**：

`mean_loss_pp = sum(max(0, 100 * (native_accuracy - converted_accuracy))) / 5`。

增益不抵消其他项下降；AIME 两年分开等权。不额外要求每项都小于 0.4。逐任务报告原分、下降和按题配对区间，单轮达标不称跨 seed 稳定性证明。

用户已明确：**2–3 小时仅约束与原约 20 小时同口径的优化阶段**。历史基准为 [training_costs.json](../../results/module_design_ablation/formal_20261008T074812Z/training_costs.json) 中 matrix_norm 的 72,146.3059 秒。首要目标 ≤3 小时，争取约2小时（约6.68–10.02倍）；合理探索后若无法达到，最低要求精度不变且至少2倍加速（≤10.0203小时）。达到最低线不代表达到理想目标。

优化阶段从初始 embedding/cache 开始，覆盖48层教师目标、训练、验证、选优保存、传播及原流程末尾源码核对。使用 `legacy_comparable_seconds` 对照旧 `seconds`，不能只算梯度更新。另报 GPU 型号/数量和 GPU-hours，以及筛选、轨迹生成、搜索、导出、开发评测和最终评测。复用轨迹已有成本也注明；不把研发总成本或评测时间塞入2–3小时目标，也不隐藏 OPD 的生成成本。

## 推进顺序

1. **数据高效与独立短程重训**：保留 M 与 norm 联合梯度更新，比较半 token 预算的随机/覆盖筛选与全数据短程对照。
2. 必要时做有限真实 HiF4 学生轨迹再校准，由原生教师在学生前缀上监督。一次固定轨迹不称持续 on-policy；保留失败状态和广域数据。
3. 证据支持时再做短程端到端 OPD，并设置同损失、参数、题池与预算的离线蒸馏对照。OPD 不预设必然更快或泛化更好。达标即可结束，不强制走完所有阶段。

解析 M 更新留作有依据时讨论的备选；不为去梯度而冻结 norm。

## 第一阶段的具体对照

复用 [native trace 源运行](../../results/reasoning_trace_calibration/continue_20261016T135141Z/REPORT.md) 的128条 train、32条 val、独立64 math＋64 code开发题。只筛选 train；验证和开发题固定，正式五任务不参与样本价值计算或超参选优。

| 候选 | 训练数据 | 独立训练轮数 |
| --- | --- | --- |
| `full_e5` | 全部 native traces | 5 |
| `random50_e5` | 分层随机，约50% token上限 | 5 |
| `coverage50_e5` | 分层文本覆盖，约50% token上限 | 5 |

各候选从同一原生模型初始化全部48层，独立5轮 cosine 调度；batch4、AdamW、学习率3e-4、weight decay0。仍为学生前缀上同输入完整 block-delta NMSE，router权重冻结、router loss权重0。每层选验证NMSE最佳训练轮，再传播学生隐状态。固定 activation R64 不意味着冻结经 norm 改变的激活值。

每个领域按完整序列长度排序分成3个等人数桶。领域预算为原token总数的50%向下取整，桶预算按token量的最大余数法分配，不跨桶借预算。随机方案按固定seed42派生顺序装入放得下的完整序列；覆盖方案以题干词集合和字符5-gram的平均Jaccard相似度定义 facility-location 覆盖，每次选单位token边际收益最高且可装入的序列。各桶至少保留1条；不截断序列、不按答案正确性筛选。选择顺序及最终训练源顺序均记录。

文本覆盖只是便宜的代表性代理，并未测量专家路由或激活子空间。完整序列约束使两组实际token不同；覆盖方案可能产生更多batch，必须报告实际token、batch数与墙钟，不能直接声称2倍加速。

开发评测保持原 prompt/thinking/生成参数，按math和code等权平均选优，同分优先实测优化时间更短者，再按上表顺序。开发题打分在生成结束后读取参考答案。先冻结候选，再进入完整正式五任务；若后续继续设计，以独立开发证据为依据。

## 评测与必要验证

| 任务 | 正式规模 | 指标 |
| --- | --- | --- |
| MMLU-Pro | 12,032 | extractive match |
| LCB | 冻结v6队列175题 | 沿用实际单次生成的pass@1定义 |
| GSM8K | 1,319 | extractive match |
| AIME24 | 30题，每题5次 | 最终答案avg5 |
| AIME25 | 30题，每题5次 | 最终答案avg5 |

五项通过根目录 `main.py`、vLLM/lighteval，TP2，沿用同一原生 tokenizer/thinking 与采样协议，固定 Arrow 内容及revision。MMLU-Pro 本研究全量是用户指定正式条件。两年AIME保留正式原分、原始生成及最终答案复核分；教师与候选用相同规则。单题歧义只暂停依赖该分数的汇总。新五任务原生结果不由旧跨运行四项分数拼接而成。

已有源清洗审计复用，并追加 AIME24 对 train/val/dev 的题簇隔离检查；命中则停止依赖部分，不自动替换固定验证/开发题。CPU检查覆盖预算、完整序列、划分、实际注册题量、prompt/token/gold、thinking与上下文长度。

真实短测使用最长4条train＋4条val、batch4、2层×2轮，覆盖教师缓存、前反向更新、验证选优、保存重载与传播。随后导出可加载48层模型（其余层保持原生初始化的R64转换），仅用于开发/评测流程短测，禁止用于正式精度验收。推理短测保留正式TP2/长度/算法，不证明完整精度或长程收敛。导出为BF16保存的HiF4 QDQ值与W4A4运行语义，不宣称实际packed文件压缩率。

## 入口与当前产物

在仓库根目录，由调用者激活 `hif4`。薄入口 `python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.matrix_norm_efficiency.run` 提供 `initialize/prepare/bind/train/export/development/select/evaluate/report`。各步骤需 `--run`，模型步骤需 `--candidate`，评测另需 `--task`。`train/export --qualification` 与 `development/evaluate --smoke` 的产物独立存放。

本轮 RUN：`Native_NVFP4_HiF4_Linear_Puncture/results/matrix_norm_efficiency/run_20261029T023839Z`。CPU准备证据：[数据选择](../../results/matrix_norm_efficiency/run_20261029T023839Z/prepare.inputs.json)、[AIME24隔离](../../results/matrix_norm_efficiency/run_20261029T023839Z/prepare.aime24_audit.json)、[五任务准备](../../results/matrix_norm_efficiency/run_20261029T023839Z/evaluation/preparation_validation.json)。运行状态以RUN内日志、进程与manifest为准。源码在必要验证前绑定，活动验证/实验期间不改依赖。

2026-10-29 用户已明确授权第一阶段使用空闲的 GPU5/6/7，最多同时三张。正式训练于北京时间11:19在GPU7单卡启动，三候选顺序执行；后续开发及原生/胜出候选的五项评测使用GPU5/6的TP2。启动依据见[实际启动记录](../../results/matrix_norm_efficiency/run_20261029T023839Z/launch_02.json)，队列状态见[pipeline.json](../../results/matrix_norm_efficiency/run_20261029T023839Z/pipeline.json)。真实训练、导出、开发、HiF4 AIME24与原生LCB的短测已完成，核验及全部证据摘要见[qualification/complete.json](../../results/matrix_norm_efficiency/run_20261029T023839Z/qualification/complete.json)。短测不参与候选选优或正式精度验收。

分项成本见[costs.json](../../results/matrix_norm_efficiency/run_20261029T023839Z/costs.json)，由RUN内`cost_ledger.py`读取真实步骤记录生成。运行中的费用为快照下界，最终总额在任务未结束时保持空值。历史共享轨迹生成按完整外层进程单列一次；内部优化计时与顶层进程墙钟不重复累加。选择器独立CPU重放计时与原始未单列的选择器时间明确区分，不能替代或拆分原准备流程计时。

GPU短测按项目必要验证权限执行；正式训练/全量评测在具体资源范围明确后启动。不占用他人正在使用的卡，不改变正式题量来迁就资源。训练固定单卡；TP2用于真实推理。现有工具阶段入口不会自动启动后续长实验。

## 参考证据与待验假设

参考[评测经验](../../../docs/experience/lessons/evaluation_evidence.md)中的版本错配、原始AIME生成与同轮基线边界，以及[低比特执行经验](../../../docs/experience/lessons/fake_quant_boundaries.md)中的packed存储与真实执行区别。旧20轮NMSE曲线前5轮收益大，但41/48层最优仍在20轮；因此必须独立重训，不把截取曲线当5轮模型。当前轨迹比旧S1K token更多，旧20.04小时是成本参照，并非同数据因果对照。

待验：5轮是否足够，文本覆盖是否优于分层随机，约半训练token能否与固定验证/教师开销一起达到2–3小时。CPU准备通过不提供这些结论；本阶段尚无新的方法效果经验。
