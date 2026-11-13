# 评测协议与真实执行路径

证据状态：包含历史审计、30B 同协议全量评测、235B 固定数据版本诊断，以及截至 2026-10-21 的 B 模型追加对齐实验，具体来源与适用范围见各段。局部经验不自动适用于其他模型或执行条件。

## 输入不一致时，分数差不能单独归因于格式转换

[2026-09-07 正式 LCB 协议审计](../../../Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/format_conversion_lcb_mechanism_go_20260907/00_audit/FORMAL_PROTOCOL_MISMATCH.md) 发现 E0 的 175/175 输入带 chat wrapper，而 E1/E2/E3 与 E0 完全一致的输入均为 0/175。该批分数可以保留为历史观测，不能解释成相同输入下仅改变量化格式的因果效应。

比较前应检查实际输入 token、模板、thinking 设置、样本 ID、采样参数与评分口径；配置中的 null 不能被当成显式开启。优先使用适用的已有逐样本证据，不因文档整理重复评测。正式 sampled 评测与隔离 greedy 机制诊断分开解释，不互相替代。

## 校准训练集也须与正式测试隔离

2026-10-16 的[轨迹校准数据审计](../../../Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_calibration/run_20261016T132000Z/REPORT.md)对现有 S1K 128/32、六项正式测试及固定版本 MATH/CodeContests 官方测试划分共 43,374 条记录做了题干和数字掩码后的字符 5-gram 排重，发现 S1K 的 15 个测试重合题簇：6 个来自实际 MMLU-Pro，9 个来自 MATH 官方 test；涉及训练 10 条、验证 5 条。其中 12 例规范化题干完全一致，其余 3 例只是格式差异，相似度为 0.8478–0.9357。既有样本、checkpoint 及历史分数未改动，本轮依赖比较停止；样本属于名为 train 的来源、曾用于校准或旧配置可复用，都不能证明没有测试重合。重新开展无泄露比较须明确处理既有校准数据，并在单独产物中完成审计与训练；不能静默补样本，也不能把检测到重合直接解释为已量化的准确率偏差。

随后用户批准清洗补足并重训 A，[独立新 RUN](../../../Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_calibration/clean_s1k_20261016T030238Z/REPORT.md)将完整 S1K 1000 题加入同一次分簇，保留 145 条干净样本及其位置，按固定未选顺序补入 15 条；全部新样本从固定来源重建核对 token，原始文件保留。对已知官方 test 来源元数据也按整簇排除，避免仅依赖题干匹配。新划分须重新训练，原 checkpoint 不再满足复用契约；清洗本身不证明准确率改善。

## 机制证据要来自所研究的执行路径

[旧 semantic replay 复盘](../../../Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/LEGACY_SEMANTIC_REPLAY_POSTMORTEM.md) 记录了同一绝对位置在增量前缀与一次完整序列计算下 top1 不同，以及手工 TP 适配改变结果的反例。逻辑 causal 相同不足以证明数值执行路径相同。

该研究线后来采用真实 vLLM TP2 的 prefill、incremental KV-cache decode 和 worker hook 作为正式机制证据。局部 frozen-input 对照仍可解释算子行为，但不能代替真实长轨迹或候选模型的下游结果。该要求属于这条实验协议，不扩展成所有 HiF4_Sp 算法任务都必须逐位复现 TP2。

## 复用权重时检查预测缓存的实际位置

2026-09-24 接续[残差 LoRA 下游计划](../../../Native_NVFP4_HiF4_Linear_Puncture/experiments/residual_lora_compensation/PLAN.md)时检查到，当前 vendored lighteval 的 [SampleCache](../../../3rdparty/lighteval/src/lighteval/utils/cache_management.py) 用 `cache_dir / model_name / model_hash` 组成路径；当 `model_name` 是绝对路径时，缓存实际落在模型目录中。仅更换结果目录不足以隔离历史生成结果，短测也可能为之后的正式运行留下预测缓存。

该实验为正式评测创建独立的模型目录视图，只链接权重和配置文件，不链接缓存子目录；旧 artifact 和已有缓存保留。此做法隔离生成缓存，不改变权重、prompt 或评分协议。适用于当前缓存实现和本地绝对模型路径，其他后端应先检查自己的缓存键与读取规则。

2026-11-11 的[官方 BF16 GPQA 准备](../records/qwen3_bf16_gpqa_20261111.md)发现，旧
`Qwen3-30B-A3B` 缓存的16个权重分片与官方SHA-256全部一致，但配置含额外专家剪枝参数。
缓存目录名和完整权重不能证明当前配置仍为官方基线。该次保留旧目录，复用逐分片校验过的
权重，另从固定的官方revision下载配置/tokenizer，组成独立模型视图。此证据只确认来源与
BF16存储完整性，不证明旧附加字段在任意后端一定生效，也不替代实际推理验证。

## 生成接口参数须验证到实际后端

2026-11-03 MiniMax 的六卡整模型生成短测通过后，lighteval 初始化仍因
`GenerationParameters.top_k=-1` 被非负 schema 拒绝。当前 vLLM 的 0 与 -1 均关闭
top-k 截断，只有 `0 < top_k < vocab_size` 才启用该过滤。因此在独立 MiniMax 入口
使用 `top_k=0`，并通过真实 `GenerationParameters.to_vllm_dict()` 与 `SamplingParams`
构造检查其含义，不改公共依赖或采样协议。CPU 接口检查不替代真实任务生成与评分；
来源及后续状态见 [MiniMax 指南](../../guides/minimax_m3.md) 的修复核验和阶段日志。

## 共享 GPU 的自动显存预算与有界短测

2026-11-09 的 [GPQA 原生基线](../records/nvfp4_gpqa_20261109.md)在授权共享 GPU 4–7
时，首次 TP4 初始化失败：其他进程在模型加载期间释放显存，空闲量从 61.81 GiB
增到 62.61 GiB，触发 vLLM 自动预算的占用稳定性断言。按瞬时剩余显存设置
`gpu_memory_utilization` 不能消除这个前提问题。

给现有评测入口传递 vLLM 原生 `kv_cache_memory_bytes`，显式设置每卡 32 GiB KV，
保留最大批量的实际 `profile_run()` 后，TP4、batch256、上下文40960的64-token
执行检查完成，260个回答和原始缓存覆盖均验证，退出0。它只验证当前模型、预算与
资源条件；不是任意共享占用下的容量保证。默认不传该参数时保持原自动预算行为。

同次初始13题×10次完整长回答短测造成不必要等待，后经用户授权停止。并发检查应保留
正式引擎和批量形状，用少量解码步覆盖计算、通信及保存；评分正常/异常分支复用有依据的
专项检查。截断回答的分数不作为精度证据，完整正式评测仍保持全部题目、采样次数和输出上限。

## 推理主进程退出不等于 worker 和显存已释放

2026-10-29 的[Ban校准接续](../records/ban_trace_calibration_20261028.md)核查到原控制器
只等待主进程退出，随后立即进行GPU空闲检查；原报错缺少占用PID，不能据此断言历史
时刻被外部任务抢占。真实CPU子进程测试确认主进程退出后其子进程仍可存活。

本次将每个推理阶段放入独立session，主进程结束后只等待该session的非僵尸worker退出，
再有限等待残留GPU上下文清除；原空闲阈值不变，记录实际显存和PID。检测到其他活动
任务时不停止或等待它，下一阶段仍须通过空闲检查。`hif4`中使用已验证的psutil等待路径。
两次连续真实TP2、正式上下文/批容量的64-token验证均退出0，GPU已清空后再启动第二次；
这验证了阶段交接，不证明最终精度或任意后续资源条件。

## 离线缓存不自动遵守固定的数据版本

2026-10-08 的 235B 评测准备发现，当前 `datasets` 在 `HF_HUB_OFFLINE=1/HF_DATASETS_OFFLINE=1` 时选择最近修改的本地缓存，MMLU-Pro task 虽指定 `3373e0...`，实际得到 `b189ec...`。两个缓存均有 12,032 题，文档 ID 和最长 prompt 相同，但有 659 条记录变化；仅检查题数或 native/HiF4 两者相同不足以发现数据版本偏移。用相同 formatter/tokenizer 分别读取两个真实 Arrow，精确复现了原协议与离线准备的不同 prompt/token 哈希；恢复非离线加载后，原指定版本和原预检哈希恢复一致。此结论限于当前库与缓存条件，不宣称所有离线加载都会忽略 revision。固定评测应核对实际数据内容证据；恢复原加载条件即可时，不更改任务版本或默许缓存回退。见[诊断与修复记录](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_transfer/parallel_20261003T144300Z/REPORT.md)及其原始诊断。

2026-10-14 的 30B 模块消融进一步确认，即使进程未设置这些离线变量，镜像连接被拒绝也会触发同类缓存回退。5 个候选的单组 native/candidate 审计彼此一致，但跨时间与本轮原生基准相比，实际有 645 个 prompt、174 个 gold 不同，并集为 659 题；配置记录仍显示指定 revision。应把所有候选绑定到同一已核验的实际数据内容，而不能为每个候选重新构造一个可能随缓存变化的“native 参考”。本次隔离读取指定版本的真实 Arrow 并核对文件摘要、全部 12,032 题的 prompt/token/gold，原训练及推理/评分语义保持原位。重新导出的权重分片及在线旋转矩阵与原导出一致，真实 TP2 单题短测验证新的父/子进程数据入口；完整恢复仍须逐组核对实际题目和指标，进度以运行记录为准。见[跨版本诊断](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/mmlu_revision_recovery_20261014/revision_diagnostic.json)、[恢复验证](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/mmlu_revision_recovery_20261014/qualification.json)及[模块消融记录](../../../Native_NVFP4_HiF4_Linear_Puncture/experiments/module_design_ablation/PLAN.md)。

## 生成进度不等于已保存的回答

2026-11-11 的[BF16 GPQA](../records/qwen3_bf16_gpqa_20261111.md)按用户要求停止时，
进度条已经累计200/1980个回答，但原始预测缓存仍未落盘。当前同步lighteval路径等待
整次`LLM.generate`返回后才写缓存；不能把进度条数量当作可恢复、可评分的已保存数据。
新方案按用户要求改为三轮顺序全量评测，每轮独立缓存和保存分数；每轮内部仍受此保存边界影响。
分轮可更早得到一轮完整结果，不证明单条生成吞吐提高。

## 指标名称与实际生成数量

2026-10-25 的 [Ban 补测输入核验](../records/ban_matrix_norm_20261023.md)确认，当前 LCB 的
指标名称 `codegen_pass@1:16` 并不能证明每题生成 16 次。注册的 CodegenMetric 继承
SampleLevelComputation，未走 SamplingMetric 的采样数量分支；实际 Doc.num_samples=1。
同次源码和 CPU 核验中，AIME25 avg5 的实际数量为 5。复用本仓库协议时应同时保存注册
任务、实际 Doc.num_samples、原始缓存和请求输出数，不能仅按指标名称推断生成数量；
改变实际次数属于评测协议变更，不能为对齐名称而静默修改。具体运行覆盖与分数以原记录
后续产物为准。

## avg5 的实际评分输入与最终答案误判

2026-10-14 拆分 30B 的 AIME25 五次回答时确认，当前 `AvgAtN.compute` 逐个索引 `ModelResponse`，而 `__getitem__` 传递原始 `text`，没有传递 `text_post_processed`。本项目精简 details 只保存后者，因此不能仅对显示的 final 文本重新评分来还原原指标。原生模型 id=15 的保存 avg5 为 0.8，五个过滤后的回答重新评分却均为 1；改用原始 `GENERATIVE.parquet` 和同一单回答评分器后，原生及历史两阶段模型的全部 300 个回答均还原各题保存均值和总分。拆分回答序号时应使用原始预测缓存或原始单回答分数，并核对所有逐题均值，不能只检查总平均。这是当前库与保存格式的边界，没有更改本轮评分协议；也不把第 n 个回答列称为不同 seed 的独立实验。原始路径、文件摘要、逐次二元分数及反例见[逐次评分证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/aime_generation_scores.json)，相关完整指标见[模块消融报告](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/REPORT.md)。

2026-10-15 对 native_nvfp4、fixed_r64、matrix_norm、existing_two_stage 四组原正式轮的 600 个回答逐条复核，确认“能还原保存分数”不代表判分正确：原始文本中的思考候选/复述的 `boxed{ANSWER}` 会被高优先级模板先提取，最终解释里的面积或分式也可能压过最后提交的 boxed 整数。实测共有 14 次误判为错、1 次误判为对。仅去除思考后复用原提取器仍会漏判正确答案。用户要求重算后，按明确的最终提交答案与同一金标比对，独立核对 boxed 整数、单独判定非整数表达式，并人工复核未完成回答及冲突情况；不能从正文挑选碰巧正确的数值，也不把原始分数自动覆盖成诊断分数。结论限于已审计回答；原生基线与候选须同口径复核后才能比较，不将未审计的其他组混入修正排名。当前运行依赖未改动，原始指标与修正指标分开保存。见[完整复核方法、改判和边界例](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/aime_scoring_recomputed_20261015.md)及[逐回答重算结果](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/aime_scoring_recomputed_20261015.json)。

## 结论与状态分开

2026-11-10 核查 [Qwen3 GPQA 三组实验](../records/qwen3_mxfp4_20261109.md)时，三组
均完成 198×10、输入/原始评分核验通过，但 JSON-only 解析失败率达 50.96%–71.31%。
最终答案中的 Markdown `answer` 字段、大小写差异和 boxed 字母被记成缺失字段。
这说明完整生成和评分可复算并不证明评分器覆盖了模型的最终提交格式；短至 64 token、
尚未结束思考的执行检查也不能验证这一点。应保留原分数和缓存，统一复核各模型的
最终答案，明确区分任何扩展解析口径，不从思考正文挑选匹配金标的答案。不同模型
解析成功率不同时，不能把原始自动分数差直接归因于量化或旋转收益。
用户随后要求真实准确率，统一最终提交复核恢复了 3490 个此前漏识别的提交，原生／
直接／R64 得到 62.8788%／59.6970%／59.1414%。原始成功解析的答案全部保持不变；
15 个特殊案例在不显示模型和金标的队列中判定后再评分，14 个无最终提交的截断回答
保留 0。复核结果与原严格 JSON 分数分开保存，并核对各采样序号均值与整体一致。

2026-11-04 的[matrix_norm独立5轮实验](../records/matrix_norm_efficiency_20261104.md)提供了
开发集选优与正式精度脱钩的实例：三候选开发等权分同为0.734375，按时间选择的random50_e5
优化阶段3.82小时、相对旧20.04小时约5.25倍速度，但完整五任务平均正向精度损失为2.3047
个百分点，未达≤0.4目标。优化加速不能写成整体高保真转换成功；数据和训练轮数同时变化，
且其他候选未正式评测，不能单独归因于随机筛选或5轮训练。AIME统一最终提交口径并保留
原分；复核和报告收尾只需CPU，不重跑已完整生成的回答或可复用基线。

2026-11-03 的[Ban预算6.4对照](../records/ban_trace_calibration_20261028.md)在固定matrix_norm
模型、两组独立匹配lambda后，观察到自生成轨迹校准相对C4的LCB三轮均值低0.3810个百分点，
完整MMLU-Pro高0.5319个百分点；两项题目配对95%区间均跨0。两组专家预算相近，不足以
据此宣称自生成校准稳定改善；结论限于本轮校准、目标预算及生成，不推广为数据普遍优劣。
完整MMLU基线通过全部实际输入、推理协议和模型来源核验后复用；新基线启动失败的原始
控制器状态保留，与补齐复用证据后的对照报告分开，不把基线未重跑记成候选评测失败。

2026-11-05补齐[同预算其余任务](../records/ban_trace_calibration_20261028.md)后，C4在
AIME25未达98%保持率，自生成校准在统一最终提交口径下达到；这一方向与LCB的两组
排序不同。校准数据效果具有任务边界，已测任务的保持率不能外推为全部任务保证。
AIME配对重采样按30道题保留各题5个回答，不能当作150道独立题；单轮配对区间为正
也不代表跨seed稳定改善。人工复核缺少boxed的回答须读取完整最终提交，日志尾部或
固定长度摘要可能只剩空白或截断文本；同口径复核后分别保留原自动评分与最终提交评分。

2026-10-20 的 [B 模型 BF16 router 独立前缀实验](../../../Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_router_bf16/run_20261019T031744Z/REPORT.md)补充了“局部对齐不等于下游提升”的直接证据：固定 B 的普通权重和 norm，48 层只训练实际 gate，教师/学生各走自己的前缀；全部层选定验证 KL 不高于各自 epoch 0，完整正式评测的综合观测仍变差，主要来自 AIME。已验证非 router 分片一致及 BF16 参数/梯度/动量契约，因此不能把局部 KL 下降或实现契约通过当作准确率收益。该结果限于本轮配方；精度、教师前缀方式及目标本身的单独因果贡献尚未消融，不能据此概括为 router 对齐普遍无效。

2026-10-21 完成复核的 [B 末层与输出端词表 top-64＋tail KL 实验](../../../Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_last_head/run_20261020T031200Z/comparison.md)将这一边界扩展到最终词表目标：固定原 B 前缀、router 和基础权重，仅联合训练末层矩阵及 norm、final norm 和可融合 head matrix，验证 KL 降低约 12.2%，完整 TP2 评测却出现 MMLU-Pro 上升、AIME avg5 下降的任务折中。本轮四项平均正向损失观测变差，但变化的逐题配对 95% 区间跨 0，不能宣称稳定改善或稳定退化，也不能据此判断词表 KL 普遍无效。7 条歧义回答逐条复核后无二元改判；另核对了 7 条到达输出上限且没有最终提交的回答，均按既定口径保留 0。结果限于本轮 B 数据、候选 LR、验证选优及联合训练配方，未拆分各可学习组件的贡献。配置、源代码摘要、原始回答和复核依据见[实验入口](../../../Native_NVFP4_HiF4_Linear_Puncture/experiments/reasoning_trace_last_head/README.md)与[记录导航](../records/README.md)。

2026-10-15 的[15 组模块消融](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/REPORT.md)在约定数据版本恢复和逐题配对验证后确认，模块收益依赖组合及任务：固定 R64 相对直接转换的六项观测均提高；仅学习 DIAG、正交旋转或自由矩阵能提高 MMLU-Pro，但可能降低 ARC/AIME25；独立 norm、router 块矩阵和 λ=1 的 Top-mass KL 没有通用增益。应报告受控配对和每项结果，区分观察到的折中与稳定收益；不能用局部 NMSE/路由指标、某一个下游任务或单轮综合分证明全局最优。等预算 LR 筛选仍会选中不同 LR，因此比较的是各自选优配方；正交共享旋转与权重侧自由矩阵还同时改变正交约束和两侧绑定，不能只归因于其中一项。该结论限定于这轮 native 初始化、S1K 128/32、每层 20 epochs 及有限 LR 候选；没有消融的 NMSE/MSE、分组大小和校准数据优劣仍未建立。

2026-09-30 接续[五项全量精度恢复](../../../Native_NVFP4_HiF4_Linear_Puncture/experiments/downstream_accuracy_recovery/PLAN.md)时，核查到已有 group_top_mass 的 MMLU-Pro 300 题两次分数为 70.00% 与 74.33%。两份保存的模型配置仅模型路径不同，生成参数与引擎 seed 相同，实际输入一致，但 299/300 个生成文本不同。原因尚未证明；不能把差异直接归因于随机 seed，也不能只取最高分代表稳定收益。旧 E0 的实际 prompt 虽匹配，却使用不同的 batch token 容量，需在本轮共同配置下重新评测。

该目标已明确改用 MMLU-Pro 全部 12,032 题，并补全 GSM8K。验收为五项各自 `max(0, 原生分数−转换后分数)` 的百分点等权平均不超过 0.5；旧的 300 题结果和其他任务的提升都不能代替或抵消缺失/退步的任务。此要求限于本目标，不改变其他实验的默认样本范围。

2026-10-01 的[同协议全量验收报告](../../../Native_NVFP4_HiF4_Linear_Puncture/results/downstream_accuracy_recovery/paired_full_20260930T131000Z/REPORT.md)确认，既有 `group_top_mass` 在五项完整覆盖下达到该目标。其适用配置是逐层 per-G64 非等价矩阵与 norm、E4 DIAG 初始化、block-delta NMSE + top_mass router KL、既有 S1K 数据；没有新增 LoRA 或训练候选。实测支持这套完整配置在当前评测协议下的精度恢复，不单独证明 router KL 的因果贡献，也不保证每一项下降都小于 0.5 或所有随机种子均达标。可复用做法是先补齐已有强候选的同协议全量证据，再决定是否追加训练；量化后反量化的 BF16 存储仍不能当作 4-bit 存储/性能实测。

[2026-09-16 局部算术诊断](../../../Native_NVFP4_HiF4_Linear_Puncture/results/internal_error_accumulation/qwen3_30b_a3b_e0_e1_formal/runs/iea_20260909T081436Z_1076681/60_objective/corrected_objectives_v3/kernel_boundary_alignment_v1/CONCLUSIONS.md) 明确限定在固定输入、零 DIAG、单 predictor 范围，不能推广成非零 DIAG、完整反向或全序列已验证。该日的 BLOCKED / WAITING_REVIEW 同样只是当时状态。

普通 TP 归约、累加和舍入差异依 [AGENTS.md](../../../AGENTS.md) 判断是否影响算法结论；局部严格契约仍按协议验证。报告存在、进程退出或某项局部对齐通过，不自动等于全实验完成，也不授权进入后续阶段。

证据导航见 [实验记录](../records/README.md)，计划替代关系见 [方案入口](../../../Native_NVFP4_HiF4_Linear_Puncture/plans/qwen3_30b_a3b/README.md)。
