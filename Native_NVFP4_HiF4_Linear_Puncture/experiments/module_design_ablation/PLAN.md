# Qwen3-30B-A3B 模块设计消融

状态（2026-10-15 全量核查）：33 次学习率筛选、11 组正式训练和 15 组模块消融的六项评测全部完成。5 组 MMLU-Pro 固定版本恢复均已完成；全部 15 组的 reasoning 逐题 ID/prompt/gold 与中央原生参考一致，逐题均值还原正式分数，原 93 个限定源码摘要保持不变。当前 comparison.json 为 complete=true，原错误版本的分数不进入比较。历史达标组追加的三轮 AIME25 已全部完成；用户随后追加的 matrix_norm 三轮已启动、运行中，均不覆盖原消融表。完整分数、模块差值、逐题区间和成本见[同协议结果报告](../../results/module_design_ablation/formal_20261008T074812Z/REPORT.md)、[机器可读比较](../../results/module_design_ablation/formal_20261008T074812Z/comparison.json)及[逐题内容与配对分析](../../results/module_design_ablation/formal_20261008T074812Z/module_pairwise_analysis.json)。

额外三轮 AIME25 于 2026-10-15 12:31:32（北京时间）结束，均为退出码 0；正确数为 93/150、93/150、95/150，对应 62.0000%、62.0000%、63.3333%，三轮合并 281/450（62.4444%）。最终核验 96 个限定源码、75 个原消融指标文件未变；三轮各 30 题×5、逐题 ID/prompt/token/gold 与原生参考一致，指标可由保存的逐题均值还原。最好一轮的四项 reasoning 平均正向损失为 1.3751 pp；按用户要求换入原生 LCB54.2857% 和达标组旧分数后为 1.0748 pp；双方五项均换回旧成对验收分数后为 1.0569 pp，仍未达 0.5 pp。全部三轮使用 seed=1234，保留三轮合并及事后选择的最好完整一轮，不声称跨 seed 稳定达标。原生/历史模型和生成缓存继续保留，本次未新增训练或清理。见[最终核验与分数代入分析](../../results/module_design_ablation/formal_20261008T074812Z/aime_repeat_final_analysis.json)和[原运行汇总](../../results/aime25_repeats/group_top_mass_20261014T113500Z/summary.json)。

用户于 2026-10-15 追加要求 matrix_norm 重测 AIME25，按前次相同预算安排 3 轮，每轮 30 题×5、seed=1234、全新模型视图/预测缓存，最后选择 avg5 最高的完整一轮；保留全部轮次和三轮合并，不跨轮逐题拼接，不覆盖原始 64% 或原消融比较。原导出分片已清理，本次在独立目录重导出；49 个权重分片与原清理记录 SHA256 完全一致，48 层 selected 参数、96 个限定源码、运行时配置以及 AIME25 prompt/token/gold 均已核对，复用同一模型既有完整真实 TP2 路径证据。后台编排于 2026-10-15 17:01:51（北京时间）启动，PID 3018096，GPU 0,1 顺序运行；首次交付状态为“已启动，运行中”，不把排队或原有分数当成新结果。固定其余三项后，本轮正式基线需至少 101/150（67.333%），仅将原生 LCB 换为 54.2857% 的代入口径需至少 99/150（66%），才达到四项 reasoning 平均正向损失≤0.5 pp。本次保留重导出文件，未新增清理授权或训练。见[配置与源码摘要](../../results/aime25_repeats/matrix_norm_20261015T081958Z/manifest.json)、[导出和原正式路径核验](../../results/aime25_repeats/matrix_norm_20261015T081958Z/qualification.json)、[启动命令](../../results/aime25_repeats/matrix_norm_20261015T081958Z/launch.json)、[状态](../../results/aime25_repeats/matrix_norm_20261015T081958Z/status.json)和[编排日志](../../results/aime25_repeats/matrix_norm_20261015T081958Z/launcher.log)。

用户于 2026-10-15 追加要求原生 native_nvfp4 的 AIME25 重测，查看分数是否低于原 67.3333%。按前两组相同预算安排 3 轮，每轮 30 题×5，seed=1234、全新模型视图/预测缓存，原生 packed NVFP4 snapshot 不变。复用原完整真实 TP2 结果作为执行证据：96 个限定源码保持一致，4 个实际权重分片解析路径、模型/量化配置、30 题 prompt/token/gold、原采样与评分协议均已核对。编排于 2026-10-15 18:35:32（北京时间）启动，PID 3067217，第一轮评测 PID 3067294，使用已授权且空闲的 GPU 6,7 顺序运行，与 GPU 0,1 的 matrix_norm 编排独立；没有停止或修改其他任务。原生上次完整一轮耗时约 1 h 56 min，三轮估计约 6 h。保留每轮原始输出、最低/最高完整轮次、合并均值及相对原始分数的变化；最低值仅作观察，不自动覆盖原基线或声称跨 seed 稳定下降。没有重新训练、导出或删除模型。见[配置与源码摘要](../../results/aime25_repeats/native_nvfp4_20261015T103435Z/manifest.json)、[原完整路径和输入核验](../../results/aime25_repeats/native_nvfp4_20261015T103435Z/qualification.json)、[启动命令](../../results/aime25_repeats/native_nvfp4_20261015T103435Z/launch.json)、[状态](../../results/aime25_repeats/native_nvfp4_20261015T103435Z/status.json)和[编排日志](../../results/aime25_repeats/native_nvfp4_20261015T103435Z/launcher.log)。

恢复编排 PID 1927298 于 2026-10-15T02:26:19Z 正常结束。current_kl 的规范 MMLU-Pro 为 74.7673%，历史两阶段参考为 75.3989%。全部 5 组已从原 selected 参数重导出、核验同一权重及约定数据内容，并完成六项指标验证和授权 BF16 分片清理。状态/任务退出码及替换证据见[恢复状态](../../results/module_design_ablation/formal_20261008T074812Z/mmlu_revision_recovery_20261014/status.json)、[替换索引](../../results/module_design_ablation/formal_20261008T074812Z/mmlu_recovery.json)和[比较有效性记录](../../results/module_design_ablation/formal_20261008T074812Z/comparison_validity.json)。原编排未经跨版本审计的 comparison/report 继续保留在恢复目录。

问题来源已验证：镜像连接失败后，datasets 回退到缓存 revision b189ec...，配置仍记录约定的 3373e0...；两版都为 12,032 题，但 645 个 prompt、174 个 gold 不同，并集 659 题。恢复从原 selected 参数重新导出，读取隔离保存、文件摘要及完整 prompt/token/gold 已核验的约定 Arrow；父进程和实际生成子进程均绑定同一内容。原训练、推理和评分的 93 个限定源码保持不变。见[版本诊断](../../results/module_design_ablation/formal_20261008T074812Z/mmlu_revision_recovery_20261014/revision_diagnostic.json)、[固定数据和真实 TP2 验证](../../results/module_design_ablation/formal_20261008T074812Z/mmlu_revision_recovery_20261014/qualification.json)及[比较有效性状态](../../results/module_design_ablation/formal_20261008T074812Z/comparison_validity.json)。

已恢复的模块配对显示：在 matrix+norm 上增加 router 块矩阵（无 KL），MMLU-Pro 为 -0.540 pp，逐项 95% 区间 [-1.072, -0.008]，其他任务除 GSM8K 外观测下降；matrix_norm_kl 六项观测均低于 matrix_norm，其中 LCB 为 -5.143 pp，逐项区间 [-10.286, -0.571]。正交 rotation 加 norm 的 MMLU-Pro 为 -1.413 pp，但 ARC-C 提高 2.730 pp；两组还选择了不同 LR。以上是本轮有限预算的选优配方比较，区间未做多重校正，current_kl 加 KL 相对 current 的 MMLU-Pro 为 -0.025 pp、AIME25 为 -6.667 pp，LCB/ARC-C 则有观测提升，仍是任务间取舍。成本和筛选边界见[训练成本](../../results/module_design_ablation/formal_20261008T074812Z/training_costs.json)。

用户于 2026-10-14 另行授权历史达标组 group_top_mass 的 3 轮 AIME25 重复评测，并于 2026-10-15 明确要求使用空闲 GPU 0,1 立即运行。已核验并仅停止原等待程序 PID 2323318（尚无评测子进程），原状态/配置/启动记录已归档；恢复编排 PID 1927298 和其 current_kl 评测不受影响。新的脱离会话编排 PID 2806450，第一轮评测 PID 2806521，三轮在 GPU 0,1 顺序运行，与 GPU 6,7 的 MMLU-Pro 并行。每轮 30 题×5 次、seed=1234、全新模型视图/预测缓存，沿用现有完整正式入口及其已完成的 TP2 证据，模型、采样和评分协议保持不变，不称为独立 seed 实验。固定本轮原生基准和其他五项后，每轮至少 100/150 正确（avg5≥66.667%），六项平均正向损失才≤0.5 pp。保留全部结果、整体均值和最佳完整一轮；挑选后的最好观测不能证明稳定达标，也不覆盖原消融结果。见[当前重复评测配置](../../results/aime25_repeats/group_top_mass_20261014T113500Z/manifest_immediate.json)、[当前启动记录](../../results/aime25_repeats/group_top_mass_20261014T113500Z/launch_immediate.json)、[调度更改及停止核验](../../results/aime25_repeats/group_top_mass_20261014T113500Z/reschedule.json)、[状态](../../results/aime25_repeats/group_top_mass_20261014T113500Z/status.json)及[日志](../../results/aime25_repeats/group_top_mass_20261014T113500Z/launcher_immediate.log)。

按每题第 1～5 个回答拆分原始 AIME25 生成缓存后，原生正确数为 20/21/21/18/21，历史组为 17/22/19/18/17；所有逐题均值和最终 avg5 均与原保存指标一致。使用原评分器及原始 GENERATIVE.parquet，不能用过滤后的 final 文本替代原评分输入。原始路径、SHA 与二元评分见[逐次评分记录](../../results/module_design_ablation/formal_20261008T074812Z/aime_generation_scores.json)。

本轮原 13 组 BF16 分片已按授权清理约 739.35 GiB；恢复完成的 5 组仅在规范六项验证后清理可再导出的 BF16 分片。训练参数、旋转矩阵、配置、日志、原始指标/生成缓存和原生/历史模型保留，current_kl 的训练参数及 AIME 重复引用的历史模型受保护。GPU 授权仍为 0/1/6/7；没有停止或修改正在运行的实验。

2026-10-08 的编排修复记录保留于[替换操作、实际命令和推进证据](../../results/module_design_ablation/driver_repair_20261008T084419Z/replacement.json)及[编排日志](../../results/module_design_ablation/driver_repair_20261008T084419Z/driver.log)。当时保留的训练 PID `1938866` / `1936340` 已分别完成 rotation_norm_kl / current_kl；二者非新编排子进程，退出码保持 null，完成依据为进程退出及完整训练产物核验，不把无法观测的退出码改写为 0。

本次故障原因：`hif4` Python 3.11.13 未提供 `os.pidfd_open`，原 GPU 0/1 等待线程在领取任务前失败；异常未进入 job 错误记录，线程池退出等待其他线程，原状态仍显示 running。用户明确授权后，仅向原编排 PID `1926160` 发送 SIGTERM，没有向进程组或训练进程发信号。修复仅更改 `run.py`，启动前核验已完成的原生 AIME 短测，接管既有训练，并及时记录线程级异常；训练、校准、损失、导出和评测源码均保持原哈希。见[修复补丁](../../results/module_design_ablation/driver_repair_20261008T084419Z/driver.patch)、[修复验证](../../results/module_design_ablation/driver_repair_20261008T084419Z/validation.json)及[修订后的验证记录](../../results/module_design_ablation/driver_repair_20261008T084419Z/qualification/qualification.json)。

[原启动命令](../../results/module_design_ablation/formal_20261008T074812Z.launch.json)与[配置及源码哈希](../../results/module_design_ablation/formal_20261008T074812Z/manifest.json)保留原始记录；manifest 的 driver_revisions 记录编排源码修订。原生 AIME 的 [metrics](../../results/module_design_ablation/qualification_20261008T064919Z/aime25_native/metrics.json)为 1 题×5 次的执行验证，不能作为正式精度结果。

常规新运行入口为 `python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.module_design_ablation.run --output_dir <new-run> --qualification <qualification-dir>`。本次实际接管使用同一 RUN 加 `--resume_driver`，完整命令见替换记录；已有编排运行时会拒绝再次接管，不应重复启动。本次已确认四卡实际更新，交付后停止主动监控；完整 15 组、每层 20 epochs、等预算 LR 筛选及六项完整评测协议保持不变。

## 已澄清的主要目标（2026-10-15）

用户随后明确要求原生指标整套换回 LCB=54.2857% 的旧验收版本：MMLU-Pro/GSM8K 也采用该版，AIME25 因旧验收未测而沿用后来实测值。延续历史达标组旧五项与最好完整 AIME 轮次的选择；matrix_norm 追加三轮尚未全部完成时保留原消融分数并另列已完成轮次。最新重算见[完整旧原生基线比较](../../results/module_design_ablation/formal_20261008T074812Z/historical_native_comparison.json)，报告开头展示该表；原 comparison.json 与同轮配对证据保留，活动重测配置不改动。

格式转换后的主要目标是 reasoning 数据集平均损失降低：MMLU-Pro、LCB、GSM8K、AIME25 avg5 四项各占 1/4，沿用每项 max(0, native−candidate) 的正向下降定义。ARC-E/C 继续保留完整结果作辅助，不计入主要排名。六项评测协议、模型和生成数据保持不变，原五项/六项均值仅作补充。完整 15 组按四项主目标重算后，matrix_norm 为 1.2250 pp，历史两阶段为 1.7084 pp，固定 R64 为 1.8024 pp，current_kl 为 4.1282 pp；仅 MMLU-Pro 的提升不足以认定模块有益。主目标配对区间与完整来源见[reasoning 分析](../../results/module_design_ablation/formal_20261008T074812Z/reasoning_objective_analysis.json)和[当前报告](../../results/module_design_ablation/formal_20261008T074812Z/REPORT.md)。

按四项 reasoning 均值≤0.5 pp 判断历史组的额外 AIME 重复，至少需 101/150 正确（67.333%）；先前 100/150 是原六项口径的门槛。重复队列的生成和评分未改动，原六项汇总保留，新主目标由报告另行重算。用户另行要求的历史分数代入口径至少需 99/150（66%），仍高于最好一轮的 95/150；历史分数代入只作敏感性分析，不覆盖同轮正式比较或复用其配对区间。

## 目标与已知边界

比较 norm、可融合 DIAG、激活/权重两侧共用的可学习 G64 旋转、当前权重侧非等价矩阵，以及 top_mass router KL 的贡献。所有正式候选都需要 MMLU-Pro、LCB、GSM8K、ARC-E、ARC-C、AIME25 avg5 的完整约定结果，不能只给局部重构指标。

目标模型沿用当前报告的 `nvidia/Qwen3-30B-A3B-NVFP4`，源 snapshot 为 `2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`。实验代码与结果独立；旧训练、评测和活动任务依赖保持原位。

结论只能说明给定数据、训练预算和所比较候选中的相对优势，不能据有限消融证明全局最优。同时比较下游效果、训练时间、可训练参数量，以及在线旋转的部署成本。

## 当前实现核查

- [已有 group manifest](../../results/non_equivalent_reconstruction/full48_20260915T084300Z/group_top_mass/manifest.json)：先使用 E4 adopted DIAG，再学习两组 norm 和 per-G64 权重侧矩阵；128/32 条 S1K 完整文本，seed=42，48 层、20 epochs/层、batch=4、lr=1e-4。
- [参数定义](../non_equivalent_reconstruction/transforms.py)：普通矩阵从固定 R64 初始化，router 矩阵从 I64 初始化；训练期间没有正交约束。称它为“学习后的旋转矩阵”不准确，应称权重侧自由矩阵。
- [主损失](../non_equivalent_reconstruction/train.py)：同一 progressive student 输入上取原生 NVFP4 teacher 的层输出；block-delta NMSE 加 top_mass router KL。
- [router 辅助分支](../non_equivalent_reconstruction/student.py)：在 MoE norm 前 detach。KL 直接更新独立 MoE norm/router 矩阵，不直接更新 attention。NMSE 仍更新整个当前层的可训练参数。
- [DIAG 融合](../e2e_diag_reconstruction/core/moe_fold.py)：D_QKV、D_GU、D_VO、D_UD 成对补偿，router 使用 D_GU 的逆补偿。DIAG 组的 norm 数值可因融合而改变，但不等于增加独立 norm 自由参数。
- [现有导出](../non_equivalent_reconstruction/materialize.py)与 vLLM runtime 使用固定在线 R64；目前不能直接部署任意学习后的 G64 旋转。对称旋转候选需要真实训练、导出和推理路径，不能训练后换回固定 R64。
- [AIME 本地任务](../../../tasks/aime.py)已定义 `aime25_avg5|0`；`AvgAtN.compute` 对五次回答逐一评分后取均值。历史明细核对为 30 题、每题 5 个回答，但历史分数不自动进入本次对照。

## 已确认：共同起点（2026-10-08）

用户明确选择“原生 NVFP4 起步”。所有用于归因的新候选从同一个原生 NVFP4 出发，不继承已学习 E4 DIAG；已有 E4 → group_top_mass 另列完整两阶段参考。这一选择已确定，后续不再重复确认。

具体初始化边界：

- 基础权重来自上述固定的原生 packed NVFP4 snapshot，按现有 NVFP4 定义解码；teacher 保持原生 NVFP4 量化语义。不得改用 BF16 原始模型充当 teacher。
- 独立 norm 从原生对应向量初始化；可融合 DIAG 从单位缩放初始化（log2 参数为 0）；需要 R64 的普通投影矩阵从固定 R64 初始化；可学习 router 矩阵从 I64 初始化。
- 新组不读取旧 E4 `conversion_state.pt` 或旧 group 的 learned 参数；固定 R64 只作为明确列出的确定性初始化变换，不携带校准学习结果。
- Direct HiF4 组不加 R64；其余带固定 R64 的无训练/可训练组按各自参数定义启动。每个新配置独立保存来源、初始化定义和输出，不从另一个消融组接续更新。
- 旧 `non_equivalent_reconstruction.train` 的默认 E4 加载不适合作为新消融入口；新实现应直接使用 native layer state。保留旧入口及其默认值，避免影响其他任务。

现有两阶段模型与从头组的差异同时包含初始化和额外训练预算；如需要单独证明初始化收益，另做预算匹配对照，不能用这一个差值直接归因。

按当前源 config 与现有 per-G64 定义推导，48 层的独立 norm 共 196,608 个参数；可融合 DIAG 共 4,939,776 个；普通投影自由矩阵共 1,944,059,904 个；router 自由矩阵另有 6,291,456 个。当前组合共 1,950,547,968 个可训练参数。对称旋转采用每 G64 2,016 个 Cayley 坐标，共 956,841,984 个参数，不含可选独立 norm。这些是结构推导，不是运行资源实测。

## 已确认的对照矩阵（2026-10-08）

共同的固定旋转底座为：普通投影激活侧与权重侧均用 R64，router 保持原生 BF16。未列为可学习的参数均冻结。普通投影包含 Q/K/V/O 与各专家 gate/up/down。

| 组 | 可学习部分 | router 独立矩阵 | KL 设置 | 回答的问题 |
| --- | --- | --- | --- | --- |
| Native NVFP4 | 无 | 无 | 无 | 原生基准 |
| Direct HiF4 | 无；不加 R64 | 无 | 无 | 直接格式转换损失 |
| Fixed R64 HiF4 | 无；固定 R64 | 无 | 无 | 固定旋转本身的贡献 |
| Norm only | input norm、MoE norm | 无 | 0 / 1 配对 | 只调 norm 的效果与 KL 贡献 |
| Fusable DIAG only | D_QKV、D_GU、D_VO、D_UD | 无；保留严格逆补偿 | 主组为 0 | 可融合等价缩放的贡献 |
| Tied rotation only | 普通投影两侧共用的正交 Q64 | 无 | 主组为 0 | 单独学习对称旋转 |
| Tied rotation + norm | Q64 与两组 norm | 无 | 0 / 1 配对 | norm 增益，以及与自由矩阵的对照 |
| Weight matrix only | 普通投影权重侧自由矩阵 | 无 | 主组为 0 | 单独学习权重侧矩阵 |
| Weight matrix + norm | 普通投影权重侧自由矩阵与两组 norm | 无 | 0 / 1 配对 | 与 tied rotation + norm 对照 |
| Current parameterization | 普通投影/router 权重侧自由矩阵与两组 norm | 有 | 0 / 1 配对 | 当前参数化、router 矩阵增益与 KL 增益 |
| Existing two-stage reference | 已有 E4 → group_top_mass | 有 | 已有 λ=1 | 与实际已有达标模型比较 |

用户已明确选择完整 15 组：3 个无训练基准、11 个新训练配置、1 个历史权重参考。GPU 范围为 0/1/6/7；2–5 有其他活动任务，不纳入本轮调度。

主比较按匹配参数化和训练预算做成对差值。Norm only 与 Fixed R64、DIAG only 与 Fixed R64、Tied rotation only 与 Tied rotation + norm、Weight matrix only 与 Weight matrix + norm、无 router/有 router 两组，以及同参数化 λ=0/1 各自回答不同问题。

Tied rotation 与自由权重矩阵的比较同时改变两侧绑定和正交约束，可判断两种设计整体效果，不能再细分成其中某一个约束的独立因果效应；只有结果需要进一步归因时才提出额外对照。

## 对称旋转的数学定义与 KL 边界

用户已明确选择“保持正交，比较等价的可学习旋转”。每个 G64 的激活与权重共用同一个可学习正交 Q，初始化为 R64：

```text
activation: xQ
weight:     WQ
Q Q^T = I
未量化时：(xQ)(WQ)^T = xW^T
```

“两侧共用”不要求 Q 本身等于 Q^T。未选择仅共享自由矩阵的非等价变体。实现采用反对称参数的 Cayley 映射：`A = skew(theta)`，`Q = solve(I - A/2, I + A/2) @ R64`；theta=0 时准确恢复 R64，每组 2,016 个独立参数。这个连续参数化始终保持实数意义的正交，有限精度误差需验证；其 chart 不覆盖相对 R64 恰有 -1 特征值的边界，不能称为无坐标限制的全空间优化。优化器与学习率的正式选择仍需结合验证预算确定。不同投影/专家的 G64 粒度与当前自由矩阵保持一致，O 投影不跨 head，TP 切分遵循实际权重分片。

当前源码约束：vLLM 已有在线 DIAG 分支能分别处理 Q/K/V 及专家 gate/up/down 的不同变换，但其计算仍限定为逐通道 scale 与固定 R64。新的学习矩阵需要覆盖这些分支，并正确切分 O/down 的 TP 输入组；不能为了沿用单次共享激活量化而暗中把不同投影或专家的可学习矩阵绑定为同一矩阵。不修改活动实验正在依赖的代码。

router 保留 BF16，主对称旋转组不人为给 router 增加一对能相互抵消的旋转。其独立自由矩阵贡献用上表的无/有 router 矩阵配对解释。

沿用当前 MoE norm 前 detach 边界时：纯普通投影旋转/矩阵组没有 router KL 直接可达参数；纯 DIAG 组的 D_GU 与 router 逆补偿在实数运算下抵消，有限精度影响需要实际梯度核查，不能预先宣称严格零梯度，也不能把只靠舍入产生的差异当成普遍的 KL 优化收益。这些组不盲目增加同名 KL 训练重复。若要允许 KL 反传 attention，必须作为独立梯度范围变体确认，不能暗改当前定义。

## 数据、损失与训练预算

- 沿用现有 S1K 完整文本缓存：128 train / 32 validation，seed=42，无截断；实际分别为 1,253,457 / 312,662 tokens，源行互斥。各组复用相同样本和批次顺序。
- 逐层 0→47，各组形成自己的 progressive student 输入；teacher 与 student 当前层消费同一个输入。使用一致的 causal attention 和有效 token mask。旧 E4 历史路径不能未经核查直接当成这个协议下的纯 DIAG 结果。
- `L_NMSE = sum(||h_s - h_t||²) / sum(||h_t - x||²)`，求和覆盖当前 batch 有效 token 与通道。
- `L = L_NMSE + λ L_router`；top_mass 使用全部专家归一化、teacher top-8 加外部概率质量、T=1、有效 token 平均，λ 配对为 0/1。
- 用户已确认正式 20 epochs/层、等预算学习率筛选；batch=4、AdamW、weight decay=0、cosine 至 0。每个训练配置独立筛选 3 个 LR multiplier（0.3/1/3），每个候选从原生第 0 层开始，完整 128/32 条、3 epochs，使用同一批次顺序和各自验证总目标选优。norm/自由矩阵基础 lr=1e-4，DIAG=5e-3，Cayley 坐标=1e-3；组合组的 norm 仍用其自身基础 lr。DIAG 沿用 log2 范围 [-4,4]。不使用下游测试集选择。各组筛选额外优化步数相同，但实测耗时可以不同。
- 筛选结束后按选中 multiplier 从原生初始化重训完整 48 层；不得接续筛选权重。第 0 层短程排序不保证迁移到后续层和 20 epochs，应记录正式曲线中的排序/稳定性问题；这种有限搜索不能证明最优学习率。KL 配对包含等预算调参和各自验证目标选择的整体效果。
- 所有候选使用同一 checkpoint 选择规则。若沿用当前各自验证总目标最优，则 KL 对比包括训练与选择规则的整体效果，须记录；不能把不同 λ 的总目标数值直接跨组排名。
- 保留每层 NMSE、router KL、路由重合率、训练/验证曲线、选中 epoch、可训练参数数目及计时。原始 NVFP4 bits、冻结状态、保存/重载契约保持可核验。

## 六项下游协议

用户已明确选择 MMLU-Pro 全量 12,032 题；本次按全量执行，不使用项目通用默认的 300 题。其余任务采用下表覆盖。

| 任务 | 覆盖 | 指标与路径 |
| --- | --- | --- |
| MMLU-Pro | 全部 12,032 题 | extractive match；根目录 main.py + vLLM/lighteval |
| LCB | 固定 v6 队列 175 题 | 官方 checker，单次 pass@1；同上 |
| GSM8K | test 全部 1,319 题 | extractive match；同上 |
| ARC-E | test 全部 2,376 题 | raw accuracy；lm_eval + vLLM |
| ARC-C | test 全部 1,172 题 | raw accuracy；lm_eval + vLLM |
| AIME25 | 全部 30 题，每题 5 次，共 150 个回答 | 数学匹配后的平均正确率，非 pass@5/多数投票；main.py + vLLM/lighteval |

沿用 TP2、BF16 KV、eager、4096 batch tokens、reasoning thinking 开启、temperature=0.6/top-p=0.95/top-k=20/min-p=0、seed=1234。AIME 沿用现有 38,912 输出上限；MMLU-Pro/GSM8K 为 32,768；LCB 为 38,912；reasoning context=40,960、max_num_seqs=128。ARC seed=42、context=4096。

AIME 新接入五项 wrapper 时需把固定的“一题一个回答”检查推广成任务定义的生成次数，并复核每题五个输出、逐次评分和最终聚合。AIME 不应把同题五次回答当成 150 道独立题来报告题目维度的不确定性。

所有候选核对实际数据版本、样本 ID、prompt/token/template、采样与评分配置；模型视图隔离预测缓存。可复用已有五项结果仅限这些证据及实际模型/协议完全匹配。旧 AIME 分数先作历史定位，不能直接填入当前表。

报告六项逐任务分数、相对 Native 的带符号百分点差值、各消融配对差值及成本。不把旧五项 ≤0.5 门槛自动扩展成新的六项总目标，不用小任务涨分掩盖另一个任务的回退。AIME 题数少；微小差异不能直接判定更优。

## 已完成的独立 AIME 准备

- [aime.py](aime.py)：复用既有 reasoning 引擎配置；校验真实任务请求每题五次生成，记录实际数据指纹及逐题 prompt/token/gold 哈希，检查 30 题覆盖、ID 唯一、五个回答、官方 avg5 指标和逐题均值。
- [evaluate_aime.py](evaluate_aime.py)：独立 CPU audit / TP2 evaluate 入口，复用根目录 main.py 和已有 scoring/runtime；每次新建只链接模型资产的视图，隔离预测缓存，记录实际命令、源码哈希、退出码和检查结果。未修改旧五项入口或任务定义。
- [回归测试](tests/test_aime.py)：16 项 CPU 检查通过，包含真实官方数学评分器对三对两错回答得到 0.6，及缺题、重复、prompt/答案变化、生成次数、均值错误和缓存隔离。
- 原生与既有 group 模型的完整 AIME 数据、tokenizer、prompt/token/gold 审计一致；30 题，最长输入 847 tokens，38912 输出上限符合 40960 context。
- 历史真实明细的 30 题/150 回答及聚合通过检查，但历史引擎使用 TP1、2048 batch tokens、auto KV，与拟议 TP2 协议不同，不能作为当前正式结果。共享 `max_samples=300` 对仅有 30 题的 AIME 不缩小覆盖，不以这个数值差异单独拒绝复用。
- 证据与源码哈希：[CPU 准备记录](../../results/module_design_ablation/preparation_20261008T052348Z/validation.json)；[测试日志](../../results/module_design_ablation/preparation_20261008T052348Z/pytest.log)。这些只证明数据/评分检查与入口准备，不代表 GPU 生成通过，更不代表六项实验完成。

已激活 `hif4` 的调用者 shell 可使用：

```bash
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.module_design_ablation.evaluate_aime audit \
  --model_dir /absolute/path/to/model --output /absolute/path/to/new/aime_audit.json
```

2026-10-08 新增执行证据：

- [真实第 0 层参数诊断](../../results/module_design_ablation/layer_diagnostic_20261008T064500Z/validation.json)：13 个新转换配置前向，11 个训练配置各两次更新和严格保存重载；新 current 初值输出与旧 Student 在同一原生 base 上精确一致。使用 64-token 片段，只验证参数/梯度性质，不充当最长 batch 验证。
- [最长 batch 与两层衔接短测](../../results/module_design_ablation/qualification_20261008T064919Z/transitions/)：rotation_norm_kl、current_kl、diag 均使用最长 4 条训练和最长 4 条验证样本、完整序列、batch=4，逐层 0→1 各两轮更新/验证，覆盖所选参数重载和 progressive hidden 传播。对应 `summary.json` 完成；这是新原生入口证据，不是正式训练。rotation_norm_kl 的峰值 allocated 显存约 46.25 GiB，current_kl 约 31.97 GiB。
- [真实 MoE kernel 对照](../../results/module_design_ablation/qualification_20261008T064919Z/kernels.json)：512 个真实嵌入 token、128 experts、更新后的独立旋转；激活 QDQ 在本次输入上相同，MoE 输出相对 L2 差异约 0.00090。新路径保留 SiLU 后 BF16 舍入，再乘 up；直接复用旧 Online-DIAG fused helper 会省掉这一边界，已在新路径中修正。
- CPU 回归为 20 项通过。完整旋转诊断模型已导出；[TP2 ARC](../../results/module_design_ablation/qualification_20261008T064919Z/arc_v2/validation.json) 与 [AIME 一题五次生成](../../results/module_design_ablation/qualification_20261008T064919Z/aime25_v3/metrics.json) 均通过，后者保留 38912 输出上限和正式引擎容量。实际 worker/量化配置从引擎对象记录，不依赖 INFO 日志格式。诊断模型只有第 0 层经过短测更新，不作为正式候选结果。

## 资源与导出保留

用户授权 GPU 0/1/6/7。训练以独立单卡任务安排；评测使用 0,1 与 6,7 两对 TP2，正式调度前再检查占用。阶段未使用的卡不额外占卡。

[原启动验证记录](../../results/module_design_ablation/qualification_20261008T064919Z/qualification.json) 保留启动时事实，其中原生 AIME 为 pending。该短测现已完成；原先依赖 `os.pidfd_open` 的等待分支在本机环境不可用，修复候选改为启动编排前核验已完成的原生 AIME 结果，正式六项评测前仍检查该结果。接管已有训练使用 Linux `/proc` 的命令和进程开始时间核对身份，等待其结束后验证全部训练产物；无法取得的非子进程退出码记为 null，不伪造 exit=0。修复已按用户授权应用，GPU 6/7 原训练已接管、GPU 0/1 已完成筛选并进入正式训练；非子进程退出后的最终产物校验尚未触发，不能写成完整训练已通过。

当前 /home 实测剩余约 740 GB，全部 BF16 导出加训练参数预计超过 800 GB。用户明确授权：本轮每组完成六项评测并核验后删除该组可重建的 BF16 导出权重；保留原生模型、所选训练参数、学习后的旋转矩阵、配置、日志和全部明细，不删除历史产物。实现须按输出索引列出精确文件并记录删除前大小/哈希，仅清理本轮独立 RUN 内导出的 BF16 shards。每次至多两组同时导出/评测，控制峰值空间。

## 实现与验证顺序

1. 共同原生起点、正交旋转、15 组矩阵、全量评测、训练预算、GPU 和导出清理已确认；完成必要真实路径验证，已有确定项不重复问。
2. 独立实现参数选择、训练和保存；复用 checkpoint、量化和损失底层。评测用统一六项入口和覆盖检查，不修改活动实验的依赖。
3. 对称旋转补齐实际导出/runtime，验证量化前等价、量化后的真实算子、参数/梯度边界、正交约束、保存重载及 TP2 prefill/decode；仅局部 loss 下降不足以通过。
4. 正式形状的短测保留完整模型、最长样本/批量、实际量化、更新和保存流程，只缩短迭代次数。针对所选正式路径检查多步推进与显存，不穷举无关配置。
5. 在正式代码/配置稳定且短测充分后，按授权 GPU 范围启动独立后台任务，确认主循环推进即交付；不轮询到实验结束，不占用或停止他人进程。
6. 完成后核实所有组、六项结果及配对协议，输出比较表和适用范围，再根据证据提出有针对性的改进方向。

## 参考经验与尚待证明的内容

参考[评测证据经验](../../../docs/experience/lessons/evaluation_evidence.md)：实际 prompt 一致、预测缓存隔离、固定 revision 内容核验、单次采样不代表统计显著。条件变化为新增 AIME 五次采样和学习后的在线旋转，因此相应输出覆盖与 runtime 需要新证据。

参考[训练执行经验](../../../docs/experience/lessons/model_transfer_execution.md)：NMSE 的全局能量分母与 router KL 的有效 token 平均是不同归一化；若将来并行不能简单平均各卡 NMSE，也不能给未使用专家补零梯度改变 Adam 状态。当前尚未选择并行训练配置。

经验增量：已记录 SiLU 的 BF16 中间舍入和本机 Python 缺失 pidfd API 时的编排失败。两者均为执行证据，没有新下游实测收益；编排修复已完成正式接管，11 个训练配置均已完成；完整下游有效性仍待 15 组结果。2026-10-12 新增的是三个基准的正式指标，通用验证经验无新增。
