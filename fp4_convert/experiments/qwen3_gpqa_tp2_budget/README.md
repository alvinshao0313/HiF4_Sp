# G512 与 UOS 的 TP2 并行及 KV 容量验证

2026-11-13 用户授权：若扩大显存预算能加快速度，可以重启 G512；GPU0–1
运行 G512，GPU2–3 补跑此前停止的 UOS c7.25；GPU4–7 不安排本任务。
G256 继续取消。两个候选各自执行完整 GPQA Diamond 198题×10，不分摊回答。

沿用 AISBench prompt、本地固定数据、thinking、seed1234、temperature0.6、
top_p0.95、top_k20、min_p0、最大生成32768、上下文40960、batch256、
8192 batched tokens、BF16 KV 和 eager。量化算法、checkpoint、非专家路径不变。
本次明确改变的条件为 UOS 的 TP4→TP2，以及通过吞吐比较后选择的 KV 容量。

`memory_probe.py` 使用既有 G512 真实输出构建256条4096-token续写前缀，比较
每卡32/56 GiB KV；相同输入、预热、两次完整计时、每条固定256个输出token。
为固定工作量，该计时使用 `min_tokens=256, ignore_eos=True`，只用于性能诊断，
不作为准确率或正式采样。记录引擎指标、显存峰值、并发任务及输入摘要。
此实验检验缓存接近满载时的吞吐，不保证任意长度分布下相同加速。

UOS runtime 已支持每rank K=384，复用原实现。`validate_uos.py` 在真实TP2下
核验 QDQ、两次GEMM、原生SwiGLU、MoE及输出SUM，覆盖tokens=1/256/8192；
沿用 NMSE<1e-4 和输出贡献幅度归一化误差标准。CPU核验首末层TP2切片与相邻
TP4切片合并相同，并生成全部48层×2rank摘要。G512沿用已经通过的TP2算子证据。

`prepare.py` 创建独立模型视图和缓存；`workflow.py` 对每个候选顺序执行
26题×10、64-token正式配置短测及完整GPQA×10。保存配置只允许已授权的TP/KV变化，
四项采样参数、实际prompt/token、全部专家字节及非专家方法均核验。
失败即退出，不自动重试或变更协议。原始JSON评分保留，最终答案解析与待盲审标记
仍由既有 `qwen3_aisbench_gpqa.report` 处理。

本次产物：[tp2_budget_20261113T093133Z](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/aisbench_gpqa_20261113T023500Z/tp2_budget_20261113T093133Z)。
授权、性能输入和计时、CPU/GPU验证、预算决策、停止记录、启动命令及退出状态
保存在该目录；每个候选的 `probe/` 与 `formal/` 独立。

参考经验：[预测缓存位置、显式KV预算与worker释放](../../../docs/experience/lessons/evaluation_evidence.md)。
短回答计时不能证明长序列缓存收益，因此本次增加固定真实长前缀的容量对照。
性能结论及当前运行标识以结果目录中的 `budget_decision.json`、`handoff.json` 为准。

本次有效对照：32 GiB两次227.94/229.41秒，56 GiB两次207.84/208.70秒，
中位吞吐提升9.80%；每轮请求抢占由8次降为0，PyTorch峰值保留显存由43.38增至67.38 GiB。
采用56 GiB；UOS复用相同结构的KV容量并单独验证TP2/正式配置，未独立测得UOS提速比例。
首次32 GiB诊断在生成结束后因callable RPC序列化失败，保留失败日志；改为named worker
extension RPC后重新计时，失败轮不计入以上结果。

2026-11-13正式评测已完成：G512于北京时间20:30、UOS于21:23退出且保存1980个回答。
用户要求核对答案并计算方差后，两组复核准确率分别为59.1414%和60.4040%；同口径
七组结果、采样方差、逐条复核理由和数据摘要见[最终复核报告](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/aisbench_gpqa_20261113T023500Z/answer_review_20261113T144526Z/REPORT.md)。
旧`handoff.json`及`progress_summary`保留交付时快照；完成状态看各候选`workflow_status.json`，
准确率看独立复核报告，原自动解析结果不覆盖。
