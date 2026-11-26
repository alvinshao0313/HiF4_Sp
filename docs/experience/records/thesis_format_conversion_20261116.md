# 2026-11-16：格式转换补充实验实施

2026-11-23 论文回填。Qwen3-30B-A3B 的独立校准消融采用各方法共同完成的第一轮，列出五项全量任务的绝对得分。比较固定 R64、加入逐组尺度搜索、加入 GPTQ，以及采用 S1K 或自生成语料的逐层重构。ARC-C/E 统一从原评测文件取 `acc_norm,none`。转换及导出用成功作业的实际卡时，独立开发评测和共享语料生成分别计时。九任务主结果及原定五次均值仍保留各自的完成条件。[首轮原始分数和计时索引](../../../LaTeX/tables/qwen30b_calibration_ablation.json) · [论文逐表复核](../../../LaTeX/tables/README.md#逐表复核与补测范围2026-11-23)

2026-11-20 北京时间15:49：用户明确要求停止当前两组旧协议补轮。已核对PID、启动时间及所属进程树，停止GPU0、1的GPTQ接续控制器2868819和GPU4、5的matrix_norm接续控制器2868820及其子进程，共14个进程；全部已退出，无需SIGKILL。两份接续`status.json`已标记`stopped_by_user`，不再推进剩余队列。保留GPTQ前3轮、A20/B20各前4轮、当前未完成MMLU-Pro输出、模型和所有日志；未生成最终五轮均值。停止证据：[GPTQ](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/s1k_all_experts_20261118T032650Z/stops/user_stop_20261120T074912Z/stop_record.json)、[matrix_norm](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/matrix_norm_repeats_20261118T034732Z/stops/user_stop_20261120T074912Z/stop_record.json)。没有停止其他GPU任务或启动替代评测。

论文适用范围：已有完整结果仍可按实际轮数和旧协议作为补充证据；当前固定seed1234五轮、ARC raw acc/seed42、LCB输出38912、eager/batched4096，与[新版九任务主表](../../../Native_NVFP4_HiF4_Linear_Puncture/plans/2026-11-19-fp4-paper-experiments-cn.md)的三种seed、ARC acc_norm、统一输出32768和执行设置不同，不能直接混表。模型、转换和训练成本仍可保留复用。此次停止是用户的执行决定，不是方法失效的实验结论。下述“运行中”均为历史时点记录；本次无新增通用经验。

2026-11-20：R64＋scale已于北京时间07:41完成五任务各5轮，20个评测进程均退出0，完整输入/原始缓存/协议核验及算术均值汇总通过。[正式五次结果](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/c4_20261118T024647Z/REPORT.md)、[独立scale结果与成本](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/c4_20261118T024647Z/scale_mean5.json)及[运行记录](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/c4_20261118T024647Z/RUN.md)已更新；原成功转换计时433.744秒，正式评测阶段耗时合计44.788小时，GPU6、7已释放。GPTQ与两版matrix_norm仍在各自队列补轮，暂不作最终五次结果的跨方法优劣结论。无新增通用经验，不删除模型或原始证据。

2026-11-19：[同A20 S1K、全token全专家GPTQ的转换与评测](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/s1k_all_experts_20261118T032650Z/RUN.md)已确认48/48层转换完成，全部6,144个专家校准覆盖通过，49个导出分片摘要一致，退出0，wall time约15.476小时。用户要求直接评测，导出模型TP2短测已通过，GPU0、1已启动五任务各五轮评测（控制器1221133），保留独立预测缓存，完成后自动汇总五次均值。正式状态以该RUN的`evaluation_handoff.json`及jobs为准；转换完成不代表下游评测完成。GPU4、5的matrix_norm与GPU6、7的scale原队列继续，无新增方法效果结论。

2026-11-18两版matrix_norm补测：[GPU4、5的A20/B20五任务五次均值](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/matrix_norm_repeats_20261118T034732Z/RUN.md)。用户明确两版都需运行；各自复用已核验首轮、补第2至5轮，保留原完整模型及数据/推理/评分协议。新增独立GPU入口，保护活动GPTQ/scale依赖。启动及完成状态以该RUN的jobs、原始输出与最终均值文件为准；与GPTQ同数据比较对应A20，B20使用自生成校准数据，单列解释。无新增方法效果经验。

2026-11-18 最新续行：[同A20 S1K、全token全专家GPTQ](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/s1k_all_experts_20261118T032650Z/RUN.md)。用户将GPTQ改回与干净matrix_norm完全一致的128条S1K完整train（1,272,746 token），授权LLM Compressor式全专家统计、原top8输出聚合；另明确同意停止C4检查。已核对并停止C4控制器502364及两子进程，部分记录保留，scale的GPU6、7评测不变。新代码使用独立模块，避免修改scale活动依赖。短测使用最长两条完整train、真实全部388投影，检验统计输入、原路由、GPTQ及直接保存重载；短测和正式完成情况分别以新RUN的qualification/jobs为准，不将校准覆盖解释为效果改善。下述C4/850条段落为历史过程。

2026-11-18 续行：[scale五次评测](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/c4_20261118T024647Z/RUN.md)与[纯C4的512×512覆盖检查](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/c4_512x512_20261118T025938Z/RUN.md)。实测旧GPTQ850完成21层，在索引21层expert29零命中退出；正交A20已完成48层训练和49分片导出。用户选择纯C4、要求先检测全专家覆盖，并把首拟1024完整文档缩减为512段×512 token；此前本轮临时扫描已停止并保留。GPU0、1各扫描256个独立窗口、各自传播全48层，按真实gate/up/down输入核对路由；扫描结束自动合并并退出，不启动GPTQ。GPU6、7的scale模型通过reasoning及ARC短测后已启动五任务各五次评测。启动与当前完成状态不能混淆，最终状态以RUN的jobs和原始记录为准。

本次执行[格式转换计划](../../../Native_NVFP4_HiF4_Linear_Puncture/plans/qwen3_30b_a3b/2026-11-16-thesis-format-conversion-supplement-plan-cn.md)，只分配 GPU0、1；GPU2–7 的其他评测保持运行。独立入口与命令见[实现说明](../../../fp4_convert/experiments/thesis_format_conversion_supplement/README.md)，实际运行状态见 [RUN](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/run_20261116T103300Z/RUN.md)。

执行期间用户要求统一测试为五次均值。每个报告任务各做五次完整运行、保留原任务 seed、独立缓存；匹配历史独立轮可计入，同次生成的重复引用不重复计数。未满五次不能输出正式均值或 L5，旧 AIME 和旧平均值不进入本轮。

准备核实：A20/B20 和三个 e5 的48层训练、模型分片均可用；逐分片摘要已核验。A20 的参数化、student、训练循环、loss、ModelOpt 解码和传播关键源码与历史训练摘要一致。full_e5/B20 的完整训练数据摘要一致，random/coverage 保留原完整序列选择。开发64 math＋64 code IDs、隔离清单、实际评测 Arrow 和 prompt/token/gold 已绑定到 RUN。

native/direct/R64/A20/B20 的五任务首轮、random50_e5 的三项 reasoning 首轮经核查可复用。额外 native MMLU-Pro 独立轮已核验；另外的 LCB/GSM8K 引用指向同一原始生成，不能增加重复次数。direct/R64 旧导出的49个权重分片已缺失，评测记录和缓存仍在；补评测须从同源有效权重重新固定转换，无需训练。

已完成的必要验证：scale 第0层388个投影采用 full 搜索，MSE不劣于所含标准候选，BF16值和保存重载完全一致；最长四条完整序列的 attention/实际专家 gate、up、down 运行时 Linear 检查一致。正交组两层×两轮的完整最长 train/val batch 通过前向、反向、AdamW更新、验证选优、严格重载和传播；峰值约34.6 GB，FP32正交性最大误差约1.43e-6。CPU检查覆盖变长输入的 GPTQ Gram 累计、G64核心输出和full搜索独立枚举。GPTQ第0层以128条完整序列、1,272,746有效token完成388投影，无未访问投影，整层导出重载输出完全一致，退出码0。正式scale/GPTQ转换已分别在GPU0/1启动并确认计算，控制器PID为3633721/3633722；最终完成状态仍以RUN对应JSON/日志为准。

工程经验：主线 `_state_to_tensors()` 会执行标准权重QDQ，因此搜索/GPTQ选定值必须直接保存；新导出器复用 tokenizer、非层张量和runtime sidecar协议，另走已量化权重保存路径。此证据说明导出语义，**不是权重MSE改善带来下游精度改善的证明**。边界见[量化执行经验](../lessons/fake_quant_boundaries.md)与[评测证据经验](../lessons/evaluation_evidence.md)。

后续：两条强基线转换及完整五任务×五次；固定在线R64正交重构和五任务×五次；full/coverage先补LCB与MMLU-Pro各五次；机制32条验证序列、代表层0/15/31/47与真实vLLM路径接入仍按计划推进。无新增方法效果结论，不把短测当最终准确率。

## 2026-11-17 GPTQ校准扩充

核实scale转换48/48完成、退出0；GPTQ转换在完成前4层后，于层4的expert35零命中退出1。保留原失败记录。用户明确允许扩大GPTQ校准，新的[续行](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/gptq_expanded_20261117T062000Z/RUN.md)从原已分簇的固定S1K来源得到850条完整干净序列，保留全部原128条，排除原val/dev/test及已知test来源簇。校准量是8,568,064 token，最长26,969 token；这只改变GPTQ的数据预算，不再声称与A20等校准量。

已复现旧失败层：实际router计数与GPTQ输入采集计数一致，原128条下只有expert35为零，因此该观察是路由覆盖缺失，并非采集遗漏。扩充最长序列在32专家gate/up Hessian驻留下执行两次真实层前向，峰值分配9,625,620,480字节，两次释放后均2,507,175,424字节；计数核对通过。CPU四项测试通过。其余扩充样本的覆盖与正式续行状态以新RUN记录为准，尚不构成48层专家全覆盖或精度提升证据。

扩充850条的旧失败层核对已完成：全部128专家命中，expert35为6个token，router与采集次数逐专家一致，两份追加扫描退出0。该计数仅对应已重现的旧失败层；新正式GPTQ从原生第0层重转，模型级完成条件要求48×128全覆盖。GPU0已启动扩大校准的GPTQ（控制器3932932、计算3932964）；GPU1按原授权启动正交A20（控制器3931981、计算3931998），原128/32数据保持。源代码被冻结，实际推进和终态以续行RUN为准；新增校准量必须计入比较边界和成本。
