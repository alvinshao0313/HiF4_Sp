# 大模型迁移的训练执行验证

证据状态：2026-10-08，真实 Qwen3-235B-A22B-NVFP4 的 E4 与 Group 均已完成 94 层并通过完整训练产物复核；完整导出结构与 CPU prompt 准备已通过，下游精度仍待验证。首层 E4 与 Group-top-mass 执行对比、最长批量短测已有实测。实验定义和具体指标集中在[迁移记录](../../../Native_NVFP4_HiF4_Linear_Puncture/experiments/qwen3_235b_transfer/PLAN.md)，不从短测或局部重构损失推断模型效果。

## 模型几何应覆盖变换与融合

扩大 hidden size 和层数不足以完成 GQA 模型迁移。本次两个模型的 KV-head 数和 head_dim 相同，Q-head 数不同；旧 VO scale 展开固定使用 32 个 Q heads，235B 的 O 投影实际接收 64 个头。应从 checkpoint 规格向训练状态传入 Q/KV/head_dim，使训练、融合、初始化和导出共享同一展开定义，而不是在失败位置补广播或另造参数。真实权重 header、GQA 重复次序、梯度与保存重载回归见[修复证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_transfer/transfer_20261001T050000Z/fix_vo_expansion/validation.json)。

## 暂存反向张量也要核对布局与更新

PyTorch 2.10 当前实现的 `save_on_cpu(pin_memory=True)` 用连续张量承接保存值；本次前向 loss 相同，但反向梯度和 Adam 更新差异明显大于 resident 重复基线。关闭 pin_memory 后，格式保留路径的差异回到通常计算精度范围。这个观察针对本次 BF16/FP32 混合训练和实际算子，不意味着所有 pinned copy 都有错误，也不要求全链路逐位一致。应比较同一真实输入、损失、梯度和更新，并用重复运行辨别通常浮点差异；不能仅凭前向 loss 相同判断执行变更没有影响。原始比较见[CPU 暂存证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_transfer/transfer_20261001T122200Z/memory_execution_fix/execution_review.json)。

暂存可以用已授权的其他 GPU，不必改变 batch、文本长度或 loss 归一化。本次保持一张卡计算，两张卡按固定轮转保存张量，保留 dtype/stride 并恢复到原设备；这是存储位置变化，不是 tensor parallel。真实短完整样本的数学执行对比通过后，另以最长完整 batch 验证更新、验证阶段、参数边界、保存重载、完整文本传播和实际导出。见[实现对比](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_transfer/transfer_20261001T122200Z/memory_execution_fix/production_gpu_context_trial/status.json)和[最长批量两阶段汇总](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_transfer/transfer_20261001T124000Z/smoke_summary.json)。

单步传输耗时和首层显存只支持对应执行条件，不代替持续训练吞吐、后续全部层或整模型下游评测；QDQ 后 BF16 导出也不构成实际 4-bit 存储或推理加速证据。

## 拆分 batch 须保留 padding 和全局损失

2026-10-03，真实 235B E4 第 0 层短/长完整 batch 的四卡候选实测：原 batch=4 分为每卡一个样本时，保留原全局 padding。当前 E4 runtime 调用不传 attention mask，且默认非 causal，独立裁短各样本会改变原计算，不能作为等价提速。block-delta NMSE 应汇总全局分母，局部分子除以全局分母后反传，再 SUM 梯度；不能平均各卡独立归一化后的 NMSE。更新仍按一个原全局 batch 一次执行。

用同形状重复、串行拆分和多卡拆分区分通常数值波动、batch 形状变化和通信归约。此次完整 batch 与四卡候选的首步各参数梯度相对 L2 差异约 1%–7%，串行拆分与四卡候选的差异约 1e-9–1e-6，支持主要差异来自计算形状的解释。短测证实同步、容量与单步提速，尚未验证收敛、epoch 选择或下游影响；梯度差异不能直接解释成精度下降。具体配置、重复基线、原始指标和适用范围见[短测报告](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_transfer/acceleration_20261003T121157Z/REPORT.md)，后续阶段依照[迁移计划](../../../Native_NVFP4_HiF4_Linear_Puncture/experiments/qwen3_235b_transfer/PLAN.md)。

四卡入口后续沿用 rank 0 原 teacher、验证、选优和传播，仅分发训练 batch；跨 epoch 必须保留各副本的 Adam 与 scheduler 状态。独立恢复需要复制已完成的 adopted 前缀和同一校准集，并按每层原生/学生选择重放前缀，不能只改 start_layer。最长样本、保存重载、前缀恢复和 workers 连续换层已实测通过，见[启动验证报告](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_transfer/parallel_20261003T144300Z/REPORT.md)；完整收敛仍待比较。

2026-10-05 的 Group-top-mass 四卡短测还需分别保留两个损失的归一化：NMSE 汇总目标能量分母，router KL 按每卡有效 token 数加权，不能按卡简单等权。辅助梯度仍在 MoE norm 前 detach；MoE 中全局未使用参数保持 `grad=None`，不能无条件填零改变 Adam 的更新状态。真实完整 E4 初始化、最长 batch 两次更新对照中，首步全梯度相对 L2 差异约 0.19%，四副本参数/梯度/Adam/scheduler 精确同步，token/expert 总覆盖一致；这是执行路径证据，不是下游精度结论。来源同上报告及其 Group 对照数据。

## 先测实际热点再选加速方向

真实 235B 首层短 batch 的一次 E4 profile 中，HiF4 QDQ 调用归因约占 CUDA kernel 时间 62%，Flash Attention 前反向约占 3.4%，见[原始归因](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_transfer/bottleneck_20261003T144100Z/kernel_attribution.json)。这是带 profiler 的单步、特定形状证据，不是稳态或长 batch 的普遍比例。优先验证保持 S0、BF16/E6M2 舍入和 STE 的量化融合、同一步相同 activation 复用；不能跨更新复用依赖可学习 DIAG 的量化权重，也不能把热点占比直接当成可获得的整体加速。

本次默认 `torch.compile` 与仅启用 `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1` 的原语试验均改变同一真实权重约 1.66% 的量化输出，因此均未采用；该选项未解决本例，根因尚未确认。量化固定舍入属于算法契约，不能仅因编译成功就当作等价执行。试验边界和来源见[独立候选报告](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_transfer/parallel_20261003T144300Z/REPORT.md)。

2026-10-08，30B 模块消融接入分支独立的可学习旋转时，直接复用 Online-DIAG 的融合 `SiLU × up` helper 会省掉重构路径要求的 SiLU 后 BF16 舍入；函数名与输入输出 dtype 相同不足以证明中间精度契约一致。新路径保留该舍入后，真实第 0 层、512 个 token、128 experts 的局部 MoE 对照差异缩小。源码核查确定了舍入边界差别，局部实测支持此次修复；不能据此推断完整模型下游已恢复。配置与修复范围见[模块消融方案](../../../Native_NVFP4_HiF4_Linear_Puncture/experiments/module_design_ablation/PLAN.md)，原始指标见[kernel 对照](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/qualification_20261008T064919Z/kernels.json)。

2026-10-08，同轮 30B 消融的 GPU 0/1 等待线程调用 `os.pidfd_open`，但本机 `hif4` Python 3.11.13 没有该属性；Linux 与 Python 版本号本身不能证明可选 API 可用。该异常发生在 job 的 try/except 之前，`ThreadPoolExecutor` 的退出又等待 GPU 6/7 工作线程，导致磁盘状态仍为 running、部分 GPU 空闲。真实环境重现及候选检查见[编排诊断](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/driver_repair_20261008T084419Z/validation.json)，状态与修复记录见[模块消融方案](../../../Native_NVFP4_HiF4_Linear_Puncture/experiments/module_design_ablation/PLAN.md)。应在启动前验证实际使用的进程等待路径，并在 future 完成时记录线程级异常；仅确认部分 worker 已开始训练不能覆盖尚未触发的接续分支。修复先通过 CPU 真实进程等待及产物检查，随后仅替换编排进程；[正式接管记录](../../../Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/driver_repair_20261008T084419Z/replacement.json)确认 GPU 6/7 原进程持续训练，GPU 0/1 完成学习率筛选并进入正式更新。该证据覆盖接续与推进。2026-10-10 复核时，被接管的两项训练均已完成 48 层且产物验证通过；非子进程退出码仍保留 null，以完整 manifest/summary 及逐层 selected 文件作为完成证据，不能事后伪造 exit=0。下游实验仍未完成。
