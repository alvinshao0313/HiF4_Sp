# 原 B 的完整 KL 与轨迹 next-token 硬 CE 对照

2026-10-21，用户已授权两组实验，并明确选择由独立开发集决选最终模型。原 top-mass RUN 及原 A/B/C、router RUN 保持原样。

## 固定输入与可学习参数

原 B 是 `reasoning_trace_calibration/continue_20261016T135141Z` 的 `matrix_norm_native_traces`，不使用后续蒸馏 checkpoint 初始化。复用其原 128 条训练、32 条验证完整轨迹，含原始 prompt、thinking 和最终回答，不重新生成或按正确性筛选。

两组都继续训练第 47 层已有普通投影 per-G64 matrix 与 input/post-attention norm，训练 B 的 final RMSNorm；head 新增 32 个单位初始化 64×64 输入侧 matrix。共 40,638,464 个参数。前 47 层、router、Q/K norm、embedding、全部基础投影及基础 head 冻结。参数、梯度及 AdamW 动量 FP32，前向与导出 BF16。沿用 HiF4 激活 R64/QDQ 与末层可微权重融合；head 不量化、不增加在线变换。

## 两个目标

| 候选 | 唯一目标 | 有效位置 |
| --- | --- | --- |
| `B_last_head_full_kl` | 完整 151936 词表的 `KL(native teacher || student)`，T=1 | 原轨迹全部有效位置，与原 top-mass 相同 |
| `B_last_head_hard_ce` | 原 NVFP4 轨迹的 next-token 硬 CE | 学生位置 t 预测原 token[t+1]，沿用下一位置的原 loss mask；prompt 仍参与，末位置无标签、padding 排除 |

完整 KL 保留每个词的概率，不合并 tail、不截断词表、不增加其他目标。教师与学生各走自己的完整前缀。CE 不使用教师 argmax 代替原轨迹 token，不使用正式评测答案；两组末位置的自然标签边界差异明确记录，不把 CE 称为只改 top-mass 近似的单变量对照。

完整 KL 缓存原生教师完整 48 层与 final norm 的 BF16 输出，约 6.83 GB；按 token 分块现场重算教师词表分布，不保存约 1 TB 的完整 FP32 targets。复用原 B 前 47 层输入缓存的只读硬链接，绑定模型、数据、token、旧缓存及源码摘要。用原 top-64＋tail 目标回放核对教师隐藏状态缓存。完整轨迹、词表及 batch 不缩减。

## 相同训练预算与开发集决选

batch=4，seed=42，AdamW betas=(0.9,0.999)、eps=1e-8、weight decay=0，恒定 LR。分别以 1e-5、3e-5、1e-4 各训练 3 epochs，按各自完整校准验证集的有效 token 平均损失保存最优点，允许 epoch 0，平手留更早 epoch。以最低验证损失选择一个 LR，平手取较小 LR；随后恢复原 B/单位 head matrix，重建优化器，训练 20 epochs，保留验证最优点。

每组最终候选池为三个 LR 筛选最优点、20 epochs 正式训练最优点及原 B（最多五个）。完全相同参数可去重。只用原有、已隔离的共同开发集（64 math＋64 code）各生成一次，沿用完整 thinking、原采样与真实 TP2，数学正确率与代码 pass@1 等权决选。平手优先原 B，否则更早 epoch、较小 LR。原 B 的开发集结果在逐题 prompt/token 校验后复用。正式测试不选择 LR、checkpoint 或候选池。

每组相同训练预算为 3×3＋20 epochs，不将每个 LR 扩成 20 epochs。开发集选择使本轮与旧 top-mass 的比较同时包含选优策略差异，因此最终报告须区分：两组之间的受控比较、旧 top-mass 的参考结果。

## 必要验证、资源和执行

独立代码、RUN、manifest、源码快照、训练、模型和评测记录。只用 GPU 6/7：完整 KL 用 6，硬 CE 用 7 并行训练；TP2 在两组训练都结束后串行使用 6/7，避免与另一组训练抢卡。准备阶段可在 GPU 6 补算教师前缀，同时在 7 做 CE 短测。不停止既有任务，不安装或更换环境。

词表计算沿 token 轴 128 分块，使用激活重计算；每 batch 只构造一次可微融合 head，统一损失、一次 backward/update。训练进程使用 PyTorch 可扩展显存分配段，以处理原运行显存预留接近上限的分配碎片；不改变计算精度、batch、数据或算法。以最长和最短实际完整 batch 的重复更新/验证短测覆盖不同形状，记录实际峰值。

启动前：完整词表 FP32 KL/CE 参考损失和梯度（atol=1e-6、rtol=1e-5）、软 CE 与 KL 梯度等价、CE shift/padding、真实 BF16 分块梯度、仅指定参数梯度和更新、冻结张量、FP32 优化器、epoch 0、参数/优化器重置、选优保存读取及导出。独立开发集的候选去重/平手规则做 CPU 检查。两组最长/最短完整轨迹短测，零步及更新模型实际 TP2；正式模型导出后再做实际 TP2。已确认 AIME 案例沿用原 600 条回放。

代码/配置完成并通过短测后脱离会话启动，保存命令、GPU、PID、stdout/stderr、退出码及结果。确认两组正式训练有有效更新后交付，停止主动监控。

## 正式评测与报告

两组候选先按开发集锁定，再分别完整评测 MMLU-Pro 12032、LCB 175、GSM8K 1319、AIME25 30×5；ARC-E/C 辅助。复用原 B 的已验证正式结果，逐组核对题目、prompt/token/gold 摘要。AIME 按最终提交评分，歧义逐条复核后再汇总；保留 raw cache、原评分和改判记录。

报告两组分别减 B、硬 CE 减完整 KL 的逐任务变化、四项平均正向损失及逐题配对 95% 区间（10000 bootstrap，seed=42，AIME 按题目 avg5），同时报告去掉 GSM8K 的三项口径与 ≤0.5 pp 是否达标。旧原生基线与原 B 保持不变。区间跨 0 不宣称稳定改善，局部训练损失不作为下游收益。

参考：[评测与局部损失边界](../../../docs/experience/lessons/evaluation_evidence.md)、[训练执行验证](../../../docs/experience/lessons/model_transfer_execution.md)。上轮 top-64 外平均质量只有约 0.0241%，完整 KL 的新增信号可能较小；CE 可能强化原错误轨迹。这些是待验假设，不能预先写成实验结论。
