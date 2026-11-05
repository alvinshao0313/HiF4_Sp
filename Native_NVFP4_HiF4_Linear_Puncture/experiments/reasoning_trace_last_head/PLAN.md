# B 最后一层与输出端的词表 Top-mass KL 蒸馏

2026-10-20，用户已确认并授权实施与正式运行。候选仅 `B_last_head_vocab64`，原 B 为 `matrix_norm_native_traces`；不使用后续 router 校准模型初始化。

## 固定数据、参数与损失

沿用原 B 的 128 条训练、32 条验证完整轨迹，固定题目、token IDs、loss mask 和划分。不重新生成、筛选或截断。教师为原固定版本 NVFP4；学生 B 与教师使用相同 token 序列，分别通过自己的完整前缀。

训练第 47 层已有普通 Attention/MoE per-G64 matrix 和 input/post-attention norm，从 B 已选 checkpoint 恢复；训练 B 的 final RMSNorm；新增 32 个 64×64 的 head 输入侧 matrix，单位初始化。共 40,638,464 个可学习参数。前 47 层、router、Q/K norm、embedding、所有基础投影及 head 权重冻结。可学习参数、梯度、AdamW 动量 FP32，模型前向与导出 BF16；head 不量化、不增加在线变换。

唯一目标为教师到学生的最终词表 top-mass KL。对完整 151936 词表归一化，教师 top-64 为单独桶，其余所有 token 为 tail，T=1，在有效 token 上平均。不重新归一化 top-64，无 NMSE、router KL 或其他正则。梯度通过最终输出联合进入所有指定参数。

## 训练与缓存

batch=4，seed=42，AdamW betas=(0.9,0.999)、eps=1e-8、weight decay=0，恒定学习率。三个 LR（1e-5、3e-5、1e-4）分别 3 epochs，以完整验证集有效 token 平均 KL 选优，平手取较小 LR。随后恢复原 B 与 head 单位矩阵，新建优化器，正式 20 epochs。联合 checkpoint 允许 epoch 0，以严格更低验证 KL 更新；平手保留更早 epoch。

一次缓存 B 前 47 层输出和独立原生教师最终 top-64 IDs、FP32 log-probability、tail log-mass。绑定题目/token、B 导出及第 47 层参数、教师版本、源码摘要。仅缓存稀疏教师目标，不落盘完整词表 logits。GPU 6/7 分别准备两个独立前缀。

词表沿 token 轴按 128 分块并重计算，每 batch 仅构造一次可微 BF16 融合 head 权重。各块形成同一完整目标，一次 backward/optimizer update；不截断轨迹或缩词表。普通投影的激活重计算也保留原矩阵融合/QDQ/STE，避免保存大量量化中间值。

## 导出与启动验证

第 47 层按原 matrix_norm 融合并 HiF4 QDQ，head matrix 右乘融入原 head 后保存 BF16。只改写末层与非层分片；其他分片硬链接原 B。非层分片只出现一次，避免重复 key 加载。冻结 embedding、Q/K norm、router 与前缀保持原值。零步导出必须与 B 张量一致。

启动前执行：FP32 缓存/65 桶/分块损失与梯度对照（atol=1e-6、rtol=1e-5）；真实 BF16 分块梯度对照；最长真实训练与验证 batch 的完整前后向、两次更新、验证、epoch 0、选优保存读取与导出；指定梯度与 FP32 动量、冻结张量检查；零步和更新模型真实 vLLM TP2 短测；已确认的 600 条 AIME 回答评分回放。正式导出后再做 TP2 短测。

资源只限 GPU 6/7；默认 6 训练、6/7 TP2，按实时余量和最长 batch 峰值检查共用条件。不停止现有任务，不改变协议。独立 RUN 保存固定 manifest、源码哈希、命令、PID、退出码、日志、缓存及 checkpoint。必要验证通过后后台正式运行，确认真实训练推进即交付并停止主动监控。

## 评测与报告

完整 MMLU-Pro 12032、LCB 175、GSM8K 1319、AIME25 30×5；ARC-E/C 辅助。沿用 B 固定正式评测组件并核对题目、prompt/token/gold 摘要。AIME 按纠正后的最终提交评分，未解决歧义时暂停汇总，人工复核不自动猜测。

主比较新候选减原 B：逐任务变化、历史原生基线下四项平均正向损失变化及逐题配对 95% 区间（10000 bootstrap，seed=42，AIME 以题目 avg5 抽样）；补充去掉 GSM8K 的三项损失与区间，≤0.5pp 单列。基线为 MMLU-Pro 75.3740%、LCB 54.2857%、GSM8K 95.4511%、AIME 70.6667%。区间跨 0 不宣称稳定改善，局部 KL 不作为下游收益证据。

参考经验：[局部对齐与评测边界](../../../docs/experience/lessons/evaluation_evidence.md)、[训练执行与精度边界](../../../docs/experience/lessons/model_transfer_execution.md)。本轮目标改为词表 KL，原 router 结果仅供设计参考；是否改善尚待实测。
