# B 模型追加 BF16 Router 校准

用户于 2026-10-19 确认并授权实施。新增 `B_router_bf16`；固定源为已完成的 `matrix_norm_native_traces`，不再用开发集选起点。

## 固定算法

- 使用 B 原有 128 train / 32 val 完整原生轨迹，不再生成、过滤或换题。教师为固定原生 NVFP4；学生直接加载 B 已融合 HiF4 QDQ 导出值，不重复变换或量化普通权重，保留其激活 R64/HiF4 路径。
- 教师与学生分别维护自己的完整前缀，在相同 token 位置对齐 48 层实际 gate；教师使用原生 NVFP4 层运行逻辑，学生使用自己的 router 输入与已选 router 层输出。
- 唯一损失为 teacher→student top-mass KL：128 专家完整 softmax，教师 top-8 加聚合 tail，T=1，按有效 token 平均。所有普通投影、专家、norm、已融合变换冻结，不加入 NMSE 或硬路由损失。
- router 参数、梯度、AdamW 一二阶动量都为 BF16，无 FP32 主权重；softmax、KL 和统计归约 FP32。优化器 step 标量计数不属于主权重。
- batch=4，seed=42，AdamW betas=(0.9,0.999)、eps=1e-8、weight_decay=0、恒定 LR；第一层 1e-5/3e-5/1e-4 各筛 3 epochs，验证 KL 选优、平手小 LR，重置权重/优化器后正式 48 层各 5 epochs。每层允许 epoch 0。
- 缓存 detached 学生实际 router 输入与教师 logits；记录验证 KL、top-8 重合率/集合一致率。选定后完整传播学生层，教师独立传播原生层。

## 验证与评测

绑定源导出全部分片、数据/题目/隔离摘要和新旧源码。最长真实 batch 的两层短测覆盖 BF16 更新、冻结、缓存 KL/梯度一致、保存重读、独立前缀与层间传播。零步导出张量与 B 相同，非 router 分片始终与 B 完全一致。真实 TP2 预填充/解码验证后启动正式训练，正式导出再次执行真实 TP2 验证。

完整 MMLU-Pro 12032、LCB 175、GSM8K 1319、AIME25 30×5，ARC-E/C 2376/1172 辅助；复用 B 完整结果，跨组核对固定实际 prompt/token/gold。AIME 保留原始缓存和分数，按已纠正最终提交口径评分，待审项逐条复核后再汇总。主比较 `B_router_bf16 − B`，使用旧原生基线、四项平均正向损失、逐题配对区间（10000 bootstrap，seed=42；AIME 单位为题目 avg5）。区间跨 0 不宣称稳定改善，≤0.5 pp 单列。旧 C 不作单变量对照。

## 执行

在独立 RUN 中保存 manifest、源码摘要、命令/PID/日志/退出状态。默认 GPU 6 训练，6/7 真实 TP2 验证及评测；每次启动前核对资源，不停止其他任务。短测输出独立且不带入正式初始化；验证适用后后台推进，确认真实主循环后交付并停止主动监控，不自动重试或占卡。

参考已有经验：[评分与执行证据边界](../../../docs/experience/lessons/evaluation_evidence.md)。旧实验的验证不替代本次 BF16 更新和独立教师前缀短测。
