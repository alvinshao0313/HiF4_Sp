# Qwen3 路由专家：G256／G512 二级 scale MXFP4 对照计划

日期：2026-11-10。模型：`nvidia/Qwen3-30B-A3B-NVFP4`。

本计划整理已讨论的算法、实现边界与执行顺序。最初按用户要求先形成计划；用户随后明确要求实施本计划，恢复后续验证与评测。GPU4–7 已获授权；GPU0–3 上的 UOS 任务保持运行并保护其依赖。第8节保留计划形成时的历史快照；后续状态见运行记录。

## 1. 实验目标与候选

只新增两个无旋转候选，权重与激活都采用相同的大组粒度：

| 显示名称 | 实验标识 | 外层分组 | 内层分组与规则 |
| --- | --- | --- | --- |
| Dual-scale G256 | `dualscale_g256` | 沿全局 K 连续 256 个元素，FP32 scale | group32，原 floor 规则 |
| Dual-scale G512 | `dualscale_g512` | 沿全局 K 连续 512 个元素，FP32 scale | group32，原 floor 规则 |

主要比较已有直接 MXFP4 的 59.6970% 和原生 NVFP4 的 62.8788%，复用其完整结果及输入证据。已有旋转结果保留；UOS 完整结束并核验后可另列对照。本候选内层不使用 UOS c7.25，也不加入 R64、FHT 或 PNQ。

旧结果来源：[五组完整汇总](../../results/qwen3_mxfp4/r16x2_gpu03_20261110T040449Z/five_model_summary.json)。这些数值是 GPQA Diamond 1980 个回答的最终答案复核准确率，不是多数投票准确率。

## 2. 确定的量化定义

对激活每一行、专家权重每一输出行，分别沿完整 K 维分组。记外层组为 g、内层 32 元素组为 j。

1. 外层 scale：`t_g = FP32(absmax(x_g) / 6)`；全零组 `t_g = 1`。
2. 用 FP32 计算 `z_g = FP32(x_g) / t_g`，归一化后不先舍入为 BF16。
3. 对 z 的每个 32 元素组求 amax：
   `e_j = clamp(floor(log2(amax_j)) - 2, -127, 127)`，`s_j = 2**e_j`。
   全零内层组取 `e_j=0, s_j=1`。
4. `q_j = E2M1_RNE(z_j / s_j)`，最近偶数舍入，饱和到 ±6。
5. 保存 E2M1 packed 值、group32 E8M0 scale，以及外层 FP32 scale。数学重建为 `t_g * s_j * q_j`。

指数使用二进制指数或 `frexp` 实现，避免近似 log2 在二次幂边界跳档。尾组仅使用有效元素求最大值；补零只用于实现，不扩大逻辑维度。非有限输入明确报错。

权重统一从同一个原始 NVFP4 snapshot 独立生成：
`2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`。
先按原 Qwen 运算次序计算 E4M3 组 scale × `weight_scale_2`，再乘 E2M1，明确舍入 BF16，然后进入上述二级量化。不得从旧 MXFP4／UOS checkpoint 再量化。

## 3. 转换与推理边界

仅修改 48 层 × 128 个路由专家 × gate/up/down 三个投影，共 **18,432 个矩阵**，以及对应的两处专家输入激活。

- gate/up 共享一次输入归一化和 MXFP4 QDQ；Router 接收原始 hidden states。
- down 的激活量化位于原生 SwiGLU 之后，保留现有 SiLU 后一次 BF16 舍入、乘 up 后再一次 BF16 舍入。
- Attention、Router、Embedding、Norm、LM Head 保留原张量和原计算路径。
- top-8、路由归一化、路由加权、专家求和及现有输出 SUM 通信保持原位置和顺序；量化导致后续 hidden states 和路由自然变化。

只新增 `qwen3_mxfp4_dualscale/` 隔离实现与独立量化注册，不修改 UOS 或旧候选活动依赖。

导出专家张量：`.weight` 为 packed uint8，`.weight_scale` 为 E8M0 uint8，新增 `.weight_scale_fp32` 为 `[out, ceil(K/G)]` 的 FP32 scale；替代旧专家 `.weight_scale_2`、`.input_scale`。非专家张量名称、形状、dtype、字节摘要及 tokenizer／生成配置全部保持一致。核验覆盖、保存重载后才发布完整 checkpoint index。

完整大组的名义负载为 `4 + 8/32 + 32/G` bit/元素：G256 为 **4.375**，G512 为 **4.3125**。实际 down 的 K=768 在 G512 下有尾组，须按实际 scale 数及文件大小另报，不能直接用完整组公式冒充实际存储量。

## 4. BF16 GEMM 与效率要求

矩阵乘法使用 **BF16 操作数的 Tensor Core GEMM**，允许其正常 FP32 累加。FP32 仅用于 scale、归一化、部分和及逐元素缩放，生产路径不使用 FP32／TF32 GEMM。

令 `Abar_g = BF16(s_A * q_A)`，`Wbar_g = BF16(s_W * q_W)`。每个输出元素按以下次序计算：

`Y[m,n] = BF16(sum_g ((t_A[m,g] * t_W[n,g]) * BF16_GEMM_FP32_ACC(Abar_g, Wbar_g.T)[m,n]))`

外层 scale 随 K 组变化，必须作用在各组部分和上，不能在整个 GEMM 结束后只乘一个 scale。down 的路由权重保留在原输出舍入边界之前。TP 分片只计算各组与本 rank 的交集，随后使用原输出归约。

实现要求：

- gate/up 将 absmax、FP32 归一化、内层 QDQ 融合；down 在必要的 MAX 通信前后各完成对应计算。
- packed 权重在 GEMM 中按需解码为 BF16；保留融合的各大组部分和和 FP32 缩放，避免每组单独启动一次完整 GEMM，避免持久化完整反量化权重。
- 不将外层 scale 提前乘回 BF16 操作数，以免改变本次确定的舍入边界。
- 检查已执行核的 PTX／编译信息，确认 BF16 MMA；数值参考可以使用 FP32 运算，但须与生产执行路径分开记录。
- 实测 QDQ、额外 MAX 通信、两次 GEMM、完整 MoE 的耗时及峰值显存，比较旧直转；不以必须更快作为准确率评测条件。

## 5. TP4 通信与全局分组

使用已授权 **GPU4–7、TP4**；两候选顺序执行。UOS 保留 GPU0–3。

gate/up 的输入 K=2048 在各 rank 上复制，外层 scale 在本地计算即可。down 的全局 K=768 被切成每 rank 192；G256／G512 必须按全局 K 起点分组，不能在每个 rank 上重新从零分组。

| 候选 | down 全局组区间 | 每条路由激活的 FP32 scale 数 |
| --- | --- | --- |
| G256 | `[0,256)`、`[256,512)`、`[512,768)` | 3 |
| G512 | `[0,512)`、`[512,768)` | 2 |

每个 rank 为各全局组计算本地 absmax，无交集贡献零；在真实 TP process group 上进行一次 `all_reduce(MAX)`，得到全局最大值，再计算外层 scale 和本地内层 QDQ。同一规约行必须对应同一 token／top-8 路由槽位，实际 TP4 验证包含该对应关系。

权重外层 scale 离线按全局 K 计算，加载后保留所需组 scale；权重不增加在线通信。激活只规约最大值，不 all-gather 完整激活。原 MoE 输出的 SUM all-reduce 保持不变。

batch256、top-8 时 MAX 缓冲区大小分别为 **24 KiB／16 KiB 每 rank**，这是规约张量大小，不是网络流量估计。每层新增一次 MAX collective，必须测量真实四卡同步开销，不能只凭缓冲区小认定代价可忽略。

## 6. 验证内容与执行顺序

沿用此前参考与误差标准，复用已通过且依赖未改变的证据。仅对受修改影响的路径补验。

1. **格式与导出**：零组、负值、有符号零、E2M1 中点／饱和、指数截断及二次幂邻域、尾组、打包顺序；真实首末层三种专家权重；全部矩阵覆盖和非专家字节一致性。冻结实际转换依赖摘要。
2. **实际 TP4 算子与 MoE**：G256/G512 都覆盖 tokens=1/256/8192、gate/up K2048 和 down 局部 K192，top-8。独立参考检查 scale、编码及 QDQ 重建值；验证两次 GEMM、原生 SwiGLU、完整 MoE 和通信。GEMM／MoE 相对参考 NMSE `<1e-4`，并沿用按输出贡献幅度归一化的误差检查；量化损失不能解释实现错误。
3. **性能**：固定输入、路由、形状，预热后 CUDA events 重复计时，记录中位数、波动、吞吐、显存及 GPU 争用。新旧路径用相同计时范围，明确是否包含 MAX／原输出 SUM；量化器和完整 MoE 分别报告。
4. **正式配置短测**：每个候选一次 26 题×10、最多 64 个新 token；TP4、batch256 和其余正式资源参数不变。核对四 rank、全部 48 层实际方法、权重及两种 scale 切片、非专家方法、真实生成和缓存。短测分数不作为准确率。
5. **正式评测**：两候选必要验证通过后，在 GPU4–7 顺序运行 G256 GPQA×10 → G512 GPQA×10；每阶段成功且本任务 worker 释放后才接续。使用有限流程，失败退出，不自动重试或修改协议。

用户明确授权恢复实施后，按第 2–5 步接续；先核查现有产物及活动进程，避免重复转换或重复评测。

## 7. 正式 GPQA 协议与交付

复用原生／旧直转相同数据、实际 prompt token、选项顺序、chat template 和评分代码；每候选使用独立缓存。

| 项目 | 固定值 |
| --- | --- |
| GPQA Diamond | 198 题×10＝1980 回答，平均正确率 |
| thinking／seed | 开启／1234 |
| 采样 | temperature 0.6、top_p 0.95、top_k 20、min_p 0 |
| 长度 | max_new_tokens 32768、上下文 40960 |
| 并行／批量 | GPU4–7、TP4、batch256、8192 batched tokens |
| KV／执行 | 每卡 32 GiB、BF16 KV、eager、A800 软件量化与 BF16 GEMM |

正式任务实际生成推进后交付 PID、命令、日志、结果路径和已验证范围，标记“运行中”并停止主动监控。

用户再次查询或收到完成通知后，核验 1980 回答覆盖、退出状态和缓存再汇总。表中增加两个候选，报告总准确率、10 个采样序号准确率、相对旧直转与原生的百分点变化、截断率、解析失败／未决率；保留原严格 JSON 分数和统一最终答案复核结果。人工复核不静默改写原分数。另保存 JSON／CSV、全部原始回答、输入与源码摘要、实际存储大小及算子性能。

## 8. 本次整理时的实际进度

只读核查时间：2026-11-10 09:01 UTC（北京时间 17:01）。下列状态不代表后续阶段已通过。

| 项目 | 状态与证据 |
| --- | --- |
| 隔离实现初稿 | 已写入本目录，尚未完成整体验证 |
| CPU 格式／注册 | 6 项测试通过，见 [cpu_tests.log](../../results/qwen3_mxfp4/dualscale_gpu47_20261110T085025Z/cpu_tests.log) |
| 四卡 QDQ／MAX 通信 | G256/G512、tokens=1/256/8192 的四 rank 检查通过；scale、编码、BF16 重建与参考一致，见 [验证报告](../../results/qwen3_mxfp4/dualscale_gpu47_20261110T085025Z/communication_validation_fixed.json) |
| 首次通信验证 | Triton 编译期常量处理失败；修正后以上检查通过，原失败日志保留。未改变分组或通信算法 |
| CPU 权重准备 | PID `2837794` 已在运行，当时处于 G256；有限流程随后准备 G512，不含 GPU／正式评测启动逻辑。见 [状态](../../results/qwen3_mxfp4/dualscale_gpu47_20261110T085025Z/weights_status.json) |
| 完整 GEMM／MoE | 尚未完成编译和数值／性能验证 |
| 新候选 TP4 整模型短测／GPQA | 尚未启动，无准确率结果 |
| UOS | GPU0–3 正式评测仍在运行，控制器 PID `2797582`；本轮未修改其依赖或停止进程 |

本轮只补齐计划与导航；不清理既有产物，不停止既有任务。CPU 准备状态随后台执行变化，以链接内状态和日志为准。

## 9. 参考经验与尚待验证事项

复用 [fake quant 与实际低比特执行](../../../docs/experience/lessons/fake_quant_boundaries.md) 的 Qwen NVFP4 乘数语义、group32/E8M0 边界及 SwiGLU 两次 BF16 舍入；复用 [评测协议与真实执行路径](../../../docs/experience/lessons/evaluation_evidence.md) 的固定 KV 预算、正式批量短解码和 worker 释放核查。

本候选增加了跨 rank 的外层 scale 和分组部分和缩放，旧单级 MXFP4 的 GEMM／MoE 结果不能直接证明这些新增路径正确。四卡 QDQ／MAX 通过只覆盖通信与量化边界；完整性能与 GPQA 收益仍待测。本轮没有新增算法有效性结论，不把计划或局部通过写成准确率提升。
