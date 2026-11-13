# fake quant 与实际低比特执行

证据状态：包含早期 dense fake quant 实现记录，以及 2026-10-16 原生 packed NVFP4 的真实 TP2 性能短测；不代表正式准确率或原生 FP4 硬件计算核的性能。

[HiF4 实现记录](../records/implementation/hif4_vllm_fake_act_quant.md) 和 [NVFP4 实现记录](../records/implementation/nvfp4_vllm_fake_act_quant.md) 描述 dense Linear 输入上的 quant-dequant，再交给原浮点 GEMM。它们不等于 packed 低比特计算核，也不覆盖全部 MoE 路径。

- fake quant 用于研究数值扰动，不能据此声称模型实际位宽、存储压缩率或推理速度已经改善。存储口径需另说明 scale、zero-point 等额外开销。
- HiF4 记录包含按 64 分组和非整组 padding 等处理；应按实际实现核对形状和 lm_head 排除条件，不能把该局部入口等同于完整 W4A4 模型。
- NVFP4 记录要求对应 activation-scale sidecar 与 layer prefix；缺失或不匹配应明确失败，不能无依据改成其他 scale 或静默降级。
- 文末的“建议验证”描述待做检查，不是通过记录。精度与吞吐结论仍需要相应运行证据。

2026-11-10 的 [NVFP4 算子核查](../../../NVFP4/reports/operator_flow_audit_20261110T092657Z/REPORT.md)
区分了两条常被混称为“BF16＋QDQ”的路径：当前 `native_nvfp4_linear` 直接复用 vLLM
QDQ，在 A800 上走 PyTorch reference；`--fake_act_quant nvfp4` 使用另一套 Triton QDQ。
6 个真实 W2 输入上，前者与普通 Linear emulation 输出完全相同；更明显的外围差异是研究
MoE 将 router 权重转 BF16、在 W2 输出转 BF16 后加权并逐专家累加，而 vLLM 保留 FP32
router 权重、在 FP32 accumulator 上加权并调用 top8 sum。独立隔离测试已确认这些差异，
但未归因下游准确率。另一路 Triton fake QDQ 的 66 个真实张量中有 257/86016 个输出不同；
已定位倒数计算舍入跨 E2M1 中点的反例，不能把该结果归给没有调用它的研究路径，也不能
从位不一致直接推断准确率下降。当前真实 BF16 SiLU 后乘 up 与 vLLM 相同，不应擅自
改成 FP32 合并后仅舍入一次。原始日志、源码/输入哈希、逐例指标与适用边界均在核查记录。

2026-10-16 的[轨迹校准运行](../../../Native_NVFP4_HiF4_Linear_Puncture/results/reasoning_trace_calibration/continue_20261016T135141Z/REPORT.md)进一步核实：原生 packed NVFP4 checkpoint 在 A800 上实际使用 Linear/MoE emulation，attention 已是 FlashAttention 2；packed 存储本身不代表获得原生 FP4 硬件加速。16 条整批等待还会因长尾题造成低并发，应先检查真实请求队列和吞吐。独立 TP2 小规模诊断中，CUDA Graph 的 16 请求/64 步吞吐为 eager 的约 1.32 倍，1024 个 greedy token 一致；此证据不覆盖正式批量 128、完整长轨迹或 HiF4 自定义路径，不能直接改正式引擎并声称等价或加速。

## 动态专家数须保留实际量化与累加边界

2026-10-23 的 [matrix_norm Ban 接入](../records/ban_matrix_norm_20261023.md)实测确认：
独立 vLLM 校准入口需像根 main.py 一样传递 HiF4 sidecar 环境变量，不能仅靠构造期
current-config；嵌套 MoE helper 的无路径查询可能遗漏 R64。应在实际 worker 核对
显式与隐式 sidecar 一致，不直接以普通 Transformers 加载 BF16 导出替代 HiF4 校准。

同一次验证发现，改变 top-k 也可能切换求和实现：当前 vLLM top-3/4 专用核逐次 BF16
舍入，而 top-8 使用通用求和，最小数值对照已证实不同。Ban 紧凑分发保留原 QDQ/GEMM，
并在末端补零到原 top-8 后求和；真实 TP2 masked/compact 的最后位置 logits 一致。
此证据限于本次 2×2048 输入和短 decode，不证明所有模型/内核的低精度求和都相同。

参考 PyTorch 路由重算与原 fused 路由的微小局部差异，在该 HiF4 模型上仍产生可见的
完整模型 logits 差异。因此须将原生路由 no-op、参考路由未剪枝、实际剪枝三者分开，
避免把重算路由或分发实现变化当成剪枝效果；不据局部一致宣称下游准确率或吞吐提升。

当前调用方式见 [仓库 README](../../../README.md)，底层 HiF4 量化过程见 [量化流程](../../guides/hif4_gpu_quant_flow.md)。

## 量化残差代理的闭式下降不保证留出误差下降

2026-11-05的[H-only CPU真实q_proj诊断](../records/matrix_only_analytic_20261105.md)中，固定norm与基础权重，
完整跨块Gram缓存配合两次闭式扫块和真实权重QDQ回溯，可显著降低训练误差，但留出验证误差反而升高；
初始R64参与验证选优后，解析候选被舍弃。此证据限于真实4096×2048 Linear、256训练token及固定阻尼，
既不证明闭式方法普遍无效，也不支持直接用于全层转换。

实现时须区分三个量：冻结量化残差的凸二次代理、重新量化后操作数的实数乘积误差、最终BF16输出误差。
缓存恒等式可以精确评价第二个量，但不能省去第三个量及留出数据上的检查。权重采用`[in,out]`推导时，
仍应转回`[out,in]`后按输入维G64量化，不能随矩阵记号转置而改量化轴。

层0 embedding加输入norm是逐token独立的，所以可用少量真实token检验Linear机制；重复token不增加输入覆盖，
该便利不适用于attention之后或更深层。应保留末步和选优结果，并将小样本过拟合解释标作待隔离假设。
CPU低预算求解耗时、训练目标下降都不能替代等质量GPU转换或下游验证。

## 共用 E2M1 数值不等于 scale 格式相同

2026-11-04 的 [MiniMax MXFP4 转换](../../guides/minimax_mxfp4.md)中，NVFP4 与 MXFP4
均使用 E2M1，但前者是 group-16 / E4M3 scale 加全局倍率，后者是 group-32 / E8M0。
既有 packed GEMM 内核可参数化传入 group-32 和解码后的 FP32 scale，其 NVFP4 wrapper
却固定了 group-16/E4M3，不能直接用于 MXFP4。新路径的 CPU 格式/保存重载及真实
专家转换、SM80 编译已有证据；GPU 数值和正式准确率仍须依据 RUN 后续产物，不能
用权重 NMSE 或离线编译通过替代。

2026-11-09 的 [Qwen3 路由专家 MXFP4 对照](../records/qwen3_mxfp4_20261109.md)补充了
CPU 格式与真实权重证据：Qwen 的 `weight_scale_2` 是组 scale 的乘数，不能复用
MiniMax 转换器的倒数语义。应先按现有 GEMM 的顺序解码并舍入 BF16，再应用指定变换。
FP32 R64 之后直接 MXFP4 量化与先舍入 BF16 再量化不同；须把中间 dtype 写入契约。
floor scale 使用二进制指数／`frexp` 并测试二次幂邻域，避免浮点 `log2` 舍入跨指数。
直接版后续 GPU 验证补充：vLLM `activation_kernels.cu` 的 SiLU 结果先舍入 BF16，
再与 up 相乘并舍入 BF16。独立参考若把两步合并为 FP32，会改变下游 MXFP4 编码，
本例 batch256 完整 MoE NMSE 达 `4.27e-4`；保留原生两次舍入后，在真实 layer0
TP4 rank0、tokens=1/256/8192 下完整 MoE NMSE 均为 0。不能用放宽量化误差解释
这种参考边界错误。运行接入应调用现有 `apply_moe_activation`，此版本
`vllm._custom_ops` 不再导出 `silu_and_mul`。完整模型和下游准确率另按 RUN 验证。

2026-11-10 同一 Qwen3 对照新增 UOS c7.25 的 ceiling 规则时，指数采用精确阈值比较：
`amax=m*2**k, m∈[0.5,1)` 时，`ceil(log2(amax/7.25)) = k-3+int(m>29/32)`。
等于阈值时不能加一；`floor(log2(amax/7.25))+1` 在这些点不等价。CPU 用 `frexp`，
GPU 用 FP32 指数和尾数位比较，避免近似 `log2` 导致跳档。独立阈值搜索核验10,822个
随机/边界值，实际 GPU 内核核验822组精确阈值及相邻浮点数的 scale/编码/重建。
这仅证明该尺度公式及格式边界正确，不证明下游精度优于旧直转；完整证据继续归于
[同一实验记录](../records/qwen3_mxfp4_20261109.md)。

2026-11-10 的 [Qwen 二级 scale 对照](../records/qwen3_mxfp4_20261109.md)进一步发现，
相同 PyTorch 表达式 `amax / 6` 在 CUDA 与 CPU 上可能相差1 ULP；本轮真实BF16输入
使2/2048个值跨过E2M1舍入中点。Triton `tl.div_rn` 与既定CPU参考的外层scale和
QDQ结果完全一致，而CUDA参考的外层scale有3组不同。对此只修正验证参考：量化在CPU
按已确定的FP32除法边界计算，GEMM参考仍独立；不放宽编码一致性、不改生产算子。
两候选随后在真实TP4、tokens=1/256/8192下完整MoE及输出SUM参考NMSE均为0。
此结论限于本次实现及输入；不能概括为所有CPU/GPU除法都不同或局部验证证明下游收益。
