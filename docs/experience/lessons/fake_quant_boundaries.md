# fake quant 与实际低比特执行

证据状态：包含早期 dense fake quant 实现记录，以及 2026-10-16 原生 packed NVFP4 的真实 TP2 性能短测；不代表正式准确率或原生 FP4 硬件计算核的性能。

[HiF4 实现记录](../records/implementation/hif4_vllm_fake_act_quant.md) 和 [NVFP4 实现记录](../records/implementation/nvfp4_vllm_fake_act_quant.md) 描述 dense Linear 输入上的 quant-dequant，再交给原浮点 GEMM。它们不等于 packed 低比特计算核，也不覆盖全部 MoE 路径。

- fake quant 用于研究数值扰动，不能据此声称模型实际位宽、存储压缩率或推理速度已经改善。存储口径需另说明 scale、zero-point 等额外开销。
- HiF4 记录包含按 64 分组和非整组 padding 等处理；应按实际实现核对形状和 lm_head 排除条件，不能把该局部入口等同于完整 W4A4 模型。
- NVFP4 记录要求对应 activation-scale sidecar 与 layer prefix；缺失或不匹配应明确失败，不能无依据改成其他 scale 或静默降级。
- 文末的“建议验证”描述待做检查，不是通过记录。精度与吞吐结论仍需要相应运行证据。

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

## 共用 E2M1 数值不等于 scale 格式相同

2026-11-04 的 [MiniMax MXFP4 转换](../../guides/minimax_mxfp4.md)中，NVFP4 与 MXFP4
均使用 E2M1，但前者是 group-16 / E4M3 scale 加全局倍率，后者是 group-32 / E8M0。
既有 packed GEMM 内核可参数化传入 group-32 和解码后的 FP32 scale，其 NVFP4 wrapper
却固定了 group-16/E4M3，不能直接用于 MXFP4。新路径的 CPU 格式/保存重载及真实
专家转换、SM80 编译已有证据；GPU 数值和正式准确率仍须依据 RUN 后续产物，不能
用权重 NMSE 或离线编译通过替代。
