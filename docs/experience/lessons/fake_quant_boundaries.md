# fake quant 与实际低比特执行

证据状态：依据已有实现记录整理边界；本次未执行性能或精度测试。

[HiF4 实现记录](../records/implementation/hif4_vllm_fake_act_quant.md) 和 [NVFP4 实现记录](../records/implementation/nvfp4_vllm_fake_act_quant.md) 描述 dense Linear 输入上的 quant-dequant，再交给原浮点 GEMM。它们不等于 packed 低比特计算核，也不覆盖全部 MoE 路径。

- fake quant 用于研究数值扰动，不能据此声称模型实际位宽、存储压缩率或推理速度已经改善。存储口径需另说明 scale、zero-point 等额外开销。
- HiF4 记录包含按 64 分组和非整组 padding 等处理；应按实际实现核对形状和 lm_head 排除条件，不能把该局部入口等同于完整 W4A4 模型。
- NVFP4 记录要求对应 activation-scale sidecar 与 layer prefix；缺失或不匹配应明确失败，不能无依据改成其他 scale 或静默降级。
- 文末的“建议验证”描述待做检查，不是通过记录。精度与吞吐结论仍需要相应运行证据。

当前调用方式见 [仓库 README](../../../README.md)，底层 HiF4 量化过程见 [量化流程](../../guides/hif4_gpu_quant_flow.md)。
