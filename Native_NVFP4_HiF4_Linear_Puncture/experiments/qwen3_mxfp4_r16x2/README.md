# Qwen3 MXFP4：regular R16 扩展到 32 维

本目录隔离实现已确认的 `r16x2` 候选，避免修改正在运行的标准 H32 实验依赖。
详细协议与对照结果见[实验记录](../../../docs/experience/records/qwen3_mxfp4_20261109.md)。

## 固定矩阵与转换边界

`R4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]] / 2`，
`R16 = R4 ⊗ R4`，`R32 = [[R16,R16],[R16,-R16]] / sqrt(2)`。
矩阵正交、对称，元素为 `±1/sqrt(32)`。常量输入的能量分布于 16 个输出通道，
每个幅值为 `sqrt(2)`；这不证明任意真实激活的能量均衡，也不预示准确率提升。

权重从同一原始 NVFP4 snapshot 独立解码并舍入 BF16，按连续 K32 在 FP32 中变换，
直接量化为 MXFP4。激活使用同方向变换。每个 K32 的 E8M0 scale 为
`2**clamp(floor(log2(amax))-2, -127, 127)`，全零组取 1，E2M1 最近偶数舍入并饱和。
只替换路由专家 gate/up/down；SwiGLU、路由加权、求和、TP 通信及非专家路径保持原实现。
TP4 下 gate/up K=2048，down 局部 K=192，均不跨 rank 拼组。

`transform.py` 提供两级 regular H4 加一级 H2 的 FP32 实现及独立密集矩阵参考；
`kernels.py` 融合变换与 QDQ，复用原参数化 packed E2M1 GEMM。
权重名义负载为 4.25 bit/元素（含组 scale）；A800 执行软件 QDQ 和 BF16 GEMM。

## 入口与验证

由调用者先激活 `hif4`，在仓库根目录运行。`convert.py` 支持
`--source`、`--output`、`--variant r16x2`、`--threads 8`；源 snapshot 固定为
`nvidia/Qwen3-30B-A3B-NVFP4@2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`。
完整 index 只在全部转换覆盖、非专家字节一致性及保存重载校验通过后发布。

独立注册方法为 `qwen3_routed_mxfp4_r16x2`，通过 `entry.py` 在父进程和 spawn 子进程注册。
`validate_gpu.py --model PATH --output REPORT` 使用实际 TP4 专家形状，
检查 tokens=1/256/8192 下的 QDQ、两次 GEMM、原生 SwiGLU 和端到端 MoE，记录 CUDA events 性能。

`workflow.py prepare --root RUN --previous COMPLETED_THREE_MODEL_RUN` 冻结协议、代码和复用证据；
`workflow.py run --root RUN` 依次执行 GPU0 算子验证、GPU0–3 TP4 的 26×10×64 执行短测，
再运行 198×10×32768 GPQA Diamond。各阶段失败即退出，不自动重试或改变协议。
全部 48 层、4 ranks 的 packed 权重及 scale 与 checkpoint 切片逐字节核对，非专家方法同原生。

正式配置沿用 batch256、8192 batched tokens、40960 上下文、32 GiB BF16 KV/卡、eager、
thinking、seed1234、temperature0.6、top_p0.95、top_k20、min_p0。
短测分数不作为准确率。正式原始 JSON 分数保留，最终提交复核独立输出；新候选的疑难回答
进入盲审队列，复用原三组人工判定前核对文本身份，不能自动改写旧结果。

代码摘要同时覆盖本目录和实际复用的原 `qwen3_mxfp4` / vLLM 文件。
本次运行目录为 `results/qwen3_mxfp4/r16x2_gpu03_20261110T040449Z`（相对 `Native_NVFP4_HiF4_Linear_Puncture`），
正式状态和命令见 `r16x2/formal/status.json`，原始日志见 `r16x2/formal/main.log`。
