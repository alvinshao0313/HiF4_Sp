# G512 的隔离 TP2 评测

用户于2026-11-13明确要求本轮G512也改用TP2，先验证后正式测。
本入口复用原G512 checkpoint及AISBench GPQA输入，不重新转换权重；
原TP4源码、正在运行的R16及其结果保持原位。

每rank的专家中间维由192改为384，gate/up输出切片及down输入切片相应扩大。
down仍按全局K=768计算G512外层scale，两个rank用MAX归约共享组最大值，
保留原输出SUM。FP32外层`absmax/6`、group32内层`floor(log2(amax))-2`、
BF16 GEMM操作数、FP32累加和分组缩放均不变。Triton量化核心与packed GEMM直接复用原实现。

仅使用GPU0–1，等待原R16释放资源；GPU2–3不占用，GPU4–7不安排任务。
正式参数保留GPQA198题×10、thinking、seed1234、temperature0.6、top_p0.95、
top_k20、min_p0、最大生成32768、上下文40960、batch256、8192 batched tokens、
每卡32GiB BF16 KV及eager。TP2是本次明确改变的执行条件，不把它称为仅prompt变化，
也不预先断言它比TP4更快。

有限执行顺序为：等待R16成功完成 → 两rank QDQ/MAX验证 → GEMM/SwiGLU/MoE验证及计时
→ 26题×10、64-token执行检查 → 正式GPQA×10。GPU验证覆盖tokens=1/256/8192，
scale/编码/QDQ与参考完全一致，GEMM/MoE沿用NMSE<1e-4及贡献幅度误差标准，
PTX须为BF16 MMA。整模型检查两个rank、全部48层真实权重及scale字节、非专家方法、
实际输入token和保存配置。任一步失败退出，不自动重试、降配置或回退TP4。

2026-11-13北京时间16:33状态：CPU配置和spawn注册2项测试通过，首末层TP2切片与
相邻TP4切片合并完全一致，已生成全部48层×2rank的预期张量摘要。
接续控制器PID1781784已启动，等待原R16 PID1732688；GPU验证、执行检查和G512正式评测尚未开始。
真实进程等待沿用已验证的psutil路径；本次无新增量化算法经验。

运行产物位于
[g512_tp2_20261113T082519Z](../../results/qwen3_mxfp4/aisbench_gpqa_20261113T023500Z/g512_tp2_20261113T082519Z)。
`launch.json`保留实际命令；`workflow.log`、`workflow_status.json`记录阶段；
`cpu_validation.json`与`expected_runtime.json`保存CPU证据；
GPU阶段会保存`communication.json`、`operators.json`及各rank记录。
短测和正式结果分别位于`dualscale_g512/probe`、`dualscale_g512/formal`，各有独立缓存。

使用已激活的`hif4`环境运行入口：

```bash
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4_dualscale_tp2.prepare --root RUN
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4_dualscale_tp2.workflow run --root RUN
```

`RUN`须包含已核对的用户授权及R16进程身份记录；已启动目录禁止重复运行。
