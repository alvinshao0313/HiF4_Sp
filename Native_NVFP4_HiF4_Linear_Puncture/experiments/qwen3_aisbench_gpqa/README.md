# 八候选 GPQA：仅替换 AISBench prompt

用户于2026-11-13将重测范围明确为GPQA，并要求只替换prompt、其余超参数不变，
KV cache不量化。因此保留TP4、BF16 KV；初始为GPU0–3与4–7两条有限顺序队列。
用户随后收回GPU4–7并取消G256；R16继续使用TP4，G512另行适配TP2后在GPU0–1接续，详见下方记录。
八个候选为原生NVFP4、直转MXFP4、R64、H32、R16扩展R32、UOS c7.25、G256、G512。

本入口使用AISBench的零样本CoT模板，固定提交
`aff1554354fbf19dccbdf2199cc3c22119f815a2`及
`ais_bench/benchmark/configs/datasets/gpqa/gpqa_gen_0_shot_cot_chat_prompt.py`。
不安装AISBench，不更换评测框架或数据；逐题核对只有instruction文本发生变化，
原题干、选项顺序、答案、Qwen chat template和thinking行为保持一致。

- 198题，每题10个回答；seed1234，temperature0.6、top_p0.95、top_k20、min_p0。
- 最多生成32768，上下文40960；TP4、batch256、8192 batched tokens、eager。
- 每卡32GiB BF16 KV，linear/moe均为原emulation路径；无KV量化。
- 复用原始checkpoint，逐分片核对转换产物摘要；不重新量化，不修改原量化器。
- 原严格JSON评分和原`parse_final`复核规则均保留。新的`Answer: A`格式已在原复核
  规则覆盖内。歧义回答进入盲审队列；存在待审项的分数明确标为临时值。
- 短测每候选26题×10、最多64个token；短测分数不用作准确率。八组均通过后才启动正式队列。
- direct/R64的共享代码后来新增过H32分支，故对照保存的原代码补验两者QDQ的
  12个实际GPU形状，并核对packed GEMM与MoE函数AST不变。其他既有算子证据按源码摘要复用。

在已激活`hif4`的调用者shell执行：

```bash
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_aisbench_gpqa.prepare --output RUN
CUDA_VISIBLE_DEVICES=0 python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_aisbench_gpqa.validate_preservation --output RUN/direct_r64_preservation.json
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_aisbench_gpqa.workflow probe --root RUN
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_aisbench_gpqa.workflow formal --root RUN
```

本次运行目录：
[aisbench_gpqa_20261113T023500Z](../../results/qwen3_mxfp4/aisbench_gpqa_20261113T023500Z)。
实时状态以`probe_workflow_status.json`、`formal_workflow_status.json`及每阶段`status.json`为准。
准备或PID不代表通过、完成或最终准确率。失败退出，不自动重试或改协议。

2026-11-13北京时间15:42：八组短测全部通过，原生、直转、R64、H32均已完成1980个回答，
缓存和运行核验通过；各组仍有待盲审的答案，自动复核分数不是最终准确率。
用户要求释放GPU4–7、取消G256并保留G512后，旧总控PID1551493及UOS进程树已停止。
UOS停止时日志进度为180/1980，尚无完整回答缓存，不报告本轮准确率。
GPU0–3的R16原进程PID1732688保持运行；新接续控制器PID1749827等待其成功完成后，
仅在GPU0–3启动一次G512。G512仅更换物理卡号，TP4、BF16 KV、其余参数及已验证源码不变。
GPU4–7不再安排任务，G256不启动，UOS不自动重启；任何接续失败均退出。

实际命令、源码摘要、停止记录、资源变更和接续状态见
[GPU0–3接管目录](../../results/qwen3_mxfp4/aisbench_gpqa_20261113T023500Z/gpu03_takeover_20261113T073938Z)。
其中`launch.json`记录启动命令，`continuation.log`与`continuation_status.json`记录接续，
`stop_record.json`保存被停止及被保护进程身份；原`handoff.json`仅代表初始两队列交付。
复用了现有非子进程等待经验，并验证真实CPU进程等待、未完成产物拒绝及仅卡号变更；无新增算法经验。
旧八组KV dtype的逐项核验见`prior_kv_dtype_audit.json`，均为BF16。

2026-11-13北京时间16:33更新：用户明确选择“G512也改TP2，先验证再正式测”，
上述TP4接续控制器PID1749827已停止，R16原进程保持运行。
新的[隔离TP2接续入口](../qwen3_mxfp4_dualscale_tp2/README.md)PID1781784等待R16完成后，
只使用GPU0–1进行两rank算子验证、执行检查及G512正式评测；任一步失败退出。
CPU配置/子进程注册与48层切片核验已通过，GPU验证和正式G512尚未启动。
原G512 TP4目录保留为历史准备与短测证据；新TP2结果使用独立目录，汇总会注明TP差异。

旧评测已按要求停止，已完成的八组GPQA和两组LCB/ARC结果见
[停止前汇总](../../results/qwen3_mxfp4/rotation_downstream_20261111T021111Z/stopped_results_20261113.md)，
与本次AISBench prompt结果分开。MMLU-Pro没有完成结果。
