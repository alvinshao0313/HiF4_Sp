# 完整 KL / 硬 CE 的 TP2 并行评测接续

2026-10-21，用户要求使用空卡并行测试。原实验为 [reasoning_trace_full_kl_ce](../reasoning_trace_full_kl_ce/README.md)。本入口独立于其已绑定源码，保持模型、训练、开发集决选、正式数据、prompt/token/gold、引擎和评分定义。

固定使用三个互不重叠的 TP2 对：0/1、4/5、6/7；不使用有其他任务的 2/3。保留正在 6/7 运行的 CE 开发集进程及其子进程。KL 已锁定模型并通过原 TP2 短测，可以先在 0/1 和 4/5 运行正式任务；CE 完成原开发集决选、真实 TP2 验证后加入任务队列。优先 MMLU-Pro、LCB，其次 GSM8K、AIME25、辅助 ARC；每项完整任务由一个进程使用真实 TP2，不在任务内拆分题目，不改变采样或引擎设置。

调度改变了不同候选测试的开始时间。每个模型仍在测试前独立锁定；CE 决选进程只读已固定的开发集及原 B 结果，不读取 KL 正式结果；训练已全部完成，不将正式测试反馈给训练或选择。两组正式结果均保留在原 RUN 的 `evaluation/CANDIDATE/TASK/`，共同使用原报告入口。

入口使用原 `hif4` 调用者环境，先 prepare、再 qualify；并发的两组真实 TP2 保留原 40960 上下文、显存比例和其余引擎设置，只将短测 decode 限制为 64 tokens。短测同时覆盖 PID 身份绑定与真实非子进程的结束等待，使用环境已安装的 psutil。正式启动前必须取得停止旧总控的明确授权；本入口不会停止任何进程。

```bash
python -u -m Native_NVFP4_HiF4_Linear_Puncture.experiments.reasoning_trace_full_kl_ce_parallel.run \
  --run RUN --output RUN/parallel_UTC --mode prepare \
  --legacy-controller 3572585 --development-pid 3727986
python -u -m Native_NVFP4_HiF4_Linear_Puncture.experiments.reasoning_trace_full_kl_ce_parallel.run \
  --run RUN --output RUN/parallel_UTC --mode qualify
```

获得授权、核对 PID/命令/启动标识后，只停止旧总控，保留其 CE 开发集子进程。将 `--mode execute` 脱离会话启动。旧实验依赖不变，新目录绑定旧源码摘要、新接续源码快照、进程身份及锁定选择记录。`commands.jsonl` 保存每个新任务的 PID、GPU、日志和退出码。被接续的 CE 开发集不是新总控的子进程，其退出码不推测，按进程结束和完整 selection/export 验证接续。

这是有限的十项任务编排，无自动重试、占卡或守护进程。每个任务独占其输出路径；若原总控仍活着、已有正式输出或关键绑定改变，入口停止。错误保留证据，不覆盖重跑。AIME 歧义保存后等待复核，全部正式结果有效后复用原 `report` 生成比较。

实验设计参考[实际评测协议与输入绑定经验](../../../docs/experience/lessons/evaluation_evidence.md)。只改变独立任务并发，不把正常跨卡浮点差异扩成逐位一致门槛。本次不产生新的算法收益结论。

本次运行：[运行交付与验证](../../results/reasoning_trace_full_kl_ce/run_20261021T030031Z/parallel_20261021T080148Z/HANDOFF.md)。状态以该目录 manifest 和日志为准。
