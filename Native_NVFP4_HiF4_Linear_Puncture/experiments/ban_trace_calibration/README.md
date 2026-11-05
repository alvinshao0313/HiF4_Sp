# 自生成轨迹重新校准 Ban

保持当前 Qwen3-30B `matrix_norm` HiF4 导出不变，对照 top-8、C4 Ban 和自生成轨迹 Ban。
LCB 每组完整175题×3轮，MMLU-Pro 每组完整12032题×1轮；新旧 Ban 默认入口和已有300题
设置均不改变。已有同权重、实际输入和推理/评分协议一致的完整基线直接复用，并注明来源。

复用已排重原生生成轨迹的128条训练数据，在数学/代码各64条中均匀、有放回抽取256个
2048-token窗口。短轨迹排除、窗口位置及重叠覆盖均记录；终点位于生成回答，直接保留
原始 token IDs。验证集按保存顺序取数学8题、代码8题，仅匹配专家预算，不按正确性选题。
两组 Ban 的 k_min=5，lambda 在真实生成中匹配运行清单中的目标（默认6.07±0.05个专家）；
未匹配不进入该组正式评测。`prepare --target_experts` 可指定新目标，旧入口默认值不变。

在已激活 `hif4` 的仓库根目录执行，所有输出必须为全新目录。先完成代码和输入检查，
再用实时空闲的TP2卡对完成预检；示例卡号不构成资源占用事实。

```bash
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.ban_trace_calibration.prepare \
  --output /absolute/path/to/NEW_RUN
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.ban_trace_calibration.run \
  --run /absolute/path/to/NEW_RUN --gpu_pairs 0,1 --preflight
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.ban_trace_calibration.run \
  --run /absolute/path/to/NEW_RUN --gpu_pairs 0,1 2,3 4,5
```

预检完成数学/代码各一个窗口的48层校准、TP2 prefill/64步 decode，以及正式上下文、
批容量、BF16 KV条件下16个开发输入的真实生成。预检既不选正式预算，也不产生精度结论。
正常入口依赖预检完成，按每组依赖执行512窗口校准、预算匹配、正式参数的两题短测、
LCB三轮和完整MMLU一轮。最多三对卡，可只提供一对（先trace，再C4，再top8），不降协议。
不自动重试，不停止既有进程，不启动占卡程序；正式运行用脱离会话方式并保存控制器日志。

`status.json` 保存命令、PID、日志、阶段及退出状态；`manifest.json` 固定输入、源码与模型
身份；各次推理均使用独立模型文件视图和预测缓存。报告流式核对原始回答缓存、实际请求、
prompt/gold/input token、所有题目和TP单份路由统计，拒收300题冒充全量MMLU。
`report.json` 保留全部轮次、正确数、均值/范围、同轮98%保持率和自生成Ban相对C4的
题目分簇配对bootstrap区间（10000次，seed0）。相同seed重复不称为独立seed；测试集
专家数差超过0.10时不解释为相同预算收益。精度或专家数变化不代表实测吞吐提升。

```bash
python -m pytest -q Native_NVFP4_HiF4_Linear_Puncture/experiments/ban_trace_calibration/tests.py
```

## 复用校准，测试6.4个专家

目标预算仅影响lambda搜索和报告，不改变层敏感度、r范围、Ban公式或权重。`rebudget`
验证来源的模型、512窗口校准、实际校准依赖及原始输入后，只链接只读校准和输入；
不复用旧lambda、短测或正式预测。新运行仍先做真实TP2预检，再分别匹配两组预算，
通过各组短测后完成LCB175题三轮和MMLU-Pro12032题一轮；top-8已有适用完整结果时复用。

```bash
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.ban_trace_calibration.rebudget \
  --source /absolute/path/to/CALIBRATED_RUN --output /absolute/path/to/NEW_BUDGET_RUN \
  --target_experts 6.4
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.ban_trace_calibration.run \
  --run /absolute/path/to/NEW_BUDGET_RUN --gpu_pairs 0,1 --preflight
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.ban_trace_calibration.run \
  --run /absolute/path/to/NEW_BUDGET_RUN --gpu_pairs 0,1 2,3 4,7
```

开发集目标为6.40±0.05，正式任务专家数按实际生成单独报告，不能保证每个任务恰为6.4。
平均6.4相对top-8减少20%的专家调用，不直接表示整体计算或吞吐变化。

## 6.4预算的其余任务补测

`remaining`仅复用已匹配6.4预算的原`matrix_norm`模型和两份512窗口校准产物，固定
C4 lambda=0.8875、自生成轨迹lambda=0.7375、k_min=5，不用正式测试结果重新调参。
两组各补测完整GSM8K1319题、AIME24@5与AIME25@5各30题五次生成。已有同模型
GSM8K和AIME25的top-8基线核验后复用；AIME24基线尚缺，首次补测一次，共7项正式运行。
LCB和完整MMLU-Pro沿用已完成报告，不继续转换提速实验。

```bash
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.ban_trace_calibration.remaining prepare \
  --source /absolute/path/to/MATCHED_BUDGET64_RUN --output /absolute/path/to/NEW_REMAINING_RUN \
  --data_source /absolute/path/to/PINNED_TASK_DATA_RUN
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.ban_trace_calibration.remaining run \
  --run /absolute/path/to/NEW_REMAINING_RUN --wait_for_gpu
```

本次用户明确要求等GPU0、1空闲：`--wait_for_gpu`只在启动前等两卡同时无计算PID且
显存不超过1024MiB，最多24小时。它是一次实验的有限等待，不占卡、不终止既有任务、
不重试失败运行；未传该参数则资源不足立即停止。等待状态为`waiting_for_gpu`，不代表
GPU推理已开始。资源可用时再次核验冻结输入/源码，然后分别对top-8及两种Ban完成真实
TP2短测，再进入正式任务；阶段间保留既有session清理与资源检查，不自动等待其他任务。

全部新预测使用独立模型视图和缓存，保留原正式batch、上下文、BF16 KV、采样协议与
生成上限。AIME同时保存原官方分数和仅按最终提交计分的avg@5；歧义回答落盘供逐条
复核，待复核时不填最终精度或98%保持率。报告记录每项实际平均专家数和TP一致的
selected/dispatched，不将目标预算6.4冒充每项任务的实际值。

## 失败后的显式接续

阶段交接将推理子命令放入独立session，主进程退出后只等待该session的worker完成清理
（最多60秒），再等待残留GPU上下文清除（最多30秒）。空闲检查保留原阈值，并记录
显存和占用PID；出现其他活动任务时不等待或停止它。此等待用于阶段清理，不是重试评测。

本次已批准的重启使用独立结果目录，保留原失败记录。`resume`只接受尚无正式预测的
失败来源，重新校验模型、原始输入、512窗口校准、开发预算和短测原始产物；仅允许控制器
代码变化，原控制器快照必须匹配旧manifest的SHA。其余科学计算源码有变化时立即停止。

```bash
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.ban_trace_calibration.resume \
  --source /absolute/path/to/FAILED_RUN --output /absolute/path/to/NEW_CONTINUATION \
  --controller_snapshot /absolute/path/to/original_run.py
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.ban_trace_calibration.run \
  --run /absolute/path/to/NEW_CONTINUATION --gpu_pairs 0,1 --verify_restart
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.ban_trace_calibration.run \
  --run /absolute/path/to/NEW_CONTINUATION --gpu_pairs 0,1
```

接续前要求两次连续真实TP2引擎启停验证通过，保留正式上下文、批容量、BF16 KV和16个
开发prompt，仅将生成步数压到64。此验证以及正式控制器均由调用者脱离会话启动并落盘
stdout/stderr。接续产物明确标注哪些阶段复用，正式预测仍使用新视图和新缓存，不自动
重试失败或中断的控制器。
