# 两个发布模型的六任务 scale 对照

N9 对应 Nemotron Nano 9B，Q4Q 对应 QUASAR Qwen3.5-4B。已有原生 NVFP4 与 HiF4/MXFP4 直转成绩从论文的外机来源复用。新运行仅包含两模型的 H06（HiF4 full scale 搜索）、H18（HiF4 每矩阵常数）、M06（MXFP4 UOS）、M07/M08（MXFP4 外层 FP32 G256/G512）。HiF4 两项只改权重。发布源、数据文件和逐模型采样设置见正式根目录的 `plan.json` 和 `data/manifest.json`。

| 任务 | 每种子题数 | 种子 | 论文指标 |
| --- | ---: | --- | --- |
| GSM8K | 1319 | 1234、1235、1236 | extractive match |
| GPQA Diamond | 198 | 1234、1235、1236 | pass@1 |
| LiveCodeBench | 175 | 1234、1235、1236 | code generation pass@1 |
| ARC-Challenge | 1172 | 1234 | acc_norm |
| OpenBookQA | 500 | 1234 | acc_norm |
| BoolQ | 3270 | 1234 | acc |

推理使用与单项基线相同的 lighteval 数据版本、聊天模板、32768 输出和 40960 上下文，每题一个生成。各任务的三个完整种子取算术平均。选择题使用 lm_eval，0-shot，无聊天模板，batch 128。新配置使用 TP1，历史对照保留实际 TP。输入、采样、评分与其余比较条件应一致。

在已激活的 hif4 环境、仓库根目录运行。正式根目录为 `Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/table3_six_tasks_20261126`。

```bash
python -m fp4_convert.experiments.table3.run prepare
python -m fp4_convert.experiments.table3.run inventory
```

`prepare` 核对真实样本、既有推理模板生成的输入指纹与既有 QA 样本摘要，保存复用清单。选用当前明确授权且空闲的单卡，以独立配置列表运行。以下命令中的 3 是一条 lane 的 GPU 编号示例。

```bash
python -m fp4_convert.experiments.table3.run lane --gpu 3 --configs Q4Q-M06
```

`--phase export` 只导出并检查索引、覆盖和实际 safetensors 元数据。`--phase qualify` 增加三个 QA 任务各 40 题及 8 个完整 GSM8K 生成。`--phase all` 默认完成资格验证后自动运行缺少的正式任务。正式 QA 和 GSM8K 推理短测使用同一导出 checkpoint。TP1 的模型容量、128 并发、任务切换和三个种子已有原基线 TP1 资格与全量记录。新短测用真实 scoring 及原始输入核对方法变更后的整模加载、前向、解码、保存和原始评分；每个评测子进程独立退出后再发布凭据。

每次调用写入 `lanes/` 与 `attempts/<配置>/<阶段>/<尝试>/`，后者保存命令、环境、代码 SHA256、子进程 PID、退出码、日志、输出和检查使用的原始数据。资格凭据在 `qualification/`。正式凭据按任务、种子写入 `verified/<配置>--<任务>--<种子>.json`，提供摘要、原始逐题结果、其哈希及进程退出记录。单卡与配置分别加锁；已验收的导出和轮次可直接重用。若正式运行中断，先检查 PID、占用、当前退出记录和产物，再调用相同 lane 命令。成功退出但尚未发布的评测产物会先经过完整校验。正式论文同步使用 `python LaTeX/tables/sync_table3.py`。
