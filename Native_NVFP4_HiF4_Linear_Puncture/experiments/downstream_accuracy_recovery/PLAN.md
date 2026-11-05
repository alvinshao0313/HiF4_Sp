# NVFP4 → HiF4 下游精度恢复

状态（2026-10-01）：五项全量配对评测已完成并验证，`group_top_mass` 的平均正向下降为 **0.19019546 个百分点，达到 ≤0.5 的目标**。MMLU-Pro 使用全部 12,032 题。完整指标见[验收报告](../../results/downstream_accuracy_recovery/paired_full_20260930T131000Z/REPORT.md)；此结论限于本轮固定协议下的观察值。

## 用户确认的目标与边界（2026-09-30）

模型为 `nvidia/Qwen3-30B-A3B-NVFP4`。对五个任务分别计算百分制分数下降：

```text
loss_i = max(0, score_native_nvfp4_i - score_hif4_i)
mean_loss = (loss_MMLU-Pro + loss_LCB + loss_GSM8K + loss_ARC-E + loss_ARC-C) / 5
目标：mean_loss <= 0.5 个百分点
```

任务提升不能抵消其他任务退步。MMLU-Pro 必须使用全部测试集，覆盖项目默认的 300 题设置；缺少任意任务、用子集代替全量或混用不同输入协议，都不能宣告完成目标。

- 用户授权使用全部 8 张 GPU；当前机器为 8 × A800 80GB，2026-09-30 首次检查均空闲。实际占用须在启动前重新核对。
- 训练只能逐层；允许残差 LoRA / norm、对角变换、非等价变换这些参数类型，不增加其他可学习参数，不改为端到端全参数训练。
- 可以调整逐层损失、router 损失和超参数。可以使用开放且论文常用的微调数据集，须记录来源、版本、实际样本和拆分。
- 保留原始 checkpoint、数据、已完成结果和其他任务改动；新运行写独立目录。依赖安装、停止既有进程和 Git 操作不包含在本目标授权内。

## 当前证据与实验优先级

1. `group_top_mass` 已完成 48 层训练并有完整模型，来自 `non_equivalent_reconstruction/full48_20260915T084300Z/group_top_mass`。其设置是 E4 DIAG 初始化、per-G64 非等价矩阵、两个 RMSNorm、block-delta NMSE + top_mass router KL；128/32 条 S1K，20 epochs/层，batch 4，AdamW lr=1e-4。初始矩阵普通投影为 R64，router 为 I64，基础权重冻结。
2. [残差 LoRA 五组结果](../../results/residual_lora_compensation/20260922T012547Z/downstream/COMPARISON.md)中，三种 r=4、alpha=8 配置的四项下游分数均低于 `group_top_mass`。局部 NMSE 改善不足以支持继续优先扩展残差参数。
3. 本轮启动前 GSM8K 缺测，MMLU-Pro 只有 300 题结果。历史 E0 与候选实际 prompt 一致，但 E0 曾使用 2048 batched tokens，候选使用 4096；本轮统一为 4096，执行各自所需的原生 NVFP4 / HiF4 后端。
4. `group_top_mass` 的两次 MMLU-Pro 300 题结果为 70.00% 和 74.33%。已核查实际输入相同，保存的模型配置仅模型路径不同，生成参数和引擎 seed 相同；299/300 个生成文本不同。原因尚未确定，不能只挑最高分认定恢复，也不能直接归因于不同 seed。
5. Progressive 六配置已完成训练，但没有五项下游结果；方向组的局部 holdout 指标未优于 baseline。KL 方向复用目前有单层实验结论，不能当作完整逐层模型的五项收益。

本轮优先验证已有最有希望的完整模型，结果已达标，因此复用 `group_top_mass`，没有启动新增训练或后备数据方案。

## 全量评测协议

| 任务 | 数据与覆盖 | 指标 | 后端 |
| --- | --- | --- | --- |
| MMLU-Pro | 全部 test，12,032 题，0-shot | extractive match | 根目录 main.py + vLLM / lighteval |
| LiveCodeBench | `lcb:codegeneration_v6|0`，175 题 | 官方 checker，单次生成 pass@1 | 同上 |
| GSM8K | `openai/gsm8k` main test，1,319 题，0-shot | extractive match | 同上 |
| ARC-E | 全部 test，2,376 题，0-shot | raw accuracy | lm_eval + vLLM |
| ARC-C | 全部 test，1,172 题，0-shot | raw accuracy | lm_eval + vLLM |

Reasoning 固定 thinking 开启、temperature=0.6、top-p=0.95、top-k=20、min-p=0；每题生成 1 个回答；引擎 seed=1234。MMLU-Pro / GSM8K 的生成上限为 32,768，LCB 为 38,912；context=40,960，max_num_seqs=128。ARC seed=42，context=4096。全部 TP2、BF16 KV、eager、max_num_batched_tokens=4096。

原生模型直接加载 packed NVFP4 checkpoint，在 A800 使用现有 `emulation` linear/MoE 后端，实际引擎量化类型必须是 `modelopt_fp4`。HiF4 沿用权重 QDQ 后 BF16 存储、runtime HiF4 激活量化与 MoE Triton 路径；两者都是相应 4-bit 数值定义，但本目标没有把 BF16 存储的 HiF4 模型宣称为原生 4-bit 打包部署或性能提升。

每个模型、任务、短测/正式运行均使用独立的模型目录视图隔离预测缓存。权重以符号链接复用，不复制或修改原模型。正式汇总检查全量样本数、ID 唯一性、实际 prompt、tokenizer、采样参数、实际后端以及逐题指标均值。仅五项齐备后计算正向下降均值。

首次全量配对先判断观察值。若结果接近阈值或与已有结果存在影响决策的实质波动，再针对相关任务复验；不把单次采样解释为所有随机种子下的保证，也不追逐有利随机结果。

## 本轮实施与运行

代码独立放在本目录，复用已经验证的底层推理与评分实现：

- `protocol.py`：五项范围、采样、资源配置。
- `evaluate.py`：原生 NVFP4 与 HiF4 评测入口，支持只减少样本数量的 `--smoke`。
- `reporting.py`：实际输入审计、全量结果核对和目标计算。
- `run.py`：原始模型视图、配置/源码哈希、四组 TP2 并行任务及状态记录。
- `scripts/run_paired.sh`：从已激活 `hif4` 的 shell 启动正式后台任务。

准备与短测在正式启动前完成。短测覆盖原生与 HiF4 的 ARC 和 GSM8K 生成、评分、结果写入和退出；MMLU/LCB 的评分链路复用 2026-09-24 已通过证据，并核对关键源码哈希未变。全量输入审计保留正式最大长度约束，短测保留正式模型、TP2、KV 和 batch token 配置。

正式调度：GPU 0/1 跑原生 ARC → GSM8K → LCB；GPU 2/3 跑 group 的同三项；GPU 4/5 跑原生全量 MMLU-Pro；GPU 6/7 跑 group 全量 MMLU-Pro。每条队列失败即停止该队列并记录原因，不终止独立队列，不自动重试；任务结束后不占卡。

运行根目录：`Native_NVFP4_HiF4_Linear_Puncture/results/downstream_accuracy_recovery/paired_full_20260930T131000Z`。

```bash
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.downstream_accuracy_recovery.run prepare --run_root "$RUN_ROOT"
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.downstream_accuracy_recovery.run smoke --run_root "$RUN_ROOT"
bash Native_NVFP4_HiF4_Linear_Puncture/experiments/downstream_accuracy_recovery/scripts/run_paired.sh "$RUN_ROOT"
```

`manifest.json` 保存实际配置和源码哈希，`prompt_audit.json` 保存全量输入证据；`smoke/` 与 `formal/` 各自保存状态、命令、PID、日志和结果。正式结束后已生成并核对 `comparison.json` 与 `REPORT.md`，本轮完整观察值达标。

本轮已完成 8 项 CPU 目标计算/覆盖检查测试，以及原生和 HiF4 各自 ARC、GSM8K 共 4 个真实 TP2 短测。短测保留正式生成上限、128-seq 容量和4096 batch token 条件，完成生成、评分、保存、引擎退出；短测分数不进入正式比较。全量审计得到 MMLU-Pro / LCB / GSM8K 最大输入长度为 1710 / 1704 / 200 tokens。

正式启动时间为 `2026-09-30T13:13:05Z`，当时 `setsid` launcher PID `2534976`，编排 PID `2534982`。启动后两模型均加载全部 12,032 条 MMLU-Pro 请求进入生成，ARC 进入真实评分计算；确认正常推进后交付，任务在后台完成。最终状态和运行过程保存在[任务状态](../../results/downstream_accuracy_recovery/paired_full_20260930T131000Z/formal/status.json)和[launcher 日志](../../results/downstream_accuracy_recovery/paired_full_20260930T131000Z/launcher.log)，不要在此已完成目录重复启动。

数据版本与完整内容指纹另存为[数据来源记录](../../results/downstream_accuracy_recovery/paired_full_20260930T131000Z/dataset_provenance.json)。MMLU-Pro 实际 revision 为 `3373e0b32277875b8db2aa555a333b78a08477ea`，LCB 为 `89e5fc5c2a8e748f50e95bc7235fab2372d49bfa`，GSM8K 为 `740312add88f781978c0658806c59bc2815b9866`；三项内容与本 RUN 的 prompt 审计一致。下载 GSM8K 时仅为本次命令使用官方 HF endpoint，未改变全局环境或安装依赖。

## 验收结果与结论（2026-10-01）

正式运行于 `2026-10-01T04:53:45.510528Z` 完成。两模型共 8 个评测任务全部退出码为 0，最终核验时全部 GPU 已释放。完整逐项分数、正向下降与原始指标路径见[生成报告](../../results/downstream_accuracy_recovery/paired_full_20260930T131000Z/REPORT.md)和[机器可读汇总](../../results/downstream_accuracy_recovery/paired_full_20260930T131000Z/comparison.json)。

五项平均正向下降为 **0.19019545699055396 个百分点**，比目标上限低约 0.3098 个百分点，达到用户确认的验收口径。MMLU-Pro 与 LCB 的正向下降均为 0；ARC-C、ARC-E、GSM8K 分别下降约 0.5973、0.1263、0.2274 个百分点。平均达标不等于每项均下降不超过 0.5，其他任务的提升没有抵消这些下降。

最终校验重新读取了两模型全部正式指标及 reasoning 原始结果与逐题 details：MMLU-Pro 各 12,032 题、LCB 各 175 题、GSM8K 各 1,319 题，唯一 ID、每题一次生成、实际 prompt 哈希、实际模型/采样配置和逐题评分均值全部匹配；ARC-C / ARC-E 各自覆盖全部 1,172 / 2,376 题并使用 raw accuracy。全部分数与任务状态、独立重算的目标值和自动汇总一致。11 个登记的源码哈希未变，各正式模型的 config、权重索引及适用的 HiF4 runtime spec 哈希与来源 manifest 一致。

模型来源的[训练 manifest](../../results/non_equivalent_reconstruction/full48_20260915T084300Z/group_top_mass/manifest.json)与[48 层训练汇总](../../results/non_equivalent_reconstruction/full48_20260915T084300Z/group_top_mass/summary.json)均已核对。采用的配置仍是逐层 per-G64 非等价变换与两个 norm、E4 DIAG 初始化、block-delta NMSE + top_mass router KL（k=8、T=1、权重=1）、S1K 完整文本 128/32、20 epochs/层、batch=4、lr=1e-4；没有扩展可学习参数类型，也没有改为端到端训练。

本轮结果不接近验收阈值，没有为追逐有利分数追加重采样。结论限定为这一模型和固定全量协议下的观察值，不能把整套配置的结果单独归因于 router KL，也不宣称所有采样种子都达标。HiF4 使用 QDQ 后 BF16 权重存储及 runtime 4-bit 激活量化，报告精度不代表实测的 4-bit 存储或加速效果。

经验已归入[评测协议与真实执行路径](../../../docs/experience/lessons/evaluation_evidence.md)，报告已加入文档入口与证据导航。保留原模型、所有运行指标、日志和复现代码；没有删除旧产物、安装依赖或执行 Git 暂存/提交。

## 未达标时的后续实验范围

以下为本轮评测前预留的条件方案。当前完整结果已达标，因此未启动，也不是本目标的未完成事项。

按新配对结果选择具体任务短板，一次改变少量因素：先探索现有 Group-G64 训练的 router loss 权重，再考虑温度、重构目标和更贴近 math/code/general 的数据覆盖。用于宣告达标的候选必须完成五项全量验收，低成本筛选或局部指标不能代替最终结果。既有矩阵、norm、残差 LoRA 足够表达本阶段候选，不增加新的可学习参数类型。

2026-09-30 等待全量评测期间，对[既有 mass 汇总](../../results/non_equivalent_reconstruction/full48_20260915T084300Z/group_top_mass/summary.json)和[partial 汇总](../../results/non_equivalent_reconstruction/full48_20260915T084300Z/group_top_partial/summary.json)重新计算 48 层验证指标的等权平均：mass 的 top-k 集合一致率 / 重合率为 81.4716% / 97.6312%，partial 为 20.4191% / 84.0817%；student 减 teacher 的 top-k 外概率质量平均差分别为 -0.0000637 / -0.150716。这支持优先保留 mass 形式；两组有各自的 progressive 输入，不能据此单独确证损失形式的因果收益，也不能把局部路由指标等同于下游精度。mass 的 router loss / NMSE 比值中位数约为 0.00190，但损失值比例不代表梯度比例，不能据此直接认定需要增大权重。

若当前配对结果未达标，首先从以下两个单变量候选中安排正式短测和训练；这是待验证的候选定义，当前尚未启动：

| 候选 | 相对 group_top_mass 的唯一变化 | 待验证假设 |
| --- | --- | --- |
| router_w03 | `--router_loss_weight 0.3` | 减弱路由蒸馏是否能改善重构与路由之间的取舍 |
| router_w3 | `--router_loss_weight 3.0` | 加强路由蒸馏是否能减少影响下游的路由差异 |

两者沿用 [non_equivalent_reconstruction.train](../non_equivalent_reconstruction/train.py)：`matrix_sharing=group`、`router_loss=top_mass`、k=8、T=1、E4 adopted DIAG 初始化、48 层、20 epochs/层、batch=4、lr=1e-4、S1K 完整文本 128/32、seed=42。每个候选从同一 E4 起点新建独立输出；不把旧 group 训练状态当作相同起点。保留辅助 router 分支在 MoE norm 之前 detach 的梯度边界，以及每层按该配置验证目标选择已训练 epoch 的规则。不同权重的 objective 数值不可直接排名；分别检查 NMSE、路由指标及实际下游结果。

后续新候选可先评测 ARC、GSM8K、LCB，再安排最耗时的全量 MMLU-Pro。若已完成且同协议核验通过的任务满足 `sum(max(0, E0_i - candidate_i)) > 2.5` 个百分点，则即使其余任务零下降，五项平均也必定超过 0.5；此时直接淘汰该候选，不启动尚未开始的评测任务。这是已确认目标的数学下界，不缩小最终验收范围；不据此终止当前已在运行的进程。原生对应任务尚未完成时不拿旧分数代替。

已查到的后备数据包括 OLMoE 论文使用的 [OLMoE SFT mix](https://huggingface.co/datasets/allenai/tulu-v3.1-mix-preview-4096-OLMoE)（本地已缓存，包含 general/math/code）与 [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k)。当前尚未选择新的数据配比。若使用，须对实际抽样记录来源/许可与 revision，检查和本项目测试题的重复，保持训练/验证互斥；不同序列长度按实际 token 预算核算，不把同样样本条数当作同样计算预算。

数据替换的源码准备结论（只读核查，尚未实施）：[训练配置](../non_equivalent_reconstruction/config.py)和[共享校准入口](../e2e_diag_reconstruction/data/calibration.py)目前不能直接识别 OLMoE，不能仅换缓存路径或把数据命名为 S1K。最小实现是在训练实验中增加独立数据准备器及明确来源值：将固定 revision 的完整 `messages` 通过源模型 Qwen3 模板渲染为 `CalibrationSample`，`add_generation_prompt=False`，全 token loss；复用现有长度分桶、右侧 padding 和逐层训练。当前没有跨样本 sequence packing，数据集名称中的 4096 也不是当前 tokenizer 下的硬长度上限。

新准备器需记录实际行号、数据/文本/token 哈希、tokenizer/template 哈希及 token 预算，按规范化用户题面分组后固定 seed 划分 train/validation，避免同题不同答案跨集合。现有 [progressive 数据模块](../progressive_error_cancellation/data.py)的 token 哈希去重与重载检查可以借鉴，但不修改其源码。所查训练入口没有五项测试题的去污染实现，使用新数据前须补充实际抽样题面与固定版本测试题的重复检查，保留排除依据；不得用测试答案或分数筛选训练数据。训练参数结构与导出格式可以保持不变。

相关经验：[评测协议与真实执行路径](../../../docs/experience/lessons/evaluation_evidence.md)。本轮首先落实实际 prompt 一致、历史缓存隔离和全量范围；旧 LoRA 局部改善不能作为已达标证据。
