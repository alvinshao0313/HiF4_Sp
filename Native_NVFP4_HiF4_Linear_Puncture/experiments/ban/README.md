# matrix_norm HiF4 上的 Ban

本模块针对现有 Qwen3-30B `matrix_norm` 导出，支持 48 层、128 专家、原始 top-k=8、
`fusable_r64` runtime ABI 3。保留融合权重、在线 R64、HiF4 激活 QDQ 和现有 Triton GEMM。
不安装依赖，不改模型文件，不替换原评测任务。

复现来源为 [MingZwhy 仓库固定版本](https://github.com/MingZwhy/An-Empirical-Study-on-Expert-Pruning/tree/3ac1169c9a48f6cda2477975a024aadba5d02f8b)，
commit `3ac1169c9a48f6cda2477975a024aadba5d02f8b`；MIT 版权见 `UPSTREAM_LICENSE`。

## 开关与推理

使用包装入口 `python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.ban.reasoning_main`，
其余参数转交仓库根目录 `main.py` 和现有 lighteval/vLLM。

| 参数 | 行为 |
| --- | --- |
| `--expert_pruning_method none` | 默认值；走原 `main.py`，不加载校准、不安装 Ban worker、不创建模型视图 |
| `--expert_pruning_method Ban` | 显式启用 Ban，必须提供匹配的校准文件和 HiF4 sidecar |
| `--ban_artifact_path` | 本模型的 `c4_ban.pt` |
| `--ban_lambda` | 默认 0.7，范围 [0,1]；不是保留率 |
| `--ban_k_min` | 默认 3，至少为 1；大于原 top-k 时按上游钳到 top-k |
| `--ban_execution compact` | 默认；仅计算保留专家 |
| `--ban_execution masked` | 对照；仍计算原 top-k，只将被剪专家权重置零 |
| `--ban_stats_dir` | 全新统计目录；未传时在校准目录下自动新建 |

开启 Ban 需 `--enforce_eager`，当前支持 DP=1 的直接 vLLM worker 路径。DP>1 的 Ray
路径明确拒绝，避免父进程扩展未传到 Ray 后静默失效。关闭 Ban 时保持原 CLI 行为。

先在调用者 shell 激活 `hif4`。示例均为少量输入的接口检查，不能据此解释准确率。

```bash
MODEL=/home/shaoyuantian/program/HiF4_Sp/Native_NVFP4_HiF4_Linear_Puncture/results/aime25_repeats/matrix_norm_20261015T081958Z/export
export CUDA_VISIBLE_DEVICES=2,3

# 关闭；也可省略 expert_pruning_method，其他参数照常使用。
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.ban.reasoning_main \
  --expert_pruning_method none \
  --model_path "$MODEL" --hif4_runtime_spec "$MODEL/hif4_runtime_spec.pt" \
  --datasets gsm8k --max_samples 2 --max_new_tokens 128 \
  --tensor_parallel_size 2 --enforce_eager --moe_backend triton \
  --max_model_length 4096 --output_dir /absolute/path/to/new/unpruned

# 开启；将校准文件和输出目录替换为实际路径。
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.ban.reasoning_main \
  --expert_pruning_method Ban --ban_artifact_path /absolute/path/to/calibration/c4_ban.pt \
  --ban_lambda 0.7 --ban_k_min 3 --ban_stats_dir /absolute/path/to/new/routing \
  --model_path "$MODEL" --hif4_runtime_spec "$MODEL/hif4_runtime_spec.pt" \
  --datasets gsm8k --max_samples 2 --max_new_tokens 128 \
  --tensor_parallel_size 2 --enforce_eager --moe_backend triton \
  --max_model_length 4096 --output_dir /absolute/path/to/new/ban_eval
```

Ban 每次调用创建只含文件链接的独立模型视图，隔离当前 lighteval 保存于绝对模型路径下的
预测缓存，避免不同 lambda/k_min 复用旧生成。权重无拷贝，旧缓存和结果保留。

## 算法与校准

在线按排序后的原 top-k 路由权重计算 `r=mass(top3)/mass(top-k)`，
`s_token=clamp((r_max-r)/max(r_max-r_min,1e-6),0,1)`，
`s=clamp(lambda*(s_layer+s_token)/2,0,1)`，
`k=clamp(round(k_min+(K-k_min)*s),k_min,K)`。
舍入沿用 PyTorch round。保留前 k 个专家，遵守模型的 `norm_topk_prob`；prefill/decode 均启用。

每条校准输入先跑未剪枝 HiF4，再逐层仅将一层限制为 top-3，计算最后输入位置的
`KL(unpruned || pruned)`。词表支持为未剪枝输出的 top-1000，两者在该支持上各自
重新归一化，不包含 tail，不是完整词表 KL 或 router KL。先按样本平均，再对层做
min-max 归一化，得到 `s_layer`。同时从未剪枝路径收集全局 `r_min/r_max`。

校准通过真实 vLLM HiF4 路径完成。单请求、无 prefix cache、无 chunked prefill，
生成 1 token 仅为触发最后输入位置 logits；该生成 token 不进入校准数据。每个请求
必须恰有一次 `[1,vocab]` 采集，禁止用 prompt-logprobs 覆盖位置。

C4 采样沿用固定上游：词法排序明确指定的本地 train Arrow，将非空文档以空格连接，
每 256 文档检查 token 数，达到 `nsamples*seqlen+1` 后以 Python random 采窗口。
显式读取本地文件，无联网或缓存回退。保存实际 input_ids、窗口位置和 token SHA256。

```bash
C4=/home/shaoyuantian/.cache/huggingface/datasets/allenai___c4/default-b04fc8a0b8562884/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.ban.calibrate \
  --model_path "$MODEL" --hif4_runtime_spec "$MODEL/hif4_runtime_spec.pt" \
  --c4_arrow "$C4/c4-train-00000-of-00002.arrow" "$C4/c4-train-00001-of-00002.arrow" \
  --output /absolute/path/to/new/calibration \
  --nsamples 512 --seqlen 2048 --seed 0 --tensor_parallel_size 2
```

上游 Qwen3 结果使用 512 条。首轮接入运行 2 条诊断，产物标注 `phase=diagnostic`；后续
512 条正式校准及同轮 GSM8K 全量对照已完成，见下方结果记录，不能混用两类产物。
输出目录必须不存在。产物绑定 config、
sidecar、导出记录、权重索引和实际权重文件的路径/大小/mtime；后者是文件身份记录，
不是全量权重内容 SHA。模型或 sidecar 不匹配时明确失败。

## 数值与统计边界

- 校准入口显式传递 `HIF4_RUNTIME_SPEC_PATH`，与根 `main.py` 一致；worker 核对隐式和
  显式 sidecar 解析相同，确保嵌套 MoE helper 使用 R64。
- compact 按相同 k 分组，复用现有 QDQ 和 routed GEMM。最终输出补零到原 top-8
  再调用原 `moe_sum`，保留其累加方式；底层专用 top-3/4 核会逐次 BF16 舍入，不能直接替代。
- 原生 fused router 与上游 PyTorch softmax/topk 的微小算术差异可能在 HiF4 路径中放大。
  验证分别保留原 fused 路由 no-op、参考 PyTorch 路由 top-8、Ban 三种对照，避免把
  重算路由造成的变化归到剪枝或分发；不承诺关闭剪枝但重算路由时与 fused 全轨迹逐位一致。
- 统计包含每层 k 直方图、selected、dispatched 和平均专家数。TP ranks 核对后只计一份，
  初始化/profile 不计入。平均专家数和节省路由比例不等于端到端吞吐提升。
- HiF4 权重为反量化后的 BF16 存储，现有 GEMM 执行激活 QDQ 后的 BF16 计算；Ban 不
  压缩常驻权重，不声称实现原生 FP4 硬件加速。现有正式准确率证据限于记录中的同轮 GSM8K 对照。

## 验证入口

```bash
python -m pytest -q Native_NVFP4_HiF4_Linear_Puncture/experiments/ban/tests
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.ban.qualify \
  --model_path "$MODEL" --hif4_runtime_spec "$MODEL/hif4_runtime_spec.pt" \
  --c4_arrow "$C4/c4-train-00000-of-00002.arrow" "$C4/c4-train-00001-of-00002.arrow" \
  --output /absolute/path/to/new/qualification --phase diagnostic \
  --nsamples 2 --seqlen 2048 --seed 0 --tensor_parallel_size 2
```

短测覆盖 native 重复、原 fused 路由 no-op、参考路由未剪枝、全部 48 层校准、
masked/compact 比较、2048-token prefill 和 31 次 incremental decode，以及真实分发数。
保留 `qualification.json`、`diagnosis.json`、`routing.json`、`calibration/` 与日志。
接口检查沿用根 `main.py` 的 GSM8K/lighteval，仅 1 题×8 token，不解释其分数。

本轮结果见 [2026-10-23 实现记录](../../../docs/experience/records/ban_matrix_norm_20261023.md)。

## top-8 / top-6 / Ban 的同轮精度比较

`evaluate.py` 只评测 GSM8K 的 1319 条 test，三组共用未剪枝 top-8 基线，使用新建的独立
模型视图和预测缓存。`evaluation_main.py` 固定同一份本地 test Arrow 及 SHA256，最终核对
逐题 prompt、gold 和实际 `GENERATIVE.parquet` 输入 token。不开启其他下游任务。

正式配置为 TP2、eager、BF16 KV、batch/max_num_seqs=128、max_num_batched_tokens=4096、
上下文 40960、生成上限 32768、seed=1234、temperature=0.6、top_p=0.95、top_k=20、min_p=0。
三组统一关闭 prefix cache，使 Ban 统计覆盖全部输入。原 HiF4 权重与激活量化不变。

固定 top-6 使用根 `main.py` 已有的 `--num_experts_per_tok 6`。动态 Ban 使用新完成的
C4 train 512×2048、seed=0 校准，随后只用 GSM8K **train 前 8 题**匹配专家预算：本轮
固定 `k_min=5`，从 lambda=0.7 开始，必要时测 [0,1] 端点并最多二分 6 次，目标为完整
prefill+decode 的实际平均专家数 `6±0.10`。只依据路由计数选择 lambda，不计算训练准确率，
不以 test 准确率选参。推理默认 `k_min=3` 未改。test 上的实际平均数另行报告，不能保证
训练集匹配一定迁移。预算匹配失败会停止依赖它的 Ban 评测，其他两组继续保留结果。

```bash
GSM8K=/home/shaoyuantian/.cache/huggingface/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.ban.evaluate \
  --model_path "$MODEL" --hif4_runtime_spec "$MODEL/hif4_runtime_spec.pt" \
  --c4_arrow "$C4/c4-train-00000-of-00002.arrow" "$C4/c4-train-00001-of-00002.arrow" \
  --gsm8k_test_arrow "$GSM8K/gsm8k-test.arrow" \
  --gsm8k_train_arrow "$GSM8K/gsm8k-train.arrow" \
  --calibration_gpus 6,7 --evaluation_gpus 6,7 \
  --output /absolute/path/to/new/gsm8k_8to6
```

`evaluation_gpus` 运行 top-8→top-6，`calibration_gpus` 运行校准→预算匹配→Ban；短测也将
Ban 与另两组分开并行。同一 GPU 对时串行使用，两个不同 GPU 对时可并行。该入口是有限的校准和评测流程，
没有守护、自动重试或占卡任务。`--short_test --diagnostic_artifact /path/to/c4_ban.pt`
只将各组题数改为 2，保留正式配置，不做 98% 判断，也不产出正式校准。

输出 `manifest.json`、各阶段命令/PID/退出状态、`logs/`、`evaluation/{top8,top6,Ban}`、
`routing/Ban`、逐题摘要与最终 `report.json`。98% 定义为该轮 `score_pruned/score_top8≥0.98`；
同时报告整数正确数、得失题和实际专家预算。一次 GSM8K 观测不等于其他任务或统计下界保证，
也不说明吞吐提升。

## 追加 MMLU-Pro、LCB 和 AIME25@5

`supplement.py` 复用已完成的正式校准，固定 lambda=0.7、k_min=5、compact，分别运行
MMLU-Pro 300 题（项目默认）、LCB v6 全部 175 题和 AIME25 全部 30 题×5 回答。
这三项仅运行 Ban，不额外建立新的 top-8 基线，也不按测试分数重新选参。

```bash
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.ban.supplement \
  --model_path "$MODEL" --hif4_runtime_spec "$MODEL/hif4_runtime_spec.pt" \
  --artifact_path /absolute/path/to/formal/calibration/c4_ban.pt \
  --gpu_pairs 0,1 2,3 4,5 --output /absolute/path/to/new/supplement
```

GPU 对按 MMLU-Pro、LCB、AIME25 顺序分配，各对独立执行短测再正式评测；流程有限且无
自动重试。短测仅减少题数为 2/1/1，保留 TP2、batch128、40960 上下文、正式生成上限
（MMLU-Pro 32768，LCB/AIME25 38912）和 AIME 五回答语义。`additional_main.py` 绑定
已核验的本地 Arrow/SHA，仍通过 Ban 包装调用根 main.py；不同任务及阶段各用新缓存。
官方分数、原始回答、实际输入 token、终止原因和真实专家分发数分别保留并核对。

当前 LCB 的指标名 `codegen_pass@1:16` 不代表实际有 16 次生成；本仓库注册任务每题
实际为 1 次生成，补测沿用该既有路径并记录实际数量。AIME 使用 `avg@n:n=5`，正式分数
来自原始五回答，不用 pass@5 或过滤后的 final 文本替代。主报告位于输出的 `report.json`，
分任务报告和原始结果位于 `<task>/{short,formal}`，过程状态与命令各自保存。

## 重复 LCB 与 AIME25@5

`repeat.py` 以已完成的 supplement 目录为 `--source`，用必填的 `--additional_rounds`
指定每项追加轮数，`--gpu_pairs` 指定互不重叠的 TP2 卡对，`--output` 必须为新目录。
只重复 LCB 和 AIME25@5，保留来源第一轮，并为每个新增轮次创建独立模型视图、缓存和日志。

重复入口先核对原短测和正式评测已完成，来源源码、模型与校准身份未变，并复核原始结果；
相同条件下复用这些验证，不重新校准或重复短测。运行命令仅替换输出和缓存路径，保留
seed=1234、采样、生成上限、Ban 参数和评分器。固定种子的重复运行不能称为独立种子实验。

最终报告逐轮保存原始分数，汇总包括原第一轮在内的全部已完成轮次、均值、最小值、最大值
和轮间标准差；失败或缺失轮次单独记录，不择优替换第一轮。没有自动重试、占卡或等待 GPU
空闲的守护程序；正式启动前仍需核实卡对可用。
