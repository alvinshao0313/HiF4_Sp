# 单层 KL 方向复用实验

独立实验；复用 `non_equivalent_reconstruction` 的参数化和量化底层，
不读取 E4/旧 DIAG，不改变旧实验。原生 NVFP4 → 固定 R64/HiF4 为共同起点。

## 冻结协议

- Qwen3-30B-A3B-NVFP4；独立训练 0-based 层 8、24、40，不合并结果。
- `group` 64×64 输入变换、两个 RMSNorm 可学习；其余权重冻结。
- 普通矩阵 R64 初始化，router I64 初始化；无正交/逆关系/额外惩罚。
- A：完整层输出标准 MSE；B：带符号的缓存最终 KL 方向；C：直接最终 KL。
- 三组均加 `top_partial` router 辅助损失：全专家归一化、teacher top-8、T=1、λ=1。
  不截断 router 辅助梯度，允许传回 Attention；不强制 teacher 路由。
- 层输出、router 和最终分布目标均来自原生模型自身路径。
- B 为全训练集在同一参数快照采集方向，每 1/2/4 遍训练刷新；router 每步重新反向。
- S1K 原文，32/16/32 train/val/test，seed=42；不截断、不按长度筛选。
- 按长度固定四条序列的批次成员，每遍只打乱批次顺序；microbatch=1。
  MSE 按 token×channel 平均，KL/router 按固定批次 token 总数平均。
  缓存梯度已经包含此分母，局部内积不再平均。
- AdamW，恒定 lr=1e-4，weight_decay=0；单种子 42；每配置单卡 7200 秒，共15配置。
- 最终比较预算内最后完整更新，不按验证集挑选或回滚。标准 KL 为 KL(native || student)，
  全词表、T=1、全部有效 predictor；NLL 使用文本 next-token，排除末位置。
- 训练单卡模拟 TP2 分片计算；实际 vLLM TP2 测最终 KL/NLL/PPL，不跑下游任务。

三组都缓存冻结前缀产生的层输入。C 每步保留从本层到最终 KL 的反向路径；
B 刷新时计算 `g = ∂KL/∂h`，后续局部主损失为 `sum(g * (h - h_anchor))`。
这里保留梯度的符号和相对分量，不做平方、绝对值或单位化；B 不再叠加 MSE。
在刷新点、相同批次归一化下，B 的本层参数梯度应与 C 一致。
多步复用后不再保证一致，缓存记录刷新间的方向余弦、预测 KL 变化和实际变化。

## 使用

全部命令必须运行于 `hif4`。GPU 编号必须显式指定；不自动占卡、不停止其他任务。
下面 `TRAIN_GPU`、`EVAL_GPUS` 是启动前由操作者设置的编号，必须保证独占/可用。
运行训练前会检查当前源码、数据、模型与实际部署前向验证报告是否一致。

### 后台托管全部阶段

已有完整原生/基线捕获时，优先使用双卡长程队列：

```bash
bash Native_NVFP4_HiF4_Linear_Puncture/experiments/kl_direction_reuse/scripts/launch_long_pipeline.sh \
  /absolute/path/to/smoke_root \
  /absolute/path/to/new/formal_root 0,1,2,3 0,1,2,3 \
  /absolute/path/to/smoke_root.control/smoke_report.json kl_direction_gpu0123_long
```

训练阶段每批最多四个配置各占一张卡，每个仍按 7200 秒预算结束；15 个配置分四批运行。
最后一批只有三个配置时，队列用一个轻量 CUDA 占位进程保留第四张卡。全部训练完成后，
GPU0/1 与 GPU2/3 各运行一个真实 TP2 评测进程；CPU 导出可与两组 TP2 评测并行。
这只改变资源编排，不合并层的可学习参数，也不改变 batch、损失或随机种子。

脚本先把认证的不可变输入从 `SOURCE_ROOT` 复用到全新的 `RUN_ROOT`，再在 tmux 中启动正式队列。
新控制目录为 `<RUN_ROOT>.long_<TMUX_SESSION>.control/`，`status.json` 列出所有活跃任务与GPU；
`plan_*.json` 保存各阶段具体队列，`jobs/<task>/attempt_*.log` 保存独立日志。
某任务报错后停止派发并停止本控制器拥有的其他子进程；训练进程按既有逻辑保留已提交更新。
不自动重试。显式 `long_pipeline.py --resume` 只接受相同源码、控制脚本和GPU分配；
跳过已完成任务前重新校验结果哈希，失败捕获/导出/核验准备目录必须先检查并归档。
全部结果完成后检查15个配置、评测覆盖和阶段成本账本，才写 `COMPLETE`。

正式队列不接受短测参数；`RUN_ROOT` 必须不存在，且 `SOURCE_ROOT` 必须带有当前代码生成的
`smoke_report.json` 和三层 verification。脚本会显式使用 `--reuse-verified` 复用这些已核验输入。
训练和评测使用同一组四张卡，但有明确阶段屏障：先完成全部训练，再把四张卡切成两个 TP2 评测组。
捕获入口在 CUDA 检查之前显式设置 `VLLM_WORKER_MULTIPROC_METHOD=spawn`，避免 fork 继承 CUDA 状态。
控制文件在 `<RUN_ROOT>.long_<TMUX_SESSION>.control/`：

- `run.sh`：本次已展开的完整启动命令。
- `launch.json`：冻结配置、GPU UUID、源码指纹和实验队列。
- `status.json`：总进程 PID、RUNNING/COMPLETE/FAILED/INTERRUPTED 和错误。
- `pipeline.log`：阶段开始/结束和总程序异常；`jobs/<task>/attempt_*.log`：每条命令的完整输出。

遇错立即停止，不自动重试、不跳过验证、不重新开始训练；卡住时保留进程供检查。
后续由操作者检查日志并明确继续。进程存活以 PID/命令行为准，不能仅凭 RUNNING 文件判断。
运行中改动参与计算的源码，会使下一条命令报错停止，防止一组实验混用不同实现。

如果验证代码变更，不能把旧 verification 当作当前代码证据；应保留旧目录并从认证输入重新生成三层 verification。

### 单卡训练与部署算子对齐

CUDA 训练使用 `ProductionTP2Student`：按部署的 2048-token prefill 切块，使用生产 RMSNorm、
CUDA 构造的 RoPE 表、分页 FlashAttention 2、fused top-k、routed Triton GEMM 和专家求和。
BF16 分数并列时也遵循生产 top-k 的选取顺序；两个 TP rank 的局部结果分别舍入后相加。
前向不替换成缓存参考值、不添加误差补偿。QDQ 反向用 STE，其他算子使用显式数学导数；
分页注意力反向由支持相同因果范围的 PyTorch Flash SDPA 计算。CPU TP2Student 仅保留为小型数学测试对象。
按用户关于 TP 舍入差异的澄清，长短完整序列的前向误差与逐值一致性均记录，
但不以逐值不一致阻断训练，也不事后调整数值容差去追求一致。
硬性检查保留：输出有限、完整 token/logit 覆盖、非零参数导出的模型身份，
新鲜方向与直接 KL 的梯度/更新一致性，以及 router 辅助梯度范围。
最终性能始终以实际 vLLM TP2 的 KL/NLL/PPL 为准。

短测由独立入口执行，正式入口不会接受其 600 秒预算或四条样本覆盖：

```bash
conda run --no-capture-output -n hif4 python -u \
  Native_NVFP4_HiF4_Linear_Puncture/experiments/kl_direction_reuse/scripts/smoke_pipeline.py \
  --source-root /absolute/path/to/certified_input_root \
  --smoke-root /absolute/path/to/new_smoke_root \
  --control /absolute/path/to/new_smoke_root.control \
  --train-gpus 0,1,2,3 --eval-gpus 0,1,2,3
```

`--resume` 仅恢复有明确耗时记录的 `INTERRUPTED`/`FAILED`，并恢复优化器、
批次游标、缓存快照和已用预算。SIGINT/SIGTERM 提交最后完整更新；
硬杀死后仍为 RUNNING 的记录不自动推测丢失的耗时。失败/不完整输出保留，
不会自动删除、改名或当作成功跳过。正式预算从运行时初始化后开始，包含
梯度刷新、模型搬运、读写、优化和训练检查点；公共准备、核验、评测成本另列。
跨过截止时间才完成的更新不提交；未获得任何更新则记录 `NO_UPDATE_WITHIN_BUDGET`，
汇总不会把它冒充训练成功。边界退出开销与实际超时单列。

## 实现与结果

CLI `prepare / baseline / capture / verify-prepare / verify-finish / train / export / report`
均可独立使用，参数见 `--help`。`suite` 使用子进程隔离 vLLM 和训练 CUDA 环境。

关键目录：`protocol.json`、`data/`、`baseline/`、`captures/native/`、
`captures/baseline/`、`verification/Lxx/`、`runs/Lxx_*/`。
候选导出只替换目标层，冻结 safetensors 与公共基线使用硬链接，禁止改写共享权重。
Teacher logits 按256个token存储，使用真实 LM head 的完整词表结果；
student 评测 logits 流式计算 KL/NLL，不保存整个词表输出。
正式训练使用 token 分块和 activation checkpoint，保留冻结后续层的输入梯度。

每30分钟保存参数，在训练后用实际TP2跑验证集；另保存2的幂次更新检查点。
每层选择五个配置共同达到的最大2的幂次更新数，作固定步数对照，不按指标选择。
测试集只评预算终点。输出 `report.json`、`comparison.csv`、`kl_vs_gpu_hours.png`；
bootstrap以完整样本为单位、配对重采样5000次，不能把token当独立样本。

**GPU前向核验必须实际执行，CPU单测通过不代表部署对齐。** 初始及非零更新
均记录目标层、router、最终隐藏状态和 logits 的数值差异，并报告本地与实际 TP2 的 KL/NLL。
不要求逐位相同，正常 TP 累加舍入差异不是硬性失败条件。
新鲜方向与直接KL使用固定FP32梯度比较阈值（atol=1e-7，rtol=1e-5）。
失败报告保留误差，不自动放宽阈值、不回退其他计算路径；训练入口拒绝继续。
这里的梯度均采用共同的STE定义，不声称是离散生产量化的数学精确导数。

`stages.jsonl` 记录 suite 每个子进程的耗时与卡数；汇总报告同时给出预算内训练耗时
和包含准备、核验、评测、启动退出的实际 GPU 小时，两者不相加重复计费。
单独调用子命令不生成 suite 总账，报告不会据此声称已统计全部成本。

## 最小回归验证

```bash
CUDA_VISIBLE_DEVICES='' conda run --no-capture-output -n hif4 python -m pytest \
  Native_NVFP4_HiF4_Linear_Puncture/experiments/kl_direction_reuse/tests -q
```

CPU测试覆盖损失公式、符号、不同序列长度的归一化、新鲜缓存方向与直接KL梯度、
辅助梯度范围、TP2运算结构、缓存覆盖/哈希、计时截止和统计配对。
