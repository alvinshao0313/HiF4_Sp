# Qwen3-235B-A22B NVFP4 → HiF4 算法迁移验证

状态（2026-10-08）：独立四卡 E4 与 Group 均已完成全部 94 层，完整导出和 CPU 评测准备已通过；原生/HiF4 全量 prompt 与原固定版本一致。下游生成与评分尚未启动，TP8 所需的 GPU 1–5 当前被其他用户占用，尚无 235B 下游精度结论。原三卡 E4 已按用户要求停止，完整层 0–80（81/94）保留。

## 本次执行状态

2026-10-08：Group 于 10 月 7 日 13:55 UTC 正常退出，实际训练 61.04 小时；94 层、60,160 次更新、1,880 个 epoch、完整样本覆盖、选优参数、最终 hidden、四副本同步与源码哈希复核通过。完整 QDQ/BF16 导出于 02:02 UTC 正常结束，95 分片、36,945 个 tensor 的 index/shape/dtype/ABI/来源绑定通过，实际权重文件合计 470,191,875,256 bytes。CPU prepare 于 02:08 UTC 完成：16 个隔离模型视图，12,032 MMLU-Pro、175 LCB、1,319 GSM8K 的实际 prompt/token 与原生及原预检完全一致。首次离线 prepare 误选较新 MMLU 缓存，已隔离保留，并恢复原指定 revision `3373e0b32277875b8db2aa555a333b78a08477ea`；后续启动必须保持 `HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0`，沿用原数据加载路径。现在等待完整 8 卡资源，之后按原 TP8 协议短测和五项全量评测，不降低题数或并行配置。见[训练完成与导出报告](../../results/qwen3_235b_transfer/parallel_20261003T144300Z/REPORT.md)。

2026-10-05：四卡 E4 于 10 月 4 日 16:13 UTC 正常结束；新增 62 层和前缀重放共 25.44 小时，完整 94 层保存重载、原前缀/数据哈希和每层同步检查通过。Group 使用同一完整 artifact 的参考/四卡短测、冻结状态、保存重载、单层导出以及真实层 0→1 换层/checkpoint 检查均通过；完整 Group 于 00:52 UTC 在 GPU 4–7 启动，PID `2570264`，保持 94 层、20 epochs、128/32 完整样本、batch=4。见[完成与接续报告](../../results/qwen3_235b_transfer/parallel_20261003T144300Z/REPORT.md)。此状态替代下文 10 月 3 日的“完整 E4 与 Group 待启动”。后续仍为 Group 全层训练 → 完整导出 → 五项全量配对评测，不以 E4 局部损失替代验收。

2026-10-03 追加授权：用户要求用剩余卡运行加速版。新候选保留原正式配置，复制层 0–31 的最终 adopted 前缀，先重放重建 hidden，再在 GPU 4–7 训练层 32–93；原运行在 GPU 0–2 保留。新入口已通过完整最长短测、恢复及连续换层、四副本 Adam/参数同步和 artifact 精确保存重载检查。见[启动与验证报告](../../results/qwen3_235b_transfer/parallel_20261003T144300Z/REPORT.md)。这替代下面“仅完成两步、尚未实现续训入口”的执行状态；完整收敛、Group 并行及五项全量验收仍待完成。

2026-10-03 加速候选补充：已在空闲 GPU 4–7 完成真实第 0 层、原 batch=4 的短/长完整样本各两次更新，四卡样本并行相比现行三卡暂存的单步加速为 2.93×/4.74×。全局归一化和副本同步检查通过，但 batch 计算形状变化带来的梯度差异尚无收敛比较证据，未切换正式训练；Group 并行和层边界续训尚未实现。见[短测报告及原始证据](../../results/qwen3_235b_transfer/acceleration_20261003T121157Z/REPORT.md)。短测已退出并释放 GPU 4–7，原正式 PID `3473948` 在 GPU 0–2 继续运行。按用户要求，短测后挂起后续自动推进与监控。

当前运行目录为 [transfer_20261001T124000Z](../../results/qwen3_235b_transfer/transfer_20261001T124000Z)：真实最长 batch 的 E4 与 Group 两阶段 GPU 暂存短测均完成。完整 E4 已于 `2026-10-01T12:45:47Z` 后台启动，PID `3473948`，GPU 0 计算、GPU 1/2 保存反向张量；[启动记录](../../results/qwen3_235b_transfer/transfer_20261001T124000Z/launches/formal_e4.json)保存命令、正式配置、103 个依赖源码文件的快照和日志路径。12:49 UTC 已核实完整源校验通过、进程存活且完成第 1 次真实 optimizer update（lr=0.005），见[交付时状态](../../results/qwen3_235b_transfer/transfer_20261001T124000Z/handoff.json)。正常推进后停止主动监控，保持后台任务运行；后续接续先核对日志、artifact 与实际状态。

### 模型、数据和源码

用户已在默认 Hugging Face `hub` 缓存完成固定 revision，27 个权重分片共 134,118,973,816 bytes；全部通过官方 SHA256 和实际 tensor 覆盖校验，13 个非权重文件与准备记录一致。见[完整源校验](../../results/qwen3_235b_transfer/transfer_20261001T123100Z/source_validation.json)、[源选择](../../results/qwen3_235b_transfer/transfer_20261001T123100Z/source_selection.json)及[校准准备](../../results/qwen3_235b_transfer/transfer_20261001T123100Z/preparation.json)。此前所有下载均已停止，无需重新下载；原始缓存和日志保留。

独立 S1K fingerprint 为 `37170af58cd97033`：128 条训练共 1,253,457 tokens，最长 18,945；32 条验证共 312,662 tokens，最长 22,632。逐条重新编码确认完整原文和全 token mask。最长 batch 短测取每个拆分最长 4 条原文，batch=4、2 epochs、真实第 0 层；仅减少样本、轮数和层数，模型架构仍是 94 层。

每次运行的 `launches/` 保存命令、PID、环境和日志路径，`source_launch/` 保存实际依赖源码与 SHA256；`configs/` 保存正式和短测配置。初始源码快照保存在准备目录的 `source_before/`，不覆盖既有任务改动。

### 已完成的迁移与执行验证

- 94 层 metadata、训练范围、初始化覆盖和 Group 导出规格已适配；tokenizer config 中嵌入的模板被原样导出。9 项迁移检查通过，原 30B 的 8 份配对结果通过只读复核。
- 真实短测发现 VO 展开固定 Q-head=32；235B 为 64，KV scale 应从 512 展开为 8192。修复使训练、DIAG 融合、Group 初始化和导出共同按模型规格展开。真实 30B/235B 几何、GQA 等价性、梯度及保存读取等 16 项 CPU 检查通过，见[维度修复证据](../../results/qwen3_235b_transfer/transfer_20261001T050000Z/fix_vo_expansion/validation.json)。DIAG 参数形状、checkpoint 字段和算法定义未改变。
- 原始最长 batch 在训练前向实际耗尽单卡显存，PyTorch allocated 77.79 GiB，见[失败状态](../../results/qwen3_235b_transfer/transfer_20261001T122200Z/smoke/e4/status.json)。没有缩短文本、减小 batch 或修改优化目标。
- `save_on_cpu(pin_memory=False)` 在真实首层、正式数据中最短 4 条完整文本上与原执行比较：loss、有效 token、步数相同；E4 的 QKV 差异低于 resident 重复基线，其余 DIAG 精确相同；Group 梯度最大相对 L2 为 `4.29e-6`、更新为 `2.79e-5`。`pin_memory=True` 会改变部分保存张量布局，观察到更大反向差异，未采用。见[执行比较](../../results/qwen3_235b_transfer/transfer_20261001T122200Z/memory_execution_fix/execution_review.json)。
- CPU 暂存的真实最长 E4 短测完成 2 次更新、验证、rollback、保存和重载；allocated 峰值约 47.5 GiB，adopted validation NMSE 从 `5.7686e-4` 降至 `5.3940e-4`。[状态](../../results/qwen3_235b_transfer/transfer_20261001T123100Z/smoke/e4/status.json)和[实际命令](../../results/qwen3_235b_transfer/transfer_20261001T123100Z/launches/smoke_e4.json)保留。
- CPU 暂存的 Group 最长短测使用上述实际 E4 artifact，完成 2 次更新、验证、选已训练 epoch、精确保存重载、全部 8 条完整文本传播和真实单层 QDQ safetensors 导出；冻结参数保持不变，391 个学习参数发生更新。allocated 峰值约 46.3 GiB，导出 4,975,558,664 bytes。[状态](../../results/qwen3_235b_transfer/transfer_20261001T123100Z/smoke/group_top_mass/status.json)及[实际命令](../../results/qwen3_235b_transfer/transfer_20261001T123100Z/launches/smoke_group.json)记录原始指标。两轮短测不用于认定收敛或下游提升。
- CPU 暂存有明显传输开销。另一个有界执行候选以一张 GPU 计算、两张 GPU 按固定轮转保存反向张量：真实短完整样本上 E4 单步 3.90 s、Group 3.73 s，对应 CPU 为 15.89/9.02 s；损失和步数相同，梯度/更新差异处于 resident 重复基线范围。见[GPU 暂存比较](../../results/qwen3_235b_transfer/transfer_20261001T122200Z/memory_execution_fix/gpu_saved_tensor_trial/review.json)。这不代替最长 batch 容量和连续阶段验证。
- 最终三卡执行已通过最长 batch 两阶段验证：[汇总](../../results/qwen3_235b_transfer/transfer_20261001T124000Z/smoke_summary.json)。E4 完成 2 次真实更新及后续阶段；Group 使用它实际导出的 DIAG artifact，完成更新、验证、精确重载、完整传播及 4,975,558,664-byte 单层导出。计算与两张暂存卡的峰值均在各自 80GB 容量内；第二次 Group 训练 batch 实测 11.61 s，相同长样本的 CPU 暂存为 36.90 s。这是本机首层短测，不作为全部层的 ETA。正式训练使用相同三卡入口，从协议初始状态重新训练，不使用短测更新。

### 下游与下一步

TP8 参数已贯通 reasoning 和 ARC；全量 prompt 预检覆盖 MMLU-Pro 12,032、LCB 175、GSM8K 1,319 条，源与导出 tokenizer 的 prompt/token 哈希相同，见[预检](../../results/qwen3_235b_transfer/transfer_20261001T123100Z/preflight_prompt_audit.json)。实际 vLLM 配置识别为 `modelopt_fp4`、BF16、94 层，TP8 的 KV-head 复制满足模型规格；这些静态证据不能替代真实加载、生成、评分和退出短测。最终 Group 模型准备时会重做实际模型目录的 prompt 核对。

当前正式 E4 覆盖层 0–93，128/32 条完整 S1K、20 epochs/层、batch=4、lr=0.005，完整配置与短测分开保存。完成后执行 Group-top-mass 94 层训练、完整 QDQ 导出和 TP8 配对评测。GPU 暂存只调整反向张量存放位置；计算、参数、loss、batch 和 optimizer 更新规则保持固定。正式入口为 `train_e4 --mode formal` → `train_group` → 原 `non_equivalent_reconstruction.materialize` → `evaluate --phase prepare/smoke/formal/report`，各阶段单独启动，不创建重试或监控守护程序。

本次已激活 `hif4` 后的正式命令为：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u -m Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_235b_transfer.train_e4 --run_root Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_transfer/transfer_20261001T124000Z --mode formal
```

新增[训练执行经验](../../../docs/experience/lessons/model_transfer_execution.md)记录 GQA 几何、保存张量布局和执行等价性的适用边界；短测通过不等于 235B 下游验收通过。

历史诊断分别保留：准备及维度错误在 `transfer_20261001T050000Z`，未暂存显存失败和执行对比在 `transfer_20261001T122200Z`，pinned CPU 最长短测在 `transfer_20261001T122700Z`。两项旧 `test_moe_materialize.py` fixture 因缺少所有 tokenizer assets 失败，用修改前函数同样复现；[旧诊断](../../results/qwen3_235b_transfer/transfer_20261001T050000Z/legacy_fixture_diagnostic.json)保留，不列为本次通过项。

## 目标和固定算法

将 [30B 已达标方案](../downstream_accuracy_recovery/PLAN.md)迁移至 `nvidia/Qwen3-235B-A22B-NVFP4`。验收仍为 MMLU-Pro 全部 12,032 题、LCB v6 175 题、GSM8K 全部 1,319 题、ARC-E/C 全部 2,376/1,172 题；以 235B 原生 NVFP4 的同协议实测为基准，五项 `max(0, native_pct - hif4_pct)` 等权平均不超过 0.5 个百分点。30B 分数、模型卡分数和局部损失不能代替这次验收。

保持两阶段逐层算法，重新在 235B 上训练模型专属参数：

1. E4 初始化：fusable DIAG + 固定 R64，block-delta NMSE，S1K 完整文本 128 train / 32 validation、seed=42，20 epochs/层，batch=4，AdamW、lr=0.005、weight decay=0、cosine，log2 DIAG clamp [-4,4]。沿用原 E4 的 progressive student 输入和 adopted DIAG / rollback 定义。
2. Group-top-mass：冻结融合 DIAG 后的基础权重，训练每个输入 G64 独立 64×64 非等价矩阵及两个 RMSNorm；普通矩阵初始化 R64，router 为 I64，在线 R64 固定。block-delta NMSE + top_mass router KL，k=8、T=1、λ=1；辅助梯度在 MoE norm 之前 detach。相同 S1K 拆分、20 epochs/层、batch=4、lr=1e-4、AdamW、weight decay=0、cosine，按每层验证目标选已训练 epoch，不增加 LoRA 或其他参数类型。

模型维度变化不允许复用 30B 的 DIAG 张量、已训练矩阵或 hidden cache。235B 使用独立校准缓存，核对 tokenizer/template、实际样本 ID 和完整文本语义；不因模型规模增大截短 S1K。

## 来源和资源

- [NVIDIA 源模型](https://huggingface.co/nvidia/Qwen3-235B-A22B-NVFP4)，固定 revision `21cfa2c9e152032eb60647ee7b46a2bbcd8d76d2`，HF API 记录总文件大小 134,148,800,581 bytes。
- 实际 config：94 层，hidden=4096，attention heads=64，KV heads=4，head_dim=128，128 experts、top-8、expert intermediate=1536，context=40960。架构仍是 Qwen3MoeForCausalLM。
- 2026-10-01 检查：8×A800 80GB 均空闲，可用主机内存约 1.3 TiB，磁盘可用约 1.5 TB。HiF4 BF16 存储的完整导出权重预计约 470 GB，属于估算；实际大小与峰值显存须实测。
- 推理按 TP8 设计，原生与 HiF4 使用同一并行和生成配置，依次评测。其余口径沿用 30B 验收：BF16 KV、eager、thinking、seed=1234、temperature=0.6、top-p=0.95、top-k=20、min-p=0；ARC raw accuracy / seed=42。正式资源容量在真实短测中验证，不能通过减少题数或生成上限绕过资源问题。
- 原生走现有 NVFP4 emulation；HiF4 仍为 QDQ 后 BF16 权重存储加 runtime HiF4 激活量化，不宣称真实 4-bit 存储或加速。

## 实施顺序和验证

根据真实模型元数据，将现有加载、训练、初始化覆盖与导出中的 48 层/30B 维度常量改为经过校验的模型规格；保留 30B 的实际语义及已有 artifact。下载期间可针对已验证完整的首层源分片执行原定短测；正式训练与评测必须等待完整模型。独立实验目录负责 235B 的配置、运行和结果索引，不复制底层训练算法。

正式启动前，使用真实 235B 层和完整正式 batch/长度条件验证两个训练阶段的前向、反向、更新、参数边界、保存读取及阶段衔接；覆盖实际长样本峰值。若内存不足，先定位并采用有依据、保持目标函数和一次优化更新语义的执行方式，不降低已确认算法条件。正式推理前执行真实 TP8 的生成、评分、保存与退出短测。

短测与正式产物分开。正式训练只在必要验证通过后启动，记录实际命令、配置、源码哈希、PID、日志和完成/失败状态；失败明确停止受影响阶段，不创建守护、自动重试或持续占卡机制。后台正常推进后交付并停止主动监控。

## 当前发现

迁移前 `moe_model_spec.py` 的 checkpoint 校验只接受 30B，DIAG config、非等价训练的层循环/初始化覆盖/导出 runtime spec 存在 48 层常量；本次已按模型规格适配。数学变换依赖的 G64、head_dim=128、top-8 条件在 235B 仍成立，但这不构成运行和下游正确性的证明。

已参考[评测证据经验](../../../docs/experience/lessons/evaluation_evidence.md)：隔离预测缓存，核对真实 prompt 和完整覆盖，以原生/HiF4 同协议配对分数验收，不从单层损失推断整模型精度，也不把 router KL 的贡献从完整配置中单独归因。
