# KL 方向指引端到端下游验证计划

状态：计划已确认，代码尚未实现，正式实验尚未启动。

本实验与旧的 `experiments/kl_direction_reuse` 及其结果目录独立成立。旧结果只读复用，不覆盖、不删除。

## 1. 实验目标

验证最终 KL 的局部方向是否能在固定的非等价补偿参数空间内，帮助逐层 MSE 更新降低最终转换损失，并观察这种改善是否传递到真实下游任务。

正式比较三个完整 48 层模型：

1. `full_mse`：标准层输出 MSE + router 辅助损失。
2. `full_mse_direction`：标准 MSE + `lambda * KL 局部线性方向` + router 辅助损失。
3. `full_direct_kl`：完整最终 KL + router 辅助损失；只优化当前层参数，作为直接端到端 KL 参考。

三组模型都按 layer `0 -> 47` 顺序逐层累积更新。

## 2. 参数和目标定义

- 可学习参数严格复用 `non_equivalent_reconstruction` 的 `LayerParameters(group)`：输入 G64 group 矩阵参数，以及当前层的两个 RMSNorm 参数。
- base quantized weights、激活侧 R64 和其它非指定参数全部冻结。
- 当前层训练完成后，重新生成更新学生前缀下的后续层输入；后续层不能继续使用旧学生前缀的激活。
- native teacher 的最终 logits、逐层 output 和 router target 固定为原始 native targets。
- `full_mse` 使用标准 MSE，不使用 NMSE。
- `full_direct_kl` 使用固定 native 最终 logits 的完整词表、全部有效 token KL，不使用 token 子采样或当前层 KL。
- 三组均使用现有 router loss，router 权重保持 `1.0`。

方向项定义为：

```text
K = mean_valid_token(final_KL(native_logits, student_logits))
g = dK / d(layer_output)       # 在刷新点计算并 detach
L_dir = sum(g * (layer_output - anchor_output))
L_mse_direction = L_mse + lambda * L_dir + L_router
```

方向项与 MSE 使用同一个当前层输出张量。最终 KL 按有效 token 平均，方向内积不再二次除以 token 数。`lambda` 默认固定为 `1.0`，代码保留 `--direction-weight` 可调入口；不做梯度范数归一化。

## 3. 数据和训练协议

- 数据集：S1K 原始完整序列，不截断有效序列。
- 固定 `seed=42`，按 token hash 去重并保持 split 互斥。
- 数据划分：`128 train + 32 val + 32 test`。
- 有效 batch size：`4`；三组使用完全相同的 batch membership 和遍历顺序。
- 学习率：`1e-4`。
- 每层训练预算：`1800` 秒，方向刷新、参数更新、保存和必要的流式 teacher 前向均计入预算。
- 每层结束保存该层账本，验证曲线只解释训练过程，不触发回滚或 checkpoint 选择。
- 最终性能使用每种方法在第 48 层结束时的最终参数，不使用中间验证集最优点。

### 方向刷新

- `full_mse_direction` 在每个 epoch 开始刷新一次方向，epoch 内复用，epoch 结束释放。
- 刷新按固定 batch 流式计算，覆盖 128 条训练样本的全部有效 token。
- `anchor` 和 `gradient` 只保留在 CPU 内存，缓存 dtype 为 BF16，仅保留当前 epoch。
- manifest 记录方向覆盖、dtype、finite 检查、刷新耗时和方向 hash，不保存历史方向张量文件。
- BF16 缓存属于已确认的数值近似；不改变完整序列和完整词表 KL 的语义。

### Teacher 目标和磁盘边界

- native layer target 和 native final logits 按 batch 流式重算，用完释放。
- 不落盘完整 native final logits；词表维度过大，落盘会产生不可接受的存储开销。
- 不保存逐样本方向 `.pt`、历史 epoch 方向、训练中间 logits 或完整下游 logits。

### Checkpoint

- 每个方法只保留一个滚动 `resume.pt`，每完成一层覆盖一次。
- 每个方法保留最终 `final.pt`。
- 每层只保留轻量 JSON 账本、loss、gradient norm、时间和 hash；不保留每层完整模型副本。
- 中断恢复最多重做当前层，不自动重试或静默降级。

## 4. 代码和目录

新增独立代码目录：

```text
experiments/kl_direction_reuse_downstream/
  PLAN.md
  config.py
  data.py
  objectives.py
  direction_cache.py
  train.py
  downstream_eval.py
  smoke_pipeline.py
  long_pipeline.py
  report.py
  tests/
```

结果根目录：

```text
results/kl_direction_reuse_downstream/<run_id>/
  protocol.json
  source_fingerprint.json
  data/
  runs/full_mse/
  runs/full_mse_direction/
  runs/full_direct_kl/
  evaluation/
  report.json
  report.md
```

关键 CLI 默认值：

```text
--direction-weight 1.0
--direction-cache-dtype bf16
--direction-refresh epoch
--layer-budget-seconds 1800
--seed 42
--lr 1e-4
--train-samples 128
--val-samples 32
--test-samples 32
```

## 5. 短测门禁

`smoke_pipeline.py` 与正式 `long_pipeline.py` 分离，短测参数不污染正式默认值。

短测使用 L24 的三个方法、4 条完整训练序列和 600 秒预算，必须完成至少两次参数更新，并覆盖真实生产前向、反向、梯度累积、router loss、方向缓存和 checkpoint reload。

短测必须通过：

- 三种目标的 loss、gradient、router loss 和方向 cache 全部有限；
- cached direction 完整覆盖 4 条训练样本；
- checkpoint 保存、重新加载后参数 hash 一致；
- 可学习参数发生非零变化；
- 最短和最长测试样本完成真实 vLLM TP2 candidate capture；
- logits 覆盖完整，KL/NLL 有效且有限；
- 重复迭代无显存或 CPU 内存持续增长、无阶段阻塞。

任一条件失败，不启动正式 3×48 层实验。

## 6. 下游评测

三组训练全部完成后，统一使用真实 vLLM TP2 评测；训练和下游评测不重叠。TP2 归约和累加顺序造成的正常数值差异只记录，不要求逐位一致。

### 任务和样本

- ARC-E、ARC-C：官方测试集全量，使用现有 `lm_eval + vLLM TP2` 入口。
- MMLU-Pro：固定 `300` 条，使用仓库 `main.py + lighteval`。
- GSM8K：官方测试集全量，沿用 MMLU-Pro reasoning 生成协议。
- LiveCodeBench：仅在 prompt、chat template、数据版本、任务版本和 TP2 配置指纹一致时运行完整锁定 split；否则明确标记 skipped，不阻断前三项。

### 生成协议

MMLU-Pro 和 GSM8K 沿用：

```text
thinking=true
temperature=0.6
top_p=0.95
top_k=20
max_new_tokens=32768
```

同时在 32 条独立 test split 上评测最终 KL、文本 NLL 和 PPL。

### Native baseline

原生 NVFP4 baseline 优先复用既有结果，但必须核对模型、tokenizer/chat template、任务版本、数据 split、生成参数、TP2 配置和评测入口指纹。匹配的指标才进入主比较表；缺少 GSM8K 或指纹不匹配的 baseline 必须按当前协议补测，不能直接混用旧数字。

### 评测输出和失败处理

- 保存逐题轻量结果、汇总均值、标准误、协议 hash、模型 hash 和日志。
- 不保存完整 logits。
- 核心任务运行失败或数据完整性失败时，停止该模型评测；不自动重试。
- 仅 LiveCodeBench 协议不一致允许明确跳过。
- 统计只报告均值和标准误，不做显著性检验、不使用 bootstrap。

## 7. 有效性判定和验收

方向方案相对 `full_mse` 的描述性有效条件：

- 最终 KL 更低；
- 核心下游任务没有明显均值回退；
- 至少一项核心下游任务均值改善。

不设固定百分点阈值，结合均值差和标准误描述改善、回退或无明显变化；不把描述性差异表述为统计显著结论。`full_direct_kl` 用于显示方向近似与直接最终 KL 优化之间的差距。

正式完成要求：

- 三个方法均完成 48 层，每层有有效更新；
- 三个最终 checkpoint、训练账本和源码指纹完整；
- 32 条 test split 的真实 TP2 KL/NLL/PPL 完整；
- 核心下游任务结果完整，LiveCodeBench 状态明确；
- native baseline 只使用通过指纹核验的结果；
- 报告包含训练成本、刷新耗时、方向覆盖、缓存 dtype、下游均值/标准误和结果 hash；
- 结果目录没有大规模历史方向张量或完整 logits 缓存。
