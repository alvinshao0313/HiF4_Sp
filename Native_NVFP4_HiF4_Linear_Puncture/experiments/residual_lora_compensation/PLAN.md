# 残差分支 LoRA 补偿实验大纲

状态（2026-09-24）：三组 48 层训练、模型导出、真实 TP2 推理验证及 ARC/MMLU-Pro/LCB 短测均已完成；五组正式下游评测已在物理 GPU 2、3 后台启动，运行中。局部重构结果与结论范围见[训练报告](../../results/residual_lora_compensation/20260922T012547Z/TRAINING_REPORT.md)，本轮接续记录见第 13 节，最新状态以[任务状态文件](../../results/residual_lora_compensation/20260922T012547Z/downstream/status.json)为准。

## 1. 实验目标

现有非等价 `group G64` 主分支可以优化投影变换，但 Transformer 的残差旁路仍然保留每层转换误差。本实验在注意力残差和 MoE 残差各加入一个低秩 LoRA，让残差分支与主分支联合优化完整 block 输出误差。

本实验独立于 `non_equivalent_reconstruction`，不修改旧实验的 CLI 或训练协议。旧实验的底层模型、校准数据、量化和评测工具可以通过 import 复用。

## 2. 已确认的 block 定义

令 `h` 为当前层输入，主分支仍使用当前 `group G64` 非等价变换：

```text
h1 = h + Attention(h) + Delta_attn(h)
h2 = h1 + MoE(h1) + Delta_moe(h1)
```

其中：

```text
Delta(x) = (alpha / r) * B @ (A @ x)
r = 4
alpha = 8
```

- `Delta_attn` 和 `Delta_moe` 参数独立；
- 两处 LoRA 都使用对应残差点的原始 hidden state，hidden size=2048；
- 不按 G64 切分，不跨层共享；
- `A` 使用 Kaiming uniform 初始化；
- `B` 全零初始化，使训练初始点与无 LoRA 的 group 主分支严格一致；
- 不使用 dropout，不增加 LoRA 范数惩罚；
- 两处 LoRA 每层 32,768 个参数，48 层约 1.57M 个参数。

## 3. 主分支定义

保持已有非等价重建定义：

- 每个 Linear 的每个输入 G64 使用独立矩阵；
- 复用 E4 DIAG 初始化；
- 普通投影矩阵初始化为 R64；
- router 矩阵初始化为 I64；
- 在线 R64 和现有 HiF4 数值路径保持不变；
- 主分支不改为线性共享矩阵，也不增加新的矩阵约束。

## 4. 训练目标和优化器

```text
L = block_delta_NMSE + router_loss
```

- 主重建损失使用完整 block 输出 `h2` 与原生 teacher 输出的差异；
- router loss 固定为 `top_mass`；
- router top-k=8；
- router temperature=1；
- router loss weight=1；
- 保留现有 router 辅助分支的 detach 边界：在 MoE norm 之前 detach，只更新 MoE norm 和 router 参数；
- 主 block 重建损失更新全部 group 主分支和两处 LoRA 参数；
- 不新增残差单独 MSE、LoRA 正则或其他辅助项；
- AdamW，所有参数统一 learning rate=1e-4；
- weight decay=0；
- cosine schedule；
- 不给 LoRA 单独设置学习率、warmup 或梯度裁剪。

## 5. 实验矩阵

最终报告包含五组：

1. `E4`：已有 E4 初始模型；
2. `group_top_mass`：已有无 LoRA group 对照；
3. `attention_lora_only`：只启用注意力残差 LoRA；
4. `moe_lora_only`：只启用 MoE 残差 LoRA；
5. `both_lora`：两处残差都启用，作为主实验。

已有对照 artifact：

```text
Native_NVFP4_HiF4_Linear_Puncture/results/non_equivalent_reconstruction/full48_20260915T084300Z/group_top_mass
```

该 artifact 的 manifest 已确认 48 层全部完成，配置与本协议一致。新实验只训练后三组，并在 manifest 中记录对照 artifact 的路径、配置和 SHA256；不覆盖旧结果。

## 6. 训练协议

- 48 层全部训练；
- progressive layerwise 输入：当前层使用前面已训练层产生的 student hidden；
- `s1k_original` 校准数据；
- 128 条训练样本、32 条验证样本；
- seed=42；
- batch size=4；
- 20 epochs/layer；
- 复用当前 `hif4` 环境和现有数据缓存；
- 训练、验证、resume 和 checkpoint 提交机制独立保存在本实验目录对应的 run root 中。

## 7. 实现范围

新建独立目录：

```text
Native_NVFP4_HiF4_Linear_Puncture/experiments/residual_lora_compensation/
```

计划包含：

- 配置和协议校验；
- LoRA 参数和两处 residual forward；
- layerwise progressive trainer；
- artifact、resume 和 manifest；
- materialize/export；
- 局部重建指标和完整模型汇总；
- ARC、MMLU-Pro、LCB 评测入口；
- 单测、smoke 和实际 vLLM TP2 检查。

materialize 时保留 LoRA sidecar，并扩展当前 vendored vLLM 的 Qwen3-MoE residual path，使正式推理实际执行两处 residual LoRA。无 LoRA 配置必须保持旧行为；不使用 semantic replay 或 CPU fallback 代替正式推理。

## 8. 验证顺序和阻断条件

1. 单测：LoRA 形状、零初始化恒等性、梯度、启用组合、保存恢复；
2. 关闭 LoRA 时，在代表层验证与已有 group 实现的数值一致性；
3. 真实模型短训：检查有限 loss、有限梯度和 LoRA 非零更新；
4. materialize 后重新加载，检查参数哈希；
5. 真实 vLLM TP2 执行 zero-LoRA identity/no-op 检查；
6. 真实 vLLM TP2 执行非零 LoRA candidate smoke，确认两处 residual adapter 实际生效；
7. 以上通过后，启动后三组正式 48 层训练；
8. 正式模型完成后统一导出和下游评测。

正常 TP 归约、BF16 累加和舍入差异只记录，不要求逐位一致。违反算法定义、LoRA 没有实际进入 vLLM 路径、identity/no-op 失败、checkpoint 不完整或协议不一致时，不启动正式下游评测。

## 9. 局部指标

每层保存：

- attention residual 点误差；
- MoE residual 点误差；
- 完整 block 输出 NMSE；
- router top-k 集合匹配率和重合率；
- teacher/student outside mass；
- 两处 LoRA 参数范数和输出能量；
- 相对无 LoRA group 对照的变化。

## 10. 下游评测

五组最终都评测：

- ARC-C；
- ARC-E；
- MMLU-Pro，固定 300 samples；
- LiveCodeBench `lcb:codegeneration_v6|0` 全量。

LCB 固定使用当前仓库的正式 vLLM/lighteval 路径：

- temperature=0.6；
- top-p=0.95；
- top-k=20；
- min-p=0；
- `max_new_tokens=38912`；
- thinking 开启；
- 真实 vLLM TP2；
- 官方 checker；
- 五组使用同一份 `chat_template.jinja` 和同一套 prompt 构造。

历史上 prompt 不一致的 LCB 分数不作为本实验结果或因果证据。

## 11. 结果交付

最终独立结果目录保存：

- 协议和源码指纹；
- 对照 artifact provenance；
- 每组 manifest、checkpoint、materialized model 和日志；
- 每层局部指标；
- ARC、MMLU-Pro、LCB 原始和汇总结果；
- 五组对比报告；
- 运行状态、失败原因和未完成范围。

本文件只记录已确认协议。任何改变 LoRA 插入位置、rank、alpha、损失、router 定义、训练范围或下游评测口径的后续修改，都需要单独记录为协议变更。

## 12. 训练启动记录（2026-09-22，历史状态）

- 2 层真实 CUDA smoke 已完成三种配置（attention、moe、both），结果保存在 `validation/smoke_validation_20260922.json`；三种配置都完成了两轮更新和第 0→1 层传递，启用分支产生非零 LoRA 更新。
- 正式 48 层训练已启动，运行目录为 `Native_NVFP4_HiF4_Linear_Puncture/results/residual_lora_compensation/20260922T012547Z/`。
- attention 使用物理 GPU2，moe 使用物理 GPU3；两者结束后 both 按启动脚本在 GPU2 运行。launcher PID 为 3217821，临时 launcher 日志为 `/tmp/residual_lora_compensation_20260922T012547Z.launcher.log`。
- 启动确认时 attention/moe 均已进入第 0 层第 0 轮的真实 batch 计算；正式结果尚未完成，materialize、TP2 vLLM 检查和下游评测需在训练完成后继续。

## 13. 下游接续记录（2026-09-24）

继续使用 RUN `20260922T012547Z`；正式评测使用物理 GPU 2、3，`hif4` 环境。训练协议保持不变，复用 E4 和 `group_top_mass` 权重，五组分数在同一套评测协议下重新计算。

已完成三组正式导出、LoRA 参数回读哈希核验，以及三组真实 vLLM TP2 的 zero/disabled 恒等性、非零适配器执行和重复推理检查。单测 6 项通过，包括 router 辅助损失的梯度边界与 checkpoint 恢复。五组实际 reasoning prompt 一致：MMLU-Pro 300 题、LCB 175 题，模板默认行为等于显式 `enable_thinking=True`。证据保存在 RUN 的 `downstream_validation/`，短测结果与正式结果分开保存。

ARC 各 2 题、MMLU-Pro 1 题和 LCB 1 题的真实评测均完成评分、引擎退出与结果写入；reasoning 保留正式生成上限和 TP2 配置。三组重复推理观察到的每卡 allocated memory 保持稳定。通过范围、源码 SHA256、参数哈希及证据路径见[验证汇总](../../results/residual_lora_compensation/20260922T012547Z/downstream_validation/summary.json)。这些短测证明执行链路可用，不代表下游效果或长轨迹等价。

MoE 双请求诊断曾观察到关闭 LoRA 的重复推理自身存在 logprob 差异（共同 token 最大 0.25，16 个生成 token 均一致）；其 zero/disabled 比较和已训练参数的重复比较通过。Attention/both 的机制探针逐请求独立提交，恒等性检查通过。该观察保留在验证汇总中，不归因于 LoRA，也未据此改变正式评测的 batch 或数值路径。

LCB 当前任务配置每题生成 1 个回答；指标键名为 `codegen_pass@1:16`，其中 `:16` 不代表本次实际生成了 16 个回答。保持现有任务定义、采样参数和官方 checker，不自行更改评价口径。

正式入口（调用前激活 `hif4`）：

```bash
bash Native_NVFP4_HiF4_Linear_Puncture/experiments/residual_lora_compensation/scripts/run_downstream.sh \
  Native_NVFP4_HiF4_Linear_Puncture/results/residual_lora_compensation/20260922T012547Z
```

`downstream.py` 依次运行五组 ARC、五组 MMLU-Pro、五组 LCB，共 15 个任务；任何任务失败即记录原因并退出，不自动重跑。每个任务验证完整样本数和有效分数，reasoning 额外验证样本 ID 唯一性以及实际输入与审计 prompt 一致。正式评测使用独立的模型路径视图隔离预测缓存；权重文件仍指向原 artifact。

正式输出位于 RUN 的 `downstream/`：`status.json` 记录实际命令、PID、当前任务和完成情况，各模型目录保存日志、指标及逐题 details。全部完成后自动生成 `comparison.json` 和 `COMPARISON.md`。这是新输出目录，入口拒绝覆盖已存在目录；已启动后不要重复执行上述命令。

实际启动时间：`2026-09-24T08:47:13Z`；使用 `setsid` 脱离交互会话。launcher PID `3985430`，Python 编排进程 PID `3985436`；首个任务为 `moe / ARC`。launcher 输出保存为 RUN 的 `downstream_launcher.log`，任务日志见[首个 ARC 日志](../../results/residual_lora_compensation/20260922T012547Z/downstream/moe/arc.log)。状态为“已启动，运行中”，尚无完整下游结论。

交付前已确认正式任务完成初始化并持续计算，首个 ARC 任务已处理 5,624 / 14,188 个评分请求（约 40%，不是题目数）。随后停止主动监控，后台流水线继续；下次接续先读取状态和日志，不重复启动。
