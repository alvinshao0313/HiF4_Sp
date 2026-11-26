# 四种格式转换基线

此入口读取发布的 NVFP4 权重，分别执行 BRQ 随机块旋转、GPTQ、AWQ 或 SmoothQuant，然后按 HiF4（每 64 个值一组）或 MXFP4（每 32 个值一组）模拟权重和激活。转换覆盖文本推理使用的全部 NVFP4 投影。本阶段先运行 Nemotron Nano 9B 和 QUASAR Qwen3.5-4B；两个 30B MoE 单项实验暂缓。QUASAR 使用 compressed-tensors 存储，适配解码时需将全局尺度转换为内部的反量化尺度；其运行状态以验证记录为准。

| 方法 | 权重处理 | 推理输入 |
| --- | --- | --- |
| 在线随机块旋转（`brq`，第四章对照） | 组内使用带随机符号的归一化 Walsh–Hadamard 矩阵。当前实现按种子生成一块矩阵，在各组重复使用，权重旋转后量化。 | 在线执行相同的旋转后量化。此基线只测试随机块旋转。 |
| 完整融合 BRQ（`brq_fused`，第三章基线） | norm 增益并入输入权重，embedding、残差输出、路由及输出头配套旋转；共享输入侧随机符号 DH 块。 | 残差流已在旋转空间，直接做目标激活量化。 |
| GPTQ | 从校准输入估计二阶统计量，按列补偿权重误差。按目标分组选择尺度，处理块宽 128，阻尼系数 0.01。 | 直接按目标格式量化。 |
| AWQ 通道尺度搜索 | 以平均绝对激活为统计量，对 20 个指数候选计算仅量化权重的输出误差。共用输入的融合投影共同选尺度。实验只启用通道缩放。 | 先将激活各通道除以选定尺度，再按目标格式量化。 |
| SmoothQuant | 取激活和权重的逐通道峰值，当前采用 α = 0.5。将尺度乘入权重后量化。 | 先除以对应通道尺度，再按目标格式量化。 |

校准默认采用固定的 128 条完整 S1K 训练文本。四个模型共享样本编号，各用自己的分词器。MoE 的校准输入覆盖所有专家，推理仍按模型原有的 top-k 路由选择专家。三个校准入口逐层保存权重，AWQ 和 SmoothQuant 同时保存输入的通道尺度。导出直接读取校准产物中的权重值，检查模型版本、方法、目标格式和完整投影清单。

## 运行

以下命令在仓库根目录执行。示例选用 Qwen3-30B，历史模型验证范围见[验证记录](../../../docs/experience/records/fp4_paper_baselines_20261119.md)。为便于比较，每个模型、方法和目标格式使用独立目录。

```bash
source /home/shaoyuantian/anaconda3/etc/profile.d/conda.sh
conda activate hif4
P=fp4_convert.experiments.paper_baselines
SOURCE="$HOME/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3"
RUN="$PWD/Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/qwen3-30b"
```

融合 BRQ 直接导出（旧 `--method brq` 仍指在线版本）。下面的 17 是旋转种子，与下游生成种子独立。

```bash
python -m "$P.export" "$SOURCE" "$RUN/brq-fused-mxfp4-seed17" --method brq_fused --target mxfp4 --seed 17
```

GPTQ、AWQ 和 SmoothQuant 先校准再导出。分别设置 `METHOD=gptq`、`METHOD=awq` 或 `METHOD=smoothquant`，`TARGET` 取 `hif4` 或 `mxfp4`。以下校准命令使用默认的完整数据，仅在正式实验获准且有相应资源时执行。

```bash
METHOD=awq
TARGET=hif4
python -m "$P.calibrate" "$SOURCE" "$RUN/$METHOD-$TARGET-fit" --method "$METHOD" --target "$TARGET" --device cuda
python -m "$P.export" "$SOURCE" "$RUN/$METHOD-$TARGET" --method "$METHOD" --target "$TARGET" --calibration "$RUN/$METHOD-$TARGET-fit"
```

工程短测在校准命令中加 `--smoke-text 'One plus one equals two.'`，使用新的输出目录。可加 `--smoke-layers 1` 验证首层。短文本会在 `recipe.json` 中标为 `qualification`。只覆盖部分投影的目录无法导出为全量模型。

双卡推理短测通过模块入口启动 vLLM，在知道有两张空闲卡时设置 `CUDA_VISIBLE_DEVICES`。以下命令使用前述 BRQ 导出，两次独立请求各生成六个 token，记录所选 token 的对数概率。

```bash
CUDA_VISIBLE_DEVICES=4,5 python -m "$P.smoke" "$RUN/brq-fused-mxfp4-seed17" "$RUN/brq-fused-mxfp4-seed17-tp2.json"
```

正式评测复用仓库入口的参数。生成任务通过 `python -m "$P.entry"` 传入原 `main.py` 参数，选择题通过 `python -m "$P.entry" lm_eval` 传入 `lm_eval` 参数。模型目录选导出目录。目标格式的激活量化已由模型执行，生成入口使用 `--fake_act_quant none`。统一题目、三轮生成、MMLU-Pro 全量与评分口径见[论文实验规划](../../../Native_NVFP4_HiF4_Linear_Puncture/plans/2026-11-19-fp4-paper-experiments-cn.md)。

导出的 safetensors 以 BF16 保存目标格式解码后的操作数，推理时对激活执行目标格式的量化与反量化。该路径用于数值精度研究，磁盘占用和运行时间按实际 BF16 仿真实现解释。GPU 双卡执行与各阶段已取得的证据见[验证记录](../../../docs/experience/records/fp4_paper_baselines_20261119.md)。

正式长度短测可用 `--smoke-sequence-index 35` 选取一条完整 S1K 序列，并以 `--smoke-layers` 减少覆盖层数。它与 `--smoke-text` 互斥，结果始终标为 qualification。Nemotron 校准采用有界显存的完整 SSD 递推，保留跨 chunk 状态及 Linear 输入采集，不截断文本。

普通 QA：`python -m fp4_convert.experiments.paper_baselines.entry lm_eval`，任务为 `arc_challenge,arc_easy,openbookqa,boolq`，`--num_fewshot 0 --batch_size 128 --seed 1234 --log_samples`，不加 `--apply_chat_template`。本机 lm_eval CLI 没有新版 `run` 子命令。引擎参数按用户外机设置固定为 TP2、4096 context、BF16 KV、`enforce_eager=False`、`gpu_memory_utilization=0.90`、`max_num_seqs=128`、`max_num_batched_tokens=8192`；目标模型由 `paper_fp4` 负责 W4A4，无需给它叠加原生 NVFP4 的 linear/moe emulation 开关。五项推理仍走 lighteval。

最新的两个小模型执行范围、融合BRQ、正式QA与下载状态见[2026-11-20记录](../../../docs/experience/records/fp4_single_methods_20261120.md)。

本轮固定队列由 `batch.py` 读取实验目录中的 `all_single_methods/plan.json`：16个配置、304个任务/种子轮次，其中已核验的N9融合BRQ→MXFP4四项QA直接复用。先完成其余普通QA与12次完整校准，再执行五项推理任务的三个种子；同一校准权重跨种子复用。两项校准可在各自单卡并行，评测固定TP2。队列不轮询空闲卡、不自动重试，已有执行目录不能重复启动。

三轮推理使用 `reasoning.py`，沿用根目录 `main.py` 的结果保存与相同的lighteval Pipeline；独立传入引擎及采样种子、模型推荐参数，避免修改其他活动实验依赖的公共入口。每个任务/种子使用独立模型视图与预测缓存，数据路径及摘要固定在 `data/manifest.json`，GPQA使用ModelScope的198题。默认任务prompt和评分保持不变；本机LCB指标的历史名称含`:16`，但实际任务的`num_samples=1`，执行时再次核验每题只有一份回答。两模型的输出上限均为32768、上下文40960，FA2、编译/CUDA Graph、max_num_seqs=128、max_num_batched_tokens=8192。

`results.py` 核对完整题数、逐题分数与汇总一致性、生成次数和实际采样参数后，写入 `scores.csv` 与 `LaTeX/tables/single_methods_results.tex`。推理任务只有三个种子均完成才填均值；未完成项保持空白。QA同时保留acc和acc_norm，以便与外机基线对齐。短测目录不参与正式结果汇总。
