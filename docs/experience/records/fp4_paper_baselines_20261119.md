# 2026-11-19 四模型格式转换基线环境验证

BRQ 随机块旋转、独立 GPTQ、AWQ 通道尺度搜索和 SmoothQuant 共用全量投影范围，各自支持 HiF4 与 MXFP4。运行命令、方法参数和产物格式见[运行说明](../../../fp4_convert/experiments/paper_baselines/README.md)。本次使用本机 `hif4` 环境，Python 3.11.13、PyTorch 2.10.0+cu128、Transformers 5.8.1，以及仓库内的 vLLM 0.19.2.dev0+gb1388b1fb.d20260430。[工作区代码 SHA-256](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/readiness_20261119T084223Z/implementation-sha256.json)

## 权重与校准文本

实际快照路径、发布文件名、大小和摘要保存在[文件清单](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/readiness_20261119T084223Z/release_files.json)。Qwen3-30B 复用了本地快照，其余三个模型从与 NVIDIA 固定版本逐文件匹配的镜像补齐。本地逐文件校验了 LFS SHA-256 或 Git blob SHA-1，并核对了 safetensors 索引和 NVFP4 侧信息。

| NVIDIA 模型 | revision 前 12 位 | 发布文件数 | 原有 NVFP4 投影数 | 128 条 S1K 的 token 总数 |
| --- | --- | ---: | ---: | ---: |
| NVIDIA-Nemotron-Nano-9B-v2-NVFP4 | `8556c9164ddb` | 34 | 96 | 1,274,376 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 | `6efb4a2a1c1f` | 18 | 5,968 | 1,269,802 |
| Qwen3-30B-A3B-NVFP4 | `2538ded2a4ed` | 18 | 18,624 | 1,272,746 |
| Qwen3.8-27B-NVFP4-QAD | `8158ac567c23` | 78 | 307 | 1,297,673 |

Qwen3.8 的 307 个投影包含输出头，另外 94 个投影使用发布的 FP8 配置。四个模型的 S1K 样本编号一致，上表 token 数由各自的分词器计算。这次读取了正式校准数据以验证输入，模型短测另行使用了少量文本。[分词与样本编号核验](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/readiness_20261119T084223Z/corpus-check.json)

## 实际执行范围

- **BRQ 和 SmoothQuant 的整模型链路。** Nemotron 9B 用种子 17 旋转并导出为 MXFP4。另用一条 16 token 短文本校准 SmoothQuant 并导出为 HiF4，56 层走完，96 份权重和通道尺度逐一核对，与校准产物完全一致。两个导出均在 Transformers CPU BF16 参考路径重载，三步贪心生成的 logits 有限，每步实际执行全部 96 个目标投影。[BRQ 生成](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/readiness_20261119T084223Z/9b-brq-generation.json) · [SmoothQuant 生成](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/readiness_20261119T084223Z/9b-smoothquant-generation.json) · [重载核对](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/readiness_20261119T084223Z/weight-reload.log)
- **GPTQ 和 AWQ 的校准与保存。** Nemotron 9B 的真实前 3 层各运行一次，覆盖其中的两个 NVFP4 投影，分别保存 HiF4-GPTQ 与 MXFP4-AWQ。Qwen3-30B 的首层 Q/K/V 三个完整投影在 6 token 上共同执行 AWQ 的 20 个候选搜索，随后以选出的共用尺度启用 MXFP4 激活量化并完成注意力前向。[GPTQ 配置与覆盖](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/readiness_20261119T084223Z/9b-gptq-hif4-layer2/recipe.json) · [AWQ 配置与覆盖](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/readiness_20261119T084223Z/9b-awq-mxfp4-layer2/recipe.json) · [Q/K/V 共用尺度](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/readiness_20261119T084223Z/awq-fused.json)
- **两种目标格式与模型结构。** 在 Qwen3-30B 的首层查询投影（16 个输出通道，输入宽度 2,048）上，用一条完整 S1K 训练序列校准 GPTQ、AWQ 和 SmoothQuant，再用独立的一条完整验证序列检查输出。三种方法与两种目标格式的六组结果均为有限值。另在四模型 24 类真实投影上检查了 BRQ 的两个目标格式。SmoothQuant 还跑过 Qwen3-30B 的首层 388 个投影、Nemotron-3-30B 的前两层 260 个投影，以及 Qwen3.8 首层的 3 个 NVFP4 投影。[校准数值](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/readiness_20261119T084223Z/cpu-calibration.json) · [投影与格式](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/readiness_20261119T084223Z/cpu-projections.json) · [解码器层](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/readiness_20261119T084223Z/block-load.json)
- **双卡状态。** 两个 Nemotron 9B 基线导出均通过 TP2 引擎配置解析。GPU 4、5 各有其他任务占用约 74,247 MiB，本次尚未完成双卡 worker 的模型加载、专家核执行和文本生成。上述 CPU 结果用于验证转换与推理的数值路径，论文正式精度仍待评测。[引擎配置](../../../Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines/readiness_20261119T084223Z/baseline-config.json) · [执行边界](../lessons/fake_quant_boundaries.md)
