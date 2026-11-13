# Qwen3-30B-A3B routed-expert MXFP4

隔离实验：原生 NVFP4、直接 MXFP4、R64＋MXFP4，以及新增的 FHT32＋MXFP4 对照。仅替换 48 层、128 个专家的
gate/up/down 共 18,432 个矩阵；Attention、Router、Embedding、Norm、LM Head、路由、
top-8、归一化、专家求和及 TP 通信使用原实现。完整模型并非全 MXFP4。

## 数值契约

- 源固定为 `nvidia/Qwen3-30B-A3B-NVFP4` 的
  `2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3` snapshot。
- NVFP4：先 `E4M3_scale.float() * weight_scale_2.float()`，再乘 E2M1，最后舍入 BF16。
  `weight_scale_2` 是乘数，不能套用 MiniMax 的倒数语义。
- MXFP4：沿 K 每 32 元素，E2M1 最近偶数舍入并饱和到 ±6；
  `e=clamp(floor(log2(amax))-2,-127,127)`，全零组 `e=0`；E8M0 字节为 `e+127`。
  CPU 用 `frexp`、GPU 用 FP32 指数位实现 floor，避免近二次幂的 log2 舍入。
- R64 使用项目 `diag_gradient/r64_transform.py` 中的固定 regular Hadamard，
  `[out,in]` 权重右乘 R，输入右乘 R。源 BF16 权重／激活旋转时为 FP32，直接量化
  FP32 旋转结果，中间不舍入 BF16。gate/up 与 SwiGLU 后的 down 分开处理。
- FHT32 使用标准 Sylvester `H2⊗5/sqrt(32)`，自然顺序，无随机符号或置换。
  权重和激活沿连续 K32 块做五级 FP32 蝶形加减，最后归一化并直接量化；在线
  FHT 与 group32 QDQ 融合。它与 regular R64 的块大小、矩阵结构均不同。
- 在线 Triton QDQ 输出 BF16，packed GEMM 内部解码权重为 BF16、FP32 累加；
  down 的路由权重在 BF16 输出舍入前相乘，最后使用原 `moe_sum`。
- SwiGLU 调用原生 `apply_moe_activation`；独立参考保留 SiLU 先舍入 BF16、
  再乘 up 并舍入 BF16 的两个边界，不能合并成一次 FP32 计算。
- TP4 的输入 K 为 2048／192，均同时对齐 32 和 64，不跨 rank 或专家拼组。

专家 `.weight` 两个 E2M1 打包一个 uint8，低 nibble 为偶数 K；`.weight_scale` 为
K/32 个 E8M0 字节，移除专家 `.weight_scale_2` 和 `.input_scale`。名义负载为
4.25 bit/元素（含 scale），实际文件大小单独记录。A800 执行软件 QDQ/BF16 GEMM，
不是原生 MXFP4 硬件性能。

## 入口

在仓库根目录、调用者已激活 `hif4` 后：

```bash
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4.convert \
  --source /home/shaoyuantian/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3 \
  --output /path/to/new/direct_checkpoint --variant direct --threads 8
```

R64 用 `--variant r64` 和独立输出目录，仍从同一 NVFP4 源转换。转换器拒绝覆盖，
逐片重载核验全部输出，保留非专家名称、形状、dtype、字节摘要和辅助文件；全部成功后
才写完整 `model.safetensors.index.json`。`conversion.json` 保存覆盖、校验、来源及代码摘要。
FHT32 使用 `--variant fht32`；元数据另外冻结实际 H32 矩阵摘要，仍从原始 NVFP4 独立转换。

`entry.py` 在主进程及 spawn 子进程注册 `qwen3_routed_mxfp4` 后调用原 `main.py`。
只有专家改用新方法；非专家由原 `ModelOptNvFp4Config` 选择。全局 NVFP4 group16 配置
保持不变，专家 group32 单独记录在 `qwen3_routed_mxfp4` 元数据中。无 Quark 安装。

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest \
  Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4.test_format \
  Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4.test_integration -v
```

`compile.py --output PATH` 离线编译实际 SM80 特化，不执行 GPU；
`validate_gpu.py --model CHECKPOINT --output PATH` 测真实 layer-0、全部 128 专家 TP4
rank-0 局部形状，覆盖 decode=1、batch=256、prefill=8192。纯 QDQ 对参考要求编码、
scale 和重建一致；旋转误差独立记录；GEMM 与完整 MoE 使用 BF16 边界参考，
NMSE<1e-4 且满足按输出贡献幅度缩放的误差限。性能使用固定输入/路由、5 次预热、
20 次 CUDA events 计时，记录分布、吞吐、显存及共享 GPU 占用。

## 本次顺序执行与证据

权重与 CPU 证据 RUN：`Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/20261109T092759Z`。
GPU 算子与短测 RUN：`Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/direct_gpu03_20261109T095013Z`。
当前正式 RUN：`Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/direct_gpu03_formal_20261109T100620Z`。
[运行记录](../../../docs/experience/records/qwen3_mxfp4_20261109.md)说明已验证范围及待执行阶段。

`prepare_weights.py --root RUN` 仅 CPU 顺序生成两组权重。`workflow.py prepare --root RUN
--baseline NATIVE_RUN` 冻结协议、源码摘要及原生进程身份；随后 `workflow.py run --root RUN`
为一次性顺序控制器，拒绝重复运行，不自动重试。它等待原生任务完整结束及自身 worker
释放，再执行 direct GPU 检查 → direct 短测／正式 → R64 GPU 检查 → R64 短测／正式。
发生失败即退出，保留日志和未完成状态，不修改协议或停止其他任务。

用户追加指定直转 GPU0–3，并授权仅替换旧等待控制器后，使用 `parallel_workflow.py`
接续：先完成直接版 GPU 验证，在 GPU0–3 执行 direct 短测／正式，等待原生结束并释放
自身 worker，再在 GPU4–7 执行 R64 验证／短测／正式。两个候选复用已验证权重，
不修改原生源码、输出或进程；失败即退出，不自动重试。新控制器实际命令与 PID 见
当前 RUN 的 `workflow_launch.json`，当前阶段见 `workflow_status.json`。
本次短测已实际完成；修正后置 LM Head 方法核验后，通过实际推理源码哈希及证据摘要
复用该短测，正式回答写入独立新目录。旧失败状态和原始短测输出完整保留。

每组短测仅 26 题×10×64 token，正式为 198 题×10×32768 token；TP4、
direct 使用 GPU0–3，原生及 R64 使用 GPU4–7；
batch256、8192 batched tokens、40960 上下文、32 GiB/卡 BF16 KV、eager。
thinking 开启，seed1234、temperature0.6、top_p0.95、top_k20、min_p0。
复用原生实际数据、prompt、选项顺序、tokenizer 和评分，所有缓存独立。

短测完成后核验四个 worker 各 48 层选用新专家方法，逐层 packed 权重/scale 字节与
导出的 TP4 切片相同，非专家量化方法保持原选择。正式阶段同样记录实际加载证据。
`evidence.py` 复核实际输入 token/prompt 哈希和全部原回答评分，输出主平均值、10 个
采样序号的准确率、解析失败率和达到生成长度上限的比例；原缓存不保存 finish_reason，
因此“截断率”明确按达到长度上限统计。短测分数不作为准确率。

最终 `comparison.json` 只有三组 1980 个回答均完整验证后才生成。后台状态为
`waiting_baseline` 时，GPU 验证与两组正式评测尚未开始，不能视为通过或完成。

## 已保存回答的最终提交复核

本次三组正式生成均已完成。原严格 JSON 评分漏识别了 Markdown/LaTeX answer 字段
及 boxed 字母。`review_answers.py` 在 CPU 上独立复核已有缓存，不修改原 `task.py`、
原自动分数或生成文本，不启动模型。`extract --root RUN --output NEW_REVIEW_DIR`
提取最终提交并生成不显示模型／金标的特殊案例队列；逐条核对后，
`score --output NEW_REVIEW_DIR --decisions DECISIONS_JSON` 验证源摘要、原始评分和
全部实际输入，再写独立汇总。拒绝覆盖已有汇总。

本次结果在当前正式 RUN 的 `answer_review_20261110_v2/reviewed_summary.json`：
原生 62.8788%、直接 59.6970%、R64 59.1414%；分母均为 1980，未提交答案仍计 0。
特殊判定、逐回答分数、原始分数和全部摘要均保留，详见运行记录。

## FHT32 四卡对照

2026-11-10 用户确认新增 FHT32，并再次指定四卡并行推理。新增 RUN 为
`Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/fht32_tp4_20261110T032038Z`。
前三组完整结果复用，GPU4–7、TP4、batch256、32k 输出上限及数据/实际输入保持原协议。

转换及 CPU 测试完成后，运行：

```bash
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4.fht32_workflow prepare \
  --root /path/to/fht32_run --previous /path/to/completed_three_model_run
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4.fht32_workflow run \
  --root /path/to/fht32_run
```

有限流程依次执行单卡真实专家算子验证/计时、TP4 26×10×64 执行检查、TP4 198×10 正式评测，
各阶段失败即退出。每次模型阶段检查四个 rank 全部 48 层的实际方法及 packed 字节，
非专家选择保持原生；正式参数没有短测上限。后台启动后确认正式推进即可交付。

生成完成后，沿用现有最终提交解析规则生成 `answer_review/`，复用前三组逐条判定前先检查
原证据摘要和提取结果完全相同。新候选的特殊答案进入隐藏模型/金标的队列，流程记为
`awaiting_answer_review`；原严格 JSON 分数与最终提交分数分开，不能在复核完成前宣称准确率。
