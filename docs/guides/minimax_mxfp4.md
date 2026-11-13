# MiniMax NVFP4 → MXFP4 权重与激活对比

2026-11-04：用户确认将路由专家的权重和两处输入激活都改为 MXFP4。
转换、独立推理入口和有限评测接续已实现；23 项 CPU 检查、真实专家矩阵转换
及 SM80 离线编译已有证据。完整转换和 GPU 实跑状态以本页链接的 RUN 为准，
离线编译与权重 NMSE 不代表下游准确率已验证。

## 转换定义

仅转换主模型 57 个 MoE 层 × 128 个专家 × `w1/w2/w3`，共 21,888 个矩阵。
原始 checkpoint 固定为 `2f33a8b9ae49ce7b51c994df97166a7fd059bc2e`，不从未知的
原始 BF16 模型重新量化，不训练、不校准或搜索额外缩放参数。

每个 NVFP4 矩阵按以下次序转换：

1. 解包 E2M1；将原 E4M3 组 scale 乘以 FP32 全局量化倍率的倒数。
2. 按现有反量化计算次序生成矩阵，并明确舍入为 **BF16**。
3. 沿 K 维每 32 个 BF16 元素取 `amax`，令共享 scale 指数为
   `clamp(floor(log2(amax)) - 2, -127, 127)`；全零组取指数 0。
4. 除以这个二的幂 scale，按 E2M1 最近偶数舍入并饱和到 ±6，两个值打包为一个字节；
   scale 保存为 E8M0 的 uint8 位模式（指数加 127，不使用保留值 255）。

这一 scale 选择与 [NVIDIA NeMo MXFP4 量化说明](https://docs.nvidia.com/nemo/automodel/latest/nemo-automodel/nemo_automodel/components/quantization/mxfp4)
中的 `floor(log2(amax))-2` 一致。E2M1、32 元素分组和 E8M0 scale 的格式定义见
[OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)。
格式本身不等于所有量化器都使用相同的 scale 选择；本轮固定上述规则，不与其他
ceil/even/误差搜索规则混用。

导出保留 `.weight_packed` 字段；`.weight_scale` 改为 K/32 个 E8M0 字节，删除专家
原有 `.weight_global_scale` 和 `.input_global_scale`。后者是原 checkpoint 的占位值，
不能作为新激活 scale。每个非专家张量按原 dtype/位模式保留，并校验写盘后的摘要；
主权重 index 仅在全部分片转换、保存重载核验完成后发布。

MXFP4 权重的名义负载为 `4 + 8/32 = 4.25 bit/元素`，包含组 scale；完整文件体积
还须计入未转换模块、header 等，以最终导出统计为准。BF16 中间矩阵只短暂存在于
内存，每次处理一份分片，不额外导出全模型 BF16 checkpoint。

## 推理范围与隔离

路由专家第一组 GEMM（gate/up）之前的 hidden states，以及第二组 GEMM（down）
之前的 SwiGLU 输出，都动态执行 MXFP4 group-32 QDQ。它们不再执行原 NVFP4 的
group-16 / E4M3 scale / per-token global-scale 量化。

其他部分维持原模型：Attention、稀疏索引投影、前三层 FFN、共享专家仍为 MXFP8
权重；Embedding、LM Head、Norm 权重及 KV/index cache 为原 BF16 路径；Router
和路由修正偏置为 FP32。图像/视频、DSpark 不参与本轮纯文本评测。

A800 执行软件量化/反量化与 BF16 GEMM，不代表原生 MXFP4 硬件吞吐。
新实现调用既有参数化 E2M1 GEMM 核，明确传入 `group_size=32` 和已解码 E8M0 的
FP32 scale；不调用硬编码 NVFP4 group-16/E4M3 的 wrapper。末端仍用原 `moe_sum`。

所有新增逻辑位于 `tools/minimax_mxfp4/`，入口为 `tools/minimax_m3_mxfp4.py`，
量化注册名为 `minimax_m3_mxfp4`。不修改仍被基线依赖的 MiniMax NVFP4 模块、
公共量化器、原转换算法或 Ban 调用链。新模型与各阶段预测缓存使用独立目录。

## 运行与结果

当前 RUN：`results/minimax_m3/mxfp4_20261104T015806Z/`。

- [转换与评测配置](../../results/minimax_m3/mxfp4_20261104T015806Z/config.json)
- [真实权重 CPU 转换短测](../../results/minimax_m3/mxfp4_20261104T015806Z/real_weight_cpu_probe.json)
- [CPU 检查日志](../../results/minimax_m3/mxfp4_20261104T015806Z/cpu_tests.log)
- [SM80 编译结果](../../results/minimax_m3/mxfp4_20261104T015806Z/sm80_compile.json)
- [转换进程记录](../../results/minimax_m3/mxfp4_20261104T015806Z/conversion_launch.json)
- [转换状态](../../results/minimax_m3/mxfp4_20261104T015806Z/model/conversion_status.json)
- [转换日志](../../results/minimax_m3/mxfp4_20261104T015806Z/conversion.log)
- [接续进程记录](../../results/minimax_m3/mxfp4_20261104T015806Z/launch.json)
- [接续状态](../../results/minimax_m3/mxfp4_20261104T015806Z/state.json)
- [启动前协议与源码核验](../../results/minimax_m3/mxfp4_20261104T015806Z/prelaunch_validation.json)

2026-11-07：全部 21,888 个专家矩阵已完成转换。按用户要求，已停止仍在等待 NVFP4
基线的 MXFP4 接续控制器 PID 965825；GPU 数值/整模型短测和正式评测均未启动，
也不会在 NVFP4 LCB 完成后自动接续。当前 NVFP4 LCB 保持运行，转换权重、日志和
已有结果保留。停止身份核验与执行结果见
[停止记录](../../results/minimax_m3/mxfp4_20261104T015806Z/stop_record.json)。
[原交付快照](../../results/minimax_m3/mxfp4_20261104T015806Z/handoff.json) 仅记录当时状态。

以下命令描述已启动的独立转换，不能对当前输出目录重复执行：

```bash
python -m tools.minimax_mxfp4 convert \
  --source /home/shaoyuantian/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M3.1-Flash/snapshots/2f33a8b9ae49ce7b51c994df97166a7fd059bc2e \
  --output results/minimax_m3/mxfp4_20261104T015806Z/model --threads 8
```

评测接续仅等待已记录身份的转换进程与现有 NVFP4 基线控制器退出，然后核对 GPU
2–7 空闲；不停止或占用其他任务。GPU 阶段先对真实转换专家的正式 TP2 切片进行
E8M0 解码、激活 CPU/GPU 对照及两次 GEMM 数值检查，再进行 TP2 × PP3 整模型
生成和五任务各两题短测。GPU 关键计算或结果核验失败即停止，不进入正式阶段。

正式条件复用 [NVFP4 基线](../../results/minimax_m3/continuation_20261103T015711Z/config.json)：
GSM8K 1,319 题；AIME24/25 各 30 题、每题五回答 avg@5；MMLU-Pro 300 题；
LCB v6 175 题、每题一回答 checker pass@1。上下文 40,960，batch 16，
max_num_batched_tokens 4096，seed 1234、temperature 1.0、top_p 0.95，
top-k 不截断。生成上限 GSM8K/MMLU-Pro 32,768，AIME/LCB 38,912，不沿用短测
的 256-token 限制。数据、实际 prompt/token 和评分与基线相同。

正式结果通过原始缓存覆盖、输入 token、回答数与无 prompt 截断核验后，写入 RUN 的
`accuracy.json`；`comparison.json` 保留原基线来源、新模型来源及百分点差。
基线已有完整同协议结果直接复用，不因新候选再跑原模型。
