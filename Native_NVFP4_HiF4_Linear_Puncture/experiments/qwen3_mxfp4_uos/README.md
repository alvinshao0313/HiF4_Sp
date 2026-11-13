# Qwen3 路由专家 UOS c7.25

隔离的无旋转候选 `direct_uos_c7p25`。仅把 MXAttention 的 UOS scale 规则移植到
Qwen3-30B-A3B 路由专家的权重和激活，不包含 Attention 修改或 PNQ。
算法名称为 **UOS c7.25**，采用用户确认的 **ceil**：

`e = clamp(ceil(log2(amax / 7.25)), -127, 127); s = 2**e`。
零组取 `e=0, s=1`；K/group32、E2M1 最近偶数舍入、饱和到 ±6、E8M0 scale。
CPU 用 `frexp` 得到 `amax=m*2**k`，计算 `k-3+int(m>29/32)`；Triton 用 FP32
指数与尾数位的等价严格比较。精确阈值及相邻浮点数须保持正确，不用近似 `log2`。

源 snapshot 固定为 `nvidia/Qwen3-30B-A3B-NVFP4` 的
`2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`。从源 NVFP4 解码到 BF16 后独立量化
全部 18,432 个专家矩阵，不能使用旧 MXFP4 权重作为输入。其余张量逐字节保留。
packed 权重与 scale 名义负载为 4.25 bit/元素；实际文件大小由 `conversion.json` 记录。

gate/up 共用输入 QDQ，down 在原生 SwiGLU 后单独 QDQ。保留 SiLU→BF16→乘 up→BF16
的两个舍入边界。复用旧参数化 packed GEMM（显式 group32、FP32 解码的 E8M0 scale）、
路由加权、求和及 TP 通信。A800 执行软件 QDQ/BF16 GEMM，不是原生 MXFP4 硬件。
量化注册名 `qwen3_routed_mxfp4_uos`，元数据记录 c=7.25、ceil、无旋转和格式版本。

## 入口

由调用者先激活 `hif4`；以下命令在仓库根目录执行。`SOURCE` 必须是固定原始 snapshot，
`RUN` 必须是独立新目录，`PREVIOUS` 是已完成的原三组 RUN，`PRIOR_TABLE` 是五组复核结果 JSON。

```bash
python -m unittest Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4_uos.test_format Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4_uos.test_integration Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4.test_review_answers -v
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4_uos.convert --source "$SOURCE" --output "$RUN/checkpoints/direct_uos_c7p25" --threads 8
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4_uos.workflow prepare --root "$RUN" --previous "$PREVIOUS" --prior-table "$PRIOR_TABLE"
python -u -m Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4_uos.workflow run --root "$RUN"
```

`prepare` 要求 `RUN/cpu_tests.log` 完整通过、全量转换和保存重载通过，以及
`authorization.json` 中保护的旧源码摘要仍匹配。它冻结实际命令、原生数据与源码、五组
复核结果和新源码；后续禁止修改运行依赖。实际启动命令及进程保存在 `workflow_launch.json`。

有限流程为：GPU0 算子检查 → GPU0–3 TP4/batch256 的 26×10×64 执行检查 →
198×10 正式评测 → 独立最终答案复核队列。任一步失败退出，不自动重试或修改协议。
短测覆盖实际四个 rank 各 48 层的 packed 字节与非专家路径；短测分数不作准确率。

正式协议：thinking、seed1234、temperature0.6、top_p0.95、top_k20、min_p0，最大
生成32768、上下文40960、batch256、8192 batched tokens、每卡32 GiB BF16 KV、eager。
实际 prompt/token/选项及严格 JSON 评分沿用原生；每个模型使用独立缓存。

## 输出与复核

- `checkpoints/direct_uos_c7p25/conversion.json`：源与输出摘要、转换覆盖、非专家与保存重载验证。
- `direct_uos_c7p25_gpu_validation.json`：tokens=1/256/8192 的数值、旧直转/UOS CUDA events 计时及显存。
- `direct_uos_c7p25/{probe,formal}/main.log`、`status.json`：各阶段实际推理与退出状态。
- `*_runtime_audit.json`：四个 rank 的完整权重切片及方法选择。
- `comparison.json`：六组原严格 JSON 结果；不替代最终答案复核。
- `answer_review/pending_blind_review_queue.json`：仅新增待审提交，不显示模型或金标。

人工复核沿用[既有最终提交规则](../qwen3_mxfp4/review_answers.py)，每条决定需记录
`answer`、`reason`、`final_sha256`。原五组的决定通过文本摘要和完整提取记录核验后复用，
不能覆盖或改变。将新决定与 `reused_manual_decisions.json` 合并后执行：

```bash
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4_uos.report --root "$RUN" --decisions "$DECISIONS"
```

该入口保留原严格 JSON 分数，生成独立 `reviewed_summary.json` 和 `reviewed_scores.jsonl`，
复现原五组分数后写出 `six_model_summary.json/csv`：1980 回答均值、十个采样序号、
相对原生/旧直转的百分点变化、截断及解析情况。正式推理结束不等于最终复核已完成。

证据与阶段状态统一见[实验记录](../../../docs/experience/records/qwen3_mxfp4_20261109.md)。
