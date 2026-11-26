# Qwen3 路由专家组合实验

算法与资源已确认，详见 [PLAN.md](PLAN.md)。四个候选分别组合 R16 扩展 R32、
UOS c7.25 和 G512；权重、激活顺序都是旋转 → 外层归一化 → 内层 MXFP4。
只改变路由专家，其余张量和推理路径沿用原生 NVFP4。A800 上为软件 QDQ 和 BF16 GEMM。

本轮结果目录：
[`combinations_tp2_20261113T154617Z`](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/combinations_tp2_20261113T154617Z)。
2026-11-14四组均已完成并复核，结果见
[最终准确率、方差和单方法对照](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/combinations_tp2_20261113T154617Z/answer_review_20261114/REPORT.md)。
以目录中的实际状态、退出码及验证文件为准；没有完整正式结果时不报告准确率。

入口均由已激活 `hif4` 的 shell 从仓库根目录执行：

```bash
python -m fp4_convert.experiments.qwen3_mxfp4_combinations.convert \
  --source /home/shaoyuantian/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3 \
  --output /path/to/new-checkpoint --variant r16x2_uos_c7p25 --threads 8
```

`format.py` 定义离线格式，`kernels.py` 提供融合激活 QDQ，`runtime.py` 注册独立专家方法。
G512 复用既有 BF16 packed GEMM；TP2 down 只通信全局组最大值，保持原输出 SUM。
`test_format.py` 检查独立 CPU 参考，`validate_qdq.py` 检查真实 TP2 边界，
`validate_gpu.py` 检查真实专家两次 GEMM、SwiGLU、MoE、BF16 MMA 和性能。

`prepare.py --root RUN` 冻结协议与全部 48 层 TP2 权重摘要。
`workflow.py --root RUN --lane 0|1 --phase qualify` 要求已保存的算子验证通过，再各自顺序执行两个候选的
26题×10、64-token 正式配置短测。两路全部通过后，`formal.py --root RUN --lane 0|1` 顺序执行各路
198题×10 正式评测，并保存十个采样序号的方差、标准差及相对单方法的变化。状态已存在时拒绝重复启动，失败退出，无自动重试。

| lane | GPU | 第一个候选 | 第二个候选 |
| --- | --- | --- | --- |
| 0 | 0–1 | `r16x2_uos_c7p25` | `r16x2_dualscale_g512` |
| 1 | 2–3 | `dualscale_g512_uos_c7p25` | `r16x2_dualscale_g512_uos_c7p25` |

所有候选 TP2、每卡56GiB BF16 KV，其余 AISBench prompt、采样与长度协议复制上一轮。
`manifest.json` 保存每个实际命令，`formal/main.log` 为生成日志，原始缓存和严格 JSON
分数始终保留。`formal/final_submission_summary.json` 使用旧最终答案规则；有待盲审项时
分数标记为暂定，不能冒充已复核结果。

复用的单方法对照见
[七组答案复核与方差](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/aisbench_gpqa_20261113T023500Z/answer_review_20261113T144526Z/REPORT.md)。
单 R32 为 TP4/32GiB；UOS、G512 为 TP2/56GiB，比较时保留此执行配置差异。
