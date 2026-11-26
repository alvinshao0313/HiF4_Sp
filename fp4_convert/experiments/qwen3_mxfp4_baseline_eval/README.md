# 原生 NVFP4／直接 MXFP4：AISBench prompt 下游对照

复用相同原始 NVFP4 snapshot 和已完成的直接 MXFP4 checkpoint，不改权重或量化算法。
使用[四组合评测](../qwen3_mxfp4_combination_eval/README.md)已核验的 AISBench 零样本模板、
固定数据、chat template 和原单回答评分器，仍通过根目录 main.py／lighteval＋vLLM 执行。
这不是完整 AISBench 框架运行。

用户本轮授权 GPU2–7，两卡 TP2 一组，每卡56GiB BF16 KV；GPU0–1既有任务保持运行。
LCB v6为175题×5，MMLU-Pro为全量12,032题×5。seed1234、thinking、temperature0.6、
top_p0.95、top_k20、min_p0、context40960、batch256、8192 batched tokens、eager保持一致。
LCB最大生成38912，MMLU-Pro最大生成32768。指标是全部五次单回答准确率的算术平均，
不去极值、不投票，不称为pass@5或五个独立seed。

| GPU | 有限队列 |
| --- | --- |
| 2–3 | 原生 NVFP4 MMLU-Pro |
| 4–5 | 直接 MXFP4 MMLU-Pro |
| 6–7 | 原生 NVFP4 LCB → 直接 MXFP4 LCB |

准备核对权重SHA256、tokenizer一致性、已有源码和实际输入摘要，并生成48层直接版TP2
packed切片参考。直接版补验K=384的QDQ/GEMM/SwiGLU/MoE，tokens=1/256/8192；
实际GEMM沿用BF16操作数、FP32累加。原生使用原ModelOpt NVFP4软件emulation路径。
原生加载观察器仅调用原方法后记录实际backend、方法和TP形状，不替换前向计算。

四项正式资源短测各52题×5、最多64新token，验证两rank、48层、BF16 KV、输入和评分
落盘。全部短测成功后才能启动正式队列；每阶段失败即退出所在队列，不自动重试或降配置。
每个任务和阶段使用独立模型视图，避免lighteval缓存复用。原始回答、逐题五次得分、
五列准确率、方差、截断及来源摘要保留；完整汇总为comparison.json/csv。

在已激活hif4的仓库根目录执行，RUN为新的绝对输出路径：

```bash
python -m fp4_convert.experiments.qwen3_mxfp4_baseline_eval.prepare --output RUN
CUDA_VISIBLE_DEVICES=4 python -m fp4_convert.experiments.qwen3_mxfp4_baseline_eval.validate_gpu --root RUN
python -m fp4_convert.experiments.qwen3_mxfp4_baseline_eval.workflow qualify --root RUN
python -m fp4_convert.experiments.qwen3_mxfp4_baseline_eval.workflow formal --root RUN
```

实际后台命令、PID、日志及阶段状态见[本轮运行目录](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/native_direct_downstream_20261116T063859Z)。
复用[评测经验](../../../docs/experience/lessons/evaluation_evidence.md)：固定真实数据版本、
检查实际请求token和输出数量，隔离缓存并保留评分输入边界；短测不代表正式精度。
本轮没有新增量化研究算法；正式生成确认推进后交付并停止主动监控。
