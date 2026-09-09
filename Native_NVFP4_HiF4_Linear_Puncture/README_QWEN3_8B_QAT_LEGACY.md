# Native NVFP4 → HiF4 Linear Puncture

基于 `ISTA-DASLab/Qwen3-8B-FPQuant-QAT-NVFP4` 的 Linear 局部穿刺实验：
保留 checkpoint 在线 block rotation，捕获 `X_rot`，离线比较 NVFP4 / MXFP8 / HiF4。

## 环境

在仓库根目录、`hif4` conda 环境中运行。

## 运行顺序

```bash
# 1) checkpoint preflight（本地 cache 不完整会直接失败）
python -m Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint \
  --config Native_NVFP4_HiF4_Linear_Puncture/configs/qwen3_8b_native_nvfp4_linear_puncture.yaml \
  --run-id RUN_ID

# 2) smoke capture（3 modules）或 formal capture（35 modules）
python -m Native_NVFP4_HiF4_Linear_Puncture.src.capture \
  --config Native_NVFP4_HiF4_Linear_Puncture/configs/qwen3_8b_native_nvfp4_linear_puncture.yaml \
  --run-id RUN_ID --mode smoke --device cuda

python -m Native_NVFP4_HiF4_Linear_Puncture.src.capture \
  --config Native_NVFP4_HiF4_Linear_Puncture/configs/qwen3_8b_native_nvfp4_linear_puncture.yaml \
  --run-id RUN_ID --mode formal --device cuda

# 3) 离线 Linear cases（建议先释放 8B 模型显存）
python -m Native_NVFP4_HiF4_Linear_Puncture.src.linear_cases \
  --config Native_NVFP4_HiF4_Linear_Puncture/configs/qwen3_8b_native_nvfp4_linear_puncture.yaml \
  --run-id RUN_ID --device cuda

# 4) 图与中文报告
python -m Native_NVFP4_HiF4_Linear_Puncture.src.report \
  --config Native_NVFP4_HiF4_Linear_Puncture/configs/qwen3_8b_native_nvfp4_linear_puncture.yaml \
  --run-id RUN_ID
```

结果写入 `Native_NVFP4_HiF4_Linear_Puncture/results/<RUN_ID>/`。

## 约束

- 禁止 import `Inference_Paradigm_Conversion.*`
- 禁止用 lm_eval 跑本实验；本实验不做下游指标
- checkpoint 未下完时 preflight / capture 立即失败，不做 fallback
