# NVFP4 vLLM Fake Activation Quantization Log

## 目的

`--nvf4_fake_act` 用于在 vLLM 普通 dense linear 的 GEMM 前执行 NVFP4 activation fake quant-dequant。

当前后端使用 `NVFP4/torch_fake.py` 的 PyTorch 实现。它不是 packed NVFP4 kernel，也不会带来低比特 GEMM 加速。

## 行为

- 开启 `--nvf4_fake_act` 时，`main.py` 要求 `model_path` 是本地目录。
- activation scale 固定从 `model_path/nvfp4_activation_scales.safetensors` 读取。
- 每个 linear 通过 vLLM layer prefix 查找对应的 `input_global_scale`。
- Qwen3.5 `linear_attn.*` 没有转换保存的 activation scale，不执行 NVFP4 fake act。
- 找不到 sidecar 文件、找不到某层 scale、fused linear 的多个 shard scale 不一致时，直接报错。
- `--hif4_fake_act` 和 `--nvf4_fake_act` 不能同时开启。

## 已知限制

- 只覆盖 vLLM `UnquantizedLinearMethod` 的普通 dense linear。
- 不覆盖 packed NVFP4 linear、fused MoE expert kernel、KV cache、embedding 和 `lm_head`。
