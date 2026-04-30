# HiF4 vLLM Fake Activation Quantization Log

## 目的

本次修改给 vLLM 的普通 FP16/BF16 dense linear 推理路径增加 HiF4 activation fake quantization 开关。目标是做数值实验：先把 linear 输入激活按 HiF4 `hifx4` 规则量化再反量化，然后继续使用 vLLM 原来的标准 GEMM。

这不是 GPTQ packed linear，也不是低比特 GEMM 加速路径；权重仍按普通 vLLM 权重格式加载。

## 修改内容

- 新增 `3rdparty/vllm/vllm/model_executor/layers/quantization/hif4_fake.py`。
- 在 `hif4_fake_quantize_hifx4()` 中复刻 `HiFloat4/hif4_gpu/quant_cy/base/QFuncs/hifx.py` 的 HiF4 `hifx4` fake quant-dequant 逻辑。
- 支持 hidden dimension 不是 64 倍数的输入：先在最后一维补零到 64 对齐，量化后再裁回原 shape。
- 修改 `3rdparty/vllm/vllm/model_executor/layers/linear.py` 的 `UnquantizedLinearMethod`：
  - 从 `VllmConfig.additional_config` 读取 `hif4_fake_act`。
  - 开启时，在普通 linear GEMM 前对输入 `x` 执行 HiF4 fake quant。
  - 默认跳过 `lm_head`。
  - 只接受 `hif4_act_qtype=hifx4`，传入其他值时直接报错。

## 为什么这样修改

用户目标是 vLLM 推理中的 fake activation quantization，不是 GPTQ 权重 packed kernel。因此最小、最直接的接入点是 `UnquantizedLinearMethod.apply()`：这里覆盖普通 dense linear 的计算入口，模型文件和权重加载格式都不需要改。

把开关放进 `--additional-config` 的原因是 vLLM 已有这个通用运行时配置入口，且它参与配置 hash，适合控制会改变计算图/数值路径的实验开关。

跳过 `lm_head` 是为了对齐当前 HiF4 脚本默认排除 `lm_head` 的策略，避免 logits 端额外扰动。

## 使用方法

开启：

```bash
vllm serve /path/to/model \
  --dtype float16 \
  --additional-config '{"hif4_fake_act": true}'
```

显式指定当前唯一支持的量化类型：

```bash
vllm serve /path/to/model \
  --dtype float16 \
  --additional-config '{"hif4_fake_act": true, "hif4_act_qtype": "hifx4"}'
```

关闭：

```bash
vllm serve /path/to/model --dtype float16
```

或：

```bash
vllm serve /path/to/model \
  --dtype float16 \
  --additional-config '{"hif4_fake_act": false}'
```

## 验证建议

所有 Python 命令应在 `hif4` conda 环境中运行。

建议先做最小验证：

1. 直接调用 `hif4_fake_quantize_hifx4()`，确认输出 shape/dtype 不变，非 64 倍数 hidden dim 可正常返回，无 NaN。
2. 构造普通 vLLM linear，关闭开关时确认输出等于原路径，开启开关时确认 shape/dtype 正常且数值发生扰动。
3. 用小模型分别关闭/开启 `hif4_fake_act` 做 vLLM 生成冒烟测试。

## 已知限制

- 首版只支持 `hifx4`。
- 首版只覆盖普通 dense linear，不覆盖 GPTQ/AWQ/FP8 等量化 linear method。
- 首版不覆盖 fused MoE expert kernel、KV cache、embedding 和 `lm_head`。
- 这是 fake quant-dequant 数值路径，不会带来低比特 GEMM 性能收益。
