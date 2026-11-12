# NVFP4 Ascend 伪量化算子包

包名：`nvfp4-ascend-fakequant`；Python 导入名：`nvfp4_npu`；版本：`0.1.0`。

这是当前 Qwen GPQA 实验所用算子的独立打包版。两个核心实现文件保持原样，来源 SHA256 位于 `src/nvfp4_npu/provenance.json`。不依赖原项目、vLLM 或模型文件。

## 安装与环境

已验证：Linux aarch64、Ascend A3、Python 3.11.18、CANN 9.1.0、PyTorch 2.10.0+cpu、torch-npu 2.10.0.post4、triton-ascend 3.2.2。此环境同时报告 Triton 前端版本 3.5.0。其他组合尚未验证。

先进入已配置好 CANN 和 PyTorch NPU 的环境，再安装 wheel：

```bash
conda activate hif4
python -m pip install --no-deps nvfp4_ascend_fakequant-0.1.0-py3-none-any.whl
```

`--no-deps` 用于保留已经配套的环境；包不包含驱动、CANN、PyTorch 或 Triton。只有源码时，在源码目录运行：

```bash
python -m pip install --no-deps --no-build-isolation .
```

wheel 是 Python 源码包，NPU 内核在首次调用时由 Triton Ascend 编译。源码包中的示例可以这样运行：

```bash
python examples/basic.py --device cpu
ASCEND_RT_VISIBLE_DEVICES=0 python examples/basic.py --device npu:0
```

## API

| 函数 | 用途 | 输入与返回 |
| --- | --- | --- |
| `nvfp4_fake_quant(x, scale)` | NVFP4 激活量化、反量化 | BF16 二维 x，K 是 16 的倍数；输出同形状 BF16 |
| `fp8_fake_quant(x, scale)` | FP8 E4M3 KV 量化、反量化 | BF16 二维 x；输出同形状 BF16 |
| `dequantize_nvfp4(weight, scale, global_scale, *, divisor=False, dtype=torch.bfloat16)` | 已打包 NVFP4 权重解码 | uint8 `[out,K/2]`、E4M3 `[out,K/16]`、FP32 全局标量；默认输出 BF16 `[out,K]` |
| `nvfp4_reference(x, scale)` / `fp8_reference(x, scale)` | 独立 CPU 参考实现 | CPU 输入，参数规则同上 |
| `e4m3_values(raw)` | E4M3FN 字节解码 | uint8 或 float8_e4m3fn → FP32 |
| `bf16_linear(x, decoded_weight, bias=None)` | BF16 矩阵乘法 | x 和 weight 都为 BF16 |

激活和 KV 的 `scale` 必须是与 x 同设备的 FP32 张量，元素数为 1 或 x 的行数。推荐形状 `[1]` 或 `[M]`。每行 scale 可用于已按专家排列的 MoE 输入。输入必须有限，scale 必须有限且为正；NPU 热路径不会逐次同步检查这些数值条件。输入需要关闭梯度，这些算子用于推理，不提供训练用 STE 或 Triton backward。

NPU 权重 block scale 建议传原始 uint8 E4M3 字节：在 CPU 上对 `float8_e4m3fn` 做 `.view(torch.uint8)` 后再搬到 NPU。scale 布局是普通行主序，不接受经过硬件 swizzle 的布局。

## 激活与 KV 示例

```python
import torch
import torch_npu
from nvfp4_npu import nvfp4_fake_quant, fp8_fake_quant

x = torch.randn(4, 2048, device='npu:0', dtype=torch.bfloat16)
# 示例数值；实际使用模型检查点中对应层/专家的标定 scale。
input_scale = torch.tensor([0.01], device=x.device, dtype=torch.float32)
with torch.inference_mode():
    x_qdq = nvfp4_fake_quant(x, input_scale)

    k_after_rope = torch.randn(4, 128, device=x.device, dtype=torch.bfloat16)
    k_scale = torch.tensor([0.02], device=x.device, dtype=torch.float32)
    k_qdq = fp8_fake_quant(k_after_rope, k_scale)
    # 将 k_qdq 写入 BF16 KV cache；V 同理，使用对应 v_scale。
```

Qwen 的接入位置：QKV 投影、attention 输出投影、专家 gate/up 和 down 投影的输入；K 在 QK norm 和 RoPE 之后、进入缓存之前伪量化，V 在进入缓存之前伪量化。包提供算子，调用方负责在模型的这些位置接入。

## 权重的两种 global scale 约定

```python
from nvfp4_npu import dequantize_nvfp4

# NVIDIA ModelOpt：weight_scale_2 是乘数。
w = dequantize_nvfp4(weight, weight_scale_bytes, weight_scale_2)

# compressed-tensors / MiniMax NVFP4 专家：weight_global_scale 是除数。
w = dequantize_nvfp4(weight_packed, weight_scale_bytes,
                    weight_global_scale, divisor=True)
```

两个 E2M1 数值放在一个字节中，低半字节对应前一个元素。幅值表为 `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`，bit 3 为符号。块大小固定 16，解码在 FP32 中完成，最后一次转换到 BF16。

## 数值口径

NVFP4 激活使用标定全局 scale `g`，每 16 个数计算：

```text
s = round_E4M3(min(amax / (6*g), 448)) * g
y = BF16(round_E2M1(x / s) * s)
```

两次舍入均为最近偶数舍入（RNE）；s 为零时返回零。FP8 KV 使用：

```text
y = BF16(round_E4M3(clamp(x / g, -448, 448)) * g)
```

结果模拟低精度操作数的误差，实际 GEMM 和 KV 存储仍是 BF16。它不是原生 FP4 GEMM，也不是 FP4/FP8 缓存压缩实现；不保证与 NVIDIA 原生 CUDA kernel 逐位一致。

## 验证与来源

源码包的 `provenance/original_activation_validation.json` 保存此前 CPU/NPU 对照验证，包括随机输入、零块、极值、FP4 中点、E4M3 编码及边界、逐行 scale。它绑定原始算子源码哈希。

本次发行另外验证 wheel 可独立安装、包内文件与原评测源码一致、CPU 参考结果及小规模 NPU 调用。发行验证结果随交付包提供。原始来源、改写范围和许可证见 `THIRD_PARTY_NOTICES.md`。
