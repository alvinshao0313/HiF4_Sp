# 来源与第三方声明

此包将项目中用于 GPQA 评测的两个算子文件逐字节复制到独立命名空间：

- `nvfp4_eval/activation_quant.py` → `nvfp4_npu/activation_quant.py`
- `nvfp4_eval/quant.py` → `nvfp4_npu/checkpoint_quant.py`

来源哈希和验证环境记录在 `nvfp4_npu/provenance.json`。原文件中的项目相对路径为历史来源说明；安装包不依赖这些路径。

NVFP4 数值规则参考 NVIDIA Model Optimizer 的固定提交：
`9aeff419f3afa1b27e7bcc10dc6a90cc68df4d77`。

- `modelopt/torch/kernels/quantization/common/nvfp4_quant.py`
- `modelopt/torch/kernels/quantization/gemm/fp4_kernel.py`
- `modelopt/torch/kernels/quantization/gemm/tensor_quant_mx.cu`

源码发行包的 `provenance/modelopt/` 保留了原始快照、版权声明、来源 URL 和 SHA256。这些上游文件用于查阅，不在运行时导入。

Copyright (c) 2023–2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
上游文件按 Apache License 2.0 分发，许可证全文见 `licenses/Model-Optimizer-APACHE-2.0.txt`。该许可证文件说明上述第三方来源的许可，不为其他来源的项目代码额外指定许可证。

Ascend 移植使用本地 PyTorch / Triton Ascend 实现 E4M3 舍入和权重字节解码；不依赖 NVIDIA CUDA libdevice。
