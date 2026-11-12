"""Run a small weight + activation + KV emulation example, without a model."""
import argparse

import torch

from nvfp4_npu import bf16_linear, dequantize_nvfp4, fp8_fake_quant, nvfp4_fake_quant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'npu:0'], default='cpu')
    args = parser.parse_args()
    if args.device.startswith('npu'):
        import torch_npu  # Register Ascend operations in the configured environment.
    torch.manual_seed(42)
    device = args.device
    x = torch.randn(2, 32, dtype=torch.bfloat16).to(device)
    # Demonstration scales only; real inference uses checkpoint calibrations.
    input_scale = torch.tensor([0.01], dtype=torch.float32, device=device)
    packed = torch.randint(0, 256, (16, 16), dtype=torch.uint8).to(device)
    # E4M3FN byte 0x38 represents 1.0; raw bytes avoid unsupported FP8 casts.
    block_scales = torch.full((16, 2), 0x38, dtype=torch.uint8, device=device)
    weight_global_scale = torch.tensor([0.01], dtype=torch.float32, device=device)
    weight = dequantize_nvfp4(packed, block_scales, weight_global_scale)
    output = bf16_linear(nvfp4_fake_quant(x, input_scale), weight)
    kv_scale = torch.tensor([0.02], dtype=torch.float32, device=device)
    kv = fp8_fake_quant(x, kv_scale)
    print({'device': str(output.device), 'output_shape': list(output.shape),
           'kv_shape': list(kv.shape), 'dtype': str(output.dtype),
           'finite': bool(torch.isfinite(output).all() and torch.isfinite(kv).all())})


if __name__ == '__main__':
    main()
