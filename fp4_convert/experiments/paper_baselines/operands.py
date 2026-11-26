"""Weight and input operand rules for the direct and random-block baselines."""
from functools import lru_cache
import math
import torch

from ..qwen3_mxfp4.format import dequantize, nvfp4_to_bf16, quantize


@lru_cache(maxsize=16)
def random_block(size, seed, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(0, 2, (size,), generator=generator) * 2 - 1
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < size:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)))
    return (signs[:, None] * h / math.sqrt(size)).to(device)


def operand(x, target, method="direct", seed=None):
    size = {"mxfp4": 32, "hif4": 64}[target]
    if x.shape[-1] % size:
        raise ValueError(f"Input width {x.shape[-1]} splits a {target} block")
    if method == "uos" and target == "mxfp4":
        from .scale_methods import uos_operand
        return uos_operand(x)
    if method == "brq":
        if seed is None:
            raise ValueError("BRQ requires a recorded sign seed")
        x = (x.float().unflatten(-1, (-1, size)) @ random_block(size, seed, x.device)).reshape_as(x)
    elif method != "direct":
        raise ValueError(f"Unsupported operand transform {method}")
    if target == "hif4":
        from vllm.model_executor.layers.quantization.hif4_fake import hif4_fake_quantize_hifx4
        return hif4_fake_quantize_hifx4(x).bfloat16()
    if x.is_cuda:
        from ..qwen3_mxfp4.kernels import activation_kernel
        # This kernel quantizes two independent G32 groups per program. Pad
        # the unused group on odd widths, leaving real group scales intact.
        width = x.shape[-1]
        work = x.reshape(-1, width)
        if width % 64:
            work = torch.nn.functional.pad(work, (0, 32))
        work = work.contiguous()
        result = torch.empty_like(work, dtype=torch.bfloat16)
        if work.numel():
            activation_kernel[(work.shape[0], work.shape[1] // 64)](
                work, result, result, result, result, work.shape[1], False, False,
                FHT32=False, num_warps=1, enable_fp_fusion=False)
        return result[:, :width].reshape_as(x).contiguous()
    return dequantize(*quantize(x))


def converted_weight(source, name, target, method, seed, device="cpu"):
    packed = source.tensor(name + ".weight")
    scales = source.tensor(name + ".weight_scale")
    global_scale = source.tensor(name + ".weight_scale_2").to(device)
    # Bound temporaries for the vocabulary head and large dense projections.
    weight = torch.empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.bfloat16, device="cpu")
    for start in range(0, len(weight), 256):
        rows = slice(start, start + 256)
        w = nvfp4_to_bf16(packed[rows].to(device), scales[rows].to(device), global_scale)
        weight[rows].copy_(operand(w, target, method, seed))
    return weight
