"""Offline Table 3 weight selection, before vLLM fusion or tensor parallelism."""
import time

import torch

from HiFloat4.hif4_scale_threshold_optimization.src.formats import S1P2_MAX
from HiFloat4.hif4_scale_threshold_optimization.src.quantizer import HiF4QuantConfig, quantize_hif4
from HiFloat4.hif4_scale_threshold_optimization.src.weight_search import search_weight_groups
from ..qwen3_mxfp4.format import nvfp4_to_bf16


METHOD_TARGET = {"scale_search": "hif4", "matrix_c": "hif4", "uos": "mxfp4",
                 "g256": "mxfp4", "g512": "mxfp4"}
GROUP_SIZE = {"g256": 256, "g512": 512}
CANDIDATES = tuple(step / 4 for step in range(24, 33))


def uos_operand(x):
    from ..qwen3_mxfp4_uos.format import VARIANT, activation_reference
    if x.is_cuda:
        from ..qwen3_mxfp4_uos.kernels import activation_qdq
        return activation_qdq(x.reshape(-1, x.shape[-1]), VARIANT).reshape_as(x)
    return activation_reference(x)


def normalized_operand(x, group):
    """Return exact BF16 inner values and FP32 outer scales separately."""
    if x.is_cuda:
        from ..qwen3_mxfp4_dualscale.kernels import activation_qdq
        return activation_qdq(x, group)
    from ..qwen3_mxfp4_dualscale.format import normalized_reference
    return normalized_reference(x, group)


def matrix_constant(blocks, weight):
    # A chunk splits only output rows. Every candidate sees the same decoded
    # weight and every real element contributes to the single matrix decision.
    errors = {c: 0.0 for c in CANDIDATES}
    energy = 0.0
    for _, w in blocks():
        for c in CANDIDATES:
            reconstructed = quantize_hif4(w, config=HiF4QuantConfig(s0_divisor=c)).reconstruction
            errors[c] += (w.float() - reconstructed.float()).square().sum(dtype=torch.float64).item()
        energy += w.float().square().sum(dtype=torch.float64).item()
    best = min(CANDIDATES, key=lambda c: (errors[c], abs(c - 7), c))
    changes = clipped = zeroed = nonzero = 0
    for rows, w in blocks():
        direct = quantize_hif4(w)
        selected = direct if best == 7 else quantize_hif4(w, config=HiF4QuantConfig(s0_divisor=best))
        changes += (selected.s0 != direct.s0).sum().item()
        clipped += (selected.normalized > S1P2_MAX).sum().item()
        zeroed += ((w != 0) & (selected.reconstruction == 0)).sum().item()
        nonzero += (w != 0).sum().item()
        weight[rows].copy_(selected.reconstruction)
    count = weight.numel()
    return {"selected_c": best, "candidate_mse": {str(c): val / count for c, val in errors.items()},
            "changed_s0_fraction": changes / (count // 64), "clipped_fraction": clipped / count,
            "nonzero_to_zero_fraction": zeroed / nonzero if nonzero else 0.0,
            "reference_sum_squares": energy, "direct_sum_squared_error": errors[7.],
            "selected_sum_squared_error": errors[best]}


def convert_scale_weight(source, name, method, device):
    start = time.monotonic()
    packed = source.tensor(name + ".weight")
    scales = source.tensor(name + ".weight_scale")
    scale2 = source.tensor(name + ".weight_scale_2").to(device)
    n, k = packed.shape[0], packed.shape[1] * 2
    weight = torch.empty((n, k), dtype=torch.bfloat16, device="cpu")

    def blocks():
        for i in range(0, n, 256):
            rows = slice(i, i + 256)
            yield rows, nvfp4_to_bf16(packed[rows].to(device), scales[rows].to(device), scale2)

    outer = None
    info = {}
    if method == "matrix_c":
        info = matrix_constant(blocks, weight)
    else:
        group = GROUP_SIZE.get(method)
        if group:
            outer = torch.empty((n, (k + group - 1) // group), dtype=torch.float32, device="cpu")
            info = {"outer_group": group, "outer_fp32_bytes": outer.numel() * outer.element_size()}
        elif method == "scale_search":
            info = {"budget": "full", "groups": n * k // 64, "reference_sum_squares": 0.,
                    "direct_sum_squared_error": 0., "selected_sum_squared_error": 0.}
        for rows, w in blocks():
            if group:
                inner, outer_chunk = normalized_operand(w, group)
                outer[rows].copy_(outer_chunk)
            elif method == "uos":
                inner = uos_operand(w)
            elif method == "scale_search":
                result = search_weight_groups(w, budget="full", device=w.device)
                inner = result.reconstruction
                direct = quantize_hif4(w).reconstruction
                info["reference_sum_squares"] += w.float().square().sum(dtype=torch.float64).item()
                for key, other in (("selected_sum_squared_error", inner), ("direct_sum_squared_error", direct)):
                    info[key] += (w.float() - other.float()).square().sum(dtype=torch.float64).item()
            else:
                raise ValueError(f"Unknown scale method {method}")
            weight[rows].copy_(inner)
    return weight, outer, {"shape": [n, k], "elapsed_seconds": time.monotonic() - start, **info}
