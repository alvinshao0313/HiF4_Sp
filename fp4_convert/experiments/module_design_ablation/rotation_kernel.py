"""FP32 G64 rotations with expert-specific routing, then existing HiF4 QDQ."""
from __future__ import annotations

import torch
from vllm.triton_utils import triton, tl
from vllm.model_executor.layers.quantization.hif4_triton import _hif4_fake_quant_block_tl


@triton.jit
def _rotate(X, Q, IDS, Y, WIDTH: tl.constexpr, SOURCE_TOP_K: tl.constexpr, ROUTED: tl.constexpr, QUANTIZE: tl.constexpr):
    row, group = tl.program_id(0), tl.program_id(1)
    expert = tl.load(IDS + row).to(tl.int32) if ROUTED else 0
    offsets = tl.arange(0, 64)
    x = tl.load(X + (row // SOURCE_TOP_K) * WIDTH + group * 64 + offsets).to(tl.float32)
    q = tl.load(Q + (expert * (WIDTH // 64) + group) * 4096 + offsets[:, None] * 64 + offsets[None, :])
    value = tl.sum(x[:, None] * q, axis=0)
    if QUANTIZE:
        value = _hif4_fake_quant_block_tl(value, offsets < 64, 64)
    tl.store(Y + row * WIDTH + group * 64 + offsets, value)


def _apply(x, q, *, route_ids, source_top_k, quantize):
    if not x.is_cuda or x.ndim != 2 or not x.is_contiguous() or x.shape[-1] % 64:
        raise ValueError("rotation kernel requires CUDA contiguous [rows,G64] activations")
    if not q.is_cuda or q.dtype != torch.float32 or not q.is_contiguous():
        raise ValueError("rotation matrices must be contiguous CUDA FP32")
    width = x.shape[-1]
    if tuple(q.shape[-3:]) != (width // 64, 64, 64):
        raise ValueError("rotation group geometry differs from activation input")
    if route_ids is None:
        if q.ndim != 3 or source_top_k != 1:
            raise ValueError("dense transform needs [groups,64,64] matrices")
        rows, ids = x.shape[0], x  # Unused pointer in the dense specialization.
    else:
        if q.ndim != 4 or not route_ids.is_cuda or not route_ids.is_contiguous():
            raise ValueError("routed transform needs [experts,groups,64,64] and CUDA IDs")
        rows, ids = route_ids.numel(), route_ids
        if type(source_top_k) is not int or source_top_k < 1 or rows != x.shape[0] * source_top_k:
            raise ValueError("route rows must match the source token/top-k layout")
    result = torch.empty((rows, width), dtype=torch.bfloat16 if quantize else torch.float32, device=x.device)
    if rows:
        _rotate[(rows, width // 64)](x, q, ids, result, width, source_top_k, route_ids is not None, quantize,
                                    num_warps=4, enable_fp_fusion=False)
    return result


def rotate(x, q, *, route_ids=None, source_top_k=1):
    return _apply(x, q, route_ids=route_ids, source_top_k=source_top_k, quantize=False)


def rotate_qdq(x, q, *, route_ids=None, source_top_k=1):
    return _apply(x, q, route_ids=route_ids, source_top_k=source_top_k, quantize=True)
