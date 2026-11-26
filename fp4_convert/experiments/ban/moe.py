"""Compact HiF4 execution with the original top-8 reduction boundary.

The QDQ and routed GEMMs follow the existing hif4_emulation_moe path. The
final buffer is padded with zeros to preserve the native top-8 moe_sum;
vLLM's specialized top-3/4 sum otherwise rounds each addition in BF16.
"""
from __future__ import annotations

import torch


def apply_compact_moe(x, w13, w2, weights, ids, reference_top_k=8):
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
    from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel, try_get_optimal_moe_config
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    from vllm.model_executor.layers.fused_moe.experts.hif4_emulation_moe import _compute_type
    from vllm.model_executor.layers.quantization.hif4_transform_triton import hif4_r64_quantize_hifx4_triton

    tokens, hidden = x.shape
    experts, twice_intermediate, _ = w13.shape
    k, intermediate = ids.shape[1], twice_intermediate // 2
    if not 1 <= k <= reference_top_k or not x.is_contiguous() or not weights.is_contiguous() or not ids.is_contiguous():
        raise ValueError("compact HiF4 requires contiguous inputs and a valid expert prefix")
    config = try_get_optimal_moe_config(w13.size(), w2.size(), k,
        FUSED_MOE_UNQUANTIZED_CONFIG.config_name(x.dtype), tokens, block_shape=None)
    workspace13 = torch.empty(tokens * k * max(twice_intermediate, hidden), device=x.device, dtype=x.dtype)
    cache1 = workspace13[:tokens * k * twice_intermediate].view(tokens, k, twice_intermediate)
    cache3 = workspace13[:tokens * k * hidden].view(tokens, k, hidden)
    workspace2 = torch.empty(tokens * max(hidden, k * intermediate), device=x.device, dtype=x.dtype)
    x_q = workspace2[:tokens * hidden].view(tokens, hidden)
    cache2 = workspace2[:tokens * k * intermediate].view(tokens * k, intermediate)
    hif4_r64_quantize_hifx4_triton(x, out=x_q)
    sorted_ids, routed_experts, padded_count = moe_align_block_size(ids, config["BLOCK_SIZE_M"], experts)
    common = dict(A_scale=None, B_scale=None, sorted_token_ids=sorted_ids, expert_ids=routed_experts,
                  num_tokens_post_padded=padded_count, config=config, compute_type=_compute_type(x.dtype),
                  use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False,
                  per_channel_quant=False, block_shape=None)
    invoke_fused_moe_triton_kernel(x_q, w13, cache1, topk_weights=None,
                                  mul_routed_weight=False, top_k=k, **common)
    torch.ops._C.silu_and_mul(cache2, cache1.view(-1, twice_intermediate))
    hif4_r64_quantize_hifx4_triton(cache2, out=cache2)
    invoke_fused_moe_triton_kernel(cache2, w2, cache3, topk_weights=weights,
                                  mul_routed_weight=True, top_k=1, **common)
    padded = torch.zeros((tokens, reference_top_k, hidden), device=x.device, dtype=x.dtype)
    padded[:, :k].copy_(cache3)
    out = torch.empty_like(x)
    ops.moe_sum(padded, out)
    return out
