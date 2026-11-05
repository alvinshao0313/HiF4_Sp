"""Independent gate/up/down rotations on the real routed Triton GEMM path."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel, try_get_optimal_moe_config
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.triton_utils import tl

from .rotation_kernel import rotate_qdq


def apply_moe(x, w13, w2, topk_weights, topk_ids, rotations):
    if x.dtype != torch.bfloat16 or x.ndim != 2 or not x.is_contiguous():
        raise ValueError("rotation MoE requires contiguous BF16 [tokens,hidden]")
    tokens, hidden = x.shape
    experts, twice_intermediate, input_width = w13.shape
    intermediate = twice_intermediate // 2
    if twice_intermediate % 2 or input_width != hidden or tuple(w2.shape) != (experts, hidden, intermediate):
        raise ValueError("invalid rotation MoE weight shapes")
    top_k = topk_ids.shape[1]
    route_ids, route_weights = topk_ids.reshape(-1, 1).contiguous(), topk_weights.reshape(-1, 1).contiguous()
    route_rows = tokens * top_k
    gate_w, up_w = w13[:, :intermediate], w13[:, intermediate:]
    config = try_get_optimal_moe_config(gate_w.size(), w2.size(), 1,
        FUSED_MOE_UNQUANTIZED_CONFIG.config_name(x.dtype), route_rows, block_shape=None)
    sorted_ids, routed_experts, padded_count = moe_align_block_size(route_ids, config["BLOCK_SIZE_M"], experts)
    common = dict(A_scale=None, B_scale=None, topk_weights=None, sorted_token_ids=sorted_ids,
                  expert_ids=routed_experts, num_tokens_post_padded=padded_count,
                  mul_routed_weight=False, top_k=1, config=config, compute_type=tl.bfloat16,
                  use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
                  use_int4_w4a16=False, per_channel_quant=False, block_shape=None)
    outputs = []
    for name, weight in (("gate_proj", gate_w), ("up_proj", up_w)):
        activation = rotate_qdq(x, rotations[name], route_ids=route_ids, source_top_k=top_k)
        branch = torch.empty((route_rows, 1, intermediate), device=x.device, dtype=x.dtype)
        invoke_fused_moe_triton_kernel(activation, weight, branch, **common)
        outputs.append(branch.view(route_rows, intermediate))
    # Match the reconstruction and native BF16 boundary: SiLU is rounded before
    # multiplying by up. The existing Online-DIAG helper fuses away that cast.
    intermediate_act = F.silu(outputs[0]) * outputs[1]
    down_q = rotate_qdq(intermediate_act, rotations["down_proj"], route_ids=route_ids)
    down_out = torch.empty((route_rows, 1, hidden), device=x.device, dtype=x.dtype)
    invoke_fused_moe_triton_kernel(down_q, w2, down_out,
        **{**common, "mul_routed_weight": True, "topk_weights": route_weights})
    result = torch.empty_like(x)
    ops.moe_sum(down_out.view(tokens, top_k, hidden), result)
    return result
