"""HiF4 MoE runtime for materialized BF16 expert weights."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_triton_kernel,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization import hif4_fake
from vllm.model_executor.layers.quantization.hif4_triton import (
    hif4_quantize_hifx4_triton,
)
from vllm.model_executor.layers.quantization.hif4_transform_triton import (
    hif4_online_routed_qdq_triton,
    hif4_r64_quantize_hifx4_triton,
    hif4_silu_mul_triton,
)
from vllm.triton_utils import tl


def _compute_type(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.float32:
        return tl.float32
    raise ValueError(f"Unsupported HiF4 MoE dtype: {dtype}")


def _apply_hif4_triton_moe(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    use_r64: bool = False,
) -> torch.Tensor:
    """CUDA HiF4 path: two routed Triton GEMMs with HiF4 activation QDQ."""
    if hidden_states.dim() != 2 or not hidden_states.is_contiguous():
        raise ValueError(
            "HiF4 Triton MoE expects contiguous [tokens, hidden] input"
        )
    if w13.stride(-1) != 1 or w2.stride(-1) != 1:
        raise ValueError("HiF4 Triton MoE weights must be contiguous on K")
    if topk_weights.dim() != 2 or topk_ids.shape != topk_weights.shape:
        raise ValueError(
            "HiF4 Triton MoE top-k tensors must have matching 2D shapes"
        )

    num_tokens, hidden = hidden_states.shape
    num_experts = w13.size(0)
    intermediate_twice = w13.size(1)
    top_k_num = topk_ids.size(1)
    if (
        w13.size(2) != hidden
        or w2.size(0) != num_experts
        or w2.size(1) != hidden
    ):
        raise ValueError("HiF4 MoE materialized weight shape mismatch")
    if intermediate_twice % 2 != 0 or w2.size(2) * 2 != intermediate_twice:
        raise ValueError("HiF4 MoE gate/up/down dimensions are inconsistent")

    config = try_get_optimal_moe_config(
        w13.size(),
        w2.size(),
        top_k_num,
        FUSED_MOE_UNQUANTIZED_CONFIG.config_name(hidden_states.dtype),
        num_tokens,
        block_shape=None,
    )
    compute_type = _compute_type(hidden_states.dtype)

    # Match vLLM's modular MoE workspace reuse: W13 output and W2 output share
    # one backing buffer; input-QDQ and post-SwiGLU QDQ share the other. CUDA
    # stream ordering makes the reuse safe because each earlier consumer is
    # enqueued before the later writer reuses the same storage.
    intermediate = intermediate_twice // 2
    cache1_numel = num_tokens * top_k_num * intermediate_twice
    cache3_numel = num_tokens * top_k_num * hidden
    workspace13 = torch.empty(
        max(cache1_numel, cache3_numel),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    cache1 = workspace13[:cache1_numel].view(
        num_tokens, top_k_num, intermediate_twice
    )
    cache3 = workspace13[:cache3_numel].view(num_tokens, top_k_num, hidden)

    x_q_numel = num_tokens * hidden
    cache2_numel = num_tokens * top_k_num * intermediate
    workspace2 = torch.empty(
        max(x_q_numel, cache2_numel),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    x_q = workspace2[:x_q_numel].view(num_tokens, hidden)
    cache2 = workspace2[:cache2_numel].view(
        num_tokens * top_k_num, intermediate
    )
    if use_r64:
        hif4_r64_quantize_hifx4_triton(hidden_states, out=x_q)
    else:
        hif4_quantize_hifx4_triton(hidden_states, out=x_q)

    sorted_ids, routed_experts, padded_count = moe_align_block_size(
        topk_ids,
        config["BLOCK_SIZE_M"],
        num_experts,
    )

    invoke_fused_moe_triton_kernel(
        x_q,
        w13,
        cache1,
        None,
        None,
        None,
        sorted_ids,
        routed_experts,
        padded_count,
        False,
        top_k_num,
        config,
        compute_type=compute_type,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=None,
    )

    torch.ops._C.silu_and_mul(cache2, cache1.view(-1, intermediate_twice))
    if use_r64:
        hif4_r64_quantize_hifx4_triton(cache2, out=cache2)
    else:
        hif4_quantize_hifx4_triton(cache2, out=cache2)

    invoke_fused_moe_triton_kernel(
        cache2,
        w2,
        cache3,
        None,
        None,
        topk_weights,
        sorted_ids,
        routed_experts,
        padded_count,
        True,
        1,
        config,
        compute_type=compute_type,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=None,
    )

    out = torch.empty_like(hidden_states)
    ops.moe_sum(cache3, out)
    return out


def _apply_hif4_online_triton_moe(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    d_gate: torch.Tensor,
    d_up: torch.Tensor,
    d_down: torch.Tensor,
    *,
    use_r64: bool,
    rot_order: str,
) -> torch.Tensor:
    """Exact Online path with independent Gate/Up/Down expert DIAGs."""
    if hidden_states.dim() != 2 or not hidden_states.is_contiguous():
        raise ValueError("Online HiF4 MoE expects contiguous [tokens, hidden]")
    num_tokens, hidden = hidden_states.shape
    num_experts = int(w13.size(0))
    intermediate_twice = int(w13.size(1))
    if intermediate_twice % 2:
        raise ValueError("w13 output dimension must be even")
    intermediate = intermediate_twice // 2
    top_k_num = int(topk_ids.size(1))
    if tuple(d_gate.shape) != (num_experts, hidden):
        raise ValueError(f"D_gate shape mismatch: {tuple(d_gate.shape)}")
    if tuple(d_up.shape) != (num_experts, hidden):
        raise ValueError(f"D_up shape mismatch: {tuple(d_up.shape)}")
    if tuple(d_down.shape) != (num_experts, intermediate):
        raise ValueError(f"D_down shape mismatch: {tuple(d_down.shape)}")

    route_ids = topk_ids.reshape(-1, 1).contiguous()
    route_weights = topk_weights.reshape(-1, 1).contiguous()
    route_rows = num_tokens * top_k_num
    gate_w = w13[:, :intermediate, :]
    up_w = w13[:, intermediate:, :]
    config = try_get_optimal_moe_config(
        gate_w.size(),
        w2.size(),
        1,
        FUSED_MOE_UNQUANTIZED_CONFIG.config_name(hidden_states.dtype),
        route_rows,
        block_shape=None,
    )
    compute_type = _compute_type(hidden_states.dtype)
    sorted_ids, routed_experts, padded_count = moe_align_block_size(
        route_ids, config["BLOCK_SIZE_M"], num_experts
    )

    gate_q = hif4_online_routed_qdq_triton(
        hidden_states,
        route_ids,
        d_gate,
        source_top_k=top_k_num,
        use_r64=use_r64,
        rot_order=rot_order,
    )
    up_q = hif4_online_routed_qdq_triton(
        hidden_states,
        route_ids,
        d_up,
        source_top_k=top_k_num,
        use_r64=use_r64,
        rot_order=rot_order,
    )
    gate_out = torch.empty(
        route_rows, 1, intermediate,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    up_out = torch.empty_like(gate_out)
    common = dict(
        A_scale=None,
        B_scale=None,
        topk_weights=None,
        sorted_token_ids=sorted_ids,
        expert_ids=routed_experts,
        num_tokens_post_padded=padded_count,
        mul_routed_weight=False,
        top_k=1,
        config=config,
        compute_type=compute_type,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=None,
    )
    invoke_fused_moe_triton_kernel(gate_q, gate_w, gate_out, **common)
    invoke_fused_moe_triton_kernel(up_q, up_w, up_out, **common)

    hidden = hif4_silu_mul_triton(
        gate_out.view(route_rows, intermediate),
        up_out.view(route_rows, intermediate),
    )
    down_q = hif4_online_routed_qdq_triton(
        hidden,
        route_ids,
        d_down,
        source_top_k=1,
        use_r64=use_r64,
        rot_order=rot_order,
    )
    down_out = torch.empty(
        route_rows, 1, hidden_states.size(1),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    invoke_fused_moe_triton_kernel(
        down_q,
        w2,
        down_out,
        None,
        None,
        route_weights,
        sorted_ids,
        routed_experts,
        padded_count,
        True,
        1,
        config,
        compute_type=compute_type,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=None,
    )
    routed = down_out.view(num_tokens, top_k_num, hidden_states.size(1))
    out = torch.empty_like(hidden_states)
    ops.moe_sum(routed, out)
    return out


def apply_hif4_online_fused_moe(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    d_gate: torch.Tensor,
    d_up: torch.Tensor,
    d_down: torch.Tensor,
    *,
    use_r64: bool,
    rot_order: str,
) -> torch.Tensor:
    if not hidden_states.is_cuda:
        raise ValueError("Online HiF4 fused MoE currently requires CUDA")
    return _apply_hif4_online_triton_moe(
        hidden_states,
        w13,
        w2,
        topk_weights,
        topk_ids,
        d_gate,
        d_up,
        d_down,
        use_r64=use_r64,
        rot_order=rot_order,
    )


def apply_hif4_fused_moe(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    use_r64: bool = False,
) -> torch.Tensor:
    """Run materialized HiF4 MoE.

    CUDA uses vLLM's routed Triton GEMM layout, eliminating the Python expert
    loop. CPU keeps the grouped PyTorch implementation as a correctness
    reference for unit tests.
    """
    if hidden_states.is_cuda:
        from vllm.model_executor.layers.quantization import hif4_runtime

        runtime_spec = hif4_runtime.current_hif4_runtime_spec()
        if runtime_spec is not None:
            algorithm = hif4_runtime.algorithm_variant(runtime_spec)
            if algorithm == "online":
                layer_idx = hif4_runtime.current_layer(hidden_states.device)
                # Gate/Up D match unreplicated hidden; Down D matches the local
                # TP shard of the expert intermediate width (w13 out/2).
                local_intermediate = int(w13.size(1)) // 2
                d_gate, d_up, d_down, online_spec = (
                    hif4_runtime.online_moe_scales_by_layer(
                        layer_idx,
                        hidden_states.device,
                        hidden_cols=int(hidden_states.shape[-1]),
                        down_cols=local_intermediate,
                    )
                )
                return _apply_hif4_online_triton_moe(
                    hidden_states,
                    w13,
                    w2,
                    topk_weights,
                    topk_ids,
                    d_gate,
                    d_up,
                    d_down,
                    use_r64=bool(online_spec.get("use_r64", False)),
                    rot_order=str(online_spec.get("rot_order", "diag_then_rot")),
                )
            use_r64 = use_r64 or bool(runtime_spec.get("use_r64", False))
            use_r64 = use_r64 or runtime_spec.get("variant") in {
                "r64",
                "fusable_r64",
            }
        return _apply_hif4_triton_moe(
            hidden_states,
            w13,
            w2,
            topk_weights,
            topk_ids,
            use_r64=use_r64,
        )

    if use_r64:
        raise ValueError("CPU HiF4 MoE reference does not implement R64")
    num_tokens, hidden = hidden_states.shape
    x_q = hif4_fake.hif4_fake_quantize_hifx4(hidden_states)
    acc = torch.zeros(
        num_tokens,
        hidden,
        device=hidden_states.device,
        dtype=torch.float32,
    )
    token_index = torch.arange(num_tokens, device=hidden_states.device)
    for slot in range(topk_ids.shape[1]):
        expert_ids = topk_ids[:, slot]
        scale = topk_weights[:, slot].to(dtype=torch.float32).unsqueeze(-1)
        slot_down = torch.empty(
            num_tokens,
            hidden,
            device=hidden_states.device,
            dtype=x_q.dtype,
        )
        for expert in torch.unique(expert_ids).tolist():
            idx = token_index[expert_ids == expert]
            x_e = x_q.index_select(0, idx)
            gu = F.linear(x_e, w13[expert])
            gate, up = gu.chunk(2, dim=-1)
            act_q = hif4_fake.hif4_fake_quantize_hifx4(F.silu(gate) * up)
            slot_down.index_copy_(0, idx, F.linear(act_q, w2[expert]))
        acc += slot_down.float() * scale
    return acc.to(dtype=hidden_states.dtype)
