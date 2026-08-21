# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NVFP4 quantization emulation for MoE.

Backported from vLLM v0.27.0 and adapted to the local TritonExperts /
FusedMoEExperts APIs. On SM80 (no Triton ``tl.float8e4nv``), block scales
are converted to float32 before the fused GEMM kernel; E2M1 / scale math
matches upstream.
"""

from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    TritonExperts,
    try_get_optimal_moe_config,
    write_zeros_to_output,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    moe_kernel_quantize_input,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    _e2m1_inline,
    _nvfp4_triton_fp8e4nv_supported,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def fused_moe_nvfp4_emulation_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    w_global_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    # Strides — A [M, K]
    stride_am,
    stride_ak,
    # Strides — B [E, N, K//2], passed as (expert, K-packed, N)
    stride_be,
    stride_bk,
    stride_bn,
    # Strides — C [M, topk, N]
    stride_cm,
    stride_cn,
    # Strides — B_scale [E, N, K//BLOCK], passed as (expert, K-scale, N)
    stride_bse,
    stride_bsk,
    stride_bsn,
    block_k_diviable: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    group_size: tl.constexpr,
    SCALES_ARE_FP32: tl.constexpr,
):
    """
    Fused MoE kernel for emulated NVFP4 weight-only dequantization + GEMM.

    Activations A are BF16 (already QDQ'd externally).
    Weights B are packed uint8 NVFP4 [E, N, K//2] — two FP4 values per byte
    along the K dimension.
    B_scale holds per-block FP8-E4M3 scales [E, N, K // group_size]
    (or float32 after SM80 pre-conversion).
    w_global_scale is a per-expert scalar global scale.

    The dequantization formula per element is:
        w_float = e2m1_decode(nibble) * (block_scale_fp8 * global_scale)
    """
    BLOCK_SIZE_K_PACKED: tl.constexpr = BLOCK_SIZE_K // 2

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_k_packed = tl.arange(0, BLOCK_SIZE_K_PACKED)

    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + offs_bn[:, None] * stride_bn
        + offs_k_packed[None, :] * stride_bk
    )

    group_size_packed: tl.constexpr = group_size // 2

    w_global_scale = tl.load(w_global_scale_ptr + off_experts).to(tl.float32)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if block_k_diviable:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )

        if block_k_diviable:
            raw_bytes = tl.load(b_ptrs)
        else:
            kp_mask = offs_k_packed[None, :] < (K // 2) - k * BLOCK_SIZE_K_PACKED
            raw_bytes = tl.load(b_ptrs, mask=kp_mask, other=0)

        low_nibble = raw_bytes & 0x0F
        high_nibble = (raw_bytes >> 4) & 0x0F

        low_decoded = _e2m1_inline(low_nibble)
        high_decoded = _e2m1_inline(high_nibble)

        b_scale_ptrs = (
            b_scale_ptr
            + off_experts * stride_bse
            + offs_bn[:, None] * stride_bsn
            + ((offs_k_packed[None, :] + BLOCK_SIZE_K_PACKED * k) // group_size_packed)
            * stride_bsk
        )
        if block_k_diviable:
            b_scale_raw = tl.load(b_scale_ptrs)
        else:
            b_scale_raw = tl.load(b_scale_ptrs, mask=kp_mask, other=0.0)

        if SCALES_ARE_FP32:
            b_scale = b_scale_raw.to(tl.float32)
        else:
            b_scale = tl.cast(b_scale_raw, tl.float8e4nv, bitcast=True).to(tl.float32)
        b_scale = b_scale * w_global_scale

        low_scaled = low_decoded * b_scale
        high_scaled = high_decoded * b_scale

        b = tl.trans(tl.interleave(low_scaled, high_scaled)).to(compute_type)

        accumulator = tl.dot(a, b, acc=accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K_PACKED * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _prepare_nvfp4_moe_block_scales(
    B_scale: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Return (scales_for_kernel, scales_are_fp32).

    SM80 cannot compile Triton ``tl.float8e4nv``; convert FP8 block scales to
    float32 once before launch. Numerical formula is unchanged.
    """
    if _nvfp4_triton_fp8e4nv_supported():
        if B_scale.dtype == torch.float8_e4m3fn:
            return B_scale.view(torch.uint8), False
        return B_scale, False

    if B_scale.dtype == torch.uint8:
        B_scale = B_scale.view(torch.float8_e4m3fn)
    return B_scale.to(torch.float32), True


def invoke_fused_moe_nvfp4_emulation_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor,
    act_global_scale: torch.Tensor,
    w_global_scale: torch.Tensor,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
):
    """Launch the fused NVFP4 emulation MoE kernel.

    B has shape [E, N, K_packed] where K_packed = K // 2 (two FP4 per byte).
    B_scale has shape [E, N, K // group_size] in FP8-E4M3 (stored as uint8)
    or float32 after SM80 pre-conversion.
    w_global_scale has shape [E] (per-expert scalar).
    """
    assert B_scale is not None and B_scale.ndim == 3
    del act_global_scale  # activations are already QDQ'd externally

    B_scale, scales_are_fp32 = _prepare_nvfp4_moe_block_scales(B_scale)

    N = B.size(1)
    K = A.size(1)

    M = A.size(0)
    num_tokens = M * top_k

    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        EM = min(
            sorted_token_ids.size(0),
            A.size(0) * top_k * config["BLOCK_SIZE_M"],
        )

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    fused_moe_nvfp4_emulation_kernel[grid](
        A,
        B,
        C,
        B_scale,
        w_global_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        # B is [E, N, K//2]: swap N and K strides so kernel indexes [K, N].
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        # B_scale is [E, N, K//group]: swap N and K strides likewise.
        B_scale.stride(0),
        B_scale.stride(2),
        B_scale.stride(1),
        block_k_diviable=K % config["BLOCK_SIZE_K"] == 0,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        group_size=16,
        SCALES_ARE_FP32=scales_are_fp32,
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
    )


class Nvfp4QuantizationEmulationTritonExperts(TritonExperts):
    """
    Extension of TritonExperts to support emulated NVFP4 MoE experts.

    It may be used for NVFP4 models when the device does not have
    native support for this dtype.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        logger.warning_once(
            "Using Nvfp4QuantizationEmulationTritonExperts MOE backend. This will"
            " dequantize weights on the fly and may be slower than native"
            " quantized MOE. Consider using a device with native quantization"
            " support (e.g. Nvidia Blackwell) for better performance."
        )

        # Packed NVFP4 weights stay uint8; keep scales for the fused kernel and
        # clear quant_config weight scales so the BF16 parent path is unused.
        self.w1_scale_val = self.quant_config.w1_scale
        self.w2_scale_val = self.quant_config.w2_scale

        self.quant_config._w1.scale = None
        self.quant_config._w2.scale = None

        self.quantization_emulation = True

    @property
    def quant_dtype(self) -> torch.dtype | str | None:
        return "nvfp4"

    @property
    def expects_unquantized_inputs(self) -> bool:
        # Prepare/finalize must not QDQ; this expert QDQ's once in apply().
        return True

    @property
    def a1_scale(self) -> torch.Tensor:
        # Used if a parent path consults a1_scale for deferred quant.
        return self.a1_gscale

    @staticmethod
    def supports_lora() -> bool:
        return False

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        if moe_config.is_lora_enabled:
            return False, "kernel does not support LoRA"
        if moe_config.has_bias:
            return False, "kernel does not support bias"

        return TritonExperts.is_supported_config(
            cls,
            moe_config,
            weight_key,
            activation_key,
            activation_format,
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kNvfp4Static, kNvfp4Dynamic)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        """
        Apply emulated quantized MoE computation.

        This dequantizes the weights on the fly and applies activation QDQ
        exactly once per GEMM (W13 and W2).
        """
        del a1q_scale, a2_scale, expert_tokens_meta
        assert w1.dtype == torch.uint8
        assert w2.dtype == torch.uint8
        assert hidden_states.is_contiguous()
        assert hidden_states.dim() == 2

        K = hidden_states.size(-1)
        assert w1.size(2) * 2 == K, f"Hidden size mismatch: {K} != {w1.size(2) * 2}"

        E, num_tokens, N, _, top_k_num = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )

        if global_num_experts == -1:
            global_num_experts = E

        config = try_get_optimal_moe_config(
            w1.size(),
            w2.size(),
            top_k_num,
            self.quant_config.config_name(hidden_states.dtype),
            num_tokens,
            block_shape=None,
        )

        if hidden_states.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif hidden_states.dtype == torch.float16:
            compute_type = tl.float16
        elif hidden_states.dtype == torch.float32:
            compute_type = tl.float32
        else:
            raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

        intermediate_cache1 = _resize_cache(workspace2, (num_tokens, top_k_num, N))
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        intermediate_cache2 = _resize_cache(
            workspace13, (num_tokens * top_k_num, activation_out_dim)
        )
        intermediate_cache3 = _resize_cache(workspace2, (num_tokens, top_k_num, K))

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids,
            config["BLOCK_SIZE_M"],
            global_num_experts,
            expert_map,
        )

        # Activation NVFP4 QDQ (once for W13).
        hidden_states_qdq, _ = moe_kernel_quantize_input(
            A=hidden_states,
            A_scale=self.quant_config.a1_gscale,
            quant_dtype="nvfp4",
            per_act_token_quant=False,
            quantization_emulation=True,
        )

        invoke_fused_moe_nvfp4_emulation_kernel(
            hidden_states_qdq,
            w1,
            intermediate_cache1,
            self.w1_scale_val,
            self.quant_config.a1_gscale,
            self.quant_config.g1_alphas,
            None,  # topk_weights — applied after w2
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,  # mul_routed_weight
            top_k_num,
            config,
            compute_type=compute_type,
        )

        self.activation(
            activation, intermediate_cache2, intermediate_cache1.view(-1, N)
        )

        # Activation NVFP4 QDQ (once for W2).
        intermediate_cache2_qdq, _ = moe_kernel_quantize_input(
            A=intermediate_cache2,
            A_scale=self.quant_config.a2_gscale,
            quant_dtype="nvfp4",
            per_act_token_quant=False,
            quantization_emulation=True,
        )

        invoke_fused_moe_nvfp4_emulation_kernel(
            intermediate_cache2_qdq,
            w2,
            intermediate_cache3,
            self.w2_scale_val,
            self.quant_config.a2_gscale,
            self.quant_config.g2_alphas,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
        )

        self.moe_sum(intermediate_cache3, output)
