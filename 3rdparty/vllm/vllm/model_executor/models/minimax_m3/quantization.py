# SPDX-License-Identifier: Apache-2.0
"""M3.1 mixed storage, isolated from the existing Qwen/HiF4 quantizers.

Checkpoint conventions follow upstream PR #64111 (dbd49af22235).
The SM80 path keeps the routed weights packed and uses the existing emulation
GEMMs, with M3.1's per-token activation scales and uninterleaved SwiGLU.
"""

import torch
from torch import nn

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
    Nvfp4QuantizationEmulationTritonExperts,
    invoke_fused_moe_nvfp4_emulation_kernel,
    try_get_optimal_moe_config,
    _resize_cache,
    moe_align_block_size,
)
from vllm.triton_utils import tl
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMxFp8Config,
    ModelOptMxFp8LinearMethod,
    ModelOptNvFp4Config,
    ModelOptNvFp4FusedMoE,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    Mxfp8LinearBackend,
    Mxfp8LinearOp,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    ref_nvfp4_quant,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped

logger = init_logger(__name__)


def swiglu_oai(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    gate = gate.float().clamp(max=7.0)
    up = up.float().clamp(min=-7.0, max=7.0)
    return (gate * torch.sigmoid(1.702 * gate) * (up + 1.0)).to(x.dtype)


def nvfp4_per_token_qdq(x: torch.Tensor) -> torch.Tensor:
    """NVFP4 QDQ with one global multiplier per token, not per tensor.

    Both routed GEMMs use this recipe in upstream's per-token NVFP4 path.
    Calling the old scalar-only Triton QDQ with a vector would silently use
    the first token's scale, so use the existing reference arithmetic here.
    """
    if x.ndim != 2 or x.shape[-1] % 16:
        raise ValueError("NVFP4 input must be a matrix with K divisible by 16")
    if x.shape[0] == 0:
        return x
    maximum = x.float().abs().amax(-1, keepdim=True)
    global_scale = torch.where(maximum == 0, torch.ones_like(maximum),
                               2688.0 / maximum).view(-1, 1, 1)
    values, scales = ref_nvfp4_quant(x, global_scale, 16)
    values = values.view(x.shape[0], -1, 16)
    return (values * (scales.unsqueeze(-1) / global_scale)).reshape_as(x).to(x.dtype)


class MiniMaxM3NVFP4Experts(Nvfp4QuantizationEmulationTritonExperts):
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
        hidden_states_qdq = nvfp4_per_token_qdq(hidden_states)

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
        intermediate_cache2_qdq = nvfp4_per_token_qdq(intermediate_cache2)

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

    def activation(self, activation, output, input):
        if activation != MoEActivation.SWIGLUOAI:
            raise ValueError("MiniMax M3.1 requires SwiGLU-OAI")
        output.copy_(swiglu_oai(input))


class MiniMaxM3ModelOptNvFp4FusedMoE(ModelOptNvFp4FusedMoE):
    """Keep the ModelOpt name required by v0.19's expert-loader dispatch."""

    def __init__(self, moe_config):
        config = ModelOptNvFp4Config(True, None, [], 16)
        super().__init__(config, moe_config)
        if self.nvfp4_backend != NvFp4MoeBackend.EMULATION:
            raise ValueError("MiniMax M3.1 backport requires --moe_backend emulation")
        self.experts_cls = MiniMaxM3NVFP4Experts

    def process_weights_after_loading(self, layer):
        if not torch.equal(layer.w13_weight_scale_2[:, 0], layer.w13_weight_scale_2[:, 1]):
            raise ValueError("M3.1 gate/up global scales differ; cannot share W13 scale")
        super().process_weights_after_loading(layer)


class MiniMaxM3MXFP8Linear(ModelOptMxFp8LinearMethod):
    def __init__(self):
        self.quant_config = ModelOptMxFp8Config(True, None, [])
        self.backend = Mxfp8LinearBackend.EMULATION
        self.mxfp8_linear_op = Mxfp8LinearOp(self.backend)
        logger.info_once("MiniMax M3.1 MXFP8 dense layers use BF16 software emulation")


class MiniMaxM3QuantizationConfig(QuantizationConfig):
    def __init__(self, ignored_layers):
        super().__init__()
        self.ignored_layers = ignored_layers

    def get_name(self):
        return "minimax_m3_mixed"

    def get_supported_act_dtypes(self):
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 80

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant):
        if (hf_quant_cfg.get("quant_method") == "mxfp8"
                and hf_quant_cfg.get("moe_quant_algo") == "NVFP4"
                and hf_quant_cfg.get("moe_quant_format") == "nvfp4-pack-quantized"):
            return "minimax_m3_mixed"
        return None

    @classmethod
    def from_config(cls, config):
        if cls.override_quantization_method(config, None) is None:
            raise ValueError("Expected MiniMax M3.1 MXFP8 + packed NVFP4 checkpoint")
        if config.get("group_size") != 16 or config.get("weight_block_size") != [1, 32]:
            raise ValueError("Expected NVFP4 group 16 and MXFP8 block [1, 32]")
        return cls(config.get("ignored_layers", []))

    def get_quant_method(self, layer: nn.Module, prefix: str):
        if isinstance(layer, FusedMoE):
            return MiniMaxM3ModelOptNvFp4FusedMoE(layer.moe_config)
        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers, self.packed_modules_mapping,
                                skip_with_substr=True):
                return UnquantizedLinearMethod()
            return MiniMaxM3MXFP8Linear()
        return None
