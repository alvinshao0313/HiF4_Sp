"""Opt-in routed experts; native ModelOpt owns every other quantized module."""
import json
import os
from pathlib import Path
import re

import torch
from vllm import _custom_ops as ops
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.activation import MoEActivation, apply_moe_activation
from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import moe_align_block_size, try_get_optimal_moe_config
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs

from .artifacts import tensor_record
from .format import FORMAT_VERSION, SCALE_RULE, VARIANTS, decode_scales, r64_matrix
from .fht32 import fht32_matrix
from .kernels import activation_qdq, packed_gemm


def audit(event):
    root = os.environ.get("QWEN3_MXFP4_AUDIT_DIR")
    if root:
        path = Path(root)
        path.mkdir(parents=True, exist_ok=True)
        with (path / f"worker-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")


def routed_experts(x, w13, w2, s13, s2, ones, weights, selected, variant, expert_map=None, diagnostics=False):
    tokens, hidden = x.shape
    if tokens == 0:
        return torch.empty_like(x)
    experts, width = w13.shape[:2]
    top_k = selected.shape[1]
    config = try_get_optimal_moe_config(w13.shape, w2.shape, top_k, None, tokens, block_shape=None)
    sorted_ids, expert_ids, padded = moe_align_block_size(selected, config["BLOCK_SIZE_M"], experts, expert_map)
    first = torch.empty(tokens, top_k, width, dtype=x.dtype, device=x.device)
    second = torch.empty(tokens, top_k, hidden, dtype=x.dtype, device=x.device)
    a1 = activation_qdq(x, variant)
    packed_gemm(a1, w13, s13, ones, first, sorted_ids, expert_ids, padded, top_k, None, False, config)
    activated = torch.empty((tokens * top_k, width // 2), dtype=x.dtype, device=x.device)
    # Same native Qwen SILU(gate)*up custom op and BF16 output boundary.
    apply_moe_activation(MoEActivation.SILU, activated, first.view(-1, width))
    a2 = activation_qdq(activated, variant)
    packed_gemm(a2, w2, s2, ones, second, sorted_ids, expert_ids, padded, 1, weights, True, config)
    output = torch.empty_like(x)
    ops.moe_sum(second, output)
    if diagnostics:
        return output, {"a1": a1, "first": first, "activated": activated, "a2": a2, "second": second}
    return output


class Qwen3ModelOptMXFP4FusedMoE(FusedMoEMethodBase):
    def __init__(self, moe, variant, prefix):
        super().__init__(moe)
        self.variant, self.prefix = variant, prefix
        if moe.moe_parallel_config.ep_size != 1 or not moe.is_act_and_mul:
            raise ValueError("The Qwen MXFP4 experiment requires TP and gated SILU experts")

    def uses_weight_scale_2_pattern(self):
        # Select ModelOpt's FP4 loader branch. No scalar expert scales are stored.
        return True

    def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **attrs):
        if hidden_size != 2048 or intermediate_size_per_partition % 64 or num_experts != 128 or params_dtype != torch.bfloat16:
            raise ValueError("Unexpected Qwen3 routed-expert shape/dtype")
        shapes = {"w13_weight": (num_experts, 2 * intermediate_size_per_partition, hidden_size // 2),
                  "w2_weight": (num_experts, hidden_size, intermediate_size_per_partition // 2),
                  "w13_weight_scale": (num_experts, 2 * intermediate_size_per_partition, hidden_size // 32),
                  "w2_weight_scale": (num_experts, hidden_size, intermediate_size_per_partition // 32)}
        for name, shape in shapes.items():
            parameter = ModelWeightParameter(data=torch.empty(shape, dtype=torch.uint8),
                input_dim=1, output_dim=2, weight_loader=attrs.get("weight_loader"))
            set_weight_attrs(parameter, {"quant_method": "block"})
            layer.register_parameter(name, parameter)

    def process_weights_after_loading(self, layer):
        self.s13 = decode_scales(layer.w13_weight_scale).contiguous()
        self.s2 = decode_scales(layer.w2_weight_scale).contiguous()
        self.ones = torch.ones(layer.w13_weight.shape[0], device=layer.w13_weight.device, dtype=torch.float32)
        audit({"event": "loaded_experts", "prefix": self.prefix, "variant": self.variant,
               "tp_rank": get_tensor_model_parallel_rank(), "method": type(self).__name__,
               "tensors": {name: tensor_record(getattr(layer, name)) for name in
                           ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")}})

    def get_fused_moe_quant_config(self, layer):
        return None

    def maybe_make_prepare_finalize(self, routing_tables=None):
        return None

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts_input):
        if layer.apply_router_weight_on_input or topk_ids.shape[1] != 8:
            raise ValueError("Expected Qwen top-8 with routing weights after down projection")
        return routed_experts(x, layer.w13_weight, layer.w2_weight, self.s13, self.s2,
                             self.ones, topk_weights, topk_ids, self.variant, layer.expert_map)


class Qwen3RoutedMXFP4Config(ModelOptNvFp4Config):
    def get_name(self):
        return "qwen3_routed_mxfp4"

    @classmethod
    def override_quantization_method(cls, config, user_quant):
        if "qwen3_routed_mxfp4" in config:
            return "qwen3_routed_mxfp4"
        return None

    @classmethod
    def from_config(cls, config):
        marker = config.get("qwen3_routed_mxfp4", {})
        if (marker.get("format") != FORMAT_VERSION or marker.get("group_size") != 32
                or marker.get("variant") not in VARIANTS or marker.get("scale_rule") != SCALE_RULE
                or marker.get("r64_sha256") != tensor_record(r64_matrix())["sha256"]):
            raise ValueError("Missing/incompatible Qwen routed MXFP4 format metadata")
        if marker["variant"] == "fht32" and marker.get("fht32_sha256") != tensor_record(fht32_matrix())["sha256"]:
            raise ValueError("Missing/incompatible normalized Sylvester FHT32 matrix")
        instance = super().from_config(config)
        if not instance.is_checkpoint_nvfp4_serialized or instance.group_size != 16:
            raise ValueError("Non-expert NVFP4 group16 configuration must remain unchanged")
        instance.variant = marker["variant"]
        return instance

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, FusedMoE):
            match = re.fullmatch(r"model\.layers\.(\d+)\.mlp\.experts", prefix)
            if not match or not 0 <= int(match[1]) < 48:
                raise ValueError(f"Unexpected expert module: {prefix}")
            return Qwen3ModelOptMXFP4FusedMoE(layer.moe_config, self.variant, prefix)
        method = super().get_quant_method(layer, prefix)
        audit({"event": "non_expert_method", "prefix": prefix, "method": type(method).__name__})
        return method
