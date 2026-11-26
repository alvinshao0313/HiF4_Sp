"""Opt-in vLLM W4A4 emulation for the BF16 operand checkpoints."""
from pathlib import Path
import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from vllm import _custom_ops as ops
from vllm.distributed import get_tensor_model_parallel_rank, tensor_model_parallel_all_gather
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.activation import apply_moe_activation
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel, try_get_optimal_moe_config
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.linear import LinearBase, RowParallelLinear, UnquantizedLinearMethod, register_weight_loader_v2_supported_method
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8KVCacheMethod, ModelOptFp8LinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.triton_utils import tl

from .operands import operand
from .scale_methods import GROUP_SIZE, METHOD_TARGET


@register_weight_loader_v2_supported_method
class TargetLinear(UnquantizedLinearMethod):
    def __init__(self, config, prefix=None):
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.divisor = None

    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)
        self.divisor = self.config.divisor(self.prefix, layer.weight.device)

    def apply(self, layer, x, bias=None):
        a = self.config.activation(x, self.divisor, sharded=isinstance(layer, RowParallelLinear))
        return F.linear(a, layer.weight, bias)


@register_weight_loader_v2_supported_method
class DualScaleLinear(TargetLinear):
    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size,
                       output_size, params_dtype, **attrs):
        super().create_weights(layer, input_size_per_partition, output_partition_sizes, input_size,
                               output_size, params_dtype, **attrs)
        group = GROUP_SIZE[self.config.method]
        if input_size_per_partition != input_size and input_size_per_partition % group:
            raise ValueError("Dual-scale row parallelism requires outer-group aligned K partitions")
        groups = (input_size_per_partition + group - 1) // group
        # Some model constructors replace the BF16 weight loader after this
        # method. Apply exactly the same row ordering to its FP32 sidecar.
        def load_outer(param, value, *shards):
            if value.dtype != torch.float32:
                raise ValueError("Expected FP32 outer scales")
            return layer.weight.weight_loader(param, value, *shards)

        values = torch.full((sum(output_partition_sizes), groups), float("nan"), dtype=torch.float32)
        layer.register_parameter("weight_scale_fp32", ModelWeightParameter(
            data=values, input_dim=1, output_dim=0, weight_loader=load_outer))

    def process_weights_after_loading(self, layer):
        if not torch.isfinite(layer.weight_scale_fp32).all():
            raise ValueError(f"Missing FP32 outer scales in {self.prefix}")
        # Keep the normalized BF16 weights and FP32 scales as two operands.
        # Their product is applied to each group partial, before output rounding.

    def apply(self, layer, x, bias=None):
        from .dualscale_linear import dualscale_linear
        return dualscale_linear(x, layer.weight, layer.weight_scale_fp32, GROUP_SIZE[self.config.method], bias)


class TargetExperts(FusedMoEMethodBase):
    # Exported weights have the same layout as the ordinary BF16 vLLM loader.
    create_weights = UnquantizedFusedMoEMethod.create_weights

    def __init__(self, moe, config, prefix):
        super().__init__(moe)
        if moe.moe_parallel_config.ep_size != 1 or moe.has_bias:
            raise ValueError("Paper TP reference supports bias-free routed experts without EP")
        self.config = config
        self.prefix = prefix
        self.up_divisors = self.down_divisors = None

    def process_weights_after_loading(self, layer):
        if self.config.method not in ("awq", "smoothquant"):
            return
        up, down = [], []
        for i in range(layer.w13_weight.shape[0]):
            base = f"{self.prefix}.{i}"
            value = self.config.divisor(base + ".up_proj", layer.w13_weight.device)
            if self.moe.is_act_and_mul:
                gate = self.config.divisor(base + ".gate_proj", value.device)
                if not torch.equal(value, gate):
                    raise ValueError(f"Gate and up scales differ in {base}")
            up.append(value)
            down.append(self.config.divisor(base + ".down_proj", value.device))
        self.up_divisors, self.down_divisors = torch.stack(up), torch.stack(down)

    def maybe_make_prepare_finalize(self, routing_tables=None):
        return None

    def get_fused_moe_quant_config(self, layer):
        return None

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts_input):
        if layer.apply_router_weight_on_input:
            raise ValueError("Expected routing weights on the second GEMM")
        tokens, hidden = x.shape
        if not tokens:
            return torch.empty_like(x)
        w13, w2 = layer.w13_weight, layer.w2_weight
        experts, width, _ = w13.shape
        k = topk_ids.shape[1]
        config = try_get_optimal_moe_config(w13.shape, w2.shape, k, None, tokens, block_shape=None)
        ids, experts_ids, padded = moe_align_block_size(topk_ids, config["BLOCK_SIZE_M"], experts, layer.expert_map)

        def gemm(a, weight, out_width, top_k, routing_weights):
            out = x.new_empty(tokens, k, out_width)
            invoke_fused_moe_triton_kernel(
                A=a, B=weight, C=out, A_scale=None, B_scale=None, topk_weights=routing_weights,
                sorted_token_ids=ids, expert_ids=experts_ids, num_tokens_post_padded=padded,
                mul_routed_weight=routing_weights is not None, top_k=top_k, config=config,
                compute_type=tl.bfloat16, use_fp8_w8a8=False, use_int8_w8a8=False,
                use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False)
            return out

        down_scale = None
        if self.up_divisors is None:
            first = gemm(self.config.activation(x), w13, width, k, None)
        else:
            selected = topk_ids.reshape(-1).long()
            routed = x[:, None, :].expand(-1, k, -1).reshape(-1, hidden)
            a = self.config.activation(routed, self.up_divisors.index_select(0, selected))
            first = gemm(a, w13, width, 1, None)
            down_scale = self.down_divisors.index_select(0, selected)
        intermediate = x.new_empty(tokens * k, w2.shape[-1])
        apply_moe_activation(layer.activation, intermediate, first.view(-1, width))
        second = gemm(self.config.activation(intermediate, down_scale, sharded=True), w2, hidden, 1, topk_weights)
        output = torch.empty_like(x)
        ops.moe_sum(second, output)
        return output


def bind_fused_scales(layer, names):
    # GDN stores Q/K/V together and Z separately. One stored scalar must fill
    # all three logical entries before ModelOpt combines their scales.
    for name in names:
        param = getattr(layer, name)
        default_loader = param.weight_loader

        def load(param, value, *shards, base=default_loader):
            if len(shards) == 1 and isinstance(shards[0], tuple):
                for shard in shards[0]:
                    base(param, value, shard)
            elif (not shards or shards == (None,)) and value.numel() == 1 and param.numel() > 1:
                # Nemotron Mamba stores one scale for its entire fused in_proj.
                param.data.copy_(value.reshape(()).expand_as(param))
            elif isinstance(layer, ParallelLMHead) and value.numel() == param.numel() == 1:
                base(param, value.reshape_as(param), *shards)
            else:
                base(param, value, *shards)

        param.weight_loader = load


@register_weight_loader_v2_supported_method
class StaticFP8Linear(ModelOptFp8LinearMethod):
    """Keep E4M3 operands and static FP32 scales on SM80 via BF16 GEMM."""

    def __init__(self, config):
        self.quant_config = config

    def create_weights(self, layer, *args, **kwargs):
        super().create_weights(layer, *args, **kwargs)
        bind_fused_scales(layer, ("input_scale", "weight_scale"))

    def process_weights_after_loading(self, layer):
        if not all(torch.isfinite(t).all() and (t > 0).all() for t in (layer.input_scale, layer.weight_scale)):
            raise ValueError("Missing static FP8 scale in packed projection")
        super().process_weights_after_loading(layer)
        # Every finite E4M3 number is exactly representable in BF16.
        layer.weight = torch.nn.Parameter(layer.weight.bfloat16(), requires_grad=False)

    def apply(self, layer, x, bias=None):
        encoded = (x.float() / layer.input_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        a = encoded.reshape(-1, x.shape[-1]).bfloat16()
        acc = torch.mm(a, layer.weight, out_dtype=torch.float32) if a.is_cuda else a.float() @ layer.weight.float()
        y = acc * (layer.input_scale * layer.weight_scale)
        if bias is not None:
            y = y + bias
        return y.to(x.dtype).reshape(*x.shape[:-1], -1)


class PaperConfig(QuantizationConfig):
    def __init__(self, config):
        super().__init__()
        if config.get("storage") != "bf16_dequantized_operands":
            raise ValueError("Expected full-scope paper QDQ checkpoint")
        self.target, self.method, self.seed = config["target"], config["method"], config["seed"]
        if self.target not in ("mxfp4", "hif4") or self.method not in ("direct", "brq", "brq_fused", "gptq", "awq", "smoothquant", *METHOD_TARGET):
            raise ValueError("Unknown target or baseline")
        if self.method in METHOD_TARGET and self.target != METHOD_TARGET[self.method]:
            raise ValueError(f"{self.method} requires {METHOD_TARGET[self.method]}")
        if (self.seed is not None) != (self.method in ("brq", "brq_fused")):
            raise ValueError("Random rotation and its seed must be specified together")
        self._init_scope(config)

    def _init_scope(self, config):
        self.kv = config["kv_cache_quant_algo"]
        self.scope = {key + ".weight": "target" for key in config["nvfp4"]}
        self.scope.update({key + ".weight": "fp8" for key in config["fp8"]})
        self.source_names = {key: key.removesuffix(".weight") for key in self.scope}
        self.quantize_lm_head = "lm_head.weight" in self.scope
        self.is_checkpoint_fp8_serialized = bool(config["fp8"])
        self._set_modules()

    def _set_modules(self):
        self.modules = {name.removesuffix(".weight"): kind for name, kind in self.scope.items()}
        self.experts = {name.split(".experts.")[0] + ".experts"
                        for name, kind in self.modules.items() if ".experts." in name and kind == "target"}

    def apply_vllm_mapper(self, mapper):
        self.scope = mapper.apply_dict(self.scope)
        self.source_names = mapper.apply_dict(self.source_names)
        self._set_modules()

    def maybe_update_config(self, model_name):
        if getattr(self, "method", None) in ("awq", "smoothquant"):
            self.scale_path = Path(model_name) / "paper_metadata" / "input_divisors.safetensors"
            if not self.scale_path.is_file():
                raise FileNotFoundError(self.scale_path)
            self._divisors = None

    def divisor(self, prefix, device):
        if self.method not in ("awq", "smoothquant"):
            return None
        # Load once in each worker. Parsing a model-wide header for every
        # routed expert would make full MoE initialization quadratic.
        if self._divisors is None:
            self._divisors = load_file(self.scale_path, device="cpu")
        values = [self._divisors[self.source_names[name + ".weight"]]
                  for name in self.logical_names(prefix)]
        if not values or any(not torch.equal(values[0], other) for other in values[1:]):
            raise ValueError(f"Missing or incompatible channel scales for {prefix}")
        return values[0].to(device)

    @classmethod
    def from_config(cls, config):
        return cls(config)

    @classmethod
    def get_name(cls):
        return "paper_fp4"

    def get_supported_act_dtypes(self):
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 80

    @staticmethod
    def get_config_filenames():
        return []

    def activation(self, x, divisor=None, sharded=False):
        size = 64 if self.target == "hif4" else 32
        # Both HiF4 scale searches operate exclusively on stored weights.
        transform = self.method if self.method in ("brq", "uos") else "direct"
        if self.method in GROUP_SIZE:
            raise ValueError("Dual-scale activation must retain FP32 outer scales for the grouped GEMM")
        if sharded and x.shape[-1] % size:
            # Nemotron-9B MLP and 30B routed down projections split a HiF4
            # block across TP2. Quantize the original group before local GEMM.
            width = x.shape[-1]
            full = tensor_model_parallel_all_gather(x, dim=-1)
            full = full if divisor is None else full.float() / divisor
            quantized = operand(full, self.target, transform, self.seed)
            return quantized.narrow(-1, get_tensor_model_parallel_rank() * width, width).contiguous()
        if divisor is not None:
            width = x.shape[-1]
            if sharded and divisor.shape[-1] != width:
                divisor = divisor.narrow(-1, get_tensor_model_parallel_rank() * width, width)
            x = x.float() / divisor
        return operand(x, self.target, transform, self.seed)

    def logical_names(self, prefix):
        if prefix in self.modules:
            return [prefix]
        parent, _, name = prefix.rpartition(".")
        children = self.packed_modules_mapping.get(name, [])
        return [f"{parent}.{child}" for child in children]

    def kind(self, prefix):
        kinds = {self.modules.get(child) for child in self.logical_names(prefix)}
        if len(kinds) > 1:
            raise ValueError(f"Fused projection has inconsistent coverage: {prefix}, {kinds}")
        return next(iter(kinds), None)

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, Attention):
            return ModelOptFp8KVCacheMethod(self) if self.kv else None
        if isinstance(layer, FusedMoE):
            if prefix not in self.experts:
                raise ValueError(f"Missing expert coverage for {prefix}")
            if self.method in GROUP_SIZE:
                raise ValueError("Paper dual-scale linear supports the two dense Table 3 models")
            return TargetExperts(layer.moe_config, self, prefix)
        if isinstance(layer, (LinearBase, ParallelLMHead)):
            kind = self.kind(prefix)
            if kind == "target":
                if self.method in GROUP_SIZE:
                    return DualScaleLinear(self, prefix)
                return TargetLinear(self, prefix)
            if kind == "fp8":
                return StaticFP8Linear(self)
            return UnquantizedLinearMethod() if isinstance(layer, LinearBase) else None
        return None
