"""Packed MXFP4 W4A4 experts on SM80, isolated from active NVFP4 inference."""
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
    fused_moe_nvfp4_emulation_kernel, moe_align_block_size, try_get_optimal_moe_config,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import _round_to_fp4
from vllm.model_executor.models.minimax_m3.quantization import (
    MiniMaxM3QuantizationConfig, swiglu_oai,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.triton_utils import triton, tl

from .format import decode_scales


@triton.jit
def _qdq_kernel(x_ptr, out_ptr, K: tl.constexpr, GROUPS: tl.constexpr):
    row = tl.program_id(0)
    group = tl.program_id(1)
    offset = row * K + group * 32 + tl.arange(0, 32)
    x = tl.load(x_ptr + offset).to(tl.float32)
    maximum = tl.max(tl.abs(x), 0)
    exponent = tl.floor(tl.log2(tl.maximum(maximum, 2. ** -125))) - 2.
    exponent = tl.where(maximum == 0., 0., tl.minimum(127., tl.maximum(-127., exponent)))
    scale = tl.exp2(exponent)
    quantized = _round_to_fp4(x / scale)
    tl.store(out_ptr + offset, quantized * scale)


def activation_qdq(x):
    if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] % 32:
        raise ValueError('MXFP4 activations require a BF16 matrix with K divisible by 32')
    if not x.is_cuda:
        from .format import quantize, dequantize
        return dequantize(*quantize(x))
    x = x.contiguous()
    output = torch.empty_like(x)
    if x.numel():
        _qdq_kernel[(x.shape[0], x.shape[1] // 32)](x, output, x.shape[1], x.shape[1] // 32, num_warps=1)
    return output


def packed_gemm(a, weight, scales, ones, output, ids, experts, padded,
                top_k, route_weights, multiply, config):
    # The existing E2M1 GEMM is parameterized by group size and accepts FP32
    # scales. Use group 32 and decoded E8M0 values; the NVFP4 wrapper hardcodes
    # group 16/E4M3 and must never be called for this format.
    n, k = weight.shape[1], a.shape[1]
    em = ids.numel()
    if a.shape[0] < config['BLOCK_SIZE_M']:
        em = min(em, a.shape[0] * top_k * config['BLOCK_SIZE_M'])
    grid = (triton.cdiv(em, config['BLOCK_SIZE_M']) * triton.cdiv(n, config['BLOCK_SIZE_N']),)
    fused_moe_nvfp4_emulation_kernel[grid](
        a, weight, output, scales, ones, route_weights, ids, experts, padded,
        n, k, em, a.shape[0] * top_k, a.stride(0), a.stride(1),
        weight.stride(0), weight.stride(2), weight.stride(1),
        output.stride(1), output.stride(2), scales.stride(0), scales.stride(2), scales.stride(1),
        block_k_diviable=k % config['BLOCK_SIZE_K'] == 0,
        BLOCK_SIZE_M=config['BLOCK_SIZE_M'], BLOCK_SIZE_N=config['BLOCK_SIZE_N'],
        BLOCK_SIZE_K=config['BLOCK_SIZE_K'], GROUP_SIZE_M=config['GROUP_SIZE_M'],
        MUL_ROUTED_WEIGHT=multiply, top_k=top_k, compute_type=tl.bfloat16,
        group_size=32, SCALES_ARE_FP32=True)


def routed_experts(x, w13, w2, s13, s2, ones, weights, selected, expert_map=None):
    tokens, hidden = x.shape
    if tokens == 0:
        return torch.empty_like(x)
    experts, width = w13.shape[:2]
    top_k = selected.shape[1]
    config = try_get_optimal_moe_config(w13.shape, w2.shape, top_k, None, tokens, block_shape=None)
    sorted_ids, expert_ids, padded = moe_align_block_size(selected, config['BLOCK_SIZE_M'], experts, expert_map)
    first = torch.empty(tokens, top_k, width, dtype=x.dtype, device=x.device)
    second = torch.empty(tokens, top_k, hidden, dtype=x.dtype, device=x.device)
    packed_gemm(activation_qdq(x), w13, s13, ones, first,
                sorted_ids, expert_ids, padded, top_k, None, False, config)
    activated = swiglu_oai(first.view(-1, width)).contiguous()
    packed_gemm(activation_qdq(activated), w2, s2, ones, second,
                sorted_ids, expert_ids, padded, 1, weights, True, config)
    output = torch.empty_like(x)
    ops.moe_sum(second, output)
    return output


class MiniMaxM3ModelOptMXFP4FusedMoE(FusedMoEMethodBase):
    def __init__(self, moe):
        super().__init__(moe)
        if moe.moe_parallel_config.ep_size != 1 or not moe.is_act_and_mul:
            raise ValueError('This MiniMax MXFP4 path requires TP experts and gated activation')

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype, **attrs):
        if hidden_size % 32 or intermediate_size_per_partition % 32:
            raise ValueError('MXFP4 TP partitions must align to groups of 32')
        shapes = {'w13_weight': (num_experts, 2 * intermediate_size_per_partition, hidden_size // 2),
                  'w2_weight': (num_experts, hidden_size, intermediate_size_per_partition // 2),
                  'w13_weight_scale': (num_experts, 2 * intermediate_size_per_partition, hidden_size // 32),
                  'w2_weight_scale': (num_experts, hidden_size, intermediate_size_per_partition // 32)}
        for name, shape in shapes.items():
            parameter = ModelWeightParameter(data=torch.empty(shape, dtype=torch.uint8),
                input_dim=1, output_dim=2, weight_loader=attrs.get('weight_loader'))
            set_weight_attrs(parameter, {'quant_method': 'block'})
            layer.register_parameter(name, parameter)

    def process_weights_after_loading(self, layer):
        self.s13 = decode_scales(layer.w13_weight_scale).contiguous()
        self.s2 = decode_scales(layer.w2_weight_scale).contiguous()
        self.ones = torch.ones(layer.w13_weight.shape[0], device=layer.w13_weight.device, dtype=torch.float32)

    def get_fused_moe_quant_config(self, layer):
        # The custom method owns both quantization boundaries. The surrounding
        # routing/communication operates on BF16 and needs no modular quantizer.
        return None

    def maybe_make_prepare_finalize(self, routing_tables=None):
        return None

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts_input):
        if layer.apply_router_weight_on_input:
            raise ValueError('MiniMax applies routing weights after the second GEMM')
        return routed_experts(x, layer.w13_weight, layer.w2_weight,
            self.s13, self.s2, self.ones, topk_weights, topk_ids, layer.expert_map)


class MiniMaxM3MXFP4Config(MiniMaxM3QuantizationConfig):
    def get_name(self):
        return 'minimax_m3_mxfp4'

    @classmethod
    def override_quantization_method(cls, config, user_quant):
        if (config.get('quant_method') == 'mxfp8' and config.get('moe_quant_algo') == 'MXFP4'
                and config.get('moe_quant_format') == 'mxfp4-pack-quantized'):
            return 'minimax_m3_mxfp4'
        return None

    @classmethod
    def from_config(cls, config):
        if (cls.override_quantization_method(config, None) is None
                or config.get('group_size') != 32 or config.get('weight_block_size') != [1, 32]
                or config.get('moe_activation_quant_algo') != 'MXFP4'):
            raise ValueError('Expected the converted MiniMax MXFP4 W4A4 checkpoint')
        return cls(config.get('ignored_layers', []))

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, FusedMoE):
            return MiniMaxM3ModelOptMXFP4FusedMoE(layer.moe_config)
        return super().get_quant_method(layer, prefix)
