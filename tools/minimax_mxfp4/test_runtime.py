import json
from pathlib import Path

import torch

from .format import dequantize, quantize
from .runtime import MiniMaxM3ModelOptMXFP4FusedMoE, activation_qdq


def test_real_moe_parameter_layout_and_e8m0_decoding(tmp_path):
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    config = FusedMoEConfig(num_experts=4, experts_per_token=4, hidden_dim=64,
        intermediate_size_per_partition=64, num_local_experts=4, num_logical_experts=4,
        activation=MoEActivation.SWIGLUOAI, device='cpu', routing_method=RoutingMethodType.DeepSeekV3,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(), in_dtype=torch.bfloat16)
    from vllm.distributed import init_distributed_environment, initialize_model_parallel, cleanup_dist_env_and_memory
    init_distributed_environment(world_size=1,rank=0,local_rank=0,
        distributed_init_method='file://' + str(tmp_path / 'gloo'),backend='gloo')
    try:
        from vllm.config import VllmConfig, set_current_vllm_config
        with set_current_vllm_config(VllmConfig()):
            initialize_model_parallel(1,1,backend='gloo')
        method = MiniMaxM3ModelOptMXFP4FusedMoE(config)
        layer = torch.nn.Module()
        method.create_weights(layer, 4, 64, 64, torch.bfloat16)
        assert layer.w13_weight.shape == (4, 128, 32)
        assert layer.w13_weight_scale.shape == (4, 128, 2)
        assert layer.w2_weight_scale.shape == (4, 64, 2)
        layer.w13_weight_scale.data.fill_(126)
        layer.w2_weight_scale.data.fill_(128)
        method.process_weights_after_loading(layer)
        assert torch.all(method.s13 == .5) and torch.all(method.s2 == 2.)
        assert all(p.dtype == torch.uint8 for p in layer.parameters())
        assert not any('global_scale' in name or 'input_scale' in name for name,_ in layer.named_parameters())
    finally:
        cleanup_dist_env_and_memory()


def test_registration_does_not_change_existing_quantizers_and_cpu_activation_grid():
    from vllm.model_executor.layers.quantization import get_quantization_config
    from .integration import register
    names = ['mxfp8', 'modelopt', 'compressed-tensors']
    previous = {name:get_quantization_config(name) for name in names}
    register()
    assert all(get_quantization_config(name) is cls for name,cls in previous.items())
    values = torch.randn(3, 96).bfloat16()
    torch.testing.assert_close(activation_qdq(values), dequantize(*quantize(values)), rtol=0, atol=0)
