"""Task 8: A800 NVFP4 backend identity + packed residency contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
VLLM_ROOT = REPO_ROOT / "3rdparty" / "vllm"
if str(VLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VLLM_ROOT))


def _emulation_vllm_config():
    from vllm.config import VllmConfig
    from vllm.config.kernel import KernelConfig

    return VllmConfig(
        kernel_config=KernelConfig(
            linear_backend="emulation",
            moe_backend="emulation",
        )
    )


def test_kernel_config_allows_emulation_backends():
    from vllm.config.kernel import KernelConfig

    cfg = KernelConfig(linear_backend="emulation", moe_backend="emulation")
    assert cfg.linear_backend == "emulation"
    assert cfg.moe_backend == "emulation"


def test_dense_modelopt_and_ct_use_emulation_kernel_not_marlin():
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.kernels.linear.nvfp4.emulation import (
        EmulationNvFp4LinearKernel,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import (
        CompressedTensorsW4A4Fp4,
    )
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
        ModelOptNvFp4LinearMethod,
    )

    with set_current_vllm_config(_emulation_vllm_config()):
        method = ModelOptNvFp4LinearMethod(
            ModelOptNvFp4Config(
                is_checkpoint_nvfp4_serialized=True,
                kv_cache_quant_algo=None,
                exclude_modules=[],
                group_size=16,
            )
        )
        scheme = CompressedTensorsW4A4Fp4()

    assert isinstance(method.kernel, EmulationNvFp4LinearKernel)
    assert isinstance(scheme.kernel, EmulationNvFp4LinearKernel)
    assert "Marlin" not in type(method.kernel).__name__
    assert "Marlin" not in type(scheme.kernel).__name__


def test_moe_modelopt_and_ct_use_emulation_experts_not_marlin():
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
        Nvfp4QuantizationEmulationTritonExperts,
    )
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        CompressedTensorsW4A4Nvfp4MoEMethod,
    )
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
        ModelOptNvFp4FusedMoE,
    )

    moe_config = FusedMoEConfig(
        num_experts=2,
        experts_per_token=1,
        hidden_dim=64,
        intermediate_size_per_partition=32,
        num_local_experts=2,
        num_logical_experts=2,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.TopK,
        moe_backend="emulation",
    )

    with set_current_vllm_config(_emulation_vllm_config()):
        modelopt = ModelOptNvFp4FusedMoE(
            ModelOptNvFp4Config(
                is_checkpoint_nvfp4_serialized=True,
                kv_cache_quant_algo=None,
                exclude_modules=[],
                group_size=16,
            ),
            moe_config,
        )
        ct = CompressedTensorsW4A4Nvfp4MoEMethod(moe_config)

    assert modelopt.nvfp4_backend == NvFp4MoeBackend.EMULATION
    assert ct.nvfp4_backend == NvFp4MoeBackend.EMULATION
    assert modelopt.experts_cls is Nvfp4QuantizationEmulationTritonExperts
    assert ct.experts_cls is Nvfp4QuantizationEmulationTritonExperts
    assert "Marlin" not in modelopt.experts_cls.__name__
    assert "Marlin" not in ct.experts_cls.__name__


def _assert_no_full_logical_bf16_weight(layer: torch.nn.Module, *, n: int, k: int):
    """No persistent Parameter may match the full logical (n, k) BF16/FP16 weight."""
    forbidden = {(n, k), (k, n)}
    for name, param in layer.named_parameters(recurse=False):
        if param.dtype not in (torch.bfloat16, torch.float16):
            continue
        assert tuple(param.shape) not in forbidden, (
            f"persistent {param.dtype} Parameter {name} has full logical weight "
            f"shape {tuple(param.shape)}"
        )


@pytest.mark.parametrize("scheme", ["modelopt", "ct"])
def test_dense_packed_weight_residency(scheme: str):
    from torch.nn.parameter import Parameter

    from vllm.config import set_current_vllm_config
    from vllm.model_executor.kernels.linear.nvfp4.emulation import (
        EmulationNvFp4LinearKernel,
    )
    from vllm.model_executor.kernels.linear.nvfp4.select import (
        init_nvfp4_linear_kernel,
    )

    n, k, group = 32, 64, 16
    weight_u8 = torch.zeros(n, k // 2, dtype=torch.uint8)
    weight_scale = torch.ones(n, k // group, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    input_gs = 0.05
    weight_gs = 0.05

    with set_current_vllm_config(_emulation_vllm_config()):
        kernel = init_nvfp4_linear_kernel()
        assert isinstance(kernel, EmulationNvFp4LinearKernel)

        layer = torch.nn.Module()
        if scheme == "modelopt":
            layer.weight = Parameter(weight_u8.clone(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale.clone(), requires_grad=False)
            layer.input_scale = Parameter(
                torch.tensor([input_gs], dtype=torch.float32), requires_grad=False
            )
            layer.weight_scale_2 = Parameter(
                torch.tensor([weight_gs], dtype=torch.float32), requires_grad=False
            )
            packed_before = layer.weight.data.clone()
            input_global_scale = layer.input_scale.max().to(torch.float32)
            layer.input_global_scale = Parameter(
                input_global_scale, requires_grad=False
            )
            del layer.input_scale
            weight_global_scale = layer.weight_scale_2.max().to(torch.float32)
            layer.weight_global_scale = Parameter(
                weight_global_scale, requires_grad=False
            )
            del layer.weight_scale_2
            layer.alpha = Parameter(
                layer.input_global_scale * layer.weight_global_scale,
                requires_grad=False,
            )
            layer.input_global_scale_inv = Parameter(
                (1.0 / layer.input_global_scale).to(torch.float32),
                requires_grad=False,
            )
            kernel.process_weights_after_loading(layer)
            assert layer.weight.dtype == torch.uint8
            assert layer.weight.shape == (n, k // 2)
            torch.testing.assert_close(
                layer.weight.data, packed_before, atol=0, rtol=0
            )
        else:
            layer.weight_packed = Parameter(weight_u8.clone(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale.clone(), requires_grad=False)
            layer.input_global_scale = Parameter(
                torch.tensor([1.0 / input_gs], dtype=torch.float32),
                requires_grad=False,
            )
            layer.weight_global_scale = Parameter(
                torch.tensor([1.0 / weight_gs], dtype=torch.float32),
                requires_grad=False,
            )
            packed_before = layer.weight_packed.data.clone()
            layer.weight = layer.weight_packed
            del layer.weight_packed
            input_global_scale_inv = layer.input_global_scale.max().to(torch.float32)
            layer.input_global_scale = Parameter(
                (1.0 / input_global_scale_inv).to(torch.float32), requires_grad=False
            )
            weight_global_scale = layer.weight_global_scale.max().to(torch.float32)
            layer.weight_global_scale = Parameter(
                1.0 / weight_global_scale, requires_grad=False
            )
            layer.input_global_scale_inv = Parameter(
                input_global_scale_inv, requires_grad=False
            )
            layer.alpha = Parameter(
                layer.input_global_scale * layer.weight_global_scale,
                requires_grad=False,
            )
            kernel.process_weights_after_loading(layer)
            assert layer.weight.dtype == torch.uint8
            assert layer.weight.shape == (n, k // 2)
            torch.testing.assert_close(
                layer.weight.data, packed_before, atol=0, rtol=0
            )

        _assert_no_full_logical_bf16_weight(layer, n=n, k=k)


def test_moe_emulation_keeps_packed_uint8_weights():
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        NvFp4MoeBackend,
        convert_to_nvfp4_moe_kernel_format,
    )

    e, hidden, inter = 2, 64, 32
    w13 = torch.zeros(e, 2 * inter, hidden // 2, dtype=torch.uint8)
    w13_scale = torch.zeros(e, 2 * inter, hidden // 16, dtype=torch.float8_e4m3fn)
    w2 = torch.zeros(e, hidden, inter // 2, dtype=torch.uint8)
    w2_scale = torch.zeros(e, hidden, inter // 16, dtype=torch.float8_e4m3fn)
    w13_gscale = torch.ones(e, dtype=torch.float32)
    w2_gscale = torch.ones(e, dtype=torch.float32)
    a13 = torch.tensor([1e-4, 2e-4], dtype=torch.float32)
    a2 = torch.tensor([3e-4, 1e-4], dtype=torch.float32)

    class _Layer:
        pass

    (
        out_w13,
        _out_w13_s,
        _out_w13_g,
        _a13_o,
        out_w2,
        _out_w2_s,
        _out_w2_g,
        _a2_o,
    ) = convert_to_nvfp4_moe_kernel_format(
        nvfp4_backend=NvFp4MoeBackend.EMULATION,
        layer=_Layer(),
        w13=w13,
        w13_scale=w13_scale,
        w13_scale_2=w13_gscale,
        a13_scale=a13,
        w2=w2,
        w2_scale=w2_scale,
        w2_scale_2=w2_gscale,
        a2_scale=a2,
        is_act_and_mul=True,
    )

    assert out_w13.dtype == torch.uint8
    assert out_w13.shape == (e, 2 * inter, hidden // 2)
    assert out_w2.dtype == torch.uint8
    assert out_w2.shape == (e, hidden, inter // 2)
    # Full logical BF16 shapes would be (e, 2*inter, hidden) / (e, hidden, inter).
    assert out_w13.shape != (e, 2 * inter, hidden)
    assert out_w2.shape != (e, hidden, inter)
