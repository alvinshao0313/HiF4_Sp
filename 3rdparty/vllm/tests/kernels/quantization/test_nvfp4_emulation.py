# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 emulation numerical primitive tests (v0.27.0 baseline).

MoE expert correctness tests are added in later backport tasks once
Nvfp4QuantizationEmulationTritonExperts is present.
"""

from __future__ import annotations

import sys
from pathlib import Path

# layers.linear imports NVFP4.torch_fake; ensure repo root is importable when
# pytest loads vllm via 3rdparty/vllm only.
_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import torch

from vllm.model_executor.layers.quantization.utils import nvfp4_emulation_utils
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    cast_to_fp4,
    dequantize_to_dtype,
    ref_nvfp4_quant,
    ref_nvfp4_quant_dequant,
    run_nvfp4_emulations,
)
from vllm.platforms import current_platform

E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


@pytest.fixture(autouse=True)
def _place_e2m1_lut_on_cpu():
    # Python reference path indexes the LUT; keep it on CPU by default.
    nvfp4_emulation_utils.kE2M1ToFloat_handle.val = E2M1_TABLE.clone()
    yield


def test_e2m1_unpack_table_values():
    # Packed bytes expand as [low_nibble, high_nibble] per byte.
    packed = torch.tensor(
        [
            [0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70],  # low=0, high=+mag
            [0x08, 0x19, 0x2A, 0x3B, 0x4C, 0x5D, 0x6E, 0x7F],  # low=-0, high=-mag
        ],
        dtype=torch.uint8,
    )
    out = nvfp4_emulation_utils.break_fp4_bytes(packed, torch.float32)
    expected = torch.tensor(
        [
            [
                0.0,
                0.0,
                0.0,
                0.5,
                0.0,
                1.0,
                0.0,
                1.5,
                0.0,
                2.0,
                0.0,
                3.0,
                0.0,
                4.0,
                0.0,
                6.0,
            ],
            [
                -0.0,
                0.0,
                -0.5,
                0.5,
                -1.0,
                1.0,
                -1.5,
                1.5,
                -2.0,
                2.0,
                -3.0,
                3.0,
                -4.0,
                4.0,
                -6.0,
                6.0,
            ],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (0.0, 0.0),
        (0.25, 0.0),
        (0.26, 0.5),
        (0.74, 0.5),
        (0.75, 1.0),
        (1.25, 1.0),
        (1.26, 1.5),
        (1.74, 1.5),
        (1.75, 2.0),
        (2.5, 2.0),
        (2.6, 3.0),
        (3.4, 3.0),
        (3.5, 4.0),
        (5.0, 4.0),
        (5.1, 6.0),
        (-1.3, -1.5),
        (-5.1, -6.0),
    ],
)
def test_cast_to_fp4_rounding_boundaries(x, expected):
    t = torch.tensor([x], dtype=torch.float32)
    out = cast_to_fp4(t)
    assert float(out.item()) == pytest.approx(expected)


def test_ref_nvfp4_quant_group_size_16_and_zero_block():
    block = 16
    x = torch.zeros(2, 32, dtype=torch.float32)
    x[0, :16] = 0.0
    x[0, 16:] = 3.0
    x[1, :] = -1.5
    global_scale = torch.tensor(1.0, dtype=torch.float32)
    q, scale = ref_nvfp4_quant(x, global_scale, block)
    assert q.shape == x.shape
    assert scale.shape == (2, 2)
    assert torch.all(q[0, :16] == 0)
    assert not torch.any(torch.isnan(q))
    assert not torch.any(torch.isinf(q))


def _pack_e2m1_pairs(values: torch.Tensor) -> torch.Tensor:
    """Pack float E2M1 values (even K) into uint8 low/high nibbles."""
    assert values.shape[-1] % 2 == 0
    abs_vals = torch.abs(values)
    codes = torch.zeros_like(values, dtype=torch.uint8)
    for code, val in enumerate(E2M1_TABLE.tolist()):
        codes = torch.where(
            torch.isclose(abs_vals, torch.tensor(val, dtype=abs_vals.dtype), atol=1e-6),
            torch.tensor(code, dtype=torch.uint8),
            codes,
        )
    signs = (values < 0).to(torch.uint8) << 3
    nibbles = codes | signs
    pairs = nibbles.view(*values.shape[:-1], values.shape[-1] // 2, 2)
    return pairs[..., 0] | (pairs[..., 1] << 4)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_run_nvfp4_emulations_packed_weight_bf16_fp16(dtype):
    torch.manual_seed(0)
    m, n, k = 4, 32, 64
    group = 16
    x = torch.randn(m, k, dtype=dtype)
    # Construct exact E2M1-representable weights so packing is lossless.
    mag = E2M1_TABLE[torch.randint(0, 8, (n, k))]
    signs = torch.where(torch.rand(n, k) < 0.5, 1.0, -1.0)
    weight_hp = (mag * signs).to(dtype)
    input_gs = torch.tensor(1.0, dtype=torch.float32)
    weight_gs = torch.tensor(1.0, dtype=torch.float32)

    weight_u8 = _pack_e2m1_pairs(weight_hp.float())
    # Unit block scales in E4M3; global scale 1 → dequant == E2M1 values.
    weight_scale = torch.ones(n, k // group, dtype=torch.float32).to(torch.float8_e4m3fn)

    out = run_nvfp4_emulations(
        x=x,
        input_global_scale=input_gs,
        weight=weight_u8,
        weight_scale_swizzled=weight_scale,
        weight_global_scale=weight_gs,
        swizzle=False,
    )
    assert out.shape == (m, n)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()


@pytest.mark.skipif(
    not current_platform.is_cuda_alike()
    or not current_platform.has_device_capability(89),
    reason="Triton NVFP4 fp8e4nv path requires CUDA SM89+.",
)
@pytest.mark.parametrize("global_scale_value", [0.5, 1.0, 0.001])
@pytest.mark.parametrize(
    ("m", "k"),
    [
        (1, 16),
        (4, 64),
        (16, 128),
        (33, 160),
    ],
)
def test_triton_nvfp4_quant_dequant_matches_python_ref(
    monkeypatch, m: int, k: int, global_scale_value: float
):
    block_size = 16
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    global_scale = torch.tensor(
        global_scale_value, dtype=torch.float32, device="cuda"
    )

    triton_result = ref_nvfp4_quant_dequant(x, global_scale, block_size)
    with monkeypatch.context() as mp:
        mp.setattr(
            nvfp4_emulation_utils,
            "_nvfp4_triton_fp8e4nv_supported",
            lambda: False,
        )
        reference = ref_nvfp4_quant_dequant(x.cpu(), global_scale.cpu(), block_size)

    torch.testing.assert_close(triton_result.cpu(), reference, atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike()
    or not current_platform.has_device_capability(89),
    reason="Triton NVFP4 fp8e4nv path requires CUDA SM89+.",
)
def test_triton_dequantize_matches_python_ref_synthetic(monkeypatch):
    block_size = 16
    m, k = 8, 64
    torch.manual_seed(1)
    fp4 = torch.randint(0, 255, (m, k // 2), dtype=torch.uint8, device="cuda")
    sf = torch.randn(m, k // block_size, device="cuda").to(torch.float8_e4m3fn)
    gs = torch.tensor(0.01, dtype=torch.float32, device="cuda")

    nvfp4_emulation_utils.kE2M1ToFloat_handle.val = (
        nvfp4_emulation_utils.kE2M1ToFloat_handle.val.to("cuda")
    )

    triton_result = dequantize_to_dtype(
        fp4, sf, gs, torch.bfloat16, block_size, swizzle=False
    )
    with monkeypatch.context() as mp:
        mp.setattr(
            nvfp4_emulation_utils,
            "_nvfp4_triton_fp8e4nv_supported",
            lambda: False,
        )
        reference = dequantize_to_dtype(
            fp4, sf, gs, torch.bfloat16, block_size, swizzle=False
        )

    torch.testing.assert_close(triton_result, reference, atol=0, rtol=0)


@pytest.mark.parametrize(
    "scale_val",
    [torch.finfo(torch.float8_e4m3fn).max, 1.0, 1e-4],
)
def test_e4m3_block_scale_extreme_and_normal(scale_val):
    m, k, block = 2, 32, 16
    fp4 = torch.zeros(m, k // 2, dtype=torch.uint8)
    # Encode +1.0 in every low nibble (code 2).
    fp4[:] = 0x02
    sf = torch.full((m, k // block), float(scale_val), dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    gs = torch.tensor(1.0, dtype=torch.float32)
    nvfp4_emulation_utils.kE2M1ToFloat_handle.val = E2M1_TABLE.clone()
    out = dequantize_to_dtype(fp4, sf, gs, torch.float32, block, swizzle=False)
    assert out.shape == (m, k)
    assert torch.isfinite(out).all()


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="CUDA required for device-path smoke",
)
def test_python_emulation_path_on_sm80_cuda_tensors():
    """A800 (SM80) must use Python QDQ/dequant; results must be finite."""
    block = 16
    m, k, n = 4, 64, 32
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    weight_u8 = torch.randint(0, 255, (n, k // 2), dtype=torch.uint8, device="cuda")
    weight_scale = torch.ones(n, k // block, device="cuda").to(torch.float8_e4m3fn)
    gs = torch.tensor(0.05, dtype=torch.float32, device="cuda")
    nvfp4_emulation_utils.kE2M1ToFloat_handle.val = E2M1_TABLE.to("cuda")

    out = run_nvfp4_emulations(
        x=x,
        input_global_scale=gs,
        weight=weight_u8,
        weight_scale_swizzled=weight_scale,
        weight_global_scale=gs,
        swizzle=False,
    )
    assert out.shape == (m, n)
    assert torch.isfinite(out).all()
    # On SM80 the Triton fp8e4nv path must stay disabled.
    if not current_platform.has_device_capability(89):
        assert not nvfp4_emulation_utils._nvfp4_triton_fp8e4nv_supported()


# ---------------------------------------------------------------------------
# Dense linear backend (Task 3): ModelOpt / CT + EmulationNvFp4LinearKernel
# ---------------------------------------------------------------------------


def _emulation_vllm_config():
    from vllm.config import VllmConfig
    from vllm.config.kernel import KernelConfig

    return VllmConfig(kernel_config=KernelConfig(linear_backend="emulation"))


def _fill_modelopt_style_layer(
    layer: torch.nn.Module,
    *,
    n: int,
    k: int,
    weight_u8: torch.Tensor,
    weight_scale: torch.Tensor,
    input_gs: float,
    weight_gs: float,
) -> None:
    layer.weight = torch.nn.Parameter(weight_u8.clone(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(weight_scale.clone(), requires_grad=False)
    layer.input_scale = torch.nn.Parameter(
        torch.tensor([input_gs], dtype=torch.float32), requires_grad=False
    )
    layer.weight_scale_2 = torch.nn.Parameter(
        torch.tensor([weight_gs], dtype=torch.float32), requires_grad=False
    )
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n


def _fill_ct_style_layer(
    layer: torch.nn.Module,
    *,
    n: int,
    k: int,
    weight_u8: torch.Tensor,
    weight_scale: torch.Tensor,
    input_gs: float,
    weight_gs: float,
) -> None:
    # CT stores reciprocal (divisor) global scales in the checkpoint.
    layer.weight_packed = torch.nn.Parameter(weight_u8.clone(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(weight_scale.clone(), requires_grad=False)
    layer.input_global_scale = torch.nn.Parameter(
        torch.tensor([1.0 / input_gs], dtype=torch.float32), requires_grad=False
    )
    layer.weight_global_scale = torch.nn.Parameter(
        torch.tensor([1.0 / weight_gs], dtype=torch.float32), requires_grad=False
    )
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n


@pytest.mark.parametrize("scheme", ["modelopt", "ct"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_dense_linear_emulation_kernel_matches_run_nvfp4_emulations(
    scheme: str, with_bias: bool
):
    """linear_backend=emulation selects EmulationNvFp4LinearKernel; packed
    uint8 weights stay packed; output matches run_nvfp4_emulations."""
    from torch.nn.parameter import Parameter

    from vllm.config import set_current_vllm_config
    from vllm.model_executor.kernels.linear.nvfp4.emulation import (
        EmulationNvFp4LinearKernel,
    )
    from vllm.model_executor.kernels.linear.nvfp4.select import (
        init_nvfp4_linear_kernel,
    )

    torch.manual_seed(0)
    m, n, k = 4, 32, 64
    group = 16
    dtype = torch.bfloat16
    x = torch.randn(m, k, dtype=dtype)
    mag = E2M1_TABLE[torch.randint(0, 8, (n, k))]
    signs = torch.where(torch.rand(n, k) < 0.5, 1.0, -1.0)
    weight_hp = (mag * signs).to(dtype)
    weight_u8 = _pack_e2m1_pairs(weight_hp.float())
    weight_scale = torch.ones(n, k // group, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    input_gs = 0.05
    weight_gs = 0.05
    bias = torch.randn(n, dtype=dtype) if with_bias else None

    with set_current_vllm_config(_emulation_vllm_config()):
        kernel = init_nvfp4_linear_kernel()
        assert isinstance(kernel, EmulationNvFp4LinearKernel)
        assert type(kernel).__name__ == "EmulationNvFp4LinearKernel"
        assert "Marlin" not in type(kernel).__name__
        assert "Legacy" not in type(kernel).__name__

        layer = torch.nn.Module()
        if scheme == "modelopt":
            _fill_modelopt_style_layer(
                layer,
                n=n,
                k=k,
                weight_u8=weight_u8,
                weight_scale=weight_scale,
                input_gs=input_gs,
                weight_gs=weight_gs,
            )
            packed_before = layer.weight.data.clone()
            # ModelOptNvFp4LinearMethod.process_weights_after_loading scale logic
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
            assert layer.weight.shape == packed_before.shape
            torch.testing.assert_close(
                layer.weight.data, packed_before, atol=0, rtol=0
            )
            assert layer.weight.numel() == n * (k // 2)
            out = kernel.apply_weights(layer, x, bias=bias)
        else:
            _fill_ct_style_layer(
                layer,
                n=n,
                k=k,
                weight_u8=weight_u8,
                weight_scale=weight_scale,
                input_gs=input_gs,
                weight_gs=weight_gs,
            )
            packed_before = layer.weight_packed.data.clone()
            # CompressedTensorsW4A4Fp4.process_weights_after_loading scale logic
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
            assert layer.weight.shape == packed_before.shape
            torch.testing.assert_close(
                layer.weight.data, packed_before, atol=0, rtol=0
            )
            assert layer.weight.numel() == n * (k // 2)
            out = kernel.apply_weights(layer, x, bias=bias)

    nvfp4_emulation_utils.kE2M1ToFloat_handle.val = E2M1_TABLE.clone()
    ref = run_nvfp4_emulations(
        x=x,
        input_global_scale=torch.tensor(1.0 / input_gs, dtype=torch.float32),
        weight=weight_u8,
        weight_scale_swizzled=weight_scale,
        weight_global_scale=torch.tensor(weight_gs, dtype=torch.float32),
        swizzle=False,
    )
    if bias is not None:
        ref = ref + bias

    # Emulation path is the same Python/Triton primitive; expect exact match.
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


def test_dense_linear_modelopt_and_ct_classes_use_emulation_kernel():
    """ModelOpt / CT scheme constructors pick EmulationNvFp4LinearKernel."""
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


def test_dense_linear_backend_emulation_not_marlin():
    """Explicit emulation must never return a Marlin kernel class."""
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.kernels.linear.nvfp4.emulation import (
        EmulationNvFp4LinearKernel,
    )
    from vllm.model_executor.kernels.linear.nvfp4.select import (
        init_nvfp4_linear_kernel,
    )

    with set_current_vllm_config(_emulation_vllm_config()):
        kernel = init_nvfp4_linear_kernel()
    assert isinstance(kernel, EmulationNvFp4LinearKernel)
    assert type(kernel).__name__ == "EmulationNvFp4LinearKernel"
    assert "Marlin" not in type(kernel).__name__
    assert "Legacy" not in type(kernel).__name__
