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


# ---------------------------------------------------------------------------
# MoE NVFP4 emulation (Tasks 4–6)
# ---------------------------------------------------------------------------


def _make_synthetic_nvfp4_moe_tensors(
    num_experts: int,
    hidden_dim: int,
    intermediate_size: int,
    device: str = "cuda",
):
    """Packed uint8 W13/W2 + FP8 block scales + per-expert global scales."""
    assert hidden_dim % 16 == 0
    assert intermediate_size % 16 == 0
    group = 16
    w13_n = 2 * intermediate_size
    w1 = torch.randint(
        0, 256, (num_experts, w13_n, hidden_dim // 2), dtype=torch.uint8, device=device
    )
    w1_scale = torch.randn(
        num_experts, w13_n, hidden_dim // group, device=device
    ).to(torch.float8_e4m3fn)
    w1_gscale = torch.rand(num_experts, dtype=torch.float32, device=device) * 0.5 + 0.1

    w2 = torch.randint(
        0,
        256,
        (num_experts, hidden_dim, intermediate_size // 2),
        dtype=torch.uint8,
        device=device,
    )
    w2_scale = torch.randn(
        num_experts, hidden_dim, intermediate_size // group, device=device
    ).to(torch.float8_e4m3fn)
    w2_gscale = torch.rand(num_experts, dtype=torch.float32, device=device) * 0.5 + 0.1

    # Non-uniform per-expert activation scales (ModelOpt-style small scales).
    a13_raw = torch.tensor(
        [1e-4 * (i + 1) for i in range(num_experts)],
        dtype=torch.float32,
        device=device,
    )
    a2_raw = torch.tensor(
        [2e-4 * (i + 1) for i in range(num_experts)],
        dtype=torch.float32,
        device=device,
    )
    a1_gscale = 1.0 / a13_raw.max().to(torch.float32)
    a2_gscale = 1.0 / a2_raw.max().to(torch.float32)

    return (
        w1,
        w1_scale,
        w1_gscale,
        w2,
        w2_scale,
        w2_gscale,
        a1_gscale,
        a2_gscale,
        a13_raw,
        a2_raw,
    )


def _torch_nvfp4_moe_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w1_gscale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_gscale: torch.Tensor,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
):
    """Single-QDQ reference: dequant weights, QDQ acts once per GEMM, silu-and-mul."""
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        dequantize_to_dtype,
        ref_nvfp4_quant_dequant,
    )

    num_tokens, hidden_dim = hidden_states.shape
    top_k = topk_ids.size(1)
    dtype = hidden_states.dtype

    w1_dq = dequantize_to_dtype(
        w1, w1_scale, w1_gscale, dtype=dtype, block_size=16, swizzle=False
    )
    w2_dq = dequantize_to_dtype(
        w2, w2_scale, w2_gscale, dtype=dtype, block_size=16, swizzle=False
    )

    hs_qdq = ref_nvfp4_quant_dequant(hidden_states, a1_gscale, block_size=16)
    out = torch.zeros_like(hidden_states)

    for t in range(num_tokens):
        acc = torch.zeros(hidden_dim, dtype=torch.float32, device=hidden_states.device)
        for k in range(top_k):
            e = int(topk_ids[t, k].item())
            # W13: [2I, H] @ [H] -> [2I]
            gemm1 = torch.nn.functional.linear(hs_qdq[t], w1_dq[e])
            gate, up = gemm1.chunk(2, dim=-1)
            act = torch.nn.functional.silu(gate) * up
            act_qdq = ref_nvfp4_quant_dequant(
                act.unsqueeze(0), a2_gscale, block_size=16
            ).squeeze(0)
            gemm2 = torch.nn.functional.linear(act_qdq, w2_dq[e])
            acc = acc + gemm2.float() * float(topk_weights[t, k].item())
        out[t] = acc.to(dtype)
    return out


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Triton NVFP4 MoE emulation requires CUDA.",
)
@pytest.mark.parametrize("num_tokens", [1, 4])
@pytest.mark.parametrize("top_k", [1, 2])
def test_nvfp4_moe_emulation_correctness(num_tokens: int, top_k: int):
    """Fused emulation experts vs single-QDQ torch reference (BF16, no NaN/Inf)."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
        nvfp4_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
        Nvfp4QuantizationEmulationTritonExperts,
    )

    torch.manual_seed(0)
    num_experts = max(2, top_k)
    hidden_dim = 64
    intermediate_size = 32
    device = "cuda"

    nvfp4_emulation_utils.kE2M1ToFloat_handle.val = E2M1_TABLE.clone().to(device)

    (
        w1,
        w1_scale,
        w1_gscale,
        w2,
        w2_scale,
        w2_gscale,
        a1_gscale,
        a2_gscale,
        _a13_raw,
        _a2_raw,
    ) = _make_synthetic_nvfp4_moe_tensors(
        num_experts, hidden_dim, intermediate_size, device=device
    )

    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden_dim,
        intermediate_size_per_partition=intermediate_size,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device=device,
        routing_method=RoutingMethodType.TopK,
        moe_backend="emulation",
    )
    quant_config = nvfp4_moe_quant_config(
        g1_alphas=w1_gscale.clone(),
        g2_alphas=w2_gscale.clone(),
        a1_gscale=a1_gscale.clone(),
        a2_gscale=a2_gscale.clone(),
        w1_scale=w1_scale.clone(),
        w2_scale=w2_scale.clone(),
    )

    experts = Nvfp4QuantizationEmulationTritonExperts(
        moe_config=moe_config, quant_config=quant_config
    )
    assert experts.expects_unquantized_inputs is True
    assert w1.dtype == torch.uint8 and w2.dtype == torch.uint8

    hidden_states = torch.randn(
        num_tokens, hidden_dim, dtype=torch.bfloat16, device=device
    )
    topk_weights = torch.randn(
        num_tokens, top_k, dtype=torch.float32, device=device
    ).softmax(dim=-1)
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)

    N = w1.size(1)
    K = hidden_dim
    ws13 = torch.zeros(
        num_tokens * top_k * max(intermediate_size, K),
        dtype=torch.bfloat16,
        device=device,
    )
    ws2 = torch.zeros(
        num_tokens * top_k * max(N, K), dtype=torch.bfloat16, device=device
    )
    output = torch.zeros(num_tokens, K, dtype=torch.bfloat16, device=device)

    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=ws13,
        workspace2=ws2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    assert torch.isfinite(output).all()
    ref = _torch_nvfp4_moe_reference(
        hidden_states,
        w1,
        w1_scale,
        w1_gscale,
        w2,
        w2_scale,
        w2_gscale,
        a1_gscale,
        a2_gscale,
        topk_weights,
        topk_ids,
    )
    torch.testing.assert_close(output, ref, atol=0.05, rtol=0.01)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Requires CUDA for expert construction.",
)
def test_nvfp4_moe_expects_unquantized_inputs_prevents_double_qdq():
    """Prepare must defer act quant; expert owns the single NVFP4 QDQ."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
        nvfp4_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
        Nvfp4QuantizationEmulationTritonExperts,
    )

    device = "cuda"
    nvfp4_emulation_utils.kE2M1ToFloat_handle.val = E2M1_TABLE.clone().to(device)
    (
        w1,
        w1_scale,
        w1_gscale,
        w2,
        w2_scale,
        w2_gscale,
        a1_gscale,
        a2_gscale,
        _a13_raw,
        _a2_raw,
    ) = _make_synthetic_nvfp4_moe_tensors(2, 64, 32, device=device)

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
        device=device,
        routing_method=RoutingMethodType.TopK,
        moe_backend="emulation",
    )
    quant_config = nvfp4_moe_quant_config(
        g1_alphas=w1_gscale,
        g2_alphas=w2_gscale,
        a1_gscale=a1_gscale,
        a2_gscale=a2_gscale,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )
    experts = Nvfp4QuantizationEmulationTritonExperts(moe_config, quant_config)

    # Without this flag, prepare would call moe_kernel_quantize_input with
    # quant_dtype="nvfp4" (native pack) before apply's emulation QDQ.
    assert experts.expects_unquantized_inputs is True
    assert experts.quant_dtype == "nvfp4"
    assert experts.quantization_emulation is True
    # Packed weights stay uint8 — no BF16 Parameter materialization.
    assert w1.dtype == torch.uint8 and w2.dtype == torch.uint8
    assert w1.shape == (2, 64, 32)  # E, 2*inter, hidden//2
    assert w2.shape == (2, 64, 16)  # E, hidden, inter//2


@pytest.mark.parametrize(
    ("config_kwargs", "expected_reason"),
    [
        ({"has_bias": True}, "kernel does not support bias"),
        ({"is_lora_enabled": True}, "kernel does not support LoRA"),
    ],
)
def test_nvfp4_moe_emulation_support_check_rejects_bias_and_lora(
    config_kwargs: dict[str, bool],
    expected_reason: str,
) -> None:
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
        Nvfp4QuantizationEmulationTritonExperts,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kNvfp4Dynamic,
        kNvfp4Static,
    )

    moe_config = FusedMoEConfig(
        num_experts=2,
        experts_per_token=1,
        hidden_dim=16,
        intermediate_size_per_partition=16,
        num_local_experts=2,
        num_logical_experts=2,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.TopK,
        **config_kwargs,
    )

    supported, reason = Nvfp4QuantizationEmulationTritonExperts.is_supported_config(
        Nvfp4QuantizationEmulationTritonExperts,
        moe_config,
        kNvfp4Static,
        kNvfp4Dynamic,
        mk.FusedMoEActivationFormat.Standard,
    )
    assert not supported
    assert reason == expected_reason


def test_nvfp4_moe_backend_emulation_mapping():
    """moe_backend='emulation' -> EMULATION -> Nvfp4QuantizationEmulationTritonExperts."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
        Nvfp4QuantizationEmulationTritonExperts,
    )
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        NvFp4MoeBackend,
        backend_to_kernel_cls,
        map_nvfp4_backend,
        select_nvfp4_moe_backend,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8StaticTensorSym,
        kNvfp4Dynamic,
        kNvfp4Static,
    )

    assert map_nvfp4_backend("emulation") == NvFp4MoeBackend.EMULATION
    assert backend_to_kernel_cls(NvFp4MoeBackend.EMULATION) == [
        Nvfp4QuantizationEmulationTritonExperts
    ]

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
    backend, cls = select_nvfp4_moe_backend(
        moe_config, kNvfp4Static, kNvfp4Dynamic
    )
    assert backend == NvFp4MoeBackend.EMULATION
    assert cls is Nvfp4QuantizationEmulationTritonExperts

    # Auto + non-NVFP4-W4A4 key must not land on EMULATION.
    auto_config = FusedMoEConfig(
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
        moe_backend="auto",
    )
    auto_backend, _ = select_nvfp4_moe_backend(
        auto_config, kFp8StaticTensorSym, None
    )
    assert auto_backend != NvFp4MoeBackend.EMULATION

    # Completely unsupported scheme raises rather than inventing emulation.
    with pytest.raises((ValueError, NotImplementedError)):
        select_nvfp4_moe_backend(auto_config, None, None)


def test_nvfp4_moe_emulation_scale_max_reciprocal():
    """convert EMULATION keeps exact a13/a2 = 1.0 / max() semantics."""
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        NvFp4MoeBackend,
        convert_to_nvfp4_moe_kernel_format,
        make_nvfp4_moe_quant_config,
    )

    device = "cpu"
    nvfp4_emulation_utils.kE2M1ToFloat_handle.val = E2M1_TABLE.clone()

    e, n, k = 3, 32, 64
    w13 = torch.zeros(e, n, k // 2, dtype=torch.uint8, device=device)
    w13_scale = torch.zeros(e, n, k // 16, dtype=torch.float8_e4m3fn, device=device)
    w13_scale_2 = torch.ones(e, dtype=torch.float32, device=device)
    w2 = torch.zeros(e, k, n // 4, dtype=torch.uint8, device=device)
    w2_scale = torch.zeros(e, k, (n // 4) // 8, dtype=torch.float8_e4m3fn, device=device)
    w2_scale_2 = torch.ones(e, dtype=torch.float32, device=device)

    a13 = torch.tensor([1e-4, 3e-4, 2e-4], dtype=torch.float32, device=device)
    a2 = torch.tensor([5e-4, 1e-4, 2e-4], dtype=torch.float32, device=device)
    expected_a13 = 1.0 / a13.max().to(torch.float32)
    expected_a2 = 1.0 / a2.max().to(torch.float32)

    class _Layer:
        pass

    (
        _w13,
        _w13_s,
        _w13_s2,
        a13_out,
        _w2,
        _w2_s,
        _w2_s2,
        a2_out,
    ) = convert_to_nvfp4_moe_kernel_format(
        nvfp4_backend=NvFp4MoeBackend.EMULATION,
        layer=_Layer(),
        w13=w13,
        w13_scale=w13_scale,
        w13_scale_2=w13_scale_2,
        a13_scale=a13,
        w2=w2,
        w2_scale=w2_scale,
        w2_scale_2=w2_scale_2,
        a2_scale=a2,
        is_act_and_mul=True,
    )
    torch.testing.assert_close(a13_out, expected_a13, atol=0, rtol=0)
    torch.testing.assert_close(a2_out, expected_a2, atol=0, rtol=0)

    qc = make_nvfp4_moe_quant_config(
        backend=NvFp4MoeBackend.EMULATION,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_scale_2=w13_scale_2,
        w2_scale_2=w2_scale_2,
        a13_scale=a13_out,
        a2_scale=a2_out,
    )
    torch.testing.assert_close(qc.a1_gscale, expected_a13, atol=0, rtol=0)
    torch.testing.assert_close(qc.a2_gscale, expected_a2, atol=0, rtol=0)


def test_modelopt_and_compressed_moe_share_emulation_backend():
    """ModelOpt + CT MoE methods both select the same EMULATION expert class."""
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
        # MoE methods read moe_backend from FusedMoEConfig, not KernelConfig,
        # but keep emulation KernelConfig for consistency with dense tests.
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
    assert modelopt.experts_cls is ct.experts_cls


def test_modelopt_ct_moe_emulation_quant_config_contract():
    """After EMULATION convert, ModelOpt-style and CT-style scales share make()."""
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        NvFp4MoeBackend,
        convert_to_nvfp4_moe_kernel_format,
        make_nvfp4_moe_quant_config,
    )

    nvfp4_emulation_utils.kE2M1ToFloat_handle.val = E2M1_TABLE.clone()
    e, hidden, inter = 2, 64, 32
    w13 = torch.zeros(e, 2 * inter, hidden // 2, dtype=torch.uint8)
    w13_scale = torch.zeros(
        e, 2 * inter, hidden // 16, dtype=torch.float8_e4m3fn
    )
    w2 = torch.zeros(e, hidden, inter // 2, dtype=torch.uint8)
    w2_scale = torch.zeros(e, hidden, inter // 16, dtype=torch.float8_e4m3fn)
    w13_gscale = torch.ones(e, dtype=torch.float32)
    w2_gscale = torch.ones(e, dtype=torch.float32)

    # ModelOpt: raw small input scales.
    modelopt_a13 = torch.tensor([1e-4, 2e-4], dtype=torch.float32)
    modelopt_a2 = torch.tensor([3e-4, 1e-4], dtype=torch.float32)

    # CT: stores reciprocal global scales; loader passes 1.0 / global.
    ct_input_global_a13 = 1.0 / modelopt_a13
    ct_input_global_a2 = 1.0 / modelopt_a2
    ct_a13_in = 1.0 / ct_input_global_a13
    ct_a2_in = 1.0 / ct_input_global_a2
    torch.testing.assert_close(ct_a13_in, modelopt_a13, atol=0, rtol=0)
    torch.testing.assert_close(ct_a2_in, modelopt_a2, atol=0, rtol=0)

    class _Layer:
        pass

    def _run(a13, a2):
        (
            out_w13,
            out_w13_s,
            out_w13_g,
            a13_o,
            out_w2,
            out_w2_s,
            out_w2_g,
            a2_o,
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
        assert out_w13.dtype == torch.uint8 and out_w13.shape[-1] == hidden // 2
        assert out_w2.dtype == torch.uint8 and out_w2.shape[-1] == inter // 2
        assert out_w13_s.dtype == torch.float8_e4m3fn
        assert out_w2_s.dtype == torch.float8_e4m3fn
        return make_nvfp4_moe_quant_config(
            backend=NvFp4MoeBackend.EMULATION,
            w13_scale=out_w13_s,
            w2_scale=out_w2_s,
            w13_scale_2=out_w13_g,
            w2_scale_2=out_w2_g,
            a13_scale=a13_o,
            a2_scale=a2_o,
        )

    qc_m = _run(modelopt_a13, modelopt_a2)
    qc_c = _run(ct_a13_in, ct_a2_in)
    torch.testing.assert_close(qc_m.a1_gscale, qc_c.a1_gscale, atol=0, rtol=0)
    torch.testing.assert_close(qc_m.a2_gscale, qc_c.a2_gscale, atol=0, rtol=0)
    assert qc_m.quant_dtype == "nvfp4"
    assert qc_c.quant_dtype == "nvfp4"
