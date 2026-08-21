"""Hard gate: (7,4,2) hardware must align with ChuanCi and quant_hifx."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
_HIFLOAT4 = _ROOT.parent
_REPO = _HIFLOAT4.parent
for p in (_ROOT, _HIFLOAT4, _REPO / "ChuanCi"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from src.fixed_thresholds import get_baseline_config  # noqa: E402
from src.quantizer import HiF4QuantConfig, quantize_hif4  # noqa: E402


def _random_tensor(shape: tuple[int, ...], seed: int, device: str = "cpu") -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(shape, generator=g, dtype=torch.float32)
    # Inject a few outliers so e8/e4 fire.
    flat = x.view(-1)
    n = flat.numel()
    idx = torch.arange(0, n, max(n // 64, 1))[:32]
    flat[idx] = flat[idx] * 20.0
    return x.to(device)


def test_align_chuaci_hardware_metadata_and_recon():
    from nvfp4_hif4_torch import HiF4Config as CCConfig
    from nvfp4_hif4_torch import quantize_hif4 as cc_quantize

    x = _random_tensor((32, 256), seed=20260730)
    ours = quantize_hif4(x, config=get_baseline_config("standard"))
    cc = cc_quantize(x, config=CCConfig(scale_mode="hardware", payload_format="s1p2"))

    assert torch.equal(ours.s0, cc.top_scale)
    assert torch.equal(ours.e8, cc.e1_per_8)
    assert torch.equal(ours.e4, cc.e1_per_4)
    assert torch.equal(ours.payload, cc.payload_magnitude)
    assert torch.equal(ours.reconstruction, cc.values)


def test_align_chuaci_on_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from nvfp4_hif4_torch import HiF4Config as CCConfig
    from nvfp4_hif4_torch import quantize_hif4 as cc_quantize

    x = _random_tensor((16, 512), seed=7, device="cuda")
    ours = quantize_hif4(x, config=get_baseline_config("standard"))
    cc = cc_quantize(x, config=CCConfig(scale_mode="hardware"))
    assert torch.equal(ours.s0, cc.top_scale)
    assert torch.equal(ours.e8, cc.e1_per_8)
    assert torch.equal(ours.e4, cc.e1_per_4)
    assert torch.allclose(ours.reconstruction, cc.values, rtol=0.0, atol=0.0)


def test_align_quant_hifx_reconstruction():
    """Reconstruction must match quant_hifx (PyTorch reference path)."""
    from hif4_gpu.quant_cy import QType, quant_dequant_float

    x = _random_tensor((8, 512), seed=99)
    ours = quantize_hif4(x, config=get_baseline_config("standard"))
    ref = quant_dequant_float(
        x.contiguous(), QType("hifx4").dim(-1), force_py=True, force_fp32=True
    )
    # Hardware E6M2 path in ChuanCi uses codebook RNE; quant_hifx uses log2 mantissa.
    # Require near-exact reconstruction; allow tiny FP noise only.
    assert torch.allclose(ours.reconstruction.float(), ref.float(), rtol=0.0, atol=0.0) or (
        torch.allclose(ours.reconstruction.float(), ref.float(), rtol=1e-5, atol=1e-6)
    )
    max_err = (ours.reconstruction.float() - ref.float()).abs().max().item()
    assert max_err == 0.0, f"quant_hifx max abs err={max_err}"


def test_custom_thresholds_change_bits():
    x = _random_tensor((16, 256), seed=3)
    std = quantize_hif4(x, config=get_baseline_config("standard"))
    mse = quantize_hif4(x, config=get_baseline_config("scalar_mse"))
    # Lower thresholds should not increase e8/e4 trigger rates.
    assert float(mse.e8.mean()) >= float(std.e8.mean()) - 1e-12
    assert float(mse.e4.mean()) >= float(std.e4.mean()) - 1e-12


def test_s0_modes_smoke():
    x = _random_tensor((4, 128), seed=1)
    for mode in ("continuous", "bf16_math", "e6m2", "hardware"):
        cfg = HiF4QuantConfig(s0_mode=mode)  # type: ignore[arg-type]
        out = quantize_hif4(x, config=cfg)
        assert out.reconstruction.shape == x.shape
        assert torch.isfinite(out.reconstruction).all()
