from __future__ import annotations

import pytest
import torch

from Inference_Paradigm_Conversion.ipc_analysis.formats.hif4 import quantize_hif4_tensor


def test_hif4_qdq_preserves_shape_bf16_out():
    x = torch.randn(32, 128, dtype=torch.float32)
    view = quantize_hif4_tensor(x, group_dim=-1, output_dtype=torch.bfloat16)
    assert view.dequantized.shape == x.shape
    assert view.dequantized.dtype == torch.bfloat16
    assert view.metadata["group_size"] == 64
    assert "top_scale" in view.metadata


def test_hif4_rejects_nondivisible():
    x = torch.randn(8, 60, dtype=torch.float32)
    with pytest.raises(ValueError, match="divisible by"):
        quantize_hif4_tensor(x)


def test_hif4_continuous_s0_differs_or_equals_full():
    torch.manual_seed(0)
    x = torch.randn(16, 128, dtype=torch.float32)
    full = quantize_hif4_tensor(x, variant="full", output_dtype=torch.float32)
    cont = quantize_hif4_tensor(x, variant="continuous_s0", output_dtype=torch.float32)
    assert full.dequantized.shape == cont.dequantized.shape
    # Deterministic; continuous may match or improve — both finite
    assert torch.isfinite(full.dequantized).all()
    assert torch.isfinite(cont.dequantized).all()
