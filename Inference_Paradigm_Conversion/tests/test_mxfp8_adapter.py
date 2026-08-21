from __future__ import annotations

import pytest
import torch

from Inference_Paradigm_Conversion.ipc_analysis.formats.mxfp8 import quantize_mxfp8_activation


def test_mxfp8_qdq_same_x():
    x = torch.randn(3, 64, dtype=torch.bfloat16)
    view = quantize_mxfp8_activation(x)
    assert view.dequantized.shape == x.shape
    assert view.dequantized.dtype == torch.bfloat16
    assert view.metadata["block_size"] == 32


def test_mxfp8_rejects_bad_dim():
    x = torch.randn(2, 48, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="divisible by 32"):
        quantize_mxfp8_activation(x)
