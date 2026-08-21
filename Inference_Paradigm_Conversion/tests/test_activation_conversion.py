from __future__ import annotations

import torch

from Inference_Paradigm_Conversion.ipc_analysis.analysis.activation_conversion import (
    fair_triple_qdq,
)
from Inference_Paradigm_Conversion.ipc_analysis.formats.hif4 import quantize_hif4_tensor
from Inference_Paradigm_Conversion.ipc_analysis.formats.mxfp8 import quantize_mxfp8_activation
from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import quantize_nvfp4_activation


def test_same_x_fed_to_three_formats():
    torch.manual_seed(0)
    x = torch.randn(8, 128, dtype=torch.bfloat16)
    scale = torch.tensor(64.0, dtype=torch.float32)
    triple = fair_triple_qdq(x, scale)
    # Recompute independently to ensure no hidden-state mixing
    a_m = quantize_mxfp8_activation(x).dequantized
    a_n = quantize_nvfp4_activation(x, scale).dequantized
    a_h = quantize_hif4_tensor(x.float(), output_dtype=torch.bfloat16).dequantized
    torch.testing.assert_close(triple["A_M"].dequantized, a_m)
    torch.testing.assert_close(triple["A_N"].dequantized, a_n)
    torch.testing.assert_close(triple["A_H"].dequantized, a_h)


def test_missing_scale_must_raise_for_nvfp4():
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    # empty scale is invalid
    try:
        quantize_nvfp4_activation(x, torch.zeros(0))
        raised = False
    except ValueError:
        raised = True
    assert raised
