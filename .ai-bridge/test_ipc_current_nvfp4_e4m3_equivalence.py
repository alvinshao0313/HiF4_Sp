from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path('/home/shaoyuantian/program/HiF4_Sp')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import (
    _nvfp4_metadata,
    quantize_nvfp4_activation,
)
from NVFP4.torch_fake import (
    FP4_E2M1_MAX,
    _fake_quant_nvfp4_activation_torch,
    cast_to_fp8_e4m3fn,
)


def test_current_adapter_is_exact_e4m3fn_reference():
    torch.manual_seed(20260812)
    x = (torch.randn(3, 64) * 1.7).to(torch.bfloat16)
    g = torch.tensor(37.25, dtype=torch.float32)

    view = quantize_nvfp4_activation(x, g, output_dtype=torch.float32, collect_metadata=True)
    ref = _fake_quant_nvfp4_activation_torch(x, g, group_size=16, output_dtype=torch.float32)
    assert torch.equal(view.dequantized, ref)

    meta = _nvfp4_metadata(x, g, group_size=16)
    grouped = x.float().reshape(3, 4, 16)
    raw = torch.clamp(g * (grouped.abs().amax(dim=-1, keepdim=True) / FP4_E2M1_MAX), min=-448.0, max=448.0)
    expected_e4m3 = cast_to_fp8_e4m3fn(raw).float().squeeze(-1)
    assert torch.equal(meta['e4m3_local_scale'], expected_e4m3)
    assert 'e8m0' not in ''.join(meta.keys()).lower()
    print('adapter/reference exact; first raw/e4m3 scales:', meta['raw_local_scale'][0].tolist(), meta['e4m3_local_scale'][0].tolist())
