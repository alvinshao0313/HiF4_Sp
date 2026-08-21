from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import (
    load_nvfp4_activation_scales,
    quantize_nvfp4_activation,
    resolve_nvfp4_scale_for_module,
)


def test_nvfp4_qdq_shape_and_scale_required():
    x = torch.randn(4, 64, dtype=torch.bfloat16)
    scale = torch.tensor(100.0, dtype=torch.float32)
    view = quantize_nvfp4_activation(x, scale)
    assert view.dequantized.shape == x.shape
    assert view.dequantized.dtype == torch.bfloat16
    assert view.metadata["block_size"] == 16
    assert "e2m1_payload" in view.metadata


def test_missing_scale_fails(tmp_path: Path):
    path = tmp_path / "nvfp4_activation_scales.safetensors"
    save_file(
        {"model.layers.0.mlp.down_proj.input_global_scale": torch.tensor(1.0)},
        str(path),
    )
    scales = load_nvfp4_activation_scales(path)
    with pytest.raises(ValueError, match="Missing NVFP4"):
        resolve_nvfp4_scale_for_module(scales, "model.layers.0.mlp.up_proj")


def test_scale_key_alias():
    scales = {
        "model.layers.0.self_attn.q_proj.input_global_scale": torch.tensor(
            2.0, dtype=torch.float32
        )
    }
    s = resolve_nvfp4_scale_for_module(
        scales, "model.language_model.layers.0.self_attn.q_proj"
    )
    assert float(s.item()) == 2.0
