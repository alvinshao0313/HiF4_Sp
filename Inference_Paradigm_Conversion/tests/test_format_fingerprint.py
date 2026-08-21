from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from Inference_Paradigm_Conversion.ipc_analysis.formats.fingerprint import (
    load_nvfp4_qat_dequant_weight,
)


def test_source_weight_bf16_semantics(tmp_path: Path):
    w = torch.randn(64, 128, dtype=torch.bfloat16)
    shard = tmp_path / "model.safetensors"
    save_file({"model.layers.0.mlp.down_proj.weight": w}, str(shard))
    view = load_nvfp4_qat_dequant_weight(
        tmp_path, "model.layers.0.mlp.down_proj.weight", device="cpu"
    )
    assert view.format_name == "nvfp4_qat_fake_dequant_bf16"
    assert view.dequantized.dtype == torch.float32
    assert view.metadata["storage_dtype"] == "bfloat16"
    assert view.metadata["hadamard_runtime"] == "disabled"
    torch.testing.assert_close(view.dequantized, w.float(), rtol=0, atol=0)


def test_rejects_non_bf16_storage(tmp_path: Path):
    w = torch.randn(64, 128, dtype=torch.float32)
    save_file({"w": w}, str(tmp_path / "model.safetensors"))
    with pytest.raises(TypeError, match="bfloat16"):
        load_nvfp4_qat_dequant_weight(tmp_path, "w")
