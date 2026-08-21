"""Source checkpoint weight loading: NVFP4-QAT fake-dequant BF16 container."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


@dataclass
class QuantizedTensorView:
    format_name: str
    dequantized: torch.Tensor
    source_shape: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def _index_path(checkpoint_path: Path) -> Path:
    index = checkpoint_path / "model.safetensors.index.json"
    if index.is_file():
        return index
    single = checkpoint_path / "model.safetensors"
    if single.is_file():
        return single
    raise FileNotFoundError(f"No safetensors index/model under {checkpoint_path}")


def locate_weight_shard(checkpoint_path: Path, tensor_name: str) -> Path:
    checkpoint_path = Path(checkpoint_path)
    index = checkpoint_path / "model.safetensors.index.json"
    if index.is_file():
        with index.open("r", encoding="utf-8") as f:
            mapping = json.load(f)["weight_map"]
        if tensor_name not in mapping:
            raise KeyError(f"tensor {tensor_name!r} not in weight_map of {index}")
        return checkpoint_path / mapping[tensor_name]
    single = checkpoint_path / "model.safetensors"
    if single.is_file():
        return single
    raise FileNotFoundError(f"Cannot locate shard for {tensor_name} in {checkpoint_path}")


def load_nvfp4_qat_dequant_weight(
    checkpoint_path: Path | str,
    tensor_name: str,
    device: torch.device | str = "cpu",
) -> QuantizedTensorView:
    """Load stored BF16 NVFP4-QAT dequant weight; convert to FP32 for analysis only.

    Forbidden:
    - packed NVFP4 decode
    - reverse-engineering E2M1 payload / E4M3 local scale from BF16 values
    - re-quantizing the weight into NVFP4
    """
    checkpoint_path = Path(checkpoint_path)
    shard = locate_weight_shard(checkpoint_path, tensor_name)
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        if tensor_name not in handle.keys():
            raise KeyError(f"{tensor_name} missing in {shard}")
        stored = handle.get_tensor(tensor_name)
    if stored.dtype != torch.bfloat16:
        raise TypeError(
            f"source weight {tensor_name} must be bfloat16 storage, got {stored.dtype}"
        )
    dequant = stored.to(device=device, dtype=torch.float32)
    return QuantizedTensorView(
        format_name="nvfp4_qat_fake_dequant_bf16",
        dequantized=dequant,
        source_shape=tuple(stored.shape),
        metadata={
            "storage_dtype": "bfloat16",
            "semantic_dtype": "nvfp4_qat_fake",
            "hadamard_runtime": "disabled",
            "tensor_name": tensor_name,
            "shard": str(shard),
        },
    )


def list_safetensor_keys(checkpoint_path: Path | str) -> dict[str, str]:
    """Return weight_map: tensor_name -> shard relative name."""
    checkpoint_path = Path(checkpoint_path)
    index = checkpoint_path / "model.safetensors.index.json"
    if index.is_file():
        with index.open("r", encoding="utf-8") as f:
            return dict(json.load(f)["weight_map"])
    single = checkpoint_path / "model.safetensors"
    if single.is_file():
        with safe_open(str(single), framework="pt", device="cpu") as handle:
            return {k: single.name for k in handle.keys()}
    raise FileNotFoundError(checkpoint_path)
