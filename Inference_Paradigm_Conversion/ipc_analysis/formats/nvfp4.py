"""NVFP4 activation adapter: read-only reuse of NVFP4.torch_fake + sidecar scales."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from Inference_Paradigm_Conversion.ipc_analysis.formats.fingerprint import QuantizedTensorView

# Read-only import of existing NVFP4 fake activation.
# Semantic Oracle uses the pure-torch math path (not Triton kernel).
from NVFP4.torch_fake import (  # noqa: E402
    FP4_E2M1_MAX,
    FP8_E4M3FN_MAX,
    _fake_quant_nvfp4_activation_torch,
    cast_to_fp4_e2m1,
    cast_to_fp8_e4m3fn,
)

ACTIVATION_SCALES_FILENAME = "nvfp4_activation_scales.safetensors"
_SCALE_CACHE: dict[str, dict[str, torch.Tensor]] = {}


def resolve_activation_scale_path(checkpoint_dir: Path | str) -> Path:
    """Same resolution as main.py when fake_act_quant=nvfp4."""
    path = Path(checkpoint_dir).resolve() / ACTIVATION_SCALES_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            "NVFP4 activation scale file not found: "
            f"{path}"
        )
    return path


def load_nvfp4_activation_scales(path: Path | str) -> dict[str, torch.Tensor]:
    """Mirror vLLM LinearMethodBase._load_nvf4_activation_scales."""
    path = os.path.abspath(str(path))
    if path in _SCALE_CACHE:
        return _SCALE_CACHE[path]
    scales: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if tensor.numel() != 1:
                raise ValueError(
                    f"NVFP4 activation scale must be scalar: {key} has shape {tuple(tensor.shape)}"
                )
            scales[key] = tensor.reshape(()).to(torch.float32).contiguous()
    _SCALE_CACHE[path] = scales
    return scales


def _nvf4_key_candidates(prefix: str) -> list[str]:
    key_prefixes = [prefix]
    if prefix.startswith("model.layers."):
        key_prefixes.append(
            "model.language_model.layers." + prefix[len("model.layers.") :]
        )
    elif prefix.startswith("model.language_model.layers."):
        key_prefixes.append(
            "model.layers." + prefix[len("model.language_model.layers.") :]
        )
    return [f"{key_prefix}.input_global_scale" for key_prefix in key_prefixes]


def resolve_nvfp4_scale_for_module(
    scales: dict[str, torch.Tensor],
    module_prefix: str,
) -> torch.Tensor:
    """Resolve per-Linear FP32 input_global_scale; missing scale fails loudly."""
    prefix = module_prefix
    if prefix.endswith(".weight"):
        prefix = prefix[: -len(".weight")]
    if prefix.endswith(".input_global_scale"):
        prefix = prefix[: -len(".input_global_scale")]
    candidates = _nvf4_key_candidates(prefix)
    found = [(key, scales[key]) for key in candidates if key in scales]
    if not found:
        raise ValueError(
            "Missing NVFP4 activation scale for linear layer prefix "
            f"{prefix}. Tried keys: {candidates}"
        )
    first_key, first_value = found[0]
    for key, value in found[1:]:
        if not torch.equal(value, first_value):
            raise ValueError(
                "Conflicting NVFP4 activation scales for "
                f"{prefix}: {first_key} and {key}"
            )
    return first_value


def _nvfp4_metadata(
    x_bf16: torch.Tensor,
    input_global_scale: torch.Tensor,
    group_size: int = 16,
) -> dict[str, Any]:
    """Observable Oracle metadata for A2 analysis (math path, not kernel)."""
    original_shape = x_bf16.shape
    hidden = original_shape[-1]
    x_2d = x_bf16.reshape(-1, hidden).to(torch.float32)
    grouped = x_2d.reshape(x_2d.shape[0], hidden // group_size, group_size)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    g = input_global_scale.reshape(()).to(device=x_2d.device, dtype=torch.float32)
    raw_block_scale = g * (amax / FP4_E2M1_MAX)
    raw_block_scale = torch.clamp(raw_block_scale, min=-448.0, max=448.0)
    e4m3_block_scale = cast_to_fp8_e4m3fn(raw_block_scale).to(torch.float32)
    output_scale = torch.where(
        e4m3_block_scale == 0,
        torch.zeros_like(e4m3_block_scale),
        g / e4m3_block_scale,
    )
    scaled = torch.clamp(grouped * output_scale, min=-FP4_E2M1_MAX, max=FP4_E2M1_MAX)
    payload = cast_to_fp4_e2m1(scaled)
    return {
        "block_size": group_size,
        "input_global_scale": float(g.item()),
        "block_amax": amax.squeeze(-1),
        "raw_local_scale": raw_block_scale.squeeze(-1),
        "e4m3_local_scale": e4m3_block_scale.squeeze(-1),
        "e2m1_payload": payload,
        "fp4_e2m1_max": FP4_E2M1_MAX,
        "fp8_e4m3fn_max": FP8_E4M3FN_MAX,
    }


def quantize_nvfp4_activation(
    x_bf16: torch.Tensor,
    input_global_scale: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
    *,
    collect_metadata: bool = True,
    use_existing_impl: bool = True,
) -> QuantizedTensorView:
    """QDQ NVFP4 activation on BF16 pre-quant activation with per-Linear global scale."""
    if x_bf16.dtype != torch.bfloat16:
        raise TypeError(f"x_bf16 must be bfloat16, got {x_bf16.dtype}")
    if x_bf16.shape[-1] % 16 != 0:
        raise ValueError(f"Last dim {x_bf16.shape[-1]} must be divisible by 16")
    if input_global_scale.numel() != 1:
        raise ValueError(
            "Only scalar per-tensor input_global_scale is supported; "
            f"got shape {tuple(input_global_scale.shape)}"
        )
    if not use_existing_impl:
        raise RuntimeError("only existing NVFP4.torch_fake math path is allowed")
    global_scale = input_global_scale.reshape(()).to(
        device=x_bf16.device, dtype=torch.float32
    )
    dequant = _fake_quant_nvfp4_activation_torch(
        x_bf16,
        global_scale,
        group_size=16,
        output_dtype=output_dtype,
    )

    meta: dict[str, Any] = {
        "format": "nvfp4",
        "block_size": 16,
        "hadamard_runtime": "disabled",
        "input_global_scale": float(input_global_scale.reshape(()).item()),
    }
    if collect_metadata:
        meta.update(_nvfp4_metadata(x_bf16, input_global_scale, group_size=16))
    return QuantizedTensorView(
        format_name="nvfp4_activation",
        dequantized=dequant,
        source_shape=tuple(x_bf16.shape),
        metadata=meta,
    )
