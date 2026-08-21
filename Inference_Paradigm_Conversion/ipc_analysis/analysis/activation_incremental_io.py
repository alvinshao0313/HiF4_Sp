"""Activation incremental experiment I/O types and builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from Inference_Paradigm_Conversion.ipc_analysis.formats.hif4 import quantize_hif4_tensor
from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import quantize_nvfp4_activation


@dataclass
class ActivationIncrementalInput:
    run_id: str
    sample_id: str
    phase: str
    layer_idx: int
    module_name: str
    projection: str
    prompt_family: str
    split: str
    token_positions: torch.Tensor
    x_bf16: torch.Tensor
    a_nvfp4: torch.Tensor
    a_hif4: torch.Tensor
    weight_fp32: torch.Tensor
    nvfp4_metadata: dict
    hif4_metadata: dict
    input_global_scale: torch.Tensor


def assert_split_isolation(split: Literal["discovery", "validation"], items: list) -> None:
    """Ensure all items belong to the requested split."""
    for item in items:
        item_split = getattr(item, "split", None)
        if item_split is None and isinstance(item, dict):
            item_split = item.get("split")
        if item_split != split:
            raise ValueError(
                f"split isolation violated: expected {split!r}, got {item_split!r} "
                f"for sample {getattr(item, 'sample_id', item)}"
            )


def build_incremental_input(
    *,
    run_id: str,
    sample_id: str,
    phase: str,
    layer_idx: int,
    module_name: str,
    projection: str,
    prompt_family: str,
    split: str,
    x_bf16: torch.Tensor,
    input_global_scale: torch.Tensor,
    weight_fp32: torch.Tensor,
    token_positions: torch.Tensor | None = None,
) -> ActivationIncrementalInput:
    """Build one incremental input from captured X + scale + weight."""
    if x_bf16.dtype != torch.bfloat16:
        raise TypeError(f"x_bf16 must be bfloat16, got {x_bf16.dtype}")
    if token_positions is None:
        n = x_bf16.shape[0] if x_bf16.ndim >= 2 else 1
        token_positions = torch.arange(n, dtype=torch.int64)

    nv_view = quantize_nvfp4_activation(
        x_bf16, input_global_scale, output_dtype=torch.bfloat16, collect_metadata=True
    )
    hf_view = quantize_hif4_tensor(
        x_bf16.float(), group_dim=-1, variant="full", output_dtype=torch.bfloat16
    )
    return ActivationIncrementalInput(
        run_id=run_id,
        sample_id=sample_id,
        phase=phase,
        layer_idx=layer_idx,
        module_name=module_name,
        projection=projection,
        prompt_family=prompt_family,
        split=split,
        token_positions=token_positions,
        x_bf16=x_bf16,
        a_nvfp4=nv_view.dequantized,
        a_hif4=hf_view.dequantized,
        weight_fp32=weight_fp32.float(),
        nvfp4_metadata=dict(nv_view.metadata),
        hif4_metadata=dict(hf_view.metadata),
        input_global_scale=input_global_scale.reshape(()).detach(),
    )
