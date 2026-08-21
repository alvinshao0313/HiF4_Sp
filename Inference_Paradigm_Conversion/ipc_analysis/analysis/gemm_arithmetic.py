"""G0–G5: minimal GEMM arithmetic chain for format-semantic Oracle (not kernel perf)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from Inference_Paradigm_Conversion.ipc_analysis.formats.hif4 import quantize_hif4_tensor
from Inference_Paradigm_Conversion.ipc_analysis.formats.mxfp8 import quantize_mxfp8_activation
from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import quantize_nvfp4_activation
from Inference_Paradigm_Conversion.ipc_analysis.metrics.tensor_metrics import compute_pair_metrics


@torch.no_grad()
def gemm_chain_p1(
    x_bf16: torch.Tensor,
    w_n: torch.Tensor,
) -> dict[str, Any]:
    """P1_semantic: A=MXFP8, W: N vs HiF4."""
    a_m = quantize_mxfp8_activation(x_bf16).dequantized.float()
    w_h = quantize_hif4_tensor(w_n.float(), output_dtype=torch.float32).metadata["values_fp32"]
    y_n = F.linear(a_m, w_n.float())
    y_h = F.linear(a_m, w_h.float())
    return {
        "path_id": "P1_semantic",
        "output": compute_pair_metrics(y_n, y_h),
        "weight": compute_pair_metrics(w_n.float(), w_h.float()),
        "activation_vs_x": compute_pair_metrics(x_bf16.float(), a_m),
    }


@torch.no_grad()
def gemm_chain_p2(
    x_bf16: torch.Tensor,
    w_n: torch.Tensor,
    input_global_scale: torch.Tensor,
) -> dict[str, Any]:
    """P2_matched_semantic: A N→H, W N→H on same X."""
    a_n = quantize_nvfp4_activation(x_bf16, input_global_scale).dequantized.float()
    a_h = quantize_hif4_tensor(x_bf16.float(), output_dtype=torch.float32).metadata["values_fp32"]
    w_h = quantize_hif4_tensor(w_n.float(), output_dtype=torch.float32).metadata["values_fp32"]
    y_n = F.linear(a_n, w_n.float())
    y_h = F.linear(a_h, w_h.float())
    return {
        "path_id": "P2_matched_semantic",
        "output": compute_pair_metrics(y_n, y_h),
        "activation_h_vs_n": compute_pair_metrics(a_n, a_h),
        "weight": compute_pair_metrics(w_n.float(), w_h.float()),
    }


def gemm_fp32_reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Unified FP32 GEMM carrier for semantic comparison."""
    return F.linear(x.float(), w.float())
