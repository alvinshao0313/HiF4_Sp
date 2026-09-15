"""BF16 fused residual-add runtime reference and high-precision residues."""
from __future__ import annotations

import torch


def bf16_fused_residual_add(residual: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
    """PyTorch-native equivalent of vLLM fused residual-add storage writeback.

    sum_fp32 = x.float() + residual
    updated_residual = sum_fp32.to(orig_dtype)  # BF16 under formal protocol
    """
    if residual.shape != branch.shape:
        raise ValueError(
            f"residual/branch shape mismatch: {tuple(residual.shape)} vs {tuple(branch.shape)}"
        )
    if residual.dtype != torch.bfloat16 or branch.dtype != torch.bfloat16:
        raise ValueError(
            f"formal residual dtype must be BF16, got residual={residual.dtype} branch={branch.dtype}"
        )
    return (residual.float() + branch.float()).to(dtype=torch.bfloat16)


def rounding_residue(updated: torch.Tensor, residual: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
    """epsilon = FP64(updated) - FP64(residual) - FP64(branch)."""
    return (
        updated.detach().to(dtype=torch.float64)
        - residual.detach().to(dtype=torch.float64)
        - branch.detach().to(dtype=torch.float64)
    )


def assert_runtime_add_identity(
    captured_updated: torch.Tensor,
    residual: torch.Tensor,
    branch: torch.Tensor,
    *,
    repeatability_max_abs: float,
    repeatability_l2: float,
) -> dict:
    """Closure against BF16 runtime-add reference within measured repeatability envelope."""
    reference = bf16_fused_residual_add(residual, branch)
    delta = captured_updated.detach().to(dtype=torch.float64) - reference.detach().to(dtype=torch.float64)
    max_abs = float(delta.abs().max())
    l2 = float(torch.linalg.vector_norm(delta))
    passed = max_abs <= float(repeatability_max_abs) and l2 <= float(repeatability_l2)
    result = {
        "identity_pass": passed,
        "max_abs": max_abs,
        "l2": l2,
        "repeatability_max_abs": float(repeatability_max_abs),
        "repeatability_l2": float(repeatability_l2),
        "reference_dtype": str(reference.dtype),
    }
    if not passed:
        raise RuntimeError(f"BF16 runtime residual-add identity failed: {result}")
    return result
