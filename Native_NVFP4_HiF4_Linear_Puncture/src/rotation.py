"""Block Hadamard / rotation helpers (no quantization)."""

from __future__ import annotations

import hashlib

import torch


def rotation_sha256(rotation_matrix: torch.Tensor) -> str:
    """Stable SHA256 over contiguous BF16/float bytes of H."""
    h = rotation_matrix.detach().to(dtype=torch.bfloat16, device="cpu").contiguous()
    return hashlib.sha256(h.view(torch.uint8).numpy().tobytes()).hexdigest()


def apply_block_rotation(
    x: torch.Tensor,
    rotation_matrix: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Apply per-block right-multiply ``x_blocks @ H`` on the last dimension.

    Formal output dtype is BF16. No activation quantization is performed here.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    k = x.shape[-1]
    if k % group_size != 0:
        raise ValueError(
            f"last dim K={k} must be divisible by group_size={group_size}"
        )
    if tuple(rotation_matrix.shape) != (group_size, group_size):
        raise ValueError(
            f"rotation_matrix shape must be ({group_size}, {group_size}), "
            f"got {tuple(rotation_matrix.shape)}"
        )

    orig_shape = x.shape
    x_f = x.reshape(-1, k // group_size, group_size).to(torch.float32)
    h = rotation_matrix.to(device=x.device, dtype=torch.float32)
    y = torch.matmul(x_f, h)
    return y.reshape(orig_shape).to(torch.bfloat16)


def rotation_orthogonality_stats(rotation_matrix: torch.Tensor) -> dict[str, float]:
    h = rotation_matrix.detach().to(torch.float64)
    eye = torch.eye(h.shape[0], dtype=torch.float64)
    orth = (h @ h.T - eye).abs().max().item()
    sym = (h - h.T).abs().max().item()
    return {
        "max_abs_orthogonality_error": float(orth),
        "symmetry_error": float(sym),
    }
