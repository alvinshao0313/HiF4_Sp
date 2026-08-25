"""Fixed 4×4 Hadamard block rotation aligned to HiF4 G4 groups.

The unnormalized matrix given by the user is never applied directly.
The operator used everywhere is R4 = H4 / 2.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

H4_UNNORMALIZED = (
    (1.0, 1.0, 1.0, -1.0),
    (1.0, 1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0, 1.0),
)
H4_GROUP_SIZE = 4
HIF4_GROUP_SIZE = 64
ORTH_ATOL = 1e-6


def r4_matrix(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return R4 = H4 / 2. Multiplication by 0.5 is exact in binary floating point."""
    h4 = torch.tensor(H4_UNNORMALIZED, dtype=torch.float32, device=device)
    return (h4 * 0.5).to(dtype=dtype)


def assert_r4_orthogonal(r4: torch.Tensor | None = None) -> None:
    """Require R4 R4^T = I and R4^T R4 = I."""
    r = r4.to(torch.float64) if r4 is not None else r4_matrix(dtype=torch.float64)
    if tuple(r.shape) != (H4_GROUP_SIZE, H4_GROUP_SIZE):
        raise ValueError(f"R4 shape must be (4, 4), got {tuple(r.shape)}")
    eye = torch.eye(H4_GROUP_SIZE, dtype=torch.float64, device=r.device)
    err_rrt = (r @ r.T - eye).abs().max().item()
    err_rtr = (r.T @ r - eye).abs().max().item()
    if err_rrt >= ORTH_ATOL or err_rtr >= ORTH_ATOL:
        raise RuntimeError(
            f"R4 is not orthogonal: max|R R^T - I|={err_rrt:.3e}, "
            f"max|R^T R - I|={err_rtr:.3e}"
        )


def apply_h4_g4(
    x: torch.Tensor,
    dim: int = -1,
    *,
    compute_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Right-multiply every contiguous G4 on ``dim`` by R4.

    Layout is HiF4 G4 aligned: ``[..., N64, 64] -> [..., N64, 16, 4] @ R4``.
    Last (or selected) dimension must be divisible by 64.
    """
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    if dim < 0:
        dim = x.ndim + dim
    if dim < 0 or dim >= x.ndim:
        raise ValueError(f"dim={dim} is out of range for ndim={x.ndim}")
    k = int(x.shape[dim])
    if k % HIF4_GROUP_SIZE != 0:
        raise ValueError(
            f"dimension {dim} length {k} is not divisible by {HIF4_GROUP_SIZE}"
        )

    moved = x.movedim(dim, -1)
    leading = moved.shape[:-1]
    n_g4 = k // H4_GROUP_SIZE
    x_g4 = moved.to(compute_dtype).reshape(*leading, n_g4, H4_GROUP_SIZE)
    r4 = r4_matrix(dtype=compute_dtype, device=x.device)
    y_g4 = torch.matmul(x_g4, r4)
    y = y_g4.reshape(*leading, k)
    out_dtype = x.dtype if output_dtype is None else output_dtype
    return y.to(dtype=out_dtype).movedim(-1, dim)


def relative_frobenius(y_hat: torch.Tensor, y_ref: torch.Tensor, eps: float = 1e-30) -> float:
    diff = (y_hat.to(torch.float64) - y_ref.to(torch.float64)).norm()
    ref = y_ref.to(torch.float64).norm()
    return float((diff / ref.clamp_min(eps)).item())


def linear_prequant_equivalence_error(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    compute_dtype: torch.dtype = torch.float32,
) -> float:
    """Relative Frobenius error of Linear(X, W) vs Linear(X R, W R).

    H4 products stay in ``compute_dtype``. Linear and the error metric
    accumulate in FP64 so a 1e-6 gate is not dominated by large FP32 GEMM
    rounding.
    """
    x_c = x.to(compute_dtype)
    w_c = w.to(compute_dtype)
    x_r = apply_h4_g4(x_c, compute_dtype=compute_dtype, output_dtype=compute_dtype)
    w_r = apply_h4_g4(w_c, compute_dtype=compute_dtype, output_dtype=compute_dtype)
    y_ref = F.linear(x_c.to(torch.float64), w_c.to(torch.float64))
    y_rot = F.linear(x_r.to(torch.float64), w_r.to(torch.float64))
    return relative_frobenius(y_rot, y_ref)
