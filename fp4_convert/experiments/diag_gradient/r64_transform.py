"""64×64 Hadamard block rotation aligned to HiF4 G64 groups.

R64 = kron(kron(H4, H4), H4) / 8.0
"""

from __future__ import annotations

import torch

H4_UNNORMALIZED = (
    (1.0, 1.0, 1.0, -1.0),
    (1.0, 1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0, 1.0),
)
R64_GROUP_SIZE = 64
ORTH_ATOL = 1e-5


def r64_matrix(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return R64 = kron(kron(H4, H4), H4) / 8."""
    h4 = torch.tensor(H4_UNNORMALIZED, dtype=torch.float32, device=device)
    h16 = torch.kron(h4, h4)
    h64 = torch.kron(h16, h4)
    return (h64 / 8.0).to(dtype=dtype)


def assert_r64_orthogonal(r64: torch.Tensor | None = None) -> None:
    """Require R64 R64^T = I and R64^T R64 = I."""
    r = r64.to(torch.float64) if r64 is not None else r64_matrix(dtype=torch.float64)
    if tuple(r.shape) != (R64_GROUP_SIZE, R64_GROUP_SIZE):
        raise ValueError(f"R64 shape must be (64, 64), got {tuple(r.shape)}")
    eye = torch.eye(R64_GROUP_SIZE, dtype=torch.float64, device=r.device)
    err_rrt = (r @ r.T - eye).abs().max().item()
    err_rtr = (r.T @ r - eye).abs().max().item()
    if err_rrt >= ORTH_ATOL or err_rtr >= ORTH_ATOL:
        raise RuntimeError(
            f"R64 is not orthogonal: max|R R^T - I|={err_rrt:.3e}, "
            f"max|R^T R - I|={err_rtr:.3e}"
        )


def apply_r64_g64(
    x: torch.Tensor,
    dim: int = -1,
    *,
    compute_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Right-multiply every contiguous G64 on ``dim`` by R64.

    Layout: ``[..., N64, 64] @ R64``. Dimension length must be divisible by 64.
    """
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    if dim < 0:
        dim = x.ndim + dim
    if dim < 0 or dim >= x.ndim:
        raise ValueError(f"dim={dim} is out of range for ndim={x.ndim}")
    k = int(x.shape[dim])
    if k % R64_GROUP_SIZE != 0:
        raise ValueError(
            f"dimension {dim} length {k} is not divisible by {R64_GROUP_SIZE}"
        )

    moved = x.movedim(dim, -1)
    leading = moved.shape[:-1]
    n_g64 = k // R64_GROUP_SIZE
    x_g = moved.to(compute_dtype).reshape(*leading, n_g64, R64_GROUP_SIZE)
    r64 = r64_matrix(dtype=compute_dtype, device=x.device)
    y_g = torch.matmul(x_g, r64)
    y = y_g.reshape(*leading, k)
    out_dtype = x.dtype if output_dtype is None else output_dtype
    return y.to(dtype=out_dtype).movedim(-1, dim)
