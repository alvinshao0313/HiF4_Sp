"""Synthetic distributions for threshold experiments."""

from __future__ import annotations

import torch


def sample_gaussian(n: int, seed: int, device: str | torch.device = "cpu") -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(n, generator=g, dtype=torch.float32).to(device)


def sample_laplace(n: int, seed: int, device: str | torch.device = "cpu") -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    u = torch.rand(n, generator=g, dtype=torch.float32) - 0.5
    # Laplace(0, 1/sqrt(2)) has unit variance: scale = 1/sqrt(2)
    scale = 0.5**0.5
    x = -scale * torch.sign(u) * torch.log1p(-2.0 * u.abs().clamp(max=1.0 - 1e-7))
    return x.to(device)


def sample_student_t3(n: int, seed: int, device: str | torch.device = "cpu") -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    # t with df=3, unit variance: Var = df/(df-2) = 3 => scale by 1/sqrt(3)? 
    # Standard student-t3 has variance 3; plan asks unit-variance Student-t3.
    df = 3.0
    # Sample via normal/chi2
    z = torch.randn(n, generator=g, dtype=torch.float32)
    g2 = torch.Generator(device="cpu").manual_seed(seed + 1)
    chi2 = torch.randn(n, generator=g2, dtype=torch.float32).pow(2)
    # For df=3, sum of 3 standard normals squared
    g3 = torch.Generator(device="cpu").manual_seed(seed + 2)
    chi2 = (
        torch.randn(n, generator=g2, dtype=torch.float32).pow(2)
        + torch.randn(n, generator=g3, dtype=torch.float32).pow(2)
        + torch.randn(n, generator=torch.Generator(device="cpu").manual_seed(seed + 3), dtype=torch.float32).pow(2)
    )
    t = z / torch.sqrt(chi2 / df)
    # Unit variance: divide by sqrt(df/(df-2)) = sqrt(3)
    t = t / (df / (df - 2.0)) ** 0.5
    return t.to(device)


def sample_outlier(n: int, seed: int, device: str | torch.device = "cpu") -> torch.Tensor:
    x = sample_gaussian(n, seed, device="cpu")
    g = torch.Generator(device="cpu").manual_seed(seed + 99)
    mask = torch.rand(n, generator=g) < 0.001
    x = x.clone()
    x[mask] = x[mask] * 20.0
    return x.to(device)


def sample_phase_boundary(n: int, seed: int, device: str | torch.device = "cpu") -> torch.Tensor:
    """Values concentrated near threshold boundaries 1.75/1.875/2/3.5/3.75/4."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    centers = torch.tensor([1.75, 1.875, 2.0, 3.5, 3.75, 4.0], dtype=torch.float32)
    idx = torch.randint(0, centers.numel(), (n,), generator=g)
    noise = torch.randn(n, generator=g, dtype=torch.float32) * 0.05
    mag = centers[idx] + noise
    signs = torch.where(torch.rand(n, generator=g) > 0.5, 1.0, -1.0)
    # Place inside groups with shared S0 ~ 1 so thresholds act on magnitude directly.
    return (signs * mag).to(device)


DISTRIBUTIONS = {
    "gaussian": sample_gaussian,
    "laplace": sample_laplace,
    "student_t3": sample_student_t3,
    "outlier_0p1pct_20x": sample_outlier,
    "phase_boundary": sample_phase_boundary,
}


def make_matrix(
    name: str,
    rows: int,
    cols: int,
    seed: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    if cols % 64 != 0:
        raise ValueError("cols must be divisible by 64")
    fn = DISTRIBUTIONS[name]
    flat = fn(rows * cols, seed, device="cpu")
    return flat.reshape(rows, cols).to(device)
