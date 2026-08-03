"""E6M2 / S1P2 / BF16 format helpers for HiF4 reference quantization."""

from __future__ import annotations

import torch

S1P2_MAX = 1.75
S1P2_STEP = 0.25
E6M2_MIN = (1.0 + 0.0 / 4.0) * (2.0 ** (0 - 48))
E6M2_MAX = (1.0 + 3.0 / 4.0) * (2.0 ** (63 - 48))


def build_e6m2_codebook() -> tuple[torch.Tensor, torch.Tensor]:
    """Unsigned E6M2 scale codebook: codes 0..254 (255 values)."""
    values: list[float] = []
    codes: list[int] = []
    for code in range(255):
        exponent = (code >> 2) & 0x3F
        mantissa = code & 0x03
        value = (1.0 + mantissa / 4.0) * (2.0 ** (exponent - 48))
        values.append(value)
        codes.append(code)
    return (
        torch.tensor(values, dtype=torch.float32),
        torch.tensor(codes, dtype=torch.int16),
    )


E6M2_VALUES, E6M2_CODES = build_e6m2_codebook()


def round_bfloat16(values: torch.Tensor) -> torch.Tensor:
    """Explicit BF16 carrier: FP32 -> BF16 -> FP32."""
    if not values.is_floating_point():
        raise TypeError("values must be floating point")
    return values.to(torch.bfloat16).to(torch.float32)


def round_positive_to_codebook(
    values: torch.Tensor,
    codebook_values: torch.Tensor,
    codebook_codes: torch.Tensor,
) -> torch.Tensor:
    """Non-negative RNE to codebook; ties pick even code."""
    if not values.is_floating_point():
        raise TypeError("values must be floating point")
    if not torch.isfinite(values).all():
        raise ValueError("values must be finite")
    x = values.to(torch.float32).clamp_min(0.0)
    book = codebook_values.to(device=x.device, dtype=torch.float32)
    codes = codebook_codes.to(device=x.device)
    hi = torch.searchsorted(book, x).clamp(0, book.numel() - 1)
    lo = (hi - 1).clamp(0, book.numel() - 1)
    d_lo = x - book[lo]
    d_hi = book[hi] - x
    choose_hi = d_hi < d_lo
    tie = d_hi == d_lo
    choose_hi = choose_hi | (tie & ((codes[hi] & 1) == 0))
    return torch.where(choose_hi, book[hi], book[lo])


def round_e6m2(x: torch.Tensor) -> torch.Tensor:
    """Round positive values to E6M2 codebook."""
    return round_positive_to_codebook(x, E6M2_VALUES, E6M2_CODES)


def e6m2_code_of(values: torch.Tensor) -> torch.Tensor:
    """Return E6M2 code index for already-quantized (or to-quantize) values."""
    rounded = round_e6m2(values)
    book = E6M2_VALUES.to(device=rounded.device, dtype=torch.float32)
    # Exact match after rounding; searchsorted finds insertion point of equal value.
    idx = torch.searchsorted(book, rounded).clamp(0, book.numel() - 1)
    # Correct ties / float noise by checking neighbors.
    lo = (idx - 1).clamp(0, book.numel() - 1)
    pick_lo = (rounded - book[lo]).abs() < (rounded - book[idx]).abs()
    return torch.where(pick_lo, lo, idx).to(torch.int64)


def e6m2_neighbor_values(base_s0: torch.Tensor, offsets: list[int] | tuple[int, ...]) -> torch.Tensor:
    """For each base S0, return neighbor E6M2 values at given code offsets.

    Returns shape ``(*base_s0.shape, K)`` with invalid/non-positive neighbors
    replaced by the nearest valid positive code (never silently invents out-of-range).
    Offsets that leave ``[0, 254]`` are clamped to the codebook boundary.
    """
    base_codes = e6m2_code_of(base_s0)
    off = torch.tensor(list(offsets), device=base_s0.device, dtype=torch.int64)
    codes = (base_codes.unsqueeze(-1) + off).clamp(0, E6M2_VALUES.numel() - 1)
    book = E6M2_VALUES.to(device=base_s0.device, dtype=torch.float32)
    return book[codes]


def quantize_s1p2_magnitude(normalized: torch.Tensor) -> torch.Tensor:
    """S1P2 magnitude: round to 0.25 grid, clamp to 1.75."""
    ratio = torch.floor(4.0 * normalized + 0.5) / 4.0
    return torch.minimum(ratio, torch.full_like(ratio, S1P2_MAX))
