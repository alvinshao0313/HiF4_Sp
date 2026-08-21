"""Block rotation oracle tests (CPU / synthetic)."""

from __future__ import annotations

import pytest
import torch

from Native_NVFP4_HiF4_Linear_Puncture.src.rotation import apply_block_rotation


def _random_orthogonal(group_size: int) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(group_size, group_size, dtype=torch.float32))
    return q.to(torch.bfloat16)


def test_rotation_operates_on_last_dim_blocks_only():
    """Convention: last-dim blocks only; each block transformed as x_block @ H."""
    group_size = 16
    h = _random_orthogonal(group_size)
    x = torch.arange(2 * 3 * 32, dtype=torch.float32).reshape(2, 3, 32).to(torch.bfloat16)
    y = apply_block_rotation(x, h, group_size=group_size)
    assert y.shape == x.shape

    x_blocks = x.float().reshape(2, 3, 2, group_size)
    y_ref = torch.matmul(x_blocks, h.float()).reshape(2, 3, 32)
    assert torch.allclose(y.float(), y_ref, rtol=1e-2, atol=1e-2)

    # Batch axis must not mix.
    y_flip = apply_block_rotation(x.flip(0), h, group_size=group_size)
    assert torch.allclose(y_flip.flip(0).float(), y.float(), rtol=1e-2, atol=1e-2)


def test_rotation_does_not_quantize():
    group_size = 8
    h = torch.eye(group_size, dtype=torch.bfloat16)
    x = torch.linspace(-3.0, 3.0, group_size, dtype=torch.bfloat16).unsqueeze(0)
    y = apply_block_rotation(x, h, group_size=group_size)
    assert y.dtype == torch.bfloat16
    assert torch.equal(y, x)
    assert bool((y < 0).any())
    assert torch.unique(y.float()).numel() >= group_size // 2


def test_rotation_fails_when_k_not_divisible_by_group_size():
    h = torch.eye(16, dtype=torch.bfloat16)
    x = torch.randn(4, 24, dtype=torch.bfloat16)
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        apply_block_rotation(x, h, group_size=16)
