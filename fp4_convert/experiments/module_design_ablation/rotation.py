"""Tied orthogonal G64 transforms, initialized exactly to the existing R64."""
from __future__ import annotations

import torch
from torch import nn

from ..diag_gradient.r64_transform import r64_matrix

GROUP_SIZE = 64


class CayleyRotation(nn.Module):
    """One orthogonal matrix per input group, with 2016 skew coordinates each.

    Finite skew coordinates cover the Cayley chart around R64. The activation
    and its projection weight must consume the SAME returned matrix, so both
    contributions participate in the gradient of a tied transform.
    """

    def __init__(self, groups: int, *, device=None):
        super().__init__()
        if type(groups) is not int or groups <= 0:
            raise ValueError("rotation groups must be a positive integer")
        indices = torch.triu_indices(GROUP_SIZE, GROUP_SIZE, offset=1, device=device)
        self.register_buffer("indices", indices, persistent=False)
        self.register_buffer("initial_r64", r64_matrix(device=device))
        self.register_buffer("identity", torch.eye(GROUP_SIZE, device=device), persistent=False)
        self.coefficients = nn.Parameter(torch.zeros(groups, indices.shape[1], device=device))

    def forward(self):
        theta = self.coefficients
        skew = theta.new_zeros(theta.shape[0], GROUP_SIZE, GROUP_SIZE)
        rows, cols = self.indices
        skew[:, rows, cols] = theta
        skew[:, cols, rows] = -theta
        half = skew * 0.5
        return torch.linalg.solve(self.identity - half, self.identity + half) @ self.initial_r64


def apply_group_matrix(x, matrix, *, head_dim=None):
    """Apply a different matrix to each contiguous input G64; retain all axes."""
    if x.ndim < 1 or x.shape[-1] % GROUP_SIZE:
        raise ValueError("input width must be divisible by 64")
    width = x.shape[-1]
    groups = width // GROUP_SIZE
    if tuple(matrix.shape) != (groups, GROUP_SIZE, GROUP_SIZE):
        raise ValueError(f"expected {groups} G64 matrices, got {tuple(matrix.shape)}")
    if head_dim is not None and (head_dim <= 0 or head_dim % GROUP_SIZE or width % head_dim):
        raise ValueError("G64 groups must remain within complete attention heads")
    blocks = x.to(matrix.dtype).reshape(-1, groups, GROUP_SIZE)
    return torch.einsum("ngk,gkj->ngj", blocks, matrix).reshape(x.shape)
