"""Normalized Sylvester H32 in natural order, with no signs or permutations."""
import math

import torch

INV_SQRT_32 = 1. / math.sqrt(32)


def fht32_matrix(*, device=None, dtype=torch.float32):
    h2 = torch.tensor([[1., 1.], [1., -1.]], device=device, dtype=dtype)
    matrix = h2
    for _ in range(4):
        matrix = torch.kron(matrix, h2)
    return matrix * INV_SQRT_32


def apply_fht32(x):
    """Right-multiply contiguous K32 blocks, retaining FP32 until quantization."""
    if x.ndim < 1 or x.shape[-1] % 32:
        raise ValueError("FHT32 requires contiguous groups of 32 along the last dimension")
    values = x.float().reshape(-1, 32)
    for stride in (1, 2, 4, 8, 16):
        pairs = values.reshape(-1, 2, stride)
        left, right = pairs[:, 0], pairs[:, 1]
        values = torch.stack((left + right, left - right), 1).reshape(-1, 32)
    return (values * INV_SQRT_32).reshape(x.shape)


def fht32_reference(x):
    """Independent dense-matrix reference for the butterfly implementation."""
    return (x.float().reshape(-1, 32) @ fht32_matrix(device=x.device)).reshape(x.shape)
