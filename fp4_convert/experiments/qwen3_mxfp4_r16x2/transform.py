"""R32 = (H2 / sqrt(2)) tensor R4 tensor R4, with regular normalized R4."""
import math

import torch

INV_SQRT_2 = 1. / math.sqrt(2)


def r16x2_matrix(*, device=None, dtype=torch.float32):
    r4 = torch.tensor([[1., 1., 1., -1.], [1., 1., -1., 1.],
                       [1., -1., 1., 1.], [-1., 1., 1., 1.]], device=device, dtype=dtype) * .5
    h2 = torch.tensor([[1., 1.], [1., -1.]], device=device, dtype=dtype) * INV_SQRT_2
    return torch.kron(h2, torch.kron(r4, r4))


def apply_r16x2(x):
    """FP32 right multiplication on contiguous K32 blocks, without BF16 rounding."""
    if x.ndim < 1 or x.shape[-1] % 32:
        raise ValueError("R16x2 requires complete contiguous K32 blocks")
    values = x.float().reshape(-1, 32)
    for stride in (1, 4):
        groups = values.reshape(-1, 4, stride)
        a, b, c, d = groups.unbind(1)
        values = (torch.stack((a+b+c-d, a+b-c+d, a-b+c+d, -a+b+c+d), 1) * .5).reshape(-1, 32)
    left, right = values[:, :16], values[:, 16:]
    return (torch.cat((left+right, left-right), 1) * INV_SQRT_2).reshape(x.shape)


def r16x2_reference(x):
    """Independent dense matrix reference, separate from the staged implementation."""
    if x.ndim < 1 or x.shape[-1] % 32:
        raise ValueError("R16x2 requires complete contiguous K32 blocks")
    return (x.float().reshape(-1, 32) @ r16x2_matrix(device=x.device)).reshape(x.shape)
