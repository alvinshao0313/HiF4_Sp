"""Dense dual-scale GEMM over separate BF16 operands and FP32 outer scales."""
import torch
from vllm.triton_utils import tl, triton

from .scale_methods import normalized_operand


@triton.jit
def _scaled_dot(A, W, SA, SW, BIAS, OUT, M, N: tl.constexpr, K: tl.constexpr,
                G: tl.constexpr, HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    groups: tl.constexpr = tl.cdiv(K, G)
    total = tl.full((BM, BN), 0., tl.float32)
    for g in range(groups):
        partial = tl.full((BM, BN), 0., tl.float32)
        for tile in tl.static_range(G // BK):
            k = g * G + tile * BK + tl.arange(0, BK)
            a = tl.load(A + rows[:, None] * K + k[None, :], (rows[:, None] < M) & (k[None, :] < K), other=0.)
            w = tl.load(W + cols[:, None] * K + k[None, :], (cols[:, None] < N) & (k[None, :] < K), other=0.)
            partial = tl.dot(a, tl.trans(w), partial)
        sa = tl.load(SA + rows * groups + g, rows < M, other=0.)
        sw = tl.load(SW + cols * groups + g, cols < N, other=0.)
        total += partial * (sa[:, None] * sw[None, :])
    if HAS_BIAS:
        total += tl.load(BIAS + cols, cols < N, other=0.).to(tl.float32)[None, :]
    tl.store(OUT + rows[:, None] * N + cols[None, :], total, (rows[:, None] < M) & (cols[None, :] < N))


def dualscale_linear(x, weight, outer, group, bias=None):
    n, k = weight.shape
    a, sa = normalized_operand(x.reshape(-1, k), group)
    if x.is_cuda:
        out = x.new_empty(a.shape[0], n)
        if out.numel():
            _scaled_dot[(triton.cdiv(a.shape[0], 16), triton.cdiv(n, 64))](
                a, weight, sa, outer, out if bias is None else bias, out, a.shape[0], n, k, group,
                bias is not None, 16, 64, 32, num_warps=4, enable_fp_fusion=False)
    else:
        out = torch.zeros((a.shape[0], n), device=x.device, dtype=torch.float32)
        for g in range(outer.shape[1]):
            cols = slice(g * group, (g + 1) * group)
            partial = a[:, cols].float() @ weight[:, cols].float().T
            out += partial * (sa[:, g, None] * outer[None, :, g])
        if bias is not None:
            out += bias.float()
    return out.to(x.dtype).reshape(*x.shape[:-1], n)
