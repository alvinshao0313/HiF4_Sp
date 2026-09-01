# SPDX-License-Identifier: Apache-2.0
"""Triton transforms for HiF4 R64 and Online-DIAG inference."""

from __future__ import annotations

import torch

from vllm.model_executor.layers.quantization.hif4_triton import (
    _HIF4_GROUP_SIZE,
    _hif4_fake_quant_block_tl,
)
from vllm.triton_utils import tl, triton


def _work_tensor(x: torch.Tensor, name: str) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError(f"{name} requires CUDA")
    if x.ndim == 0:
        raise ValueError(f"{name} requires at least one dimension")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"unsupported {name} dtype: {x.dtype}")
    return x if x.is_contiguous() else x.contiguous()


def _check_r64_width(cols: int, head_dim: int | None = None) -> None:
    if cols % _HIF4_GROUP_SIZE:
        raise ValueError(f"R64 requires width divisible by 64, got {cols}")
    if head_dim is not None:
        if head_dim <= 0 or head_dim % 64 or cols % head_dim:
            raise ValueError(
                f"invalid per-head R64 layout: width={cols}, head_dim={head_dim}"
            )


def _result_tensor(
    work: torch.Tensor,
    out: torch.Tensor | None,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if out is None:
        return torch.empty(shape, device=work.device, dtype=work.dtype)
    if tuple(out.shape) != shape:
        raise ValueError(f"output shape mismatch: {tuple(out.shape)} != {shape}")
    if out.device != work.device or out.dtype != work.dtype or not out.is_contiguous():
        raise ValueError("output must be contiguous and match input device/dtype")
    return out


@triton.jit
def _h4(a, b, c, d, digit):
    y0 = a + b + c - d
    y1 = a + b - c + d
    y2 = a - b + c + d
    y3 = -a + b + c + d
    return tl.where(
        digit == 0,
        y0,
        tl.where(digit == 1, y1, tl.where(digit == 2, y2, y3)),
    ) * 0.5


@triton.jit
def _r64_stage(
    x_ptr,
    out_ptr,
    cols: tl.constexpr,
    groups: tl.constexpr,
    STRIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // groups
    group = pid - row * groups
    lane = tl.arange(0, 64)
    digit = (lane // STRIDE) % 4
    base = (lane // (4 * STRIDE)) * (4 * STRIDE) + lane % STRIDE
    g0 = group * 64
    r0 = row * cols
    p0 = g0 + base
    a = tl.load(x_ptr + r0 + p0).to(tl.float32)
    b = tl.load(x_ptr + r0 + p0 + STRIDE).to(tl.float32)
    c = tl.load(x_ptr + r0 + p0 + 2 * STRIDE).to(tl.float32)
    d = tl.load(x_ptr + r0 + p0 + 3 * STRIDE).to(tl.float32)
    tl.store(out_ptr + r0 + g0 + lane, _h4(a, b, c, d, digit))


@triton.jit
def _r64_final_qdq(x_ptr, out_ptr, cols: tl.constexpr, groups: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // groups
    group = pid - row * groups
    lane = tl.arange(0, 64)
    digit = (lane // 16) % 4
    base = lane % 16
    g0 = group * 64
    r0 = row * cols
    p0 = g0 + base
    a = tl.load(x_ptr + r0 + p0).to(tl.float32)
    b = tl.load(x_ptr + r0 + p0 + 16).to(tl.float32)
    c = tl.load(x_ptr + r0 + p0 + 32).to(tl.float32)
    d = tl.load(x_ptr + r0 + p0 + 48).to(tl.float32)
    y = _h4(a, b, c, d, digit)
    valid = lane < 64
    y = _hif4_fake_quant_block_tl(y, valid, 64)
    tl.store(out_ptr + r0 + g0 + lane, y)


@triton.jit
def _dense_diag_qdq(
    x_ptr,
    d_ptr,
    out_ptr,
    cols: tl.constexpr,
    groups: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // groups
    group = pid - row * groups
    lane = tl.arange(0, 64)
    col = group * 64 + lane
    y = tl.load(x_ptr + row * cols + col).to(tl.float32)
    y *= tl.load(d_ptr + col).to(tl.float32)
    valid = lane < 64
    y = _hif4_fake_quant_block_tl(y, valid, 64)
    tl.store(out_ptr + row * cols + col, y)


@triton.jit
def _dense_diag_r64_stage0(
    x_ptr,
    d_ptr,
    out_ptr,
    cols: tl.constexpr,
    groups: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // groups
    group = pid - row * groups
    lane = tl.arange(0, 64)
    digit = lane % 4
    base = (lane // 4) * 4
    g0 = group * 64
    r0 = row * cols
    p0 = g0 + base
    a = tl.load(x_ptr + r0 + p0).to(tl.float32) * tl.load(d_ptr + p0).to(tl.float32)
    b = tl.load(x_ptr + r0 + p0 + 1).to(tl.float32) * tl.load(d_ptr + p0 + 1).to(tl.float32)
    c = tl.load(x_ptr + r0 + p0 + 2).to(tl.float32) * tl.load(d_ptr + p0 + 2).to(tl.float32)
    d = tl.load(x_ptr + r0 + p0 + 3).to(tl.float32) * tl.load(d_ptr + p0 + 3).to(tl.float32)
    tl.store(out_ptr + r0 + g0 + lane, _h4(a, b, c, d, digit))


@triton.jit
def _r64_final_dense_diag_qdq(
    x_ptr,
    d_ptr,
    out_ptr,
    cols: tl.constexpr,
    groups: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // groups
    group = pid - row * groups
    lane = tl.arange(0, 64)
    digit = (lane // 16) % 4
    base = lane % 16
    g0 = group * 64
    r0 = row * cols
    p0 = g0 + base
    a = tl.load(x_ptr + r0 + p0).to(tl.float32)
    b = tl.load(x_ptr + r0 + p0 + 16).to(tl.float32)
    c = tl.load(x_ptr + r0 + p0 + 32).to(tl.float32)
    d = tl.load(x_ptr + r0 + p0 + 48).to(tl.float32)
    y = _h4(a, b, c, d, digit)
    col = g0 + lane
    y *= tl.load(d_ptr + col).to(tl.float32)
    valid = lane < 64
    y = _hif4_fake_quant_block_tl(y, valid, 64)
    tl.store(out_ptr + r0 + col, y)


def hif4_r64_quantize_hifx4_triton(
    x: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    head_dim: int | None = None,
) -> torch.Tensor:
    work = _work_tensor(x, "HiF4 R64 QDQ")
    cols = int(work.shape[-1])
    _check_r64_width(cols, head_dim)
    result = _result_tensor(work, out, tuple(work.shape))
    if work.numel() == 0:
        return result
    rows = work.numel() // cols
    groups = cols // 64
    tmp1 = torch.empty_like(work, dtype=torch.float32)
    tmp2 = torch.empty_like(work, dtype=torch.float32)
    grid = (rows * groups,)
    _r64_stage[grid](work, tmp1, cols=cols, groups=groups, STRIDE=1, num_warps=4)
    _r64_stage[grid](tmp1, tmp2, cols=cols, groups=groups, STRIDE=4, num_warps=4)
    _r64_final_qdq[grid](tmp2, result, cols=cols, groups=groups, num_warps=4)
    return result


def hif4_online_dense_qdq_triton(
    x: torch.Tensor,
    d: torch.Tensor,
    *,
    use_r64: bool,
    rot_order: str,
    head_dim: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    work = _work_tensor(x, "HiF4 Online dense QDQ")
    cols = int(work.shape[-1])
    if cols % 64:
        raise ValueError(f"Online HiF4 requires width divisible by 64, got {cols}")
    if d.device != work.device or d.dtype != torch.float32 or tuple(d.shape) != (cols,):
        raise ValueError(
            f"Online D must be fp32 [{cols}] on {work.device}, "
            f"got shape={tuple(d.shape)} dtype={d.dtype} device={d.device}"
        )
    if rot_order not in {"diag_then_rot", "rot_then_diag"}:
        raise ValueError(f"invalid rot_order={rot_order!r}")
    if use_r64:
        _check_r64_width(cols, head_dim)
    result = _result_tensor(work, out, tuple(work.shape))
    if work.numel() == 0:
        return result
    rows = work.numel() // cols
    groups = cols // 64
    grid = (rows * groups,)
    if not use_r64:
        _dense_diag_qdq[grid](work, d, result, cols=cols, groups=groups, num_warps=4)
        return result
    tmp1 = torch.empty_like(work, dtype=torch.float32)
    tmp2 = torch.empty_like(work, dtype=torch.float32)
    if rot_order == "diag_then_rot":
        _dense_diag_r64_stage0[grid](
            work, d, tmp1, cols=cols, groups=groups, num_warps=4
        )
        _r64_stage[grid](
            tmp1, tmp2, cols=cols, groups=groups, STRIDE=4, num_warps=4
        )
        _r64_final_qdq[grid](
            tmp2, result, cols=cols, groups=groups, num_warps=4
        )
    else:
        _r64_stage[grid](
            work, tmp1, cols=cols, groups=groups, STRIDE=1, num_warps=4
        )
        _r64_stage[grid](
            tmp1, tmp2, cols=cols, groups=groups, STRIDE=4, num_warps=4
        )
        _r64_final_dense_diag_qdq[grid](
            tmp2, d, result, cols=cols, groups=groups, num_warps=4
        )
    return result


@triton.jit
def _routed_diag_qdq(
    x_ptr,
    route_ids_ptr,
    expert_d_ptr,
    out_ptr,
    cols: tl.constexpr,
    groups: tl.constexpr,
    source_top_k: tl.constexpr,
    expert_stride: tl.constexpr,
):
    pid = tl.program_id(0)
    route_row = pid // groups
    group = pid - route_row * groups
    source_row = route_row // source_top_k
    expert = tl.load(route_ids_ptr + route_row).to(tl.int64)
    lane = tl.arange(0, 64)
    col = group * 64 + lane
    y = tl.load(x_ptr + source_row * cols + col).to(tl.float32)
    y *= tl.load(expert_d_ptr + expert * expert_stride + col).to(tl.float32)
    valid = lane < 64
    y = _hif4_fake_quant_block_tl(y, valid, 64)
    tl.store(out_ptr + route_row * cols + col, y)


@triton.jit
def _routed_r64_stage0(
    x_ptr,
    route_ids_ptr,
    expert_d_ptr,
    out_ptr,
    cols: tl.constexpr,
    groups: tl.constexpr,
    source_top_k: tl.constexpr,
    expert_stride: tl.constexpr,
    APPLY_DIAG: tl.constexpr,
):
    pid = tl.program_id(0)
    route_row = pid // groups
    group = pid - route_row * groups
    source_row = route_row // source_top_k
    expert = tl.load(route_ids_ptr + route_row).to(tl.int64)
    lane = tl.arange(0, 64)
    digit = lane % 4
    base = (lane // 4) * 4
    g0 = group * 64
    p0 = g0 + base
    src = source_row * cols
    a = tl.load(x_ptr + src + p0).to(tl.float32)
    b = tl.load(x_ptr + src + p0 + 1).to(tl.float32)
    c = tl.load(x_ptr + src + p0 + 2).to(tl.float32)
    d = tl.load(x_ptr + src + p0 + 3).to(tl.float32)
    if APPLY_DIAG:
        ds = expert * expert_stride
        a *= tl.load(expert_d_ptr + ds + p0).to(tl.float32)
        b *= tl.load(expert_d_ptr + ds + p0 + 1).to(tl.float32)
        c *= tl.load(expert_d_ptr + ds + p0 + 2).to(tl.float32)
        d *= tl.load(expert_d_ptr + ds + p0 + 3).to(tl.float32)
    tl.store(out_ptr + route_row * cols + g0 + lane, _h4(a, b, c, d, digit))


@triton.jit
def _r64_final_routed_diag_qdq(
    x_ptr,
    route_ids_ptr,
    expert_d_ptr,
    out_ptr,
    cols: tl.constexpr,
    groups: tl.constexpr,
    expert_stride: tl.constexpr,
    APPLY_DIAG: tl.constexpr,
):
    pid = tl.program_id(0)
    route_row = pid // groups
    group = pid - route_row * groups
    expert = tl.load(route_ids_ptr + route_row).to(tl.int64)
    lane = tl.arange(0, 64)
    digit = (lane // 16) % 4
    base = lane % 16
    g0 = group * 64
    r0 = route_row * cols
    p0 = g0 + base
    a = tl.load(x_ptr + r0 + p0).to(tl.float32)
    b = tl.load(x_ptr + r0 + p0 + 16).to(tl.float32)
    c = tl.load(x_ptr + r0 + p0 + 32).to(tl.float32)
    d = tl.load(x_ptr + r0 + p0 + 48).to(tl.float32)
    y = _h4(a, b, c, d, digit)
    col = g0 + lane
    if APPLY_DIAG:
        y *= tl.load(expert_d_ptr + expert * expert_stride + col).to(tl.float32)
    valid = lane < 64
    y = _hif4_fake_quant_block_tl(y, valid, 64)
    tl.store(out_ptr + r0 + col, y)


def hif4_online_routed_qdq_triton(
    x: torch.Tensor,
    route_expert_ids: torch.Tensor,
    expert_d: torch.Tensor,
    *,
    source_top_k: int,
    use_r64: bool,
    rot_order: str,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Route-expand, apply expert-specific Online transform, then HiF4 QDQ."""
    work = _work_tensor(x, "HiF4 Online routed QDQ")
    if source_top_k <= 0:
        raise ValueError("source_top_k must be positive")
    route_ids = route_expert_ids.reshape(-1)
    if not route_ids.is_cuda or route_ids.device != work.device:
        raise ValueError("route_expert_ids must be on x.device")
    route_ids = route_ids.contiguous()
    cols = int(work.shape[-1])
    if cols % 64:
        raise ValueError(f"Online routed HiF4 requires K divisible by 64, got {cols}")
    if (
        expert_d.device != work.device
        or expert_d.dtype != torch.float32
        or expert_d.ndim != 2
        or not expert_d.is_contiguous()
        or int(expert_d.shape[1]) != cols
    ):
        raise ValueError("expert_d must be contiguous fp32 [num_experts,K] on x.device")
    route_rows = int(route_ids.numel())
    source_rows = work.numel() // cols
    if source_rows * source_top_k != route_rows:
        raise ValueError(
            f"source/top-k mismatch: {source_rows}*{source_top_k} != {route_rows}"
        )
    if rot_order not in {"diag_then_rot", "rot_then_diag"}:
        raise ValueError(f"invalid rot_order={rot_order!r}")
    if use_r64:
        _check_r64_width(cols)
    result = _result_tensor(work, out, (route_rows, cols))
    if route_rows == 0:
        return result
    groups = cols // 64
    grid = (route_rows * groups,)
    expert_stride = int(expert_d.stride(0))
    if not use_r64:
        _routed_diag_qdq[grid](
            work,
            route_ids,
            expert_d,
            result,
            cols=cols,
            groups=groups,
            source_top_k=source_top_k,
            expert_stride=expert_stride,
            num_warps=4,
        )
        return result
    tmp1 = torch.empty((route_rows, cols), device=work.device, dtype=torch.float32)
    tmp2 = torch.empty_like(tmp1)
    apply_first = rot_order == "diag_then_rot"
    _routed_r64_stage0[grid](
        work,
        route_ids,
        expert_d,
        tmp1,
        cols=cols,
        groups=groups,
        source_top_k=source_top_k,
        expert_stride=expert_stride,
        APPLY_DIAG=apply_first,
        num_warps=4,
    )
    _r64_stage[grid](
        tmp1,
        tmp2,
        cols=cols,
        groups=groups,
        STRIDE=4,
        num_warps=4,
    )
    _r64_final_routed_diag_qdq[grid](
        tmp2,
        route_ids,
        expert_d,
        result,
        cols=cols,
        groups=groups,
        expert_stride=expert_stride,
        APPLY_DIAG=not apply_first,
        num_warps=4,
    )
    return result


@triton.jit
def _silu_mul_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = (gate / (1.0 + tl.exp(-gate))) * up
    tl.store(out_ptr + offsets, y, mask=mask)


def hif4_silu_mul_triton(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if gate.shape != up.shape or gate.device != up.device or gate.dtype != up.dtype:
        raise ValueError("gate/up must have matching shape/device/dtype")
    if not gate.is_cuda:
        raise ValueError("HiF4 Online silu_mul requires CUDA")
    gate_c = gate if gate.is_contiguous() else gate.contiguous()
    up_c = up if up.is_contiguous() else up.contiguous()
    result = _result_tensor(gate_c, out, tuple(gate_c.shape))
    n = gate_c.numel()
    if n:
        block = 256
        _silu_mul_kernel[(triton.cdiv(n, block),)](
            gate_c,
            up_c,
            result,
            n,
            BLOCK=block,
            num_warps=4,
        )
    return result
