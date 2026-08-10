from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HessianSnapshot:
    matrix: torch.Tensor
    num_tokens: int
    diagonal_mean: float
    dead_columns: torch.Tensor


@dataclass(frozen=True)
class OBSSystem:
    upper_inverse_cholesky: torch.Tensor
    column_order: torch.Tensor
    inverse_column_order: torch.Tensor
    damp_value: float
    diagonal_mean: float
    dead_columns: torch.Tensor


class HessianAccumulator:
    def __init__(self, dimension: int, device: torch.device, context: str) -> None:
        if dimension < 1:
            raise ValueError(f"{context}: dimension must be >= 1, got {dimension}")
        self.dimension = int(dimension)
        self.device = torch.device(device)
        self.context = context
        self._gram = torch.zeros(
            (self.dimension, self.dimension),
            dtype=torch.float32,
            device=self.device,
        )
        self._num_tokens = 0
        self._finalized = False

    def add_batch(self, inputs: torch.Tensor) -> None:
        if self._finalized:
            raise RuntimeError(f"{self.context}: cannot add_batch after finalize")
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(f"{self.context}: inputs must be a Tensor")
        if not inputs.is_floating_point():
            raise TypeError(
                f"{self.context}: inputs must be floating, got {inputs.dtype}"
            )
        if inputs.ndim < 2:
            raise ValueError(
                f"{self.context}: inputs rank must be >= 2, got {inputs.ndim}"
            )
        if int(inputs.shape[-1]) != self.dimension:
            raise ValueError(
                f"{self.context}: last dim {int(inputs.shape[-1])} != "
                f"dimension {self.dimension}"
            )
        if not torch.isfinite(inputs).all():
            raise ValueError(f"{self.context}: inputs contain non-finite values")
        x2d = inputs.detach().float().reshape(-1, self.dimension)
        if x2d.shape[0] < 1:
            raise ValueError(f"{self.context}: empty token batch")
        self._gram.add_(x2d.t().matmul(x2d).to(device=self.device))
        self._num_tokens += int(x2d.shape[0])

    def finalize(self) -> HessianSnapshot:
        if self._finalized:
            raise RuntimeError(f"{self.context}: finalize called twice")
        if self._num_tokens < 1:
            raise RuntimeError(f"{self.context}: cannot finalize with zero batches")
        matrix = self._gram.mul(2.0 / self._num_tokens)
        matrix = 0.5 * (matrix + matrix.t())
        if not torch.isfinite(matrix).all():
            raise RuntimeError(f"{self.context}: finalized Hessian is non-finite")
        if matrix.dtype != torch.float32:
            raise RuntimeError(f"{self.context}: Hessian must be float32")
        diagonal = torch.diag(matrix)
        diagonal_mean = float(diagonal.mean().item())
        dead_columns = diagonal == 0
        self._finalized = True
        return HessianSnapshot(
            matrix=matrix,
            num_tokens=self._num_tokens,
            diagonal_mean=diagonal_mean,
            dead_columns=dead_columns.detach().clone(),
        )


def _validate_column_order(column_order: torch.Tensor, size: int, context: str) -> None:
    if not isinstance(column_order, torch.Tensor):
        raise TypeError(f"{context}: column_order must be a Tensor")
    if column_order.dtype != torch.int64:
        raise TypeError(
            f"{context}: column_order must be int64, got {column_order.dtype}"
        )
    if column_order.ndim != 1 or column_order.numel() != size:
        raise ValueError(
            f"{context}: column_order length mismatch, expected {size}, "
            f"got {tuple(column_order.shape)}"
        )
    expected = torch.arange(size, dtype=torch.int64)
    if not torch.equal(torch.sort(column_order.cpu()).values, expected):
        raise ValueError(f"{context}: column_order is not bijective")


def build_obs_system(
    hessian: HessianSnapshot,
    column_order: torch.Tensor,
    percdamp: float,
    context: str,
) -> OBSSystem:
    if not (0.0 < float(percdamp) <= 1.0):
        raise ValueError(
            f"{context}: percdamp must satisfy 0 < percdamp <= 1, got {percdamp}"
        )
    if hessian.matrix.ndim != 2 or hessian.matrix.shape[0] != hessian.matrix.shape[1]:
        raise ValueError(f"{context}: Hessian must be square")
    size = int(hessian.matrix.shape[0])
    _validate_column_order(column_order, size, context)

    order = column_order.to(device=hessian.matrix.device)
    h = hessian.matrix.index_select(0, order).index_select(1, order).clone()
    diag_idx = torch.arange(h.shape[0], device=h.device)
    dead = torch.diag(h) == 0
    h[diag_idx[dead], diag_idx[dead]] = 1.0
    diagonal_mean = float(torch.diag(h).mean().item())
    damp_value = float(percdamp) * diagonal_mean
    h[diag_idx, diag_idx] += damp_value

    try:
        lower = torch.linalg.cholesky(h)
        h_inverse = torch.cholesky_inverse(lower)
        upper_inverse_cholesky = torch.linalg.cholesky(h_inverse, upper=True)
    except Exception as exc:
        diag = torch.diag(h)
        raise RuntimeError(
            f"{context}: Cholesky failed; "
            f"diag_min={float(diag.min().item())}, "
            f"diag_max={float(diag.max().item())}, "
            f"diagonal_mean={diagonal_mean}, "
            f"damp_value={damp_value}"
        ) from exc

    inverse_order = torch.empty_like(order)
    inverse_order[order] = torch.arange(order.numel(), device=order.device)
    return OBSSystem(
        upper_inverse_cholesky=upper_inverse_cholesky,
        column_order=order.detach().cpu().clone(),
        inverse_column_order=inverse_order.detach().cpu().clone(),
        damp_value=damp_value,
        diagonal_mean=diagonal_mean,
        dead_columns=dead.detach().cpu().clone(),
    )
