"""Projection-local Adam/analytic comparison with actual HiF4 weight QDQ.

Weights here use [out, in]; H uses [group, 64, 64] and equals M.transpose.
The cached quadratic describes real-valued products of BF16 QDQ operands.
Validation always uses the BF16 Linear output, including its final rounding.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import torch
import torch.nn.functional as F

from ..diag_gradient.r64_transform import r64_matrix
from ..e2e_diag_reconstruction.core.moe_semantic_hif4 import qdq_hif4_ste_bf16
from ..non_equivalent_reconstruction.transforms import block_right
from ...src.formats import qdq_hif4_direct


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def initial_h(weight):
    if weight.ndim != 2 or weight.shape[1] % 64:
        raise ValueError("weight must have shape [out, in] with in divisible by 64")
    return r64_matrix(device=weight.device).T.repeat(weight.shape[1] // 64, 1, 1)


def quantized_weight(weight, h, *, ste=False):
    folded = block_right(weight, h.transpose(-1, -2))
    return qdq_hif4_ste_bf16(folded) if ste else qdq_hif4_direct(folded, output_dtype=torch.bfloat16)


@dataclass
class ProjectionData:
    activation: torch.Tensor
    target: torch.Tensor

    def __post_init__(self):
        a, y = self.activation, self.target
        if a.ndim != 2 or y.ndim != 2 or a.shape[0] != y.shape[0]:
            raise ValueError("projection data must contain aligned 2D matrices")
        if a.dtype != torch.bfloat16 or y.dtype != torch.bfloat16 or a.device != y.device:
            raise ValueError("projection operands/targets must be BF16 on the same device")
        if a.shape[1] % 64 or not torch.isfinite(a).all() or not torch.isfinite(y).all():
            raise ValueError("invalid projection dimension or non-finite data")

    @property
    def count(self):
        return self.activation.shape[0]


@dataclass
class Statistics:
    gram: torch.Tensor
    cross: torch.Tensor
    target_energy_per_token: torch.Tensor
    tokens: int


@torch.no_grad()
def collect_statistics(data, *, chunk_tokens=4096):
    if data.count == 0 or chunk_tokens <= 0:
        raise ValueError("statistics require nonempty input and a positive chunk size")
    a, y = data.activation, data.target
    gram = torch.zeros(a.shape[1], a.shape[1], dtype=torch.float64, device=a.device)
    cross = torch.zeros(a.shape[1], y.shape[1], dtype=torch.float64, device=a.device)
    energy = torch.zeros((), dtype=torch.float64, device=a.device)
    for start in range(0, data.count, chunk_tokens):
        aa = a[start:start + chunk_tokens].double()
        yy = y[start:start + chunk_tokens].double()
        gram.addmm_(aa.T, aa)
        cross.addmm_(aa.T, yy)
        energy += yy.square().sum()
    # Normalizing by token count makes damping independent of corpus length.
    return Statistics(gram / data.count, cross / data.count, energy / data.count, data.count)


def quadratic_mse(stats, v):
    v = v.double()
    value = (v * (stats.gram @ v)).sum() - 2 * (v * stats.cross).sum() + stats.target_energy_per_token
    return float(value / v.shape[1])


def delta_loss(gram_ii, residual_i, delta_v):
    return 2 * (delta_v * residual_i).sum() + (delta_v * (gram_ii @ delta_v)).sum()


def psd_eigh(matrix):
    values, vectors = torch.linalg.eigh(matrix)
    tolerance = 8 * matrix.shape[0] * torch.finfo(matrix.dtype).eps * values.abs().max()
    if values.min() < -tolerance:
        raise ArithmeticError("Gram eigendecomposition is materially non-PSD")
    # Remove only eigenvalue roundoff bounded by the eigensystem backward error.
    return values.clamp_min(0), vectors


def solve_increment(gram, weight_gram, rhs, damping):
    if not torch.isfinite(damping) or damping <= 0:
        raise ValueError("positive finite damping is required")
    ag, u = psd_eigh(gram)
    bk, v = psd_eigh(weight_gram)
    return u @ ((u.T @ rhs @ v) / (ag[:, None] * bk[None, :] + damping)) @ v.T


@torch.no_grad()
def evaluate_projection(data, weight, h):
    if data.count == 0:
        return None
    prediction = F.linear(data.activation, quantized_weight(weight, h))
    error = prediction.float() - data.target.float()
    numerator = error.double().square().sum()
    denominator = data.target.double().square().sum()
    return {"mse": float(numerator / error.numel()),
            "nmse": float(numerator / denominator) if denominator > 0 else None,
            "tokens": data.count, "elements": error.numel()}


@dataclass
class FitResult:
    selected_h: torch.Tensor
    final_h: torch.Tensor
    report: dict


def fit_projection(weight, training, validation, method, *, callback=None):
    """Run the approved fixed diagnostic budget; input tensors are never changed."""
    if method not in {"adam", "analytic"}:
        raise ValueError(f"unknown method {method}")
    if weight.requires_grad:
        raise ValueError("base weights must be frozen")
    if any(data.activation.shape[1] != weight.shape[1] or data.target.shape[1] != weight.shape[0]
           or data.activation.device != weight.device for data in (training, validation)):
        raise ValueError("projection data/weight shapes or devices disagree")
    synchronize(weight.device)
    started = time.monotonic()
    h = initial_h(weight)
    best = h.detach().clone()
    best_mse, best_iteration = float("inf"), 0
    curve = []
    measurement_seconds = 0.0

    def measure(iteration, **extra):
        nonlocal best, best_mse, best_iteration, measurement_seconds
        synchronize(weight.device)
        measure_started = time.monotonic()
        train_metric = evaluate_projection(training, weight, h)
        val_metric = evaluate_projection(validation, weight, h)
        if val_metric is not None and val_metric["mse"] < best_mse:
            best_mse, best_iteration = val_metric["mse"], iteration
            best = h.detach().clone()
        synchronize(weight.device)
        entry = {"iteration": iteration, "train": train_metric, "validation": val_metric,
                 "elapsed_seconds": time.monotonic() - started, **extra}
        curve.append(entry)
        if callback is not None:
            callback(entry)
        measurement_seconds += time.monotonic() - measure_started

    measure(0)
    skip = "no_training_tokens" if training.count == 0 else (
        "no_validation_tokens" if validation.count == 0 else None)
    statistics_seconds = 0.0
    if skip is None and method == "adam":
        h.requires_grad_(True)
        optimizer = torch.optim.Adam([h], lr=3e-4, weight_decay=0.)
        for iteration in range(1, 21):
            optimizer.zero_grad(set_to_none=True)
            prediction = F.linear(training.activation, quantized_weight(weight, h, ste=True))
            loss = (prediction.float() - training.target.float()).square().mean()
            if not torch.isfinite(loss):
                raise ArithmeticError("non-finite Adam projection objective")
            loss.backward()
            if h.grad is None or not torch.isfinite(h.grad).all():
                raise ArithmeticError("missing/non-finite H gradient")
            optimizer.step()
            del prediction, loss
            measure(iteration)
        h.requires_grad_(False)
    elif skip is None:
        synchronize(weight.device)
        stat_started = time.monotonic()
        stats = collect_statistics(training)
        synchronize(weight.device)
        statistics_seconds = time.monotonic() - stat_started
        v = quantized_weight(weight, h).T.double()
        residual = stats.gram @ v - stats.cross
        decomposition_started = time.monotonic()
        blocks = []
        for i in range(h.shape[0]):
            span = slice(64 * i, 64 * (i + 1))
            gi = stats.gram[span, span]
            bi = weight[:, span].T.double()
            ki = bi @ bi.T
            damping = .01 * gi.diag().mean() * ki.diag().mean()
            if damping == 0:
                blocks.append(None)
                continue
            if damping < 0 or not torch.isfinite(damping):
                raise ArithmeticError("invalid Gram damping")
            ag, u = psd_eigh(gi)
            bk, vv = psd_eigh(ki)
            blocks.append((gi, bi, u, vv, ag[:, None] * bk[None, :] + damping))
        synchronize(weight.device)
        decomposition_seconds = time.monotonic() - decomposition_started
        for sweep in range(1, 3):
            accepted, changes = 0, []
            for i, block in enumerate(blocks):
                if block is None:
                    changes.append({"block": i, "reason": "zero_input_or_weight_energy"})
                    continue
                span = slice(64 * i, 64 * (i + 1))
                gi, bi, u, vv, divisor = block
                rhs = -residual[span] @ bi.T
                delta_h = u @ ((u.T @ rhs @ vv) / divisor) @ vv.T
                if not torch.isfinite(delta_h).all():
                    raise ArithmeticError("non-finite analytic increment")
                accepted_step = 0.0
                for step in (1., .5, .25, .125, .0625):
                    proposal = (h[i].double() + step * delta_h).float()
                    # Quantize across the INPUT dimension after transposition.
                    candidate_v = qdq_hif4_direct(
                        weight[:, span].float() @ proposal.T, output_dtype=torch.bfloat16).T.double()
                    delta_v = candidate_v - v[span]
                    change = delta_loss(gi, residual[span], delta_v)
                    if not torch.isfinite(change):
                        raise ArithmeticError("non-finite cached QDQ loss change")
                    if change < 0:
                        h[i].copy_(proposal)
                        v[span].copy_(candidate_v)
                        residual.addmm_(stats.gram[:, span], delta_v)
                        accepted += 1
                        accepted_step = step
                        break
                changes.append({"block": i, "step": accepted_step,
                                "cached_mse_change": float(change / weight.shape[0]) if accepted_step else 0.})
            recomputed_residual = stats.gram @ v - stats.cross
            residual_error = float((residual - recomputed_residual).norm() /
                                   recomputed_residual.norm().clamp_min(torch.finfo(torch.float64).tiny))
            if residual_error > 1e-9:
                raise ArithmeticError(f"cross-block residual drift: {residual_error}")
            measure(sweep, accepted_blocks=accepted, block_updates=changes,
                    cached_real_product_mse=quadratic_mse(stats, v), residual_relative_error=residual_error)
    synchronize(weight.device)
    fit_seconds = time.monotonic() - started
    report = {"method": method, "status": "skipped" if skip else "complete", "skip_reason": skip,
              "selected_iteration": best_iteration, "curve": curve,
              "fit_seconds_including_validation": fit_seconds,
              "measurement_seconds": measurement_seconds,
              "solver_seconds_excluding_measurement": fit_seconds - measurement_seconds,
              "statistics_seconds": statistics_seconds,
              "eigendecomposition_seconds": decomposition_seconds if skip is None and method == "analytic" else 0.,
              "statistics_dtype": "float64" if method == "analytic" else None,
              "selection": "minimum BF16 projection validation MSE including initial R64",
              "objective_scope": "unweighted local Linear MSE; not full block loss",
              "budget": {"steps": 20, "lr": 3e-4, "weight_decay": 0., "scheduler": "constant"} if method == "adam"
                        else {"sweeps": 2, "relative_damping": .01, "steps": [1., .5, .25, .125, .0625]}}
    return FitResult(best.detach().cpu(), h.detach().cpu(), report)
