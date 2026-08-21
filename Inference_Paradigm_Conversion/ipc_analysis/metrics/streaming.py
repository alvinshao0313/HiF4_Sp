"""Streaming error accumulator: scalar stats only, no full activation residency."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from Inference_Paradigm_Conversion.ipc_analysis.metrics.tensor_metrics import compute_pair_metrics


@dataclass
class ErrorAccumulator:
    """FP64 streaming sums for NMSE/SQNR/cosine/MAE + approximate abs-error percentiles.

    Percentiles use a fixed reservoir of abs-error samples (deterministic subsample)
    so memory stays bounded for 8B analysis.
    """

    reservoir_size: int = 200_000
    numel: int = 0
    reference_energy: float = 0.0
    target_energy: float = 0.0
    error_energy: float = 0.0
    dot: float = 0.0
    abs_error_sum: float = 0.0
    signed_error_sum: float = 0.0
    max_abs_error: float = 0.0
    _reservoir: list[float] = field(default_factory=list)
    _seen_for_reservoir: int = 0

    def update(self, reference: torch.Tensor, target: torch.Tensor) -> None:
        if reference.shape != target.shape:
            raise ValueError(
                f"shape mismatch: {tuple(reference.shape)} vs {tuple(target.shape)}"
            )
        ref = reference.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
        tgt = target.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
        err = tgt - ref
        self.numel += int(ref.numel())
        self.reference_energy += float(torch.dot(ref, ref).item())
        self.target_energy += float(torch.dot(tgt, tgt).item())
        self.error_energy += float(torch.dot(err, err).item())
        self.dot += float(torch.dot(ref, tgt).item())
        abs_err = err.abs()
        self.abs_error_sum += float(abs_err.sum().item())
        self.signed_error_sum += float(err.sum().item())
        if abs_err.numel():
            self.max_abs_error = max(self.max_abs_error, float(abs_err.max().item()))
        # Deterministic reservoir: keep first N, then stride-subsample further chunks.
        flat = abs_err.tolist()
        for v in flat:
            self._seen_for_reservoir += 1
            if len(self._reservoir) < self.reservoir_size:
                self._reservoir.append(float(v))
            else:
                # Replace with decreasing probability; deterministic via index.
                # Use Vitter-like but deterministic: replace position = seen % size when
                # seen is power-aligned — simpler: every k-th after full.
                if self._seen_for_reservoir % (self._seen_for_reservoir // self.reservoir_size + 1) == 0:
                    idx = self._seen_for_reservoir % self.reservoir_size
                    self._reservoir[idx] = float(v)

    def finalize(self) -> dict[str, float]:
        n = self.numel
        if n == 0:
            return compute_pair_metrics(torch.zeros(0), torch.zeros(0))

        def _safe_div(num: float, den: float) -> float:
            if den == 0.0:
                return 0.0 if num == 0.0 else 1.0e300
            return num / den

        nmse = _safe_div(self.error_energy, self.reference_energy)
        if self.error_energy == 0.0:
            sqnr = 1.0e300 if self.reference_energy > 0 else 0.0
        elif self.reference_energy == 0.0:
            sqnr = -1.0e300
        else:
            sqnr = 10.0 * math.log10(self.reference_energy / self.error_energy)
        ref_norm = math.sqrt(self.reference_energy)
        tgt_norm = math.sqrt(self.target_energy)
        if ref_norm == 0.0 and tgt_norm == 0.0:
            cosine = 1.0
        elif ref_norm == 0.0 or tgt_norm == 0.0:
            cosine = 0.0
        else:
            cosine = self.dot / (ref_norm * tgt_norm)
        relative_norm_change = (
            _safe_div(tgt_norm - ref_norm, ref_norm) if ref_norm != 0.0 else (0.0 if tgt_norm == 0.0 else 1.0e300)
        )
        sorted_abs = torch.tensor(sorted(self._reservoir), dtype=torch.float64) if self._reservoir else torch.zeros(0)

        def _pct(q: float) -> float:
            if sorted_abs.numel() == 0:
                return 0.0
            pos = (q / 100.0) * (sorted_abs.numel() - 1)
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            if lo == hi:
                return float(sorted_abs[lo].item())
            w = pos - lo
            return float((sorted_abs[lo] * (1 - w) + sorted_abs[hi] * w).item())

        if self._reservoir and self.error_energy > 0:
            sq = torch.tensor(self._reservoir, dtype=torch.float64) ** 2
            k = max(1, int(math.ceil(0.01 * sq.numel())))
            top1 = float(torch.topk(sq, k=k).values.sum().item()) / float(sq.sum().item())
        else:
            top1 = 0.0

        return {
            "nmse": nmse,
            "sqnr_db": sqnr,
            "cosine": cosine,
            "mae": self.abs_error_sum / n,
            "mean_signed_error": self.signed_error_sum / n,
            "max_abs_error": self.max_abs_error,
            "relative_norm_change": relative_norm_change,
            "reference_energy": self.reference_energy,
            "target_energy": self.target_energy,
            "error_energy": self.error_energy,
            "error_p50": _pct(50),
            "error_p90": _pct(90),
            "error_p99": _pct(99),
            "error_p99_9": _pct(99.9),
            "top1pct_error_energy_share": top1,
            "numel": float(n),
        }
