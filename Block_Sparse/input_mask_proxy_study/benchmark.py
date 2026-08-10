from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class LatencyStats:
    median_ms: float
    p10_ms: float
    p90_ms: float
    repeats: int
    peak_memory_bytes: int


def benchmark_cuda(
    fn: Callable[[], None],
    warmup: int,
    repeats: int,
) -> LatencyStats:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark_cuda")
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    times: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    peak = int(torch.cuda.max_memory_allocated())
    t = torch.tensor(times, dtype=torch.float64)
    return LatencyStats(
        median_ms=float(t.median().item()),
        p10_ms=float(torch.quantile(t, 0.10).item()),
        p90_ms=float(torch.quantile(t, 0.90).item()),
        repeats=repeats,
        peak_memory_bytes=peak,
    )
