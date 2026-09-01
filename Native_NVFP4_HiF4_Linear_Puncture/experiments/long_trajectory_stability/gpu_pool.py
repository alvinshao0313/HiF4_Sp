from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GPUInfo:
    index: int
    free_mib: int
    total_mib: int
    utilization: int

    @property
    def free_ratio(self) -> float:
        return self.free_mib / max(self.total_mib, 1)


def query_gpus() -> list[GPUInfo]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    rows: list[GPUInfo] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        values = [x.strip() for x in line.split(",")]
        if len(values) != 4:
            raise RuntimeError(f"unexpected nvidia-smi row: {line!r}")
        rows.append(GPUInfo(*(int(x) for x in values)))
    return rows


def available_gpus() -> list[int]:
    all_gpus = query_gpus()
    by_index = {g.index: g for g in all_gpus}
    project_pool = [int(x) for x in os.environ.get("PROJECT_GPU_POOL", "0,1,2,3").split(",") if x.strip()]
    explicit = os.environ.get("GPU_POOL", "").strip()
    if explicit:
        requested = [int(x) for x in explicit.split(",") if x.strip()]
        missing = [x for x in requested if x not in by_index]
        outside = [x for x in requested if x not in project_pool]
        if missing:
            raise RuntimeError(f"GPU_POOL contains unknown GPU ids: {missing}")
        if outside:
            raise RuntimeError(f"GPU_POOL contains ids outside PROJECT_GPU_POOL: {outside}")
        return requested
    min_ratio = float(os.environ.get("GPU_MIN_FREE_RATIO", "0.90"))
    max_util = int(os.environ.get("GPU_MAX_UTIL", "10"))
    return [
        idx
        for idx in project_pool
        if idx in by_index
        and by_index[idx].free_ratio >= min_ratio
        and by_index[idx].utilization <= max_util
    ]


def cuda_env(gpu_ids: list[int]) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_ids)
    return env
