#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.config import (
    DEFAULT_FREE_RUN_MAX_NEW_TOKENS,
    DEFAULT_FREE_RUN_SAMPLES,
    DEFAULT_PHASEA_ROOT,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.gpu_pool import (
    available_gpus,
    cuda_env,
)

EXP_DIR = Path(__file__).resolve().parent
VARIANTS = ("E0", "E1", "E2", "E3", "E4")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_root", required=True)
    p.add_argument("--phasea_root", default=str(DEFAULT_PHASEA_ROOT))
    p.add_argument("--max_samples", type=int, default=DEFAULT_FREE_RUN_SAMPLES)
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_FREE_RUN_MAX_NEW_TOKENS)
    p.add_argument("--max_parallel", type=int, default=2)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def launch_one(args: argparse.Namespace, variant: str, gpu_pair: list[int]) -> tuple[subprocess.Popen, object, str]:
    run_root = Path(args.run_root)
    out_dir = run_root / "free_run" / variant
    normalized = run_root / "normalized" / f"{variant}.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized.parent.mkdir(parents=True, exist_ok=True)
    if normalized.is_file() and normalized.stat().st_size > 0 and not args.force:
        print(f"[skip] {variant}: {normalized}", flush=True)
        return None, None, variant  # type: ignore[return-value]
    log_handle = (out_dir / "capture.log").open("w", encoding="utf-8")
    cmd = [
        sys.executable,
        str(EXP_DIR / "launch_capture.py"),
        "--variant",
        variant,
        "--phasea_root",
        args.phasea_root,
        "--output_dir",
        str(out_dir),
        "--profile",
        "greedy",
        "--max_samples",
        str(args.max_samples),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--tensor_parallel_size",
        "2",
    ]
    print(f"[launch] {variant}: GPUs={gpu_pair}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=cuda_env(gpu_pair),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, log_handle, variant


def normalize(run_root: Path, variant: str) -> None:
    cmd = [
        sys.executable,
        str(EXP_DIR / "normalize_capture.py"),
        "--capture_dir",
        str(run_root / "free_run" / variant),
        "--output",
        str(run_root / "normalized" / f"{variant}.jsonl"),
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main() -> None:
    args = parse_args()
    gpus = available_gpus()
    if len(gpus) < 2:
        raise RuntimeError(f"free-run requires at least 2 available GPUs, got {gpus}")
    pairs = [gpus[i : i + 2] for i in range(0, len(gpus) - 1, 2)]
    max_parallel = min(max(args.max_parallel, 1), len(pairs))
    run_root = Path(args.run_root)
    index = 0
    while index < len(VARIANTS):
        jobs: list[tuple[subprocess.Popen, object, str]] = []
        for slot in range(max_parallel):
            if index >= len(VARIANTS):
                break
            variant = VARIANTS[index]
            index += 1
            job = launch_one(args, variant, pairs[slot])
            if job[0] is not None:
                jobs.append(job)
        for proc, log_handle, variant in jobs:
            rc = proc.wait()
            log_handle.close()
            if rc != 0:
                raise RuntimeError(
                    f"free-run {variant} failed with rc={rc}; see {run_root / 'free_run' / variant / 'capture.log'}"
                )
            normalize(run_root, variant)
            print(f"[done] {variant}", flush=True)


if __name__ == "__main__":
    main()
