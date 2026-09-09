#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
COMPARE = REPO_ROOT / "Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/compare_free_runs.py"
BUILD_PROBES = REPO_ROOT / "Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/build_probe_plan.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_root", required=True)
    p.add_argument("--num_samples", type=int, default=16)
    p.add_argument("--probes_per_bin", type=int, default=4)
    p.add_argument("--max_decode_index", type=int, default=12287)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.run_root).resolve()
    isolated = root / "isolated"
    analysis = root / "analysis"
    for variant in ("E0", "E1", "E2", "E3", "E4"):
        path = isolated / f"{variant}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
    subprocess.run(
        [
            sys.executable,
            str(COMPARE),
            "--normalized_dir",
            str(isolated),
            "--output_dir",
            str(analysis),
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(BUILD_PROBES),
            "--e0_trajectories",
            str(isolated / "E0.jsonl"),
            "--divergence_events",
            str(analysis / "divergence_events.jsonl"),
            "--output",
            str(analysis / "probe_plan.json"),
            "--num_samples",
            str(args.num_samples),
            "--probes_per_bin",
            str(args.probes_per_bin),
            "--max_decode_index",
            str(args.max_decode_index),
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )


if __name__ == "__main__":
    main()
