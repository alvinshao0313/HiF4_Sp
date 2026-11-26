#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_root", required=True)
    p.add_argument("--num_samples", type=int, default=16)
    p.add_argument("--probes_per_bin", type=int, default=4)
    p.add_argument("--max_decode_index", type=int, default=12287)
    args = p.parse_args()
    root = Path(args.run_root).resolve()
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(EXP_DIR / "compare_free_runs.py"),
            "--normalized_dir",
            str(root / "normalized"),
            "--output_dir",
            str(analysis),
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(EXP_DIR / "build_probe_plan.py"),
            "--e0_trajectories",
            str(root / "normalized/E0.jsonl"),
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
    print(analysis / "probe_plan.json")


if __name__ == "__main__":
    main()
