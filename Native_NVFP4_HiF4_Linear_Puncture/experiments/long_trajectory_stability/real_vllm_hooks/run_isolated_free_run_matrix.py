#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.gpu_pool import (
    available_gpus,
    cuda_env,
)

RUNNER = Path(__file__).with_name("run_isolated_free_run.py")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_root", required=True)
    p.add_argument("--variants", nargs="+", default=["E0", "E1", "E2", "E3", "E4"])
    p.add_argument("--prompt_source", default=None)
    p.add_argument("--max_samples", type=int, default=64)
    p.add_argument("--max_new_tokens", type=int, default=16384)
    p.add_argument("--sample_keys", nargs="*", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gpus = available_gpus()
    if len(gpus) < 2:
        raise RuntimeError(
            f"isolated TP2 free-run requires two genuinely idle project GPUs; available={gpus}"
        )
    pair = gpus[:2]
    run_root = Path(args.run_root).resolve()
    normalized = run_root / "isolated"
    normalized.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        output = normalized / f"{variant}.jsonl"
        cmd = [
            sys.executable,
            str(RUNNER),
            "--variant",
            variant,
            "--output",
            str(output),
            "--max_samples",
            str(args.max_samples),
            "--max_new_tokens",
            str(args.max_new_tokens),
        ]
        if args.prompt_source is not None:
            cmd.extend(["--prompt_source", args.prompt_source])
        if args.sample_keys:
            cmd.extend(["--sample_keys", *args.sample_keys])
        print(f"[isolated free-run] GPUs={pair} variant={variant}", flush=True)
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=cuda_env(pair), check=True)


if __name__ == "__main__":
    main()
