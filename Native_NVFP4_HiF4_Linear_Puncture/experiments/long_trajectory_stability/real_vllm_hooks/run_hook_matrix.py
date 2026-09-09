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


RUNNER = Path(__file__).with_name("run_variant_hook_capture.py")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_root", required=True)
    p.add_argument("--variants", nargs="+", default=["E0", "E1"])
    p.add_argument(
        "--mode",
        default="forced_core",
        choices=["forced_logits_only", "forced_core", "greedy_none", "greedy_core"],
    )
    p.add_argument("--probe_plan", default=None)
    p.add_argument("--smoke_focus", action="store_true")
    p.add_argument("--max_tokens", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gpus = available_gpus()
    if len(gpus) < 2:
        raise RuntimeError(
            f"real-vLLM TP2 hook capture requires two genuinely idle project GPUs; available={gpus}"
        )
    pair = gpus[:2]
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        variant_root = run_root / variant / args.mode
        cmd = [
            sys.executable,
            str(RUNNER),
            "--variant",
            variant,
            "--mode",
            args.mode,
            "--output_root",
            str(variant_root),
        ]
        if args.probe_plan is not None:
            cmd.extend(["--probe_plan", args.probe_plan])
        if args.smoke_focus:
            cmd.append("--smoke_focus")
        if args.max_tokens is not None:
            cmd.extend(["--max_tokens", str(args.max_tokens)])
        print(f"[real-vLLM hook] GPUs={pair} variant={variant} mode={args.mode}", flush=True)
        subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=cuda_env(pair),
            check=True,
        )


if __name__ == "__main__":
    main()
