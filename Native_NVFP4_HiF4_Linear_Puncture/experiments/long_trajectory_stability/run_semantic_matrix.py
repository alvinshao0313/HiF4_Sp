#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.config import DEFAULT_PHASEA_ROOT
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.gpu_pool import available_gpus, cuda_env

EXP_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_root", required=True)
    p.add_argument("--phasea_root", default=str(DEFAULT_PHASEA_ROOT))
    p.add_argument("--max_parallel", type=int, default=2)
    p.add_argument("--force", action="store_true")
    p.add_argument("--min_e0_top1_parity", type=float, default=0.99)
    return p.parse_args()


def require_existing_e0_parity_pass(parity_path: Path, min_parity: float) -> bool:
    """Return True iff a previous E0 parity file exists and passes the gate.

    A failed existing file must not be treated as skippable success: otherwise a
    later resume would launch E1-E4 after a failed E0 gate.
    """
    if not parity_path.is_file():
        return False
    payload = json.loads(parity_path.read_text(encoding="utf-8"))
    parity = float(payload["top1_parity"])
    if parity < min_parity:
        raise RuntimeError(
            f"E0 semantic causal replay parity={parity:.4f} < {min_parity:.4f}; "
            "do not continue E1-E4 semantic replay"
        )
    return True


def command(args: argparse.Namespace, variant: str, run_root: Path) -> list[str]:
    return [
        sys.executable,
        str(EXP_DIR / "causal_replay.py"),
        "--variant",
        variant,
        "--probe_plan",
        str(run_root / "analysis/probe_plan.json"),
        "--reference_dir",
        str(run_root / "semantic/E0/reference"),
        "--output_dir",
        str(run_root / "semantic" / variant),
        "--phasea_root",
        args.phasea_root,
        "--device",
        "cuda:0",
        "--min_e0_top1_parity",
        str(args.min_e0_top1_parity),
    ]


def run_e0(args: argparse.Namespace, run_root: Path, gpu: int) -> None:
    output = run_root / "semantic/E0"
    parity = output / "e0_semantic_parity.json"
    if not args.force and require_existing_e0_parity_pass(parity, args.min_e0_top1_parity):
        print(f"[skip] E0 semantic reference: {parity}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    log = output / "causal_replay.log"
    with log.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            command(args, "E0", run_root),
            cwd=str(REPO_ROOT),
            env=cuda_env([gpu]),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"E0 semantic replay/parity failed; see {log}")


def launch_variant(args: argparse.Namespace, run_root: Path, variant: str, gpu: int):
    output = run_root / "semantic" / variant
    metrics = output / "semantic_metrics.jsonl"
    if metrics.is_file() and metrics.stat().st_size > 0 and not args.force:
        print(f"[skip] {variant} semantic: {metrics}", flush=True)
        return None
    output.mkdir(parents=True, exist_ok=True)
    handle = (output / "causal_replay.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        command(args, variant, run_root),
        cwd=str(REPO_ROOT),
        env=cuda_env([gpu]),
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(f"[launch] semantic {variant}: GPU={gpu}", flush=True)
    return proc, handle, variant


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    gpus = available_gpus()
    if not gpus:
        raise RuntimeError("no available GPU for semantic replay")
    run_e0(args, run_root, gpus[0])
    variants = ("E1", "E2", "E3", "E4")
    max_parallel = min(max(args.max_parallel, 1), len(gpus))
    index = 0
    while index < len(variants):
        jobs = []
        for slot in range(max_parallel):
            if index >= len(variants):
                break
            variant = variants[index]
            index += 1
            job = launch_variant(args, run_root, variant, gpus[slot])
            if job is not None:
                jobs.append(job)
        for proc, handle, variant in jobs:
            rc = proc.wait()
            handle.close()
            if rc != 0:
                raise RuntimeError(
                    f"semantic replay {variant} failed with rc={rc}; see {run_root / 'semantic' / variant / 'causal_replay.log'}"
                )
            print(f"[done] semantic {variant}", flush=True)


if __name__ == "__main__":
    main()
