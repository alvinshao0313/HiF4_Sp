#!/usr/bin/env python3
"""Formal long-trajectory pipeline: isolated free-run + probe plan + real vLLM hooks."""

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

EXP_DIR = Path(__file__).resolve().parent
HOOKS = EXP_DIR / "real_vllm_hooks"
HIF4_PYTHON = Path(sys.executable)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_root", required=True)
    p.add_argument("--stage", choices=["smoke", "formal"], required=True)
    p.add_argument("--phasea_root", default=str(DEFAULT_PHASEA_ROOT))
    p.add_argument("--detach", action="store_true")
    return p.parse_args()


def detach_and_reexec(run_root: Path) -> None:
    if os.environ.get("HIF4_PIPELINE_DETACHED") == "1":
        return
    run_root.mkdir(parents=True, exist_ok=True)
    log_path = run_root / "driver.log"
    pid_path = run_root / "driver.pid"
    argv = [str(HIF4_PYTHON), str(Path(__file__).resolve()), *[x for x in sys.argv[1:] if x != "--detach"]]
    env = dict(os.environ)
    env["HIF4_PIPELINE_DETACHED"] = "1"
    env.setdefault("GPU_POOL", "0,1")
    reader, writer = os.pipe()
    pid = os.fork()
    if pid > 0:
        os.close(writer)
        raw = os.read(reader, 64).decode("utf-8").strip()
        os.close(reader)
        os.waitpid(pid, 0)
        print(f"detached pid={raw} log={log_path} pidfile={pid_path}", flush=True)
        return
    os.close(reader)
    os.setsid()
    pid2 = os.fork()
    if pid2 > 0:
        os.write(writer, str(pid2).encode("utf-8"))
        os.close(writer)
        os._exit(0)
    os.close(writer)
    os.chdir(str(REPO_ROOT))
    log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    os.close(log_fd)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    pid_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    os.execve(argv[0], argv, env)


def run_cmd(args: list[str]) -> None:
    print(f"[pipeline] {' '.join(args)}", flush=True)
    subprocess.run(args, cwd=str(REPO_ROOT), check=True)


def smoke(args: argparse.Namespace) -> None:
    run_root = str(Path(args.run_root).resolve())
    run_cmd(
        [
            str(HIF4_PYTHON),
            str(HOOKS / "run_isolated_free_run_matrix.py"),
            "--run_root",
            run_root,
            "--variants",
            "E0",
            "E1",
            "--max_samples",
            "2",
            "--max_new_tokens",
            "512",
            "--sample_keys",
            "n159_c94220492",
            "n346_c216392826",
        ]
    )
    run_cmd(
        [
            str(HIF4_PYTHON),
            str(HOOKS / "run_hook_matrix.py"),
            "--run_root",
            run_root,
            "--variants",
            "E0",
            "E1",
            "--mode",
            "forced_core",
            "--smoke_focus",
        ]
    )
    run_cmd(
        [
            str(HIF4_PYTHON),
            str(HOOKS / "validate_capture.py"),
            "--run_root",
            f"{run_root}/E0/forced_core",
            "--variant",
            "E0",
            "--mode",
            "forced_core",
            "--require_e0_target_top1",
        ]
    )


def formal(args: argparse.Namespace) -> None:
    run_root = str(Path(args.run_root).resolve())
    run_cmd(
        [
            str(HIF4_PYTHON),
            str(HOOKS / "run_isolated_free_run_matrix.py"),
            "--run_root",
            run_root,
            "--variants",
            "E0",
            "E1",
            "E2",
            "E3",
            "E4",
            "--max_samples",
            str(DEFAULT_FREE_RUN_SAMPLES),
            "--max_new_tokens",
            str(DEFAULT_FREE_RUN_MAX_NEW_TOKENS),
        ]
    )
    run_cmd(
        [
            str(HIF4_PYTHON),
            str(HOOKS / "prepare_isolated_analysis.py"),
            "--run_root",
            run_root,
        ]
    )


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    if args.detach:
        detach_and_reexec(run_root)
        if os.environ.get("HIF4_PIPELINE_DETACHED") != "1":
            return
    print(
        f"[pipeline] stage={args.stage} run_root={run_root} pid={os.getpid()} "
        f"gpu_pool={os.environ.get('GPU_POOL')}",
        flush=True,
    )
    if args.stage == "smoke":
        smoke(args)
    else:
        formal(args)
    print("[pipeline] done", flush=True)


if __name__ == "__main__":
    main()
