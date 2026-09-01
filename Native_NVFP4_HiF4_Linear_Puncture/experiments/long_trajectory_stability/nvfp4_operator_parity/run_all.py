#!/usr/bin/env python3
"""Run E0 NVFP4 operator parity stages sequentially: P0 → capture → P1…P6 → report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.nvfp4_operator_parity.common import (
    RESULT_ROOT,
)

HERE = Path(__file__).resolve().parent


def _run(script: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, str(HERE / script), *(extra or [])]
    print(f"[run_all] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default=str(RESULT_ROOT))
    p.add_argument("--skip_capture", action="store_true")
    p.add_argument("--model_path", default="nvidia/Qwen3-30B-A3B-NVFP4")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    common = ["--output_dir", str(out), "--model_path", args.model_path]
    device_args = ["--device", args.device, *common]

    # P0 (CPU)
    try:
        _run("audit_scales.py", ["--output_dir", str(out), "--model_path", args.model_path])
    except subprocess.CalledProcessError as exc:
        p0_path = out / "P0_scale_audit.json"
        if p0_path.exists():
            verdict = json.loads(p0_path.read_text(encoding="utf-8")).get("verdict")
            if verdict == "P0_SCALE_SEMANTIC_MISMATCH":
                print("[run_all] P0_SCALE_SEMANTIC_MISMATCH — stop before GPU stages", flush=True)
                _run("write_report.py", ["--output_dir", str(out)])
                raise SystemExit(2) from exc
        raise

    p0 = json.loads((out / "P0_scale_audit.json").read_text(encoding="utf-8"))
    if p0.get("verdict") == "P0_SCALE_SEMANTIC_MISMATCH":
        print("[run_all] P0_SCALE_SEMANTIC_MISMATCH — stop before GPU stages", flush=True)
        _run("write_report.py", ["--output_dir", str(out)])
        raise SystemExit(2)

    if not args.skip_capture:
        _run("capture_frozen_inputs.py", device_args)
    else:
        print("[run_all] skip_capture", flush=True)

    _run("run_p1_qdq.py", ["--device", args.device, "--output_dir", str(out)])
    _run("run_p2_weight_dequant.py", device_args)
    _run("run_p3_dense_linear.py", device_args)
    _run("run_p4_single_expert.py", device_args)
    _run("run_p5_fused_moe.py", device_args)
    _run("run_p6_manual_tp2.py", device_args)
    _run("write_report.py", ["--output_dir", str(out)])
    print(f"[run_all] done -> {out / 'E0_NVFP4_OPERATOR_PARITY_REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
