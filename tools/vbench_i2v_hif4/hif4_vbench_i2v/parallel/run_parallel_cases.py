"""单机多 GPU 并行评测多个 case。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run_case(case: str, device: str, args) -> int:  # type: ignore[no-untyped-def]
    case_input = Path(args.input_base) / case
    output_dir = Path(args.out_base) / case / args.mode
    cmd = [
        sys.executable, "-m", "hif4_vbench_i2v.run_eval_10dim",
        "--case-input", str(case_input),
        "--output-dir", str(output_dir),
        "--image-folder", args.image_folder,
        "--mode", args.mode,
        "--dims", args.dims,
        "--device", "cuda:0",
        "--continue-on-error",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    print(f"[START] case={case} device={device}")
    rc = subprocess.run(cmd, env=env).returncode
    print(f"[DONE] case={case} rc={rc}")
    return rc


def main() -> None:
    ap = argparse.ArgumentParser(description="单机多 GPU 并行跑多个 VBench case")
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--input-base", required=True)
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--mode", default="quant")
    ap.add_argument("--dims", default="all")
    ap.add_argument("--devices", default="0")
    ap.add_argument("--max-workers", type=int, default=1)
    ap.add_argument("--allow-device-oversubscribe", action="store_true", help="允许多个 worker 共享同一 CUDA_VISIBLE_DEVICES；默认会把 worker 数限制到设备数")
    args = ap.parse_args()

    devices = [x for x in args.devices.replace(",", " ").split() if x]
    if not devices:
        raise SystemExit("--devices 不能为空")
    max_workers = args.max_workers if args.allow_device_oversubscribe else min(args.max_workers, len(devices), len(args.cases))
    if max_workers < args.max_workers and not args.allow_device_oversubscribe:
        print(f"[INFO] max_workers 从 {args.max_workers} 限制为 {max_workers}，避免多个 VBench 进程共享同一 GPU")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for i, case in enumerate(args.cases):
            futs.append(ex.submit(run_case, case, devices[i % len(devices)], args))
        rcs = [f.result() for f in as_completed(futs)]
    if any(rc != 0 for rc in rcs):
        raise SystemExit(1)
    print("PARALLEL_CASES_OK")


if __name__ == "__main__":
    main()
