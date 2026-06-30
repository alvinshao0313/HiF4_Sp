"""调度无关的缺失维度重试器。

默认 local backend：直接在当前机器上调用 run_eval_10dim。Slurm 用户可以用 examples/slurm_ustc_template wrapper。
每轮重试后会重新扫描结果，避免反复运行已经修复的旧 missing.tsv。
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def read_missing(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def run_scan(out_base: Path, cases: list[str], modes: list[str], dims: str, scan_dir: Path) -> Path:
    cmd = [
        sys.executable, "-m", "hif4_vbench_i2v.scan_missing",
        "--out-base", str(out_base),
        "--cases", *cases,
        "--modes", *modes,
        "--dims", dims,
        "--scan-dir", str(scan_dir),
    ]
    print("SCAN", " ".join(cmd))
    subprocess.run(cmd, check=False)
    return scan_dir / "missing_jobs.tsv"


def main() -> None:
    ap = argparse.ArgumentParser(description="只重试缺失 VBench 维度")
    ap.add_argument("--missing-tsv", required=True)
    ap.add_argument("--input-base", required=True)
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--devices", default="0")
    ap.add_argument("--max-rounds", type=int, default=1)
    ap.add_argument("--mode", default="quant", help="兼容旧命令；实际优先读取 missing.tsv 中的 mode 列")
    ap.add_argument("--scan-dims", default="all", help="每轮重扫时检查的维度，默认 all")
    ap.add_argument("--scan-dir", default=None)
    ap.add_argument("--continue-on-error", action="store_true", default=True)
    args = ap.parse_args()

    devices = [x for x in args.devices.replace(",", " ").split() if x] or ["0"]
    out_base = Path(args.out_base)
    current_missing = Path(args.missing_tsv)
    scan_dir = Path(args.scan_dir) if args.scan_dir else current_missing.parent

    remaining: list[dict[str, str]] = []
    for round_id in range(1, args.max_rounds + 1):
        miss = read_missing(current_missing)
        remaining = miss
        if not miss:
            print("RETRY_MISSING_OK no missing items")
            return

        grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
        for r in miss:
            grouped[(r.get("case", ""), r.get("mode") or args.mode)].append(r["dimension"])

        print(f"=== RETRY ROUND {round_id}/{args.max_rounds}: groups={len(grouped)} missing_items={len(miss)} ===")
        for i, ((case, mode), dims) in enumerate(sorted(grouped.items())):
            if not case:
                raise SystemExit("missing.tsv 中缺少 case 列")
            device = devices[i % len(devices)]
            case_input = Path(args.input_base) / case
            output_dir = out_base / case / mode
            cmd = [
                sys.executable, "-m", "hif4_vbench_i2v.run_eval_10dim",
                "--case-input", str(case_input),
                "--output-dir", str(output_dir),
                "--image-folder", args.image_folder,
                "--mode", mode,
                "--dims", " ".join(sorted(set(dims))),
                "--device", "cuda:0",
                "--continue-on-error",
                "--skip-existing",
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device
            print("RUN", " ".join(cmd), "CUDA_VISIBLE_DEVICES=", device)
            subprocess.run(cmd, env=env, check=False)

        cases = sorted({r["case"] for r in miss if r.get("case")})
        modes = sorted({(r.get("mode") or args.mode) for r in miss})
        current_missing = run_scan(out_base, cases, modes, args.scan_dims, scan_dir)

    remaining = read_missing(current_missing)
    if remaining:
        print(f"RETRY_MISSING_STILL_MISSING={len(remaining)}")
        print(f"MISSING_TSV={current_missing}")
        raise SystemExit(1)
    print("RETRY_MISSING_OK")


if __name__ == "__main__":
    main()
