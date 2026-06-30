"""最终验收入口：输入目录 + 结果扫描。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="最终验收 HiF4 VBench-I2V 结果")
    ap.add_argument("--input-base", required=True)
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--expected-sb", type=int, default=200)
    ap.add_argument("--expected-camera", type=int, default=100)
    args = ap.parse_args()

    for case in args.cases:
        case_input = Path(args.input_base) / case
        cmd = [
            sys.executable, "-m", "hif4_vbench_i2v.validate_case_input",
            "--case-input", str(case_input),
            "--expected-sb", str(args.expected_sb),
            "--expected-camera", str(args.expected_camera),
            "--mode", "quant",
            "--forbid-symlink",
        ]
        subprocess.run(cmd, check=True)

    cmd = [
        sys.executable, "-m", "hif4_vbench_i2v.scan_missing",
        "--out-base", args.out_base,
        "--cases", *args.cases,
        "--modes", "quant",
        "--dims", "all",
    ]
    subprocess.run(cmd, check=True)
    print("VALIDATE_FINAL_OK")


if __name__ == "__main__":
    main()
