"""扫描 VBench 评测结果是否完整。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .constants import DIMS_10
from .utils import extract_score_from_file, parse_dims, write_json


def main() -> None:
    ap = argparse.ArgumentParser(description="扫描缺失 VBench eval_results.json")
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--modes", nargs="+", default=["quant"])
    ap.add_argument("--dims", default="all")
    ap.add_argument("--scan-dir", default=None)
    args = ap.parse_args()

    out_base = Path(args.out_base)
    dims = parse_dims(args.dims, DIMS_10)
    scan_dir = Path(args.scan_dir) if args.scan_dir else out_base / "_scan"
    scan_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    ok_count = 0
    expected = len(args.cases) * len(args.modes) * len(dims)

    for case in args.cases:
        for mode in args.modes:
            case_ok = 0
            for dim in dims:
                dim_dir = out_base / case / mode / dim
                files = sorted(dim_dir.glob("*_eval_results.json")) if dim_dir.is_dir() else []
                status = "MISSING"
                detail = "no_eval_results_json"
                score = ""
                chosen = ""
                for f in files:
                    try:
                        score = f"{extract_score_from_file(f, dim):.6f}"
                        status = "OK"
                        detail = "ok"
                        chosen = str(f)
                        break
                    except Exception as e:
                        detail = f"bad_json: {e!r}"
                        chosen = str(f)
                if status == "OK":
                    ok_count += 1
                    case_ok += 1
                else:
                    missing.append({"case": case, "mode": mode, "dimension": dim, "dim_dir": str(dim_dir), "detail": detail})
                rows.append({"case": case, "mode": mode, "dimension": dim, "status": status, "score": score, "file": chosen, "detail": detail})
            print(f"{case}/{mode}: ok={case_ok}/{len(dims)} missing={len(dims)-case_ok}")

    with (scan_dir / "scan_rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case", "mode", "dimension", "status", "score", "file", "detail"])
        w.writeheader(); w.writerows(rows)

    with (scan_dir / "missing_jobs.tsv").open("w", encoding="utf-8") as f:
        f.write("case\tmode\tdimension\tdim_dir\tdetail\n")
        for r in missing:
            f.write("\t".join([r["case"], r["mode"], r["dimension"], r["dim_dir"], r["detail"]]) + "\n")

    summary = {"TOTAL_OK": ok_count, "TOTAL_EXPECTED": expected, "TOTAL_MISSING": expected - ok_count, "out_base": str(out_base)}
    write_json(scan_dir / "scan_summary.json", summary)

    for k, v in summary.items():
        print(f"{k}={v}")
    print(f"SCAN_DIR={scan_dir}")
    print(f"MISSING_TSV={scan_dir / 'missing_jobs.tsv'}")

    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
