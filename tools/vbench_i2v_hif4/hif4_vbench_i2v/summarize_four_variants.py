"""四路结果汇总：bf16 / quant_empty / quant_searched / hif4_w4a4。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .constants import DIMS_10
from .utils import extract_score_from_file

VARIANTS = ["bf16", "quant_empty", "quant_searched", "hif4_w4a4"]


def find_result(base: Path, case: str, mode: str, dim: str) -> Path | None:
    files = sorted((base / case / mode / dim).glob("*_eval_results.json"))
    return files[0] if files else None


def fmt(x: float | str) -> str:
    if x == "":
        return ""
    return f"{float(x):.6f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="汇总四路 VBench-I2V 结果")
    ap.add_argument("--old-out", required=True)
    ap.add_argument("--hif4-out", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", default="42 43 44")
    ap.add_argument("--empty-case-template", default="empty_seed{seed}")
    ap.add_argument("--searched-case-template", default="searched_seed{seed}")
    ap.add_argument("--hif4-case-template", default="hif4_seed{seed}")
    args = ap.parse_args()

    old_out = Path(args.old_out)
    hif4_out = Path(args.hif4_out)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds.split()

    records = []
    missing = []
    for seed in seeds:
        empty_case = args.empty_case_template.format(seed=seed)
        searched_case = args.searched_case_template.format(seed=seed)
        hif4_case = args.hif4_case_template.format(seed=seed)
        specs = [
            ("bf16", old_out, empty_case, "bf16"),
            ("quant_empty", old_out, empty_case, "quant"),
            ("quant_searched", old_out, searched_case, "quant"),
            ("hif4_w4a4", hif4_out, hif4_case, "quant"),
        ]
        for dim in DIMS_10:
            for variant, base, case, mode in specs:
                p = find_result(base, case, mode, dim)
                if p is None:
                    missing.append([seed, dim, variant, str(base), case, mode])
                    continue
                score = extract_score_from_file(p, dim)
                records.append({"seed": seed, "dimension": dim, "variant": variant, "score": score, "file": str(p)})

    with (out_dir / "raw_scores_long.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "dimension", "variant", "score", "file"])
        w.writeheader(); w.writerows([{**r, "score": fmt(r["score"])} for r in records])

    with (out_dir / "missing_four_variant_items.tsv").open("w", encoding="utf-8") as f:
        f.write("seed\tdimension\tvariant\tbase\tcase\tmode\n")
        for r in missing:
            f.write("\t".join(r) + "\n")

    expected = len(seeds) * len(DIMS_10) * len(VARIANTS)
    print(f"FOUND={len(records)}")
    print(f"EXPECTED={expected}")
    print(f"MISSING={len(missing)}")
    if missing:
        raise SystemExit(1)

    # 均值主表。
    mean_rows = []
    for dim in DIMS_10:
        row = {"dimension": dim}
        for v in VARIANTS:
            vals = [r["score"] for r in records if r["dimension"] == dim and r["variant"] == v]
            row[v] = fmt(sum(vals) / len(vals)) if vals else ""
        mean_rows.append(row)

    with (out_dir / "four_variant_mean_by_dimension.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dimension"] + VARIANTS)
        w.writeheader(); w.writerows(mean_rows)

    # 四大块：原始值、相对 bf16、相对 empty、相对 searched。
    with (out_dir / "four_blocks_raw_and_signed_deltas.csv").open("w", newline="", encoding="utf-8") as f:
        header = (["dimension"] + [f"raw__{v}" for v in VARIANTS]
                  + [f"vs_bf16__{v}" for v in VARIANTS]
                  + [f"vs_empty__{v}" for v in VARIANTS]
                  + [f"vs_searched__{v}" for v in VARIANTS])
        w = csv.writer(f)
        w.writerow(["block"] + ["raw"]*4 + ["delta_vs_bf16"]*4 + ["delta_vs_quant_empty"]*4 + ["delta_vs_quant_searched"]*4)
        w.writerow(header)
        for r in mean_rows:
            vals = {v: float(r[v]) for v in VARIANTS}
            raw = [f"{vals[v]:.6f}" for v in VARIANTS]
            vs_bf16 = [f"{vals[v]-vals['bf16']:+.6f}" for v in VARIANTS]
            vs_empty = [f"{vals[v]-vals['quant_empty']:+.6f}" for v in VARIANTS]
            vs_searched = [f"{vals[v]-vals['quant_searched']:+.6f}" for v in VARIANTS]
            w.writerow([r["dimension"]] + raw + vs_bf16 + vs_empty + vs_searched)

    print(f"OUT_DIR={out_dir}")
    print("SUMMARIZE_FOUR_VARIANTS_OK")


if __name__ == "__main__":
    main()
