#!/usr/bin/env python3
"""Compare perm+RTN results against the existing full RTN baseline summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _lm_acc(path: Path, task: str) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    # Flat summary written by eval_lm_eval_hif4a.py: top-level floats.
    if isinstance(data.get(task), (int, float)):
        return float(data[task])
    if isinstance(data.get("scores"), dict) and isinstance(data["scores"].get(task), (int, float)):
        return float(data["scores"][task])
    results = data.get("results_raw") or data.get("results") or data
    if not isinstance(results, dict) or task not in results:
        return None
    entry = results[task]
    if isinstance(entry, (int, float)):
        return float(entry)
    if not isinstance(entry, dict):
        return None
    for key in ("acc,none", "acc", "accuracy"):
        if key in entry:
            return float(entry[key])
    for k, v in entry.items():
        if "acc" in k and isinstance(v, (int, float)):
            return float(v)
    return None


def _mmlu_pro(path: Path) -> float | None:
    files = sorted(path.glob("**/results/results_*.json"))
    if not files:
        return None
    data = json.loads(files[-1].read_text())
    r = data["results"].get("mmlu_pro|0") or data["results"].get("mmlu_pro")
    if r is None:
        return None
    return float(r.get("extractive_match", r.get("acc", 0.0)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--perm_rtn_dir", type=Path, required=True)
    p.add_argument("--baseline_summary", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    args = p.parse_args()

    baseline = json.loads(args.baseline_summary.read_text())
    if "variants" in baseline and "full" in baseline["variants"]:
        full_v = baseline["variants"]["full"]
        full = {
            "arc_easy": full_v["lm_eval"]["arc_easy"],
            "arc_challenge": full_v["lm_eval"]["arc_challenge"],
            "mmlu": full_v["lm_eval"]["mmlu"],
            "mmlu_pro(300)": full_v["mmlu_pro"]["mmlu_pro"],
        }
    elif isinstance(baseline, list):
        full = next(x for x in baseline if x.get("variant") == "full")
    elif "full" in baseline:
        full = baseline["full"]
    else:
        full = baseline

    base_row = {
        "variant": "rtn_baseline",
        "arc_easy": full.get("arc_easy"),
        "arc_challenge": full.get("arc_challenge"),
        "mmlu": full.get("mmlu"),
        "mmlu_pro(300)": full.get("mmlu_pro(300)", full.get("mmlu_pro")),
    }

    lm_json = args.perm_rtn_dir / "lm_eval_arc_mmlu.json"
    perm_row = {
        "variant": "perm_rtn",
        "arc_easy": _lm_acc(lm_json, "arc_easy"),
        "arc_challenge": _lm_acc(lm_json, "arc_challenge"),
        "mmlu": _lm_acc(lm_json, "mmlu"),
        "mmlu_pro(300)": _mmlu_pro(args.perm_rtn_dir / "mmlu_pro"),
    }

    rows = [base_row, perm_row]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2))

    lines = [
        "| variant | arc_easy | arc_challenge | mmlu | mmlu_pro(300) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        def fmt(x):
            return f"{x:.4f}" if isinstance(x, float) else "NA"

        lines.append(
            f"| {r['variant']} | {fmt(r['arc_easy'])} | {fmt(r['arc_challenge'])} | "
            f"{fmt(r['mmlu'])} | {fmt(r['mmlu_pro(300)'])} |"
        )
    md = "\n".join(lines) + "\n"
    (args.output_dir / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
