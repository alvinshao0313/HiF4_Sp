#!/usr/bin/env python3
"""Build per-threshold permutation maps from the one-shot search artifacts.

Reads results/search/summary.json + results/search/candidate_permutations.pt
and writes tau_*.pt maps plus threshold_report.{json,md}. Asserts threshold
monotonicity and full 32-layer coverage with legal permutations.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
HIFLOAT4_ROOT = SCRIPT_DIR.parents[2]
if str(HIFLOAT4_ROOT) not in sys.path:
    sys.path.insert(0, str(HIFLOAT4_ROOT))

from permutation_optimization.threshold_policy import (  # noqa: E402
    build_threshold_gated_permutations,
)


def _tau_name(threshold_pct: float) -> str:
    return f"tau_{threshold_pct:.2f}".replace(".", "p")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=str, required=True)
    ap.add_argument("--candidate-permutations", type=str, required=True)
    ap.add_argument("--thresholds", type=str, default="0,0.25,0.5,1.0,2.0")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--expect-layers", type=int, default=32)
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    candidates = torch.load(args.candidate_permutations, map_location="cpu", weights_only=True)
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError("candidate_permutations.pt must be a non-empty dict")
    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    if sorted(thresholds) != thresholds:
        raise ValueError("thresholds must be sorted ascending")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: dict[float, dict] = {}
    maps: dict[float, dict[str, torch.Tensor]] = {}
    for tau in thresholds:
        gated, report = build_threshold_gated_permutations(summary, candidates, tau)
        if report["n_layers"] != args.expect_layers:
            raise RuntimeError(
                f"tau={tau}: expected {args.expect_layers} layers, got {report['n_layers']}"
            )
        for name, perm in gated.items():
            if torch.unique(perm).numel() != perm.numel():
                raise RuntimeError(f"tau={tau}: illegal permutation at {name}")
        reports[tau] = report
        maps[tau] = gated
        torch.save(gated, out_dir / f"{_tau_name(tau)}.pt")

    # Monotonicity: higher threshold keeps a subset of reordered layers.
    for t_lo, t_hi in zip(thresholds, thresholds[1:]):
        n_lo = reports[t_lo]["n_reordered"]
        n_hi = reports[t_hi]["n_reordered"]
        if not (n_lo >= n_hi):
            raise RuntimeError(
                f"monotonicity violated: n_reordered({t_lo})={n_lo} < n_reordered({t_hi})={n_hi}"
            )
        lo_layers = {r["layer_name"] for r in reports[t_lo]["layers"] if r["use_reorder"]}
        hi_layers = {r["layer_name"] for r in reports[t_hi]["layers"] if r["use_reorder"]}
        if not hi_layers <= lo_layers:
            raise RuntimeError("monotonicity violated: reordered set not nested")

    report_json = {
        "thresholds_pct": thresholds,
        "per_threshold": {
            _tau_name(tau): {
                "threshold_pct": tau,
                "n_reordered": reports[tau]["n_reordered"],
                "n_identity": reports[tau]["n_identity"],
                "reordered_layers": [
                    r["layer_name"] for r in reports[tau]["layers"] if r["use_reorder"]
                ],
            }
            for tau in thresholds
        },
        "layers": {
            _tau_name(tau): reports[tau]["layers"] for tau in thresholds
        },
    }
    (out_dir / "threshold_report.json").write_text(json.dumps(report_json, indent=2))

    lines = [
        "# Threshold-Gated Permutation Maps",
        "",
        "| threshold(%) | reordered layers | identity layers | selected layer indices |",
        "|---:|---:|---:|---|",
    ]
    for tau in thresholds:
        idxs = sorted(
            int(name.split(".")[2])
            for name in report_json["per_threshold"][_tau_name(tau)]["reordered_layers"]
        )
        lines.append(
            f"| {tau:.2f} | {reports[tau]['n_reordered']} | {reports[tau]['n_identity']} | {idxs} |"
        )
    lines += [
        "",
        "各层 gain_pct 明细见 `threshold_report.json`。",
        "",
    ]
    (out_dir / "threshold_report.md").write_text("\n".join(lines))
    print(json.dumps(report_json["per_threshold"], indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
