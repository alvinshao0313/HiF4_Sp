"""Aggregate layer diagnostics and paired bootstrap summaries."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .artifact import read_json, write_json


def paired_bootstrap(a, b, *, seed=42, rounds=5000):
    if len(a) != len(b) or not a:
        raise ValueError("paired bootstrap requires equal nonempty samples")
    rng = random.Random(seed)
    differences = [x - y for x, y in zip(a, b)]
    estimates = []
    for _ in range(rounds):
        sample = [differences[rng.randrange(len(differences))] for _ in differences]
        estimates.append(sum(sample) / len(sample))
    estimates.sort()
    return {"mean": sum(differences) / len(differences),
            "lcb_2_5": estimates[int(0.025 * (rounds - 1))],
            "ucb_97_5": estimates[int(0.975 * (rounds - 1))],
            "n": len(differences), "rounds": rounds}


def summarize(run_root):
    root = Path(run_root).resolve()
    manifest = read_json(root / "manifest.json")
    rows = []
    for layer in manifest["completed_layers"]:
        metrics = read_json(root / f"layers/L{layer:02d}/metrics_diagnostics.json")
        selected = read_json(root / f"layers/L{layer:02d}/metrics.json")
        holdout = metrics["samples"]["holdout"]
        rows.append({"layer": layer,
                     "holdout_cumulative_mse": sum(v["cumulative"]["mse"] for v in holdout.values()) / len(holdout),
                     "holdout_cumulative_l2": sum(v["cumulative"]["l2"] for v in holdout.values()) / len(holdout),
                     "holdout_propagated_mse": sum(v["propagated"]["mse"] for v in holdout.values()) / len(holdout),
                     "holdout_local_mse": sum(v["local"]["mse"] for v in holdout.values()) / len(holdout),
                     "holdout_delta_cumulative_l2_sq": _mean(v["delta_cumulative_l2_sq"] for v in holdout.values()),
                     "holdout_direction_cosine": _mean(v.get("direction", {}).get("cosine") for v in holdout.values()),
                     "holdout_local_propagated_cosine": _mean(v.get("cosine_local_propagated") for v in holdout.values()),
                     "selected_holdout_mse": selected["holdout"]["mse"],
                     "selected_holdout_direction_coverage": selected["holdout"]["direction_coverage"],
                     "finite": bool(selected["finite"])})
    result = {"status": "COMPLETE" if manifest["completed_layers"] == list(range(48)) else "PARTIAL",
              "method": manifest["config"]["method"],
              "lambda_direction": manifest["config"]["lambda_direction"], "layers": rows}
    write_json(result, root / "summary.json")
    return result


def summarize_matrix(matrix_root, *, rounds=5000):
    root = Path(matrix_root).resolve()
    names = ("baseline", "direct_l010", "direct_l030", "jvp_l010", "jvp_l030", "shuffled_l030")
    runs = {name: summarize(root / name) for name in names
            if (root / name / "manifest.json").exists()}
    if "baseline" not in runs:
        raise RuntimeError("matrix summary requires baseline")
    baseline = runs["baseline"]
    final_layer = max(baseline["layers"], key=lambda row: row["layer"])["layer"]
    base_diag = read_json(root / "baseline" / f"layers/L{final_layer:02d}/metrics_diagnostics.json")["samples"]["holdout"]
    comparisons = {}
    for name, result in runs.items():
        if name == "baseline":
            continue
        diag_path = root / name / f"layers/L{final_layer:02d}/metrics_diagnostics.json"
        if not diag_path.exists():
            continue
        candidate = read_json(diag_path)["samples"]["holdout"]
        ids = sorted(set(base_diag) & set(candidate))
        comparisons[name] = {"final_layer": final_layer,
                             "final_cumulative_mse": _mean(candidate[s]["cumulative"]["mse"] for s in ids),
                             "paired_improvement_baseline_minus_candidate": paired_bootstrap(
                                 [base_diag[s]["cumulative"]["mse"] for s in ids],
                                 [candidate[s]["cumulative"]["mse"] for s in ids], rounds=rounds)}
    result = {"status": "COMPLETE" if all(r["status"] == "COMPLETE" for r in runs.values()) else "PARTIAL",
              "runs": runs, "comparisons": comparisons, "bootstrap_rounds": rounds}
    write_json(result, root / "matrix_summary.json")
    return result


def _mean(values):
    values = [x for x in values if x is not None]
    return sum(values) / len(values) if values else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root")
    parser.add_argument("--matrix_root")
    parser.add_argument("--bootstrap_rounds", type=int, default=5000)
    args = parser.parse_args()
    if bool(args.run_root) == bool(args.matrix_root):
        raise SystemExit("provide exactly one of --run_root or --matrix_root")
    if args.run_root:
        summarize(args.run_root)
    else:
        summarize_matrix(args.matrix_root, rounds=args.bootstrap_rounds)
