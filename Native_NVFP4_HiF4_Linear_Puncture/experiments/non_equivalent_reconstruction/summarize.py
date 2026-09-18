"""Collect separate reconstruction, routing and benchmark results for all five models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from .artifact import write_json

NAMES = ("E4", "linear_top_partial", "linear_top_mass", "group_top_partial", "group_top_mass")


def summarize(run_root):
    root = Path(run_root)
    rows = []
    for name in NAMES:
        out = root / name
        row = {"name": name}
        if name != "E4":
            layers = json.loads((out / "summary.json").read_text())["layers"]
            row["layers"] = layers
            for metric in ("reconstruction_nmse", "router_loss", "topk_set_match", "topk_overlap"):
                row[f"layer_mean_{metric}"] = sum(layer[metric] for layer in layers) / len(layers)
        row["arc"] = json.loads((out / "eval/arc/metrics.json").read_text())
        row["mmlu_pro"] = json.loads((out / "eval/mmlu_pro/metrics.json").read_text())
        rows.append(row)
    write_json({"models": rows, "note": "Compare downstream scores and reconstruction/router metrics separately; total objectives differ."},
               root / "comparison.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True)
    summarize(**vars(parser.parse_args()))
