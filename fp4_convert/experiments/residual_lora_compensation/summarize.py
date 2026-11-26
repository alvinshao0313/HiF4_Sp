"""Merge three new LoRA runs with the reused E4/group comparison rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifact import write_json


def load_json(path):
    return json.loads(Path(path).read_text())


def new_row(root, name):
    run = root / name
    summary = load_json(run / "summary.json")
    layers = summary["layers"]
    row = {"name": name, "run_dir": str(run), "layers": layers}
    for metric in ("reconstruction_nmse", "router_loss", "topk_set_match", "topk_overlap",
                   "attention_residual_nmse", "moe_residual_nmse"):
        values = [float(layer[metric]) for layer in layers if metric in layer]
        if values:
            row[f"layer_mean_{metric}"] = sum(values) / len(values)
    for task in ("arc", "mmlu_pro", "livecodebench"):
        path = run / f"eval/{task}/metrics.json"
        if path.is_file():
            row[task] = load_json(path)
    return row


def reused_row(name, run):
    run = Path(run).resolve()
    row = {"name": name, "run_dir": str(run), "reused": True}
    summary = run / "summary.json"
    if summary.is_file():
        layers = load_json(summary).get("layers", [])
        row["layers"] = layers
        for metric in ("reconstruction_nmse", "router_loss", "topk_set_match", "topk_overlap"):
            values = [float(layer[metric]) for layer in layers if metric in layer]
            if values:
                row[f"layer_mean_{metric}"] = sum(values) / len(values)
    for task in ("arc", "mmlu_pro", "livecodebench"):
        path = run / f"eval/{task}/metrics.json"
        if path.is_file():
            row[task] = load_json(path)
    return row


def summarize(run_root, baseline_run, e4_run):
    root = Path(run_root).resolve()
    rows = [reused_row("E4", e4_run), reused_row("group_top_mass", baseline_run)]
    rows.extend(new_row(root, name) for name in ("attention", "moe", "both"))
    write_json({"models": rows, "note": "Reused E4/group artifacts are provenance-linked; compare downstream and reconstruction metrics separately."},
               root / "comparison.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--baseline_run", required=True)
    parser.add_argument("--e4_run", required=True)
    summarize(**vars(parser.parse_args()))
