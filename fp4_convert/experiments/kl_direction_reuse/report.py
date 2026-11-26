"""Paired sample bootstrap; production-only results and explicit missing-data failures."""
import csv
import json
import math
from pathlib import Path

import numpy as np

from .artifacts import read_json, sha256, write_json
from .capture import CaptureStore
from .config import PROTOCOL, recipes
from .data import Dataset


def aggregate(rows):
    totals = {k: sum(r[k] for r in rows) for k in ("kl_sum", "kl_tokens", "nll_sum", "nll_tokens")}
    if totals["kl_tokens"] <= 0 or totals["nll_tokens"] <= 0 or not all(math.isfinite(v) for v in totals.values()):
        raise RuntimeError("invalid evaluation metric coverage")
    kl, nll = totals["kl_sum"]/totals["kl_tokens"], totals["nll_sum"]/totals["nll_tokens"]
    return dict(kl=kl, nll=nll, ppl=math.exp(nll), **totals)


def paired_gain(candidate, baseline, seed=42, draws=5000):
    if len(candidate) != len(baseline) or not candidate:
        raise ValueError("paired rows differ")
    a = np.array([[r["kl_sum"], r["kl_tokens"]] for r in candidate], dtype=np.float64)
    b = np.array([[r["kl_sum"], r["kl_tokens"]] for r in baseline], dtype=np.float64)
    if not np.array_equal(a[:, 1], b[:, 1]):
        raise ValueError("paired token coverage differs")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (a[:, 1] <= 0).any():
        raise ValueError("nonfinite or empty paired metrics")
    indices = np.random.default_rng(seed).integers(0, len(a), (draws, len(a)))
    aa, bb = a[indices].sum(1), b[indices].sum(1)
    gains = bb[:, 0]/bb[:, 1] - aa[:, 0]/aa[:, 1]
    point = b[:, 0].sum()/b[:, 1].sum() - a[:, 0].sum()/a[:, 1].sum()
    return dict(kl_gain=float(point), ci95=[float(x) for x in np.quantile(gains, [.025, .975])])


def rows_for(store, ds, split):
    ids = ds.ids(split)
    if not set(ids) <= set(store.manifest["samples"]):
        raise RuntimeError("evaluation sample coverage incomplete")
    rows = []
    for sid in ids:
        row = store.manifest["samples"][sid]["metrics"]
        n = len(ds.samples[sid]["input_ids"])
        if row["kl_tokens"] != n or row["nll_tokens"] != n-1:
            raise RuntimeError("evaluation prediction/target coverage mismatch")
        rows.append(row)
    return rows


def summarize(root):
    ds = Dataset(root)
    baseline = CaptureStore(ds.root / "captures/baseline", ds)
    teacher = CaptureStore(ds.root / "captures/native", ds)
    base_rows = rows_for(baseline, ds, "test")
    result = dict(native=aggregate(rows_for(teacher, ds, "test")), baseline=aggregate(base_rows), candidates=[])
    curves, step_curves, evaluated = [], [], {}
    common_steps = read_json(ds.root / "common_steps.json")
    for layer in PROTOCOL.layers:
        manifests = [read_json(ds.root / "runs" / r["name"] / "manifest.json")
                     for r in recipes() if r["layer"] == layer]
        minimum = min(m["steps"] for m in manifests)
        if minimum <= 0 or common_steps[str(layer)] != 1 << (minimum.bit_length()-1):
            raise RuntimeError("common step checkpoint was not selected by the frozen rule")
    initial_val = aggregate(rows_for(baseline, ds, "val"))
    for recipe in recipes():
        run = ds.root / "runs" / recipe["name"]
        manifest = read_json(run / "manifest.json")
        if manifest["status"] != "COMPLETE" or manifest["recipe"] != recipe:
            raise RuntimeError(f"{recipe['name']} did not complete training with a parameter update")
        store = CaptureStore(run / "evaluation/final", ds)
        export = read_json(run / "exports/final/export.json")
        if export["checkpoint_sha256"] != sha256(run / "final.pt") or store.manifest["model_files"] != export["files"]:
            raise RuntimeError("evaluation does not describe the final budget checkpoint")
        rows = rows_for(store, ds, "test")
        evaluated[recipe["name"]] = rows
        result["candidates"].append(dict(**recipe, **aggregate(rows), **paired_gain(rows, base_rows),
                                          charged_seconds=manifest["charged_seconds"], steps=manifest["steps"],
                                          peak_memory_bytes=manifest["peak_memory_bytes"],
                                          checkpoint_seconds=manifest["committed_seconds"],
                                          evaluation_seconds=store.manifest["wall_seconds"]))
        curves.extend([dict(recipe=recipe["name"], seconds=0, **initial_val),
                       dict(recipe=recipe["name"], seconds=int(PROTOCOL.budget_seconds),
                            **aggregate(rows_for(store, ds, "val")))])
        expected = {"time_1800", "time_3600", "time_5400", f"step_{common_steps[str(recipe['layer'])]:06d}"}
        points = sorted((run / "evaluation/curves").glob("*/manifest.json"))
        if {point.parent.name for point in points} != expected:
            raise RuntimeError(f"missing or unexpected validation checkpoints: {recipe['name']}")
        for point in points:
            capture = CaptureStore(point.parent, ds)
            point_export = read_json(run / "exports" / point.parent.name / "export.json")
            if (point_export["checkpoint_sha256"] != sha256(run / "checkpoints" / f"{point.parent.name}.pt")
                    or capture.manifest["model_files"] != point_export["files"]):
                raise RuntimeError("validation capture belongs to a different checkpoint")
            metric = aggregate(rows_for(capture, ds, "val"))
            if point.parent.name.startswith("time_"):
                curves.append(dict(recipe=recipe["name"], seconds=int(point.parent.name[5:]), **metric))
            elif point.parent.name.startswith("step_"):
                step_curves.append(dict(recipe=recipe["name"], step=int(point.parent.name[5:]), **metric))
    result["pairwise"] = []
    for layer in (8, 24, 40):
        for k in (1, 2, 4):
            name = f"L{layer:02d}_cached_kl_k{k}"
            for comparator in (f"L{layer:02d}_mse", f"L{layer:02d}_direct_kl"):
                result["pairwise"].append(dict(candidate=name, comparator=comparator,
                                                 **paired_gain(evaluated[name], evaluated[comparator])))
    result["validation_time_curves"], result["validation_step_curves"] = curves, step_curves
    stages_path = ds.root / "stages.jsonl"
    stages = [json.loads(line) for line in stages_path.read_text().splitlines()] if stages_path.exists() else []
    successful = [s for s in stages if s["returncode"] == 0]
    names = {s["args"][s["args"].index("--recipe")+1] for s in successful
             if s["command"] == "train" and "--recipe" in s["args"]}
    expected_counts = {"prepare": 1, "baseline": 1, "verify-prepare": 3, "verify-finish": 3,
                       "capture": 80, "export": 75}
    complete_ledger = names == {r["name"] for r in recipes()} and all(
        sum(s["command"] == command for s in successful) == count for command, count in expected_counts.items())
    result["cost"] = dict(training_charged_gpu_hours=sum(r["charged_seconds"] for r in result["candidates"])/3600,
                          stages=stages, stage_ledger_complete=complete_ledger)
    if stages:
        # Child wall time includes process startup, preparation and shutdown.
        # Never add charged training time again to this measured allocation time.
        result["cost"]["recorded_allocation_gpu_hours"] = sum(s["wall_seconds"] * len(s["devices"].split(","))
                                                   for s in stages if s["devices"])/3600
        result["cost"]["cpu_stage_seconds"] = sum(s["wall_seconds"] for s in stages if not s["devices"])
    write_json(result, ds.root / "report.json")
    rows = result["candidates"]
    with (ds.root / "comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if curves:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, layer in zip(axes, (8, 24, 40)):
            for recipe in [r for r in recipes() if r["layer"] == layer]:
                points = sorted([p for p in curves if p["recipe"] == recipe["name"]], key=lambda p: p["seconds"])
                ax.plot([p["seconds"]/3600 for p in points], [p["kl"] for p in points], marker="o", label=recipe["name"])
            ax.set(title=f"Layer {layer}", xlabel="Training GPU hours", ylabel="Validation KL")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(ds.root / "kl_vs_gpu_hours.png", dpi=180)
        plt.close(fig)
    return result
