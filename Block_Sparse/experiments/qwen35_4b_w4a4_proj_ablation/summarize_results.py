#!/usr/bin/env python3
"""汇总各变体：lm_eval ARC/MMLU + lighteval mmlu_pro(300)。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


VARIANTS = ("full", "skip_gate_up", "skip_down", "skip_o_proj", "skip_mlp")

METRIC_PREFERENCE = (
    "extractive_match",
    "exact_match",
    "loglikelihood_acc",
    "acc",
    "acc_norm",
    "quasi_exact_match",
)


def pick_metric(task_metrics: dict) -> tuple[str | None, float | None]:
    for key in METRIC_PREFERENCE:
        if key in task_metrics and isinstance(task_metrics[key], (int, float)):
            return key, float(task_metrics[key])
    for key, val in task_metrics.items():
        if key.endswith("_stderr") or key == "alias":
            continue
        if isinstance(val, (int, float)):
            return key, float(val)
    return None, None


def latest_lighteval_results(variant_dir: Path) -> Path | None:
    files = sorted((variant_dir / "mmlu_pro").rglob("results_*.json"))
    return files[-1] if files else None


def extract_mmlu_pro(results_obj: dict) -> tuple[float | None, str | None]:
    results = results_obj.get("results", results_obj)
    for task_key, metrics in results.items():
        if task_key == "all" or not isinstance(metrics, dict):
            continue
        base = task_key.split("|", 1)[0]
        if base == "mmlu_pro" or base.startswith("mmlu_pro:"):
            mk, val = pick_metric(metrics)
            if val is not None:
                return val, f"{task_key}:{mk}"
    return None, None


def load_lm_eval(variant_dir: Path) -> dict:
    path = variant_dir / "lm_eval_arc_mmlu.json"
    if not path.is_file():
        return {"status": "missing"}
    obj = json.loads(path.read_text(encoding="utf-8"))
    return {
        "status": "ok",
        "file": str(path),
        "arc_easy": obj.get("arc_easy"),
        "arc_challenge": obj.get("arc_challenge"),
        "mmlu": obj.get("mmlu"),
        "metric_keys": obj.get("metric_keys", {}),
    }


def load_mmlu_pro(variant_dir: Path) -> dict:
    rpath = latest_lighteval_results(variant_dir)
    if rpath is None:
        return {"status": "missing"}
    obj = json.loads(rpath.read_text(encoding="utf-8"))
    score, mk = extract_mmlu_pro(obj)
    return {
        "status": "ok" if score is not None else "no_score",
        "file": str(rpath),
        "mmlu_pro": score,
        "metric_key": mk,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results_root",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
    )
    args = p.parse_args()
    root = Path(args.results_root)

    summary: dict = {
        "results_root": str(root),
        "protocol": {
            "arc_mmlu": "lm_eval 0-shot (prefer acc)",
            "mmlu_pro": "lighteval mmlu_pro|0 max_samples=300 disable_thinking "
            "max_new_tokens=32768 temp=0.7 top_p=0.8 top_k=20",
        },
        "variants": {},
    }
    rows = [
        "| variant | arc_easy | arc_challenge | mmlu | mmlu_pro(300) |",
        "|---|---:|---:|---:|---:|",
    ]

    def fmt(x):
        return "—" if x is None else f"{float(x):.4f}"

    for variant in VARIANTS:
        vdir = root / variant
        lm = load_lm_eval(vdir) if vdir.is_dir() else {"status": "missing"}
        mp = load_mmlu_pro(vdir) if vdir.is_dir() else {"status": "missing"}
        summary["variants"][variant] = {"lm_eval": lm, "mmlu_pro": mp}
        rows.append(
            f"| {variant} | {fmt(lm.get('arc_easy'))} | {fmt(lm.get('arc_challenge'))} "
            f"| {fmt(lm.get('mmlu'))} | {fmt(mp.get('mmlu_pro'))} |"
        )

    out_json = root / "summary.json"
    out_md = root / "summary.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
