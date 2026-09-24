"""Run distribution capture and the existing downstream evaluation entrypoints."""
from __future__ import annotations

import argparse
from pathlib import Path

from .artifact import read_json, write_json
from .capture import capture


def evaluate_distribution(root, model_dir, output_dir):
    root, model_dir, output_dir = Path(root), Path(model_dir), Path(output_dir)
    dataset = __import__(
        "Native_NVFP4_HiF4_Linear_Puncture.experiments.progressive_error_cancellation.data",
        fromlist=["Dataset"],
    ).Dataset(root)
    native_dir = root / "captures/native"
    if not (native_dir / "manifest.json").exists():
        capture(root, dataset.snapshot, native_dir, native=True)
    candidate_dir = output_dir / "tp2_holdout"
    if (root / "captures/candidate/manifest.json").exists() and not (candidate_dir / "manifest.json").exists():
        candidate_dir = root / "captures/candidate"
    elif not (candidate_dir / "manifest.json").exists():
        capture(root, model_dir, candidate_dir, split="holdout", native=False)
    rows = read_json(candidate_dir / "manifest.json")["samples"]
    metrics = {}
    for sid, row in rows.items():
        values = row["metrics"]
        metrics[sid] = {"kl": values["kl_sum"] / values["kl_tokens"],
                        "nll": values["nll_sum"] / values["nll_tokens"] if values["nll_tokens"] else None,
                        "ppl": __import__("math").exp(values["nll_sum"] / values["nll_tokens"])
                        if values["nll_tokens"] else None}
    write_json({"status": "COMPLETE", "samples": metrics}, output_dir / "distribution_metrics.json")
    return metrics


def evaluate_downstream(model_dir, output_dir, tasks):
    """Delegate ARC/MMLU-Pro to the established project evaluators."""
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.evaluate import evaluate
    result = {}
    for task in tasks:
        result[task] = evaluate(model_dir, output_dir, task)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--downstream", action="store_true")
    args = parser.parse_args()
    evaluate_distribution(args.root, args.model_dir, args.output_dir)
    if args.downstream:
        evaluate_downstream(args.model_dir, args.output_dir, ("arc", "mmlu_pro"))
