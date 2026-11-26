"""Run one finite formal lane and save repeat-preserving accuracy statistics."""
import argparse
import csv
from fractions import Fraction
import math
from pathlib import Path

from ..qwen3_aisbench_gpqa.common import read
from .artifacts import file_hash, save
from .format import VARIANTS
from .workflow import run


def statistics(root, variant):
    root = Path(root)
    target = root / variant / "formal"
    summary = read(target / "final_submission_summary.json")
    counts = summary["correct_by_sampling_index"]
    if len(counts) != 10 or summary["generations"] != 1980 or sum(counts) != summary["correct"]:
        raise ValueError("Incomplete repeated accuracy evidence")
    rates = [Fraction(n, 198) for n in counts]
    mean = sum(rates) / 10
    squares = sum((x-mean)**2 for x in rates)
    source = Path(read(root / "authorization.json")["protocol_parent"]) / "answer_review_20261113T144526Z/reviewed_summary.json"
    baseline = read(source)
    if baseline["status"] != "completed" or baseline["pending_review_count"]:
        raise ValueError("Baseline review is incomplete")
    report = dict(summary, variant=variant, accuracy=float(mean), accuracy_percent=float(100*mean),
        accuracy_percent_by_sampling_index=[float(100*x) for x in rates],
        sample_variance_pp_squared=float(squares*10000/9),
        population_variance_pp_squared=float(squares*10000/10),
        sample_std_pp=math.sqrt(float(squares*10000/9)),
        variance_definition="Across 10 sampling-index accuracies on the same 198 questions; ddof=1, not ten independently seeded runs",
        baseline_review=str(source), baseline_review_sha256=file_hash(source),
        delta_percentage_points={name: float(100*(mean-Fraction(row["correct"], row["generations"])))
            for name, row in baseline["results"].items()},
        comparison_execution_note="R32 baseline TP4/32GiB; UOS/G512 and combinations TP2/56GiB; all BF16 KV")
    if (target / "statistics.json").exists():
        raise FileExistsError("Statistics already saved; preserve previous scoring evidence")
    save(target / "statistics.json", report)
    with (target / "sampling_accuracy.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sampling_index", "correct", "questions", "accuracy_percent", "pending_blind_review"])
        writer.writerows((i, n, 198, float(rates[i]*100), summary["pending_review_count"])
                         for i, n in enumerate(counts))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--lane", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    exit_path = Path(args.root) / f"formal_lane{args.lane}_exit.json"
    if exit_path.exists():
        raise FileExistsError(exit_path)
    result = {"returncode": 1}
    try:
        run(args.root, args.lane, "formal")
        for variant in VARIANTS[args.lane::2]:
            statistics(args.root, variant)
        result["returncode"] = 0
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save(exit_path, result)
