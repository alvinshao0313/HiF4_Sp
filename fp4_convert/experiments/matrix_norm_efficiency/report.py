"""Five-task paired accuracy loss and separately measured optimization cost."""
import math

import numpy as np

from .artifacts import read_json, sha256, validate_root, verify_sources, write_json
from .config import BASELINE_SECONDS, FORMAL_COUNTS, TASKS
from .training import training_dir


def accuracy_summary(native, candidate, *, repeats=2000, seed=42):
    if set(native) != set(TASKS) or set(candidate) != set(TASKS):
        raise ValueError("all five paired tasks are required")
    rng = np.random.default_rng(seed)
    rows, loss_draws = {}, np.zeros(repeats)
    for task in TASKS:
        if not native[task] or set(native[task]) != set(candidate[task]):
            raise ValueError(f"paired question coverage differs: {task}")
        ids = sorted(native[task])
        left = np.array([native[task][key] for key in ids], dtype=float)
        right = np.array([candidate[task][key] for key in ids], dtype=float)
        if any(not np.isfinite(x).all() or np.any((x < 0) | (x > 1)) for x in (left, right)):
            raise ValueError("question scores must be finite accuracies")
        delta = (left - right) * 100
        draws = np.empty(repeats)
        # Resample questions; five AIME responses stay together within a question.
        for start in range(0, repeats, 100):
            size = min(100, repeats - start)
            indices = rng.integers(len(ids), size=(size, len(ids)))
            draws[start:start + size] = delta[indices].mean(axis=1)
        loss_draws += np.maximum(0, draws) / len(TASKS)
        rows[task] = {"native_accuracy": float(left.mean()), "candidate_accuracy": float(right.mean()),
            "signed_loss_pp": float(delta.mean()), "positive_loss_pp": max(0., float(delta.mean())),
            "signed_loss_question_bootstrap_95pct_pp": np.quantile(draws, [.025, .975]).tolist(),
            "questions": len(ids)}
    mean = math.fsum(row["positive_loss_pp"] for row in rows.values()) / len(TASKS)
    return {"tasks": rows, "mean_positive_loss_pp": mean, "accuracy_target_met": mean <= .4 + 1e-12,
        "mean_loss_question_bootstrap_95pct_pp": np.quantile(loss_draws, [.025, .975]).tolist(),
        "uncertainty_scope": "question resampling; excludes new decoding seeds and independent training variance"}


def report(root, candidate):
    root = validate_root(root)
    verify_sources(root)
    directory = training_dir(root, candidate)
    training = read_json(directory / "manifest.json")
    all_scores, evidence = {}, {}
    for model in ("native_nvfp4", candidate):
        tasks = {}
        for task in TASKS:
            folder = root / "evaluation" / model / task
            audit = read_json(folder / "prompt_audit.json")
            central = read_json(root / "evaluation/prompt_audit.json")
            if audit["tasks"][task] != central["tasks"][task]:
                raise ValueError("report inputs differ from central evaluation audit")
            raw = read_json(folder / "raw_cache_validation.json")
            if not raw["complete"] or raw["questions"] != FORMAL_COUNTS[task]:
                raise ValueError("formal raw generation coverage is incomplete")
            path = folder / ("corrected_metrics.json" if task.startswith("aime") else "verified_scores.json")
            metric = read_json(path)
            if task.startswith("aime"):
                if not metric["complete"] or metric["scoring"] != "final_submission_avg5":
                    raise ValueError("AIME final submissions are not completely scored")
                scores = metric["per_question"]
            else:
                scores = metric["paired_scores"]
            if (len(scores) != FORMAL_COUNTS[task]
                    or set(scores) != set(central["tasks"][task]["documents"])
                    or not math.isclose(math.fsum(scores.values()) / len(scores), metric["score"], abs_tol=1e-12)):
                raise ValueError("formal score coverage or aggregation differs")
            tasks[task] = scores
            evidence[f"{model}/{task}"] = {"metrics": str(path), "sha256": sha256(path),
                                           "raw_cache_sha256": raw["sha256"]}
        all_scores[model] = tasks
    seconds = training["legacy_comparable_seconds"]
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("optimization duration must be positive")
    accuracy = accuracy_summary(all_scores["native_nvfp4"], all_scores[candidate])
    runtime = training.get("runtime", {})
    hardware_matches = (runtime.get("device_name") == "NVIDIA A800 80GB PCIe"
                        and runtime.get("training_gpu_count") == 1)
    costs = {"optimization_seconds": seconds, "optimization_gpu_hours": seconds / 3600,
        "optimization_gpu_count": 1, "runtime": runtime, "reference_hardware_matches": hardware_matches,
        "reference_optimization_seconds": BASELINE_SECONDS,
        "reference_speedup": BASELINE_SECONDS / seconds if hardware_matches else None,
        "within_three_hours": seconds <= 10800,
        "at_least_twice_reference_speed": seconds <= BASELINE_SECONDS / 2 if hardware_matches else None,
        "calibration_preparation_seconds": read_json(root / "prepare.inputs.json")["seconds"],
        "export_seconds": read_json(root / "models" / candidate / "export.json")["seconds"],
        "evaluation_seconds": {model: {task: read_json(root / "evaluation" / model / task /
                     "vllm_run/exit.json")["wall_seconds"] for task in TASKS} for model in all_scores},
        "scope": "optimization compared with the historical 20.0406h run; concurrent wall times are not additive",
        "other_costs": "shared prior trace generation, data/prompt audit, development, all candidate trials and short tests reported separately"}
    result = {"candidate": candidate, **accuracy, "costs": costs, "evidence": evidence}
    write_json(result, root / "reports" / f"{candidate}.json")
    return result
