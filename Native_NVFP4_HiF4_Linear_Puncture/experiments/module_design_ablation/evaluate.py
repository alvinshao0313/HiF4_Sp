"""Six-task evaluation, using fresh model views and the existing official scorers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..downstream_accuracy_recovery.evaluate import evaluate as evaluate_five
from ..downstream_accuracy_recovery.reporting import audit_prompts, validate_metrics
from ..non_equivalent_reconstruction.artifact import write_json
from .evaluate_aime import evaluate_aime, model_view
from .evaluation_runtime import install_evaluation_runtime
from .config import NATIVE_SNAPSHOT


TASKS = ("arc", "gsm8k", "lcb", "mmlu_pro", "aime25")


def check_export(model, *, smoke):
    marker = model / "ablation_export.json"
    if marker.is_file():
        saved = json.loads(marker.read_text())
        if not saved.get("complete") or (saved.get("smoke_only") and not smoke):
            raise ValueError("formal evaluation requires a completed formal model export")


def evaluate(model_dir, output_dir, task, *, model_kind, smoke=False):
    source = Path(model_dir).resolve()
    check_export(source, smoke=smoke)
    install_evaluation_runtime()
    if task == "aime25":
        return evaluate_aime(source, output_dir, model_kind=model_kind, smoke=smoke)
    if task not in TASKS:
        raise ValueError(task)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    model = model_view(source, output / "model")
    prompt_reference = None
    if not smoke and task != "arc":
        audit = audit_prompts({"native_nvfp4": NATIVE_SNAPSHOT, "candidate": model}, output / "prompt_audit.json")
        prompt_reference = audit["models"]["native_nvfp4"]["tasks"][task]
    metrics = evaluate_five(model, output, task, model_kind=model_kind, smoke=smoke)
    checked = validate_metrics(metrics, task, smoke=smoke, model_kind=model_kind, prompt_reference=prompt_reference)
    write_json({"task": task, "smoke": smoke, "validated": True, "validation": checked}, output / "validation.json")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--model_kind", choices=("native_nvfp4", "hif4"), required=True)
    parser.add_argument("--smoke", action="store_true")
    evaluate(**vars(parser.parse_args()))
