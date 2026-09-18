"""Evaluate an already materialized model; never invoke a legacy artifact loader."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifact import write_json


def run_arc(model, out, spec_path):
    """Own the vLLM lifecycle and finish shutdown before publishing scores."""
    from .arc_runtime import evaluate_arc
    result = evaluate_arc(model, spec_path)
    write_json(result, out / "eval/arc/metrics.json")
    return result


def evaluate(model_dir, output_dir, task):
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.vllm_runner import (
        run_main_py_lighteval,
    )
    model, out = Path(model_dir).resolve(), Path(output_dir).resolve()
    marker = json.loads((model / "non_equivalent_export.json").read_text())
    if marker["complete"] is not True:
        raise ValueError("incomplete export")
    spec_path = model / "hif4_runtime_spec.pt"
    if not spec_path.is_file():
        raise FileNotFoundError(spec_path)
    if task == "arc":
        return run_arc(model, out, spec_path)
    if task != "mmlu_pro":
        raise ValueError(task)
    result = run_main_py_lighteval(
        model_path=model, output_dir=out / "eval/mmlu_pro/vllm_run", datasets="mmlu_pro|0",
        max_samples=300, max_new_tokens=32768, temperature=0.6, top_p=0.95, top_k=20,
        min_p=0.0, fake_act_quant="none", disable_thinking=False,
        hif4_runtime_spec_path=spec_path, batch_size=128,
    )
    write_json(result, out / "eval/mmlu_pro/metrics.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task", choices=("arc", "mmlu_pro"), required=True)
    evaluate(**vars(parser.parse_args()))
