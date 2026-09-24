"""Evaluate an already materialized model; never invoke a legacy artifact loader."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifact import write_json


def run_arc(model, out, spec_path, *, smoke=False):
    """Own the vLLM lifecycle and finish shutdown before publishing scores."""
    from .arc_runtime import evaluate_arc
    result = evaluate_arc(model, spec_path, limit=2 if smoke else None)
    result["smoke_only"] = smoke
    write_json(result, out / "eval/arc/metrics.json")
    return result


def evaluate(model_dir, output_dir, task, *, smoke=False):
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.vllm_runner import (
        run_main_py_lighteval,
    )
    model, out = Path(model_dir).resolve(), Path(output_dir).resolve()
    marker_path = model / "residual_lora_export.json"
    if not marker_path.is_file():
        marker_path = model / "non_equivalent_export.json"
    marker = json.loads(marker_path.read_text())
    if marker["complete"] is not True:
        raise ValueError("incomplete export")
    if marker.get("smoke_only"):
        raise ValueError("a smoke checkpoint cannot enter formal downstream evaluation")
    if not (model / "chat_template.jinja").is_file():
        raise FileNotFoundError(model / "chat_template.jinja")
    spec_path = model / "hif4_runtime_spec.pt"
    if not spec_path.is_file():
        raise FileNotFoundError(spec_path)
    if task == "arc":
        return run_arc(model, out, spec_path, smoke=smoke)
    if task == "lcb":
        result = run_main_py_lighteval(
            model_path=model, output_dir=out / "eval/livecodebench/vllm_run",
            datasets="lcb:codegeneration_v6|0", max_samples=1 if smoke else None,
            max_new_tokens=38912, temperature=0.6, top_p=0.95, top_k=20,
            min_p=0.0, fake_act_quant="none", disable_thinking=False,
            hif4_runtime_spec_path=spec_path, batch_size=None,
        )
        result["smoke_only"] = smoke
        write_json(result, out / "eval/livecodebench/metrics.json")
        return result
    if task != "mmlu_pro":
        raise ValueError(task)
    result = run_main_py_lighteval(
        model_path=model, output_dir=out / "eval/mmlu_pro/vllm_run", datasets="mmlu_pro|0",
        max_samples=1 if smoke else 300, max_new_tokens=32768, temperature=0.6, top_p=0.95, top_k=20,
        min_p=0.0, fake_act_quant="none", disable_thinking=False,
        hif4_runtime_spec_path=spec_path, batch_size=128,
    )
    result["smoke_only"] = smoke
    write_json(result, out / "eval/mmlu_pro/metrics.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task", choices=("arc", "mmlu_pro", "lcb"), required=True)
    parser.add_argument("--smoke", action="store_true", help="Use 2 ARC / 1 reasoning samples, retaining formal generation limits")
    evaluate(**vars(parser.parse_args()))
