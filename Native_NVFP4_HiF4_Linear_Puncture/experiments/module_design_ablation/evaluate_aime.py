"""AIME25-only entrypoint using the existing main.py/vLLM scoring path."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..downstream_accuracy_recovery.evaluate import check_model
from ..residual_lora_compensation.artifact import sha256, write_json
from .aime import REPO, aime_protocol, audit_aime, validate_result
from .evaluation_runtime import run_reasoning


def model_view(source, target):
    """Link model assets only, so historical prediction-cache directories stay out."""
    source, target = Path(source).resolve(), Path(target).resolve()
    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    names = set(index["weight_map"].values()) | {"model.safetensors.index.json", "config.json"}
    names.update(path.name for path in source.iterdir() if path.is_file() and (
        path.name.startswith("tokenizer") or path.name in {
            "chat_template.jinja", "special_tokens_map.json", "added_tokens.json",
            "generation_config.json", "vocab.json", "merges.txt", "hf_quant_config.json",
            "hif4_runtime_spec.pt", "hif4_runtime_abi.json", "non_equivalent_export.json",
            "residual_lora_export.json", "ablation_export.json"}
        or path.name.startswith("hif4_rotation_layer_")))
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
        raise ValueError("model assets must use relative paths within the model directory")
    for name in names:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    target.mkdir(parents=True, exist_ok=False)
    for name in sorted(names):
        link = target / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(source / name)
    return target


def source_hashes():
    paths = [Path(__file__), Path(__file__).with_name("aime.py")]
    paths += [REPO / name for name in (
        "main.py", "tasks/aime.py", "tasks/custom_tasks.py",
        "3rdparty/lighteval/src/lighteval/metrics/metrics.py",
        "3rdparty/lighteval/src/lighteval/metrics/metrics_sample.py",
        "3rdparty/lighteval/src/lighteval/models/model_output.py",
        "3rdparty/lighteval/src/lighteval/models/vllm/vllm_model.py",
        "Native_NVFP4_HiF4_Linear_Puncture/experiments/downstream_accuracy_recovery/evaluate.py",
        "Native_NVFP4_HiF4_Linear_Puncture/experiments/downstream_accuracy_recovery/protocol.py",
        "Native_NVFP4_HiF4_Linear_Puncture/experiments/downstream_accuracy_recovery/reporting.py",
    )]
    return {str(path.relative_to(REPO)): sha256(path) for path in paths}


def evaluate_aime(model_dir, output_dir, *, model_kind, smoke=False):
    from ..e2e_diag_reconstruction.evaluation.common import require_visible_cuda_count

    require_visible_cuda_count(2)
    source = Path(model_dir).resolve()
    marker = source / "ablation_export.json"
    if marker.is_file():
        saved = json.loads(marker.read_text())
        if not saved.get("complete") or (saved.get("smoke_only") and not smoke):
            raise ValueError("formal AIME evaluation requires a completed formal export")
    from .evaluation_runtime import install_evaluation_runtime
    install_evaluation_runtime()
    check_model(source, model_kind)
    audit = audit_aime(source)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    model = model_view(source, output / "model")
    spec, metadata = check_model(model, model_kind)
    protocol = aime_protocol(smoke=smoke)
    hashes = source_hashes()
    write_json({"source_model": str(source), "model": metadata, "protocol": protocol,
                "source_sha256": hashes}, output / "manifest.json")
    write_json(audit, output / "prompt_audit.json")
    os.environ.pop("HIF4_RUNTIME_SPEC_PATH", None)
    if spec is not None:
        os.environ["HIF4_RUNTIME_SPEC_PATH"] = str(spec)
    result = run_reasoning(model, output / "vllm_run", protocol, model_kind, spec)
    result.update(protocol=protocol, model_kind=model_kind, model=metadata)
    if source_hashes() != hashes:
        raise ValueError("AIME25 evaluation sources changed during the run")
    checked = validate_result(result, audit=audit, model_kind=model_kind, smoke=smoke)
    write_json({**result, "validation": checked}, output / "metrics.json")
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    audit = sub.add_parser("audit", help="CPU-only read of AIME25 data, tokenizer and prompts")
    audit.add_argument("--model_dir", required=True)
    audit.add_argument("--output", required=True)
    run = sub.add_parser("evaluate", help="Run one model; requires an isolated output directory and TP2")
    run.add_argument("--model_dir", required=True)
    run.add_argument("--output_dir", required=True)
    run.add_argument("--model_kind", choices=("native_nvfp4", "hif4"), required=True)
    run.add_argument("--smoke", action="store_true", help="One question, five generations; retain formal resource limits")
    args = vars(parser.parse_args())
    action = args.pop("action")
    if action == "audit":
        target = Path(args.pop("output"))
        if target.exists():
            raise FileExistsError(target)
        result = audit_aime(**args)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(result, target)
        print(json.dumps({"audit": str(target), "count": result["count"],
                          "generations_per_doc": result["generations_per_doc"]}))
    else:
        print(json.dumps(evaluate_aime(**args)))


if __name__ == "__main__":
    main()
