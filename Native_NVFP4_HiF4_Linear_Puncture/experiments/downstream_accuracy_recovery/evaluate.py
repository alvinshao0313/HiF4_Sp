"""Evaluate native NVFP4 or an exported HiF4 model under one task protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from Native_NVFP4_HiF4_Linear_Puncture.experiments.residual_lora_compensation.artifact import (
    sha256, write_json,
)
from .protocol import ARC_COUNTS, TASK_DIRS, TASKS, protocol_for, reasoning_details

REPO_ROOT = Path(__file__).resolve().parents[3]


def check_model(model, model_kind):
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    index = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))
    for shard in set(index["weight_map"].values()):
        if not (model / shard).is_file():
            raise FileNotFoundError(model / shard)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
        raise ValueError("model tokenizer must define a nonempty chat template")
    quant = config.get("quantization_config") or {}
    legacy_quant = model / "hf_quant_config.json"
    if not quant and legacy_quant.is_file():
        quant = json.loads(legacy_quant.read_text(encoding="utf-8"))
    metadata = {"model_kind": model_kind, "model_dir": str(model),
                "config_sha256": sha256(model / "config.json"),
                "weight_index_sha256": sha256(model / "model.safetensors.index.json"),
                "quantization_config": quant}
    if legacy_quant.is_file():
        metadata["hf_quant_config_sha256"] = sha256(legacy_quant)
    if model_kind == "native_nvfp4":
        modern_nvfp4 = quant.get("quant_method") == "modelopt" and quant.get("quant_algo") == "NVFP4"
        legacy_nvfp4 = (quant.get("producer", {}).get("name") == "modelopt"
                        and quant.get("quantization", {}).get("quant_algo") == "NVFP4")
        if not (modern_nvfp4 or legacy_nvfp4):
            raise ValueError("native baseline must load the original ModelOpt NVFP4 checkpoint")
        if (model / "hif4_runtime_spec.pt").exists():
            raise ValueError("native baseline must not contain a HiF4 runtime sidecar")
        return None, {**metadata, "linear_backend": "emulation", "moe_backend": "emulation",
                      "weight_bits": 4, "activation_bits": 4,
                      "execution": "native_packed_nvfp4_emulation"}
    if model_kind != "hif4":
        raise ValueError(model_kind)
    if quant:
        raise ValueError("HiF4 export must not select a native checkpoint quantization method")
    spec = model / "hif4_runtime_spec.pt"
    if not spec.is_file():
        raise FileNotFoundError(spec)
    for name in ("residual_lora_export.json", "non_equivalent_export.json"):
        marker = model / name
        if marker.is_file():
            saved = json.loads(marker.read_text(encoding="utf-8"))
            if saved.get("complete") is not True or saved.get("smoke_only"):
                raise ValueError(f"not a complete formal model: {marker}")
    return spec, {**metadata, "linear_backend": "auto", "moe_backend": "triton",
                  "weight_bits": 4, "activation_bits": 4,
                  "execution": "hif4_quant_dequant_bf16_weights_with_runtime_activation_quant",
                  "hif4_runtime_spec_sha256": sha256(spec)}


def native_arc(model, *, smoke, tensor_parallel_size=2):
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.vllm_runner import (
        _patch_lm_eval_transformers, _pick_metric,
    )
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.lm_eval_vllm import (
        build_lm_eval_vllm_kwargs, patch_lm_eval_vllm_generate_compat,
    )
    _patch_lm_eval_transformers()
    patch_lm_eval_vllm_generate_compat()
    from lm_eval import simple_evaluate
    from lm_eval.models.vllm_causallms import VLLM

    kwargs = build_lm_eval_vllm_kwargs(model_path=str(model), native_nvfp4=True,
                                      max_model_len=4096, max_num_batched_tokens=4096, seed=42,
                                      tensor_parallel_size=tensor_parallel_size)
    lm = VLLM(batch_size="auto", **kwargs)
    core = lm.model.llm_engine.engine_core
    output_thread = core.output_queue_thread
    try:
        actual_quantization = lm.model.llm_engine.model_config.quantization
        if actual_quantization != "modelopt_fp4":
            raise ValueError(f"native ARC selected unexpected quantization: {actual_quantization}")
        result = simple_evaluate(model=lm, tasks=list(ARC_COUNTS), num_fewshot=0,
                                 batch_size="auto", limit=2 if smoke else None)
    finally:
        core.shutdown(timeout=60)
        output_thread.join(timeout=60)
        if output_thread.is_alive():
            raise RuntimeError("vLLM output thread did not terminate after engine shutdown")
    return {"scores": {name: _pick_metric(value)[1] for name, value in result["results"].items()},
            "raw_results": result["results"], "n_samples": result["n-samples"],
            "actual_quantization": actual_quantization}


def reasoning_command(model, output, protocol, model_kind, spec):
    command = [sys.executable, str(REPO_ROOT / "main.py"), "--model_path", str(model),
               "--datasets", protocol["task_name"], "--output_dir", str(output),
               "--tensor_parallel_size", str(protocol["tensor_parallel_size"]), "--max_model_length", "40960",
               "--max_new_tokens", str(protocol["max_new_tokens"]),
               "--temperature", "0.6", "--top_p", "0.95", "--top_k", "20", "--min_p", "0",
               "--gpu_memory_utilization", "0.9", "--fake_act_quant", "none",
               "--kv_cache_dtype", "bfloat16", "--enforce_eager",
               "--batch_size", "128", "--max_num_batched_tokens", "4096"]
    if protocol["max_samples"] is not None:
        command.extend(["--max_samples", str(protocol["max_samples"])])
    if model_kind == "native_nvfp4":
        command.extend(["--linear_backend", "emulation", "--moe_backend", "emulation"])
    else:
        command.extend(["--hif4_runtime_spec", str(spec), "--moe_backend", "triton"])
    return command


def run_reasoning(model, output, protocol, model_kind, spec):
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.vllm_runner import (
        parse_main_py_results,
    )
    output.mkdir()
    command = reasoning_command(model, output, protocol, model_kind, spec)
    write_json({"command": command, "protocol": protocol}, output / "command.json")
    log_path = output / "main_py.log"
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    if child.returncode:
        raise RuntimeError(f"main.py failed rc={child.returncode}; see {log_path}")
    result = parse_main_py_results(output)
    selected = re.findall(r"\bquantization=([^,\s]+)", log_path.read_text(encoding="utf-8"))
    expected = "modelopt_fp4" if model_kind == "native_nvfp4" else "None"
    if selected != [expected]:
        raise ValueError(f"expected one engine with quantization={expected}, got {selected}; see {log_path}")
    return {**result, "actual_quantization": selected[0], "command": command,
            "exit_code": child.returncode, "log_file": str(log_path)}


def evaluate(model_dir, output_dir, task, *, model_kind, smoke=False, tensor_parallel_size=2):
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.common import (
        require_visible_cuda_count,
    )
    protocol = protocol_for(task, smoke=smoke, tensor_parallel_size=tensor_parallel_size)
    require_visible_cuda_count(tensor_parallel_size)
    model = Path(model_dir).resolve()
    spec, metadata = check_model(model, model_kind)
    target = Path(output_dir).resolve() / "eval" / TASK_DIRS[task]
    target.mkdir(parents=True, exist_ok=False)
    # A previous HiF4 task in the caller must never select the native engine's path.
    os.environ.pop("HIF4_RUNTIME_SPEC_PATH", None)
    if spec is not None:
        os.environ["HIF4_RUNTIME_SPEC_PATH"] = str(spec)
    if task == "arc":
        if model_kind == "native_nvfp4":
            result = native_arc(model, smoke=smoke, tensor_parallel_size=tensor_parallel_size)
        else:
            from Native_NVFP4_HiF4_Linear_Puncture.experiments.residual_lora_compensation.arc_runtime import evaluate_arc
            result = evaluate_arc(model, spec, limit=2 if smoke else None,
                                  tensor_parallel_size=tensor_parallel_size)
        for name, count in protocol["sample_counts"].items():
            if result["n_samples"][name]["effective"] != count:
                raise ValueError(f"incomplete ARC evaluation for {name}")
        scores = result["scores"].values()
    else:
        result = run_reasoning(model, target / "vllm_run", protocol, model_kind, spec)
        details_file, details = reasoning_details(result, task)
        if len(details) != protocol["count"] or len({row["doc"]["id"] for row in details}) != len(details):
            raise ValueError(f"incomplete or duplicate samples for {task}")
        if any(len(row["model_response"]["text_post_processed"]) != 1 for row in details):
            raise ValueError("formal protocol requires exactly one generation per document")
        prompts = sorted(row["model_response"]["input"] for row in details)
        result.update({"details_file": str(details_file), "n_samples": len(details),
                       "actual_generations_per_doc": 1,
                       "actual_prompt_set_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest()})
        scores = [result["results"][protocol["task_name"]][protocol["metric"]]]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in scores):
        raise ValueError("invalid benchmark score")
    result.update({**protocol, "task": task, "model_kind": model_kind,
                   "native_nvfp4": model_kind == "native_nvfp4", "fake_act_quant": "none",
                   "model": metadata, "protocol": protocol})
    if spec is not None:
        result["hif4_runtime_spec_path"] = str(spec)
    write_json(result, target / "metrics.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--model_kind", choices=("native_nvfp4", "hif4"), required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="Use 2 ARC or 1 reasoning samples; retain formal generation and resource limits")
    evaluate(**vars(parser.parse_args()))
