"""Select an explicit rotation worker while retaining the existing task runner."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from ..downstream_accuracy_recovery.evaluate import REPO_ROOT, reasoning_command as original_command
from ..e2e_diag_reconstruction.evaluation.vllm_runner import parse_main_py_results
from ..non_equivalent_reconstruction.artifact import write_json


def reasoning_command(*args, **kwargs):
    command = original_command(*args, **kwargs)
    return [command[0], "-m",
            "Native_NVFP4_HiF4_Linear_Puncture.experiments.module_design_ablation.reasoning_main",
            *command[2:]]


def run_reasoning(model, output, protocol, model_kind, spec):
    output = Path(output)
    output.mkdir()
    command = reasoning_command(model, output, protocol, model_kind, spec)
    write_json({"command": command, "protocol": protocol}, output / "command.json")
    log_path = output / "main_py.log"
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    if child.returncode:
        raise RuntimeError(f"main.py failed rc={child.returncode}; see {log_path}")
    engine_rows = [json.loads(line.split("ABLATION_ENGINE_CONFIG ", 1)[1])
                   for line in log_path.read_text().splitlines() if line.startswith("ABLATION_ENGINE_CONFIG ")]
    if len(engine_rows) != 1:
        raise ValueError(f"expected one actual engine configuration, got {len(engine_rows)}")
    actual = engine_rows[0]
    expected = "modelopt_fp4" if model_kind == "native_nvfp4" else "None"
    if actual["quantization"] != expected or actual["tensor_parallel_size"] != 2:
        raise ValueError(f"actual engine configuration differs: {actual}")
    result = parse_main_py_results(output)
    return {**result, "actual_quantization": actual["quantization"], "actual_engine": actual,
            "command": command, "exit_code": child.returncode, "log_file": str(log_path)}


def install_evaluation_runtime():
    from ..downstream_accuracy_recovery import evaluate
    from .runtime import install_engine_hook
    install_engine_hook()
    evaluate.run_reasoning = run_reasoning
