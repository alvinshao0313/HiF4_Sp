"""GPU4,5 adapter; retain the supplement's engine, task, cache and validation code.

The existing entry's GPU whitelist is frozen by live scale/GPTQ jobs. Keep this
small execution adapter separate instead of changing their dependencies.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..downstream_accuracy_recovery.evaluate import check_model, reasoning_command
from ..downstream_accuracy_recovery.protocol import protocol_for
from ..e2e_diag_reconstruction.evaluation.vllm_runner import parse_main_py_results
from ..module_design_ablation.evaluate_aime import model_view
from ..matrix_norm_efficiency.evaluate import validate_result, validate_raw_cache
from ..ban_trace_calibration.processes import require_free, finish_stage
from ..thesis_format_conversion_supplement.artifacts import read_json, write_json, run_root
from ..thesis_format_conversion_supplement.config import MODULE as SUPPLEMENT, REASONING, REPO
from ..thesis_format_conversion_supplement.evaluation import audit_model, model_for
from .common import ROLES, verify


def evaluate(root, candidate, task, *, smoke=False, repetition=1):
    root = run_root(root)
    verify(root)
    pair = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if pair != "4,5" or candidate not in ROLES:
        raise ValueError("this adapter is allocated GPU4,5 for A20/B20 only")
    require_free(pair)
    if repetition not in range(1, 6):
        raise ValueError("repetition must be 1..5")
    target = root / ("qualification/evaluation" if smoke else "evaluation") / candidate / f"round_{repetition:02d}" / task
    model = model_view(model_for(root, candidate), target / "model")
    kind = "hif4"
    spec, metadata = check_model(model, kind)
    if task == "arc":
        command = [sys.executable, "-m", SUPPLEMENT + ".arc_main", "--model_dir", str(model),
                   "--output_dir", str(target), "--task", "arc", "--model_kind", kind]
        if smoke:
            command.append("--smoke")
        protocol = protocol_for("arc", smoke=smoke)
    else:
        protocol = protocol_for(task, smoke=smoke)
        audit = audit_model(root, model, (task,))
        baseline = read_json(root / "evaluation/prompt_audit.json")
        if audit["tokenizer_files"] != baseline["tokenizer_files"] or audit["tasks"][task] != baseline["tasks"][task]:
            raise ValueError("actual task inputs differ from native/GPTQ protocol")
        write_json(audit, target / "prompt_audit.json")
        command = reasoning_command(model, target / "vllm_run", protocol, kind, spec)
        command[1:2] = ["-m", SUPPLEMENT + ".reasoning_main"]
    env = {**os.environ, "HIF4_SUPPLEMENT_RUN": str(root)}
    env.pop("HIF4_RUNTIME_SPEC_PATH", None)
    env["HIF4_RUNTIME_SPEC_PATH"] = str(spec)
    output = target / "vllm_run"
    output.mkdir()
    logfile = output / "main_py.log"
    write_json({"command": command, "protocol": protocol, "model": metadata}, output / "command.json")
    started = time.monotonic()
    with logfile.open("w") as log:
        child = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json({"pid": child.pid, "session_id": child.pid, "command": command,
                    "gpus": pair, "log": str(logfile)}, output / "process.json")
        status = child.wait()
    write_json({"exit_code": status, "seconds": time.monotonic() - started}, output / "exit.json")
    write_json(finish_stage(child.pid, pair), output / "children_exit.json")
    if status:
        raise RuntimeError(f"evaluation exit={status}: {logfile}")
    if task != "arc":
        engines = [json.loads(line.split("MATRIX_EFFICIENCY_ENGINE ", 1)[1]) for line in logfile.read_text().splitlines()
                   if line.startswith("MATRIX_EFFICIENCY_ENGINE ")]
        if len(engines) != 1:
            raise ValueError("expected one engine record")
        result = {**parse_main_py_results(output), "protocol": protocol, "model_kind": kind,
                  "model": metadata, "actual_engine": engines[0], "exit_code": status,
                  "actual_quantization": engines[0]["quantization"], "log_file": str(logfile), "command": command}
        checked = validate_result(result, task, audit["tasks"][task], model_kind=kind, smoke=smoke)
        validate_raw_cache(target, task, audit["tasks"][task], smoke=smoke)
        write_json(result, target / "metrics.json")
        write_json(checked, target / "verified_scores.json")
    verify(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidate", choices=ROLES, required=True)
    parser.add_argument("--task", choices=REASONING + ("arc",), required=True)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = vars(parser.parse_args())
    args["root"] = args.pop("run")
    evaluate(**args)
