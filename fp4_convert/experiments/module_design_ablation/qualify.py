"""Record completed real-path checks before launching the finite cohort."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import torch

from ..non_equivalent_reconstruction.artifact import sha256, write_json
from .run import source_manifest


def qualify(directory, native_validation_pid=None):
    root = Path(directory).resolve()
    evidence = {}
    for name in ("rotation_norm_kl", "current_kl", "diag"):
        path = root / "transitions" / name / "summary.json"
        summary = json.loads(path.read_text())
        manifest = json.loads(path.with_name("manifest.json").read_text())
        if (summary.get("complete") is not True or summary["phase"] != "smoke" or len(summary["layers"]) != 2
                or manifest["completed_layers"] != [0, 1] or manifest["config"]["batch_size"] != 4):
            raise ValueError(f"longest-batch/two-layer training check incomplete: {name}")
        evidence[name] = {"path": str(path), "sha256": sha256(path), "seconds": summary["seconds"]}
    kernels = json.loads((root / "kernels.json").read_text())
    if kernels.get("passed") is not True:
        raise ValueError("real rotation/MoE operator verification is incomplete")
    evidence["kernels"] = kernels
    arc = json.loads((root / "arc_v2/validation.json").read_text())
    if arc.get("validated") is not True or arc.get("smoke") is not True:
        raise ValueError("rotation TP2 ARC path is incomplete")
    evidence["arc"] = arc
    aime = {}
    for name, kind in (("aime25_native", "native_nvfp4"), ("aime25_v3", "hif4")):
        path = root / name / "metrics.json"
        if name == "aime25_native" and not path.is_file() and native_validation_pid is not None:
            command = Path(f"/proc/{native_validation_pid}/cmdline").read_bytes().split(b"\0")
            words = [word.decode() for word in command if word]
            actual_output = Path(words[words.index("--output_dir") + 1])
            if not actual_output.is_absolute():
                actual_output = Path(os.readlink(f"/proc/{native_validation_pid}/cwd")) / actual_output
            if actual_output.resolve() != path.parent or "native_nvfp4" not in words or "--smoke" not in words:
                raise ValueError("pending PID is not this run's native AIME validation")
            log = (path.parent / "vllm_run/main_py.log").read_text()
            engines = [json.loads(line.split("ABLATION_ENGINE_CONFIG ", 1)[1])
                       for line in log.splitlines() if line.startswith("ABLATION_ENGINE_CONFIG ")]
            if len(engines) != 1 or engines[0]["quantization"] != "modelopt_fp4" or engines[0]["tensor_parallel_size"] != 2:
                raise ValueError("pending native AIME has not initialized the required engine")
            aime[name] = {"status": "running", "pid": native_validation_pid, "path": str(path),
                          "command": words, "engine": engines[0], "gpus": [0, 1]}
            continue
        metrics = json.loads(path.read_text())
        checked = metrics["validation"]
        if (metrics["model_kind"] != kind or checked["question_count"] != 1
                or checked["generation_count"] != 5 or checked["smoke_only"] is not True):
            raise ValueError(f"AIME25 five-generation check incomplete: {name}")
        aime[name] = {"status": "complete", "path": str(path), "sha256": sha256(path), "engine": metrics["actual_engine"],
                      "documents": checked["evaluated_documents_sha256"]}
    if len({row["documents"] for row in aime.values() if row["status"] == "complete"}) != 1:
        raise ValueError("native and rotation AIME checks used different documents")
    evidence["aime"] = aime
    result = {"passed": True, "scope": "training_and_rotation_deployment", "evidence": evidence, "source_sha256": source_manifest(),
              "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                              "gpu_inventory": subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"], text=True)},
              "limitations": "two layers establish execution/updates/transition; no formal convergence or six-task quality claim"}
    write_json(result, root / "qualification.json")
    print(json.dumps({"passed": True, "qualification": str(root / "qualification.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--native_validation_pid", type=int)
    qualify(**vars(parser.parse_args()))
