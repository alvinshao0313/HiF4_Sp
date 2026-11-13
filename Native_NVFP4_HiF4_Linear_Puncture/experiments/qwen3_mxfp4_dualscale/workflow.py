"""Fail-fast TP4 probes followed by G256 and G512 GPQA x10, without retries."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess

import torch

from .artifacts import REPO, check_hashes, code_hashes, file_hash, save
from .evidence import summarize_phase, verify_runtime_audit
from .format import VARIANTS
from .report import prepare_review, prior_evidence
from ..nvfp4_gpqa.run import run as run_evaluation
from ..qwen3_mxfp4.workflow import now, make_model_views, verify_worker_release


def check_devices():
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    target = {row.split(",")[1].strip() for row in rows.splitlines() if int(row.split(",")[0]) in (4,5,6,7)}
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"], text=True)
    occupied = [row for row in apps.splitlines() if row.split(",")[0].strip() in target]
    if len(target) != 4 or occupied:
        raise RuntimeError(f"Authorized GPU4-7 are unavailable: {occupied}")
    return {"checked_at": now(), "devices": "4,5,6,7", "compute_processes": occupied}


def prepare(root, baseline, prior_table, reports):
    root, baseline, prior_table = map(lambda path: Path(path).absolute(), (root, baseline, prior_table))
    if (root / "workflow_manifest.json").exists() or any((root / variant).exists() for variant in VARIANTS):
        raise FileExistsError("Dual-scale workflow already prepared")
    native = json.loads((baseline / "manifest.json").read_text())
    check_hashes(native["source_sha256"])
    authorization = json.loads((root / "authorization.json").read_text())
    check_hashes(authorization["protected_uos_source_sha256"])
    if not (root / "cpu_tests.log").read_text().rstrip().endswith("OK"):
        raise ValueError("CPU tests must pass")
    conversions, numerical_reports = {}, {}
    for variant, report in zip(VARIANTS, reports):
        path = root / "checkpoints" / variant / "conversion.json"
        conversion = json.loads(path.read_text())
        if (conversion["status"] != "completed" or conversion["variant"] != variant
                or conversion["converted_matrices"] != 18432 or not conversion["non_expert_byte_exact"]
                or not conversion["saved_reload_byte_exact"]):
            raise ValueError(f"Incomplete conversion: {variant}")
        # Runtime/report additions do not change already exported weights.
        check_hashes({name: digest for name, digest in conversion["source_code_sha256"].items()
                      if Path(name).name in ("convert.py", "format.py", "artifacts.py", "r64_transform.py", "fht32.py")})
        conversions[str(path)] = file_hash(path)
        report = Path(report).absolute()
        numerical = json.loads(report.read_text())
        if numerical["status"] != "passed" or numerical["variant"] != variant or len(numerical["ranks"]) != 4:
            raise ValueError(f"Missing real TP4 numerical validation: {variant}")
        if any(row["status"] != "passed" or {r["tokens"] for r in row["moe"]} != {1,256,8192}
               for row in numerical["ranks"]):
            raise ValueError("Numerical validation does not cover all ranks and token shapes")
        check_hashes({name: digest for name, digest in numerical["source_sha256"].items()
                      if Path(name).name not in ("workflow.py", "report.py")})
        numerical_reports[variant] = str(report)
    prior_paths, reviews = prior_evidence(prior_table)
    frozen = code_hashes()
    manifest = {"created_at": now(), "baseline": str(baseline), "prior_table": str(prior_table),
        "prior_reviews": reviews, "source_sha256": frozen, "baseline_source_sha256": native["source_sha256"],
        "protected_uos_source_sha256": authorization["protected_uos_source_sha256"],
        "protocol_inputs_sha256": file_hash(baseline / "protocol_inputs.json"),
        "conversion_sha256": conversions, "gpu_validation_reports": numerical_reports,
        "gpu_validation_sha256": {path: file_hash(path) for path in numerical_reports.values()},
        "reused_evidence_sha256": {str(path): file_hash(path) for path in prior_paths},
        "cpu_tests_sha256": file_hash(root / "cpu_tests.log"),
        "order": ["dualscale_g256_probe", "dualscale_g512_probe", "dualscale_g256_formal", "dualscale_g512_formal", "final_answer_review"],
        "gpu_policy": "GPU4-7 TP4; preserve UOS on GPU0-3", "failure_policy": "Exit; no retries or protocol changes"}
    for variant in VARIANTS:
        destination = root / variant
        destination.mkdir()
        config = copy.deepcopy(native)
        config["source_sha256"].update(frozen)
        config.update(created_at=now(), status="prepared", variant=variant, baseline=str(baseline),
            source_snapshot=str(root / "checkpoints" / variant), source_native_snapshot=native["source_snapshot"],
            validation_phase="probe", authorization="User approved G256/G512 on GPU4-7, preserving UOS GPU0-3")
        config["checks"] = {"cpu_tests": str(root / "cpu_tests.log"), "tp4_numerical": numerical_reports[variant]}
        config["protocol"]["quantization"] = f"Routed experts {variant}: FP32 absmax/6 outer scale, original group32 floor MXFP4; BF16 GEMM"
        config["environment"]["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
        config["protocol"]["cuda_visible_devices"] = "4,5,6,7"
        if config["protocol"]["tensor_parallel_size"] != 4:
            raise ValueError("Expected TP4 protocol")
        for phase in ("probe", "formal"):
            directory = destination / phase
            directory.mkdir()
            command = config["commands"][phase]
            command[2] = str(Path(__file__).with_name("entry.py"))
            for flag, path in (("--model_path", directory / "model"), ("--output_dir", directory / "evaluation")):
                command[command.index(flag)+1] = str(path)
        for key in ("model_files", "model_small_file_sha256", "parent_run", "initialization_fix", "resolved_blocker"):
            config.pop(key, None)
        save(destination / "manifest.json", config)
        make_model_views(root, variant)
    save(root / "workflow_manifest.json", manifest)
    for name, digest in frozen.items():
        source = REPO / name
        target = root / "source_frozen" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        if file_hash(target) != digest:
            raise ValueError("Frozen source copy differs")


def run(root):
    root = Path(root).absolute()
    if (root / "workflow_status.json").exists():
        raise FileExistsError("One-shot workflow already started")
    manifest = json.loads((root / "workflow_manifest.json").read_text())
    torch.set_num_threads(8)
    status = {"status": "starting", "pid": os.getpid(), "started_at": now(), "stages": []}
    try:
        for phase in ("probe", "formal"):
            for variant in VARIANTS:
                check_hashes(manifest["source_sha256"])
                check_hashes(manifest["baseline_source_sha256"])
                check_hashes(manifest["protected_uos_source_sha256"])
                for name, digest in {**manifest["conversion_sha256"], **manifest["gpu_validation_sha256"]}.items():
                    if file_hash(name) != digest:
                        raise ValueError(f"Qualification evidence changed: {name}")
                baseline = Path(manifest["baseline"])
                if file_hash(baseline / "protocol_inputs.json") != manifest["protocol_inputs_sha256"]:
                    raise ValueError("Original protocol inputs changed")
                resource = check_devices()
                stage = {"name": f"{variant}_{phase}", "status": "running", "started_at": now(),
                         "run_root": str(root / variant), "resources": resource}
                status["stages"].append(stage)
                status.update(status="running", phase=stage["name"])
                save(root / "workflow_status.json", status)
                os.environ["QWEN3_MXFP4_AUDIT_DIR"] = str(root / variant / phase / "audit")
                print(f"Starting {stage['name']} on GPU4-7", flush=True)
                run_evaluation(root / variant, phase)
                summarize_phase(root / variant, phase, baseline / "protocol_inputs.json",
                    root / (f"{variant}_summary.json" if phase == "formal" else f"{variant}_probe_summary.json"))
                verify_runtime_audit(root / "checkpoints" / variant, root / variant / phase / "audit",
                                     root / f"{variant}_{phase}_runtime_audit.json")
                verify_worker_release(root / variant / phase / "audit")
                stage.update(status="completed", finished_at=now())
                save(root / "workflow_status.json", status)
        prepare_review(root, manifest)
        status.update(status="awaiting_answer_review", phase="final_answer_review", inference_completed=True, finished_at=now())
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=now())
        if status["stages"] and status["stages"][-1]["status"] == "running":
            status["stages"][-1]["status"] = "failed"
        raise
    finally:
        save(root / "workflow_status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--prior-table", type=Path)
    parser.add_argument("--g256-validation", type=Path)
    parser.add_argument("--g512-validation", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if any(value is None for value in (args.baseline, args.prior_table, args.g256_validation, args.g512_validation)):
            parser.error("prepare requires baseline, prior-table, and both numerical reports")
        prepare(args.root, args.baseline, args.prior_table, (args.g256_validation, args.g512_validation))
    else:
        run(args.root)
