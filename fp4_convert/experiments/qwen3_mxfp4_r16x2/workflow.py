"""One approved R16x2 candidate: GPU validation, TP4 probe, GPQA x10, review queue."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

import torch

from .artifacts import PACKAGE, check_hashes, code_hashes, file_hash, save
from ..qwen3_mxfp4.parallel_workflow import candidate
from ..qwen3_mxfp4.review_answers import extract
from ..qwen3_mxfp4.workflow import now, run_command


def prepare(root, previous):
    root, previous = Path(root).absolute(), Path(previous).absolute()
    destination = root / "r16x2"
    if (root / "workflow_manifest.json").exists() or destination.exists():
        raise FileExistsError("R16x2 workflow already prepared")
    old = json.loads((previous / "workflow_manifest.json").read_text())
    baseline = Path(old["baseline"])
    native = json.loads((baseline / "manifest.json").read_text())
    check_hashes(native["source_sha256"])
    old_comparison = json.loads((previous / "comparison.json").read_text())
    old_review = previous / "answer_review_20261110_v2"
    reviewed = json.loads((old_review / "reviewed_summary.json").read_text())
    if (old_comparison["status"] != "completed" or reviewed["status"] != "completed"
            or set(reviewed["results"]) != {"native", "direct", "r64"}
            or any(item["generations"] != 1980 for item in reviewed["results"].values())):
        raise ValueError("The three prior results must be complete and reviewed")
    if not (root / "cpu_tests.log").read_text().rstrip().endswith("OK"):
        raise ValueError("CPU format, integration and scoring tests have not passed")
    conversion = json.loads((root / "checkpoints/r16x2/conversion.json").read_text())
    if (conversion["status"] != "completed" or conversion["variant"] != "r16x2"
            or conversion["converted_matrices"] != 18432
            or not conversion["non_expert_byte_exact"] or not conversion["saved_reload_byte_exact"]):
        raise ValueError("R16x2 conversion has not completed all checks")
    check_hashes(conversion["source_code_sha256"])
    authorization = json.loads((root / "authorization.json").read_text())
    check_hashes(authorization["protected_source_sha256"])
    check_hashes(authorization["protected_baseline_source_sha256"])
    frozen = code_hashes()
    reused = [previous / "comparison.json", baseline / "protocol_inputs.json",
              old_review / "reviewed_summary.json", old_review / "reviewed_scores.jsonl",
              old_review / "manifest.json", old_review / "manual_decisions.json",
              old_review / "extractions.jsonl", old_review / "completion_validation.json"]
    manifest = {"created_at": now(), "baseline": str(baseline), "previous_run": str(previous),
                "previous_review": str(old_review), "source_sha256": frozen,
                "baseline_source_sha256": native["source_sha256"],
                "protocol_inputs_sha256": file_hash(baseline / "protocol_inputs.json"),
                "reused_evidence_sha256": {str(path): file_hash(path) for path in reused},
                "cpu_tests_sha256": file_hash(root / "cpu_tests.log"),
                "order": ["r16x2_gpu_validation", "r16x2_probe", "r16x2_formal", "final_answer_review"],
                "gpu_policy": "GPU0 for isolated operator validation; GPU0-3 TP4 for both model phases",
                "failure_policy": "Exit on first failure; no automatic retries or protocol changes"}
    config = copy.deepcopy(native)
    config["source_sha256"].update(frozen)
    config.update(created_at=now(), status="prepared", variant="r16x2", baseline=str(baseline),
                  source_snapshot=str(root / "checkpoints/r16x2"),
                  source_native_snapshot=native["source_snapshot"], validation_phase="probe",
                  authorization="User confirmed regular-R16 expanded R32 experiment on GPU0-3 with TP4")
    config["checks"] = {"cpu_tests": str(root / "cpu_tests.log"), "gpu_inference": "not_run"}
    config["protocol"]["quantization"] = "routed experts FP32 R16x2 + MXFP4 floor group32 W4A4; non-experts unchanged"
    config["final_submission_review"] = "Reuse frozen parse_final rules and blind review; retain original JSON-only scores"
    config["environment"]["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    config["protocol"]["cuda_visible_devices"] = "0,1,2,3"
    if config["protocol"]["tensor_parallel_size"] != 4:
        raise ValueError("This candidate requires the approved GPU0-3 TP4 protocol")
    destination.mkdir()
    for phase in ("probe", "formal"):
        directory = destination / phase
        directory.mkdir()
        command = config["commands"][phase]
        command[2] = str(Path(__file__).with_name("entry.py"))
        for flag, path in (("--model_path", directory / "model"), ("--output_dir", directory / "evaluation")):
            command[command.index(flag) + 1] = str(path)
    for key in ("model_files", "model_small_file_sha256", "parent_run", "initialization_fix", "resolved_blocker"):
        config.pop(key, None)
    save(destination / "manifest.json", config)
    save(root / "workflow_manifest.json", manifest)


def prepare_review(root, manifest):
    """Reuse prior decisions by identity and final-text equality; never infer new ones."""
    for path, expected in manifest["reused_evidence_sha256"].items():
        if file_hash(path) != expected:
            raise ValueError(f"Reused evidence changed: {path}")
    previous = Path(manifest["previous_run"])
    comparison = json.loads((previous / "comparison.json").read_text())
    comparison["results"]["r16x2"] = json.loads((root / "r16x2_summary.json").read_text())
    comparison["scoring_basis"] = "Original strict JSON only; use the separate final-submission review for accuracy comparisons"
    save(root / "comparison.json", comparison)
    review = root / "answer_review"
    extract(root, review)
    old_review = Path(manifest["previous_review"])
    prior = {row["id"]: row for row in (json.loads(line) for line in (old_review / "extractions.jsonl").read_text().splitlines())}
    current = [json.loads(line) for line in (review / "extractions.jsonl").read_text().splitlines()]
    for row in current:
        if row["model"] != "r16x2" and row != prior[row["id"]]:
            raise ValueError("Reused model final-answer extraction changed")
    queue = json.loads((review / "blind_review_queue.json").read_text())
    decisions = json.loads((old_review / "manual_decisions.json").read_text())
    save(review / "reused_manual_decisions.json", {row["id"]: decisions[row["id"]] for row in queue if row["id"] in decisions})
    pending = [row for row in queue if row["id"] not in decisions]
    save(review / "pending_blind_review_queue.json", pending)
    save(review / "review_status.json", {"status": "awaiting_final_answer_review", "new_decisions_needed": len(pending),
        "prior_decisions_reused": len(queue)-len(pending), "original_scores_preserved": True})


def run(root):
    root = Path(root).absolute()
    if (root / "workflow_status.json").exists():
        raise FileExistsError("One-shot workflow already started")
    manifest = json.loads((root / "workflow_manifest.json").read_text())
    check_hashes(manifest["source_sha256"])
    check_hashes(manifest["baseline_source_sha256"])
    torch.set_num_threads(8)
    status = {"status": "starting", "pid": os.getpid(), "started_at": now(), "stages": []}
    try:
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
        run_command(root, status, "r16x2_gpu_validation", [sys.executable, "-u", "-m", PACKAGE + ".validate_gpu",
                    "--model", str(root / "checkpoints/r16x2"), "--output", str(root / "r16x2_gpu_validation.json")], environment)
        candidate(root, manifest, status, "r16x2")
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
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.previous is None:
            parser.error("prepare requires --previous")
        prepare(args.root, args.previous)
    else:
        run(args.root)
