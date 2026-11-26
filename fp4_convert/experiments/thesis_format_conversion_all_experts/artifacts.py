"""Freeze this adapter separately from the active scale-evaluation dependency set."""
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess

from ..reasoning_trace_calibration.training import samples_for
from ..thesis_format_conversion_supplement.artifacts import (
    freeze_sources, read_json, run_root, sha256, verify_sources, write_json)
from ..thesis_format_conversion_supplement.config import NATIVE, PLAN, PROTOCOL, REPO, SOURCE_RUN

MODULE = "fp4_convert.experiments.thesis_format_conversion_all_experts"


def adapter_hashes():
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += [Path(__file__).parent.parent / path for path in (
        "reasoning_trace_calibration/training.py", "ban_trace_calibration/processes.py",
        "e2e_diag_reconstruction/training/moe_layer_runtime.py",
        "e2e_diag_reconstruction/training/moe_trainer.py",
        "e2e_diag_reconstruction/evaluation/moe_materialize.py")]
    return {str(p.relative_to(REPO)): sha256(p) for p in sorted(paths)}


def verify(root):
    verify_sources(root)
    if read_json(root / "all_expert_source_hashes.json") != adapter_hashes():
        raise ValueError("all-expert adapter sources changed")
    samples, _, hashes = samples_for(SOURCE_RUN, "matrix_norm_s1k")
    saved = read_json(root / "calibration.json")
    if hashes != saved["sha256"] or [s.sample_id for s in samples] != saved["sample_ids"]:
        raise ValueError("A20-matched S1K calibration changed")
    return samples


def prepare(root, source):
    root = run_root(root)
    root.mkdir(parents=True, exist_ok=False)
    train, val, hashes = samples_for(SOURCE_RUN, "matrix_norm_s1k")
    if not all(s.loss_mask.bool().all() for s in train):
        raise ValueError("matched S1K must preserve every complete valid sequence")
    protocol = copy.deepcopy(PROTOCOL)
    protocol["gptq"].update(hessian_input="all valid tokens through every expert; R64 before activation QDQ",
                            layer_output="original top-8 router selections and weights only",
                            unvisited_expert="natural zero-route allowed; zero calibration input fails",
                            reference="https://github.com/vllm-project/llm-compressor/blob/0.9.0/src/llmcompressor/modeling/qwen3_moe.py")
    write_json(protocol, root / "protocol.json")
    write_json({"source": str(SOURCE_RUN / "calibration/s1k_clean"), "candidate": "matrix_norm_s1k",
                "sha256": hashes, "sample_ids": [s.sample_id for s in train], "count": len(train),
                "tokens": sum(s.input_ids.numel() for s in train),
                "max_length": max(s.input_ids.numel() for s in train), "full_sequences": True,
                "validation_count": len(val), "validation_used_for_gptq": False,
                "scope": "exact A20 training IDs/order/tokens; no C4, expanded S1K, retokenization or truncation"},
               root / "calibration.json")
    # Evaluation inputs and existing-result inventory were already audited. Keep
    # their provenance and independent caches; do not copy historical predictions.
    for name in ("inventory.json", "summary.json", "isolation_and_development.json", "efficiency_data.json"):
        if (source / name).exists():
            shutil.copyfile(source / name, root / name)
    shutil.copytree(source / "evaluation/data", root / "evaluation/data")
    write_json({"status": "prepared", "created_utc": datetime.now(timezone.utc).isoformat(),
                "native_snapshot": str(NATIVE), "plan": str(PLAN), "plan_sha256": sha256(PLAN),
                "allocated_gpus": [0, 1, 6, 7], "conversion_gpu": 0,
                "reuse_manifest_source": str(source), "algorithm": "r64_gptq_all_experts",
                "authorization": "2026-11-18 user selects A20 S1K and all-token all-expert calibration",
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()},
               root / "manifest.json")
    (root / "initial_worktree.diff").write_text(subprocess.check_output(["git", "diff"], text=True))
    freeze_sources(root)
    write_json(adapter_hashes(), root / "all_expert_source_hashes.json")
    verify(root)
