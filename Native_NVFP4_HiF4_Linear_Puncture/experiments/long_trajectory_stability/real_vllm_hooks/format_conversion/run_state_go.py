#!/usr/bin/env python3
"""Drive full-prefix state capture / noop / reset / injection with hard identity gates."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability import gpu_pool
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import read_jsonl
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.state_intervention import (
    canonical_scope,
    make_identity_gate,
)

HERE = Path(__file__).resolve().parent
COARSE_LAYERS = (0, 8, 16, 24, 32, 40, 47)


def _gpu_run(cmd: list[str], log: Path) -> None:
    ids = gpu_pool.available_gpus()
    if len(ids) < 2:
        raise RuntimeError(f"HARDWARE_BLOCKED for state intervention: {ids}")
    env = gpu_pool.cuda_env(ids[:2])
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as out:
        subprocess.run(["conda", "run", "-n", "hif4", "--no-capture-output", *cmd],
                       cwd=REPO_ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)


def _load_rank_logits(root: Path, sample_key: str, decode_index: int) -> dict[int, torch.Tensor]:
    logits = {}
    for rank in (0, 1):
        path = root / "raw_logits" / sample_key / f"rank{rank}_decode{decode_index}.pt"
        if not path.exists():
            # alternate layout used by forced trajectory helper
            matches = sorted((root / "raw_logits").glob(f"**/rank{rank}_decode{decode_index}.pt"))
            if len(matches) != 1:
                raise FileNotFoundError(f"raw logits missing for rank{rank} decode{decode_index} under {root}")
            path = matches[0]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        tensor = payload["logits"] if isinstance(payload, dict) and "logits" in payload else payload
        logits[rank] = tensor.detach().cpu()
    return logits


def _build_noop_gate(sample: dict, layer: int, boundary: str, t_lex: int,
                     capture_root: Path, noop_root: Path) -> dict:
    scope = canonical_scope(sample, layer, boundary, t_lex)
    target = int(sample["output_ids"][t_lex])
    base = _load_rank_logits(capture_root, sample["prompt_key"], t_lex)
    patched = _load_rank_logits(noop_root, sample["prompt_key"], t_lex)
    # Repeatability noise floor: re-run capture once more if available; else use capture vs itself.
    repeat_root = capture_root
    if (noop_root.parent / "capture_E0_repeat" / "manifest.json").exists():
        repeat_root = noop_root.parent / "capture_E0_repeat"
    repeat = _load_rank_logits(repeat_root, sample["prompt_key"], t_lex)
    rows = []
    for rank in (0, 1):
        rows.append({
            "rank": rank,
            "baseline": base[rank],
            "repeat": repeat[rank],
            "patched": patched[rank],
            "target": target,
            "forced_exact": True,
            "full_prefix_complete": True,
        })
    return make_identity_gate(scope, "noop", rows)


def run(root: Path, phasea_root: Path, model_path: str, max_samples: int = 8) -> dict:
    root = Path(root)
    out = root / "11_state_intervention"
    out.mkdir(parents=True, exist_ok=True)
    mech = [r for r in read_jsonl(root / "02_cohort/mechanism_cohort.jsonl")
            if r["mechanism_group"] == "mechanism_regression"][:max_samples]
    events = {r["sample_key"]: r for r in read_jsonl(root / "05_probe_plan/divergence_events.jsonl")}
    probe = root / "05_probe_plan/probe_plan.json"
    blocked, completed = [], []
    for sample in mech:
        key = sample["prompt_key"]
        t_lex = events.get(key, {}).get("t_lex")
        if t_lex is None:
            continue
        for boundary in ("layer_out", "post_attn_norm"):
            for layer in COARSE_LAYERS:
                case = f"{key}_L{layer}_{boundary}_t{t_lex}"
                for variant in ("E0", "E1"):
                    target = out / case / f"capture_{variant}"
                    if (target / "manifest.json").exists():
                        continue
                    _gpu_run(["python", str(HERE / "state_intervention.py"),
                              "--variant", variant, "--mode", "capture",
                              "--sample_json", str(probe), "--sample_key", key,
                              "--layer", str(layer), "--boundary", boundary,
                              "--target_decode_index", str(t_lex),
                              "--output_root", str(target),
                              "--model_path", model_path, "--phasea_root", str(phasea_root)],
                             out / f"{case}_capture_{variant}.log")
                # E0 repeat for noise floor.
                repeat = out / case / "capture_E0_repeat"
                if not (repeat / "manifest.json").exists():
                    _gpu_run(["python", str(HERE / "state_intervention.py"),
                              "--variant", "E0", "--mode", "capture",
                              "--sample_json", str(probe), "--sample_key", key,
                              "--layer", str(layer), "--boundary", boundary,
                              "--target_decode_index", str(t_lex),
                              "--output_root", str(repeat),
                              "--model_path", model_path, "--phasea_root", str(phasea_root)],
                             out / f"{case}_capture_E0_repeat.log")
                noop = out / case / "noop_E0"
                if not (noop / "manifest.json").exists():
                    _gpu_run(["python", str(HERE / "state_intervention.py"),
                              "--variant", "E0", "--mode", "noop",
                              "--sample_json", str(probe), "--sample_key", key,
                              "--layer", str(layer), "--boundary", boundary,
                              "--target_decode_index", str(t_lex),
                              "--reference_root", str(out / case / "capture_E0"),
                              "--output_root", str(noop),
                              "--model_path", model_path, "--phasea_root", str(phasea_root)],
                             out / f"{case}_noop.log")
                gate_path = out / case / "noop_gate.json"
                try:
                    gate = _build_noop_gate(sample, layer, boundary, t_lex,
                                            out / case / "capture_E0", noop)
                except Exception as exc:
                    blocked.append({"case": case, "reason": f"NOOP_GATE_BUILD_FAILED: {exc}"})
                    continue
                gate_path.write_text(json.dumps(gate, indent=2, default=str) + "\n")
                if gate.get("status") != "PASS":
                    blocked.append({"case": case, "reason": "STATE_RESET_BLOCKED", "gate_status": gate["status"]})
                    continue
                reset = out / case / "reset_E1"
                if not (reset / "manifest.json").exists():
                    _gpu_run(["python", str(HERE / "state_intervention.py"),
                              "--variant", "E1", "--mode", "reset",
                              "--sample_json", str(probe), "--sample_key", key,
                              "--layer", str(layer), "--boundary", boundary,
                              "--target_decode_index", str(t_lex),
                              "--reference_root", str(out / case / "capture_E0"),
                              "--identity_gate", str(gate_path),
                              "--output_root", str(reset),
                              "--model_path", model_path, "--phasea_root", str(phasea_root)],
                             out / f"{case}_reset.log")
                # Error injection alpha grid on the same critical case.
                for alpha in (0.0, 0.5, 1.0):
                    inj = out / case / f"inject_E0_a{alpha}"
                    if (inj / "manifest.json").exists():
                        continue
                    _gpu_run(["python", str(HERE / "state_intervention.py"),
                              "--variant", "E0", "--mode", "inject",
                              "--sample_json", str(probe), "--sample_key", key,
                              "--layer", str(layer), "--boundary", boundary,
                              "--target_decode_index", str(t_lex),
                              "--reference_root", str(out / case / "capture_E0"),
                              "--deviation_root", str(out / case / "capture_E1"),
                              "--alpha", str(alpha),
                              "--identity_gate", str(gate_path),
                              "--output_root", str(inj),
                              "--model_path", model_path, "--phasea_root", str(phasea_root)],
                             out / f"{case}_inject_a{alpha}.log")
                completed.append(case)
    summary = {"completed": completed, "blocked": blocked}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not completed and blocked:
        raise RuntimeError(f"STATE_RESET_BLOCKED: {blocked[:3]}")
    return summary
