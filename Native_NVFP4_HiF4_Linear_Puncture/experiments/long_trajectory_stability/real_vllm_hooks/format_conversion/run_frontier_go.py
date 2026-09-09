#!/usr/bin/env python3
"""Drive semantic frontier screens then full-budget official-judge points."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability import gpu_pool
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import read_jsonl, write_jsonl
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.semantic_frontier import (
    coarse_prefix_lengths,
    full_budget_points,
    summarize_frontier,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.build_probe_plan import transitions

HERE = Path(__file__).resolve().parent


def _gpu_run(cmd: list[str], log: Path) -> None:
    ids = gpu_pool.available_gpus()
    if len(ids) < 2:
        raise RuntimeError(f"HARDWARE_BLOCKED for frontier: {ids}")
    env = gpu_pool.cuda_env(ids[:2])
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as out:
        subprocess.run(["conda", "run", "-n", "hif4", "--no-capture-output", *cmd],
                       cwd=REPO_ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)


def run(root: Path, phasea_root: Path, model_path: str, max_samples: int = 8) -> dict:
    root = Path(root)
    out = root / "10_semantic_frontier"
    out.mkdir(parents=True, exist_ok=True)
    mech = [r for r in read_jsonl(root / "02_cohort/mechanism_cohort.jsonl")
            if r["mechanism_group"] == "mechanism_regression"][:max_samples]
    e0 = {r["prompt_key"]: r for r in read_jsonl(root / "03_isolated/E0.jsonl")}
    e1 = {r["prompt_key"]: r for r in read_jsonl(root / "03_isolated/E1.jsonl")}
    events = {r["sample_key"]: r for r in read_jsonl(root / "05_probe_plan/divergence_events.jsonl")}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    screen_requests = []
    for sample in mech:
        key = sample["prompt_key"]
        t_lex = events[key]["t_lex"]
        if t_lex is None:
            continue
        tr = transitions(e0[key], tokenizer)
        for kind, source in (("rescue", e0[key]), ("poison", e1[key])):
            for k in coarse_prefix_lengths(t_lex, len(source["output_ids"]),
                                           thinking_end=tr["thinking_end"], code_start=tr["code_start"]):
                screen_requests.append({"sample": sample, "kind": kind, "k": k, "stage": "screen"})
    plan = out / "screen_plan.json"
    plan.write_text(json.dumps({"requests": screen_requests}, indent=2) + "\n")
    # One process per kind to keep variant load correct.
    for kind in ("rescue", "poison"):
        subset = [r for r in screen_requests if r["kind"] == kind]
        if not subset:
            continue
        kind_plan = out / f"screen_{kind}.json"
        kind_plan.write_text(json.dumps({"requests": subset}, indent=2) + "\n")
        _gpu_run(["python", str(HERE / "semantic_frontier.py"),
                  "--request_plan", str(kind_plan),
                  "--isolated_root", str(root / "03_isolated"),
                  "--output_root", str(out / "runs"),
                  "--model_path", model_path, "--phasea_root", str(phasea_root)],
                 out / f"screen_{kind}.log")
    # Collect screens and schedule full-budget neighbors.
    full_requests = []
    summaries = []
    for sample in mech:
        key = sample["prompt_key"]
        for kind in ("rescue", "poison"):
            rows = []
            for path in sorted((out / "runs" / key).glob(f"{kind}_k*_screen.json")):
                payload = json.loads(path.read_text())
                rows.append(payload["result"])
            if not rows:
                continue
            for k in full_budget_points(rows):
                full_requests.append({"sample": sample, "kind": kind, "k": k, "stage": "full"})
    if full_requests:
        for kind in ("rescue", "poison"):
            subset = [r for r in full_requests if r["kind"] == kind]
            if not subset:
                continue
            kind_plan = out / f"full_{kind}.json"
            kind_plan.write_text(json.dumps({"requests": subset}, indent=2) + "\n")
            _gpu_run(["python", str(HERE / "semantic_frontier.py"),
                      "--request_plan", str(kind_plan),
                      "--isolated_root", str(root / "03_isolated"),
                      "--output_root", str(out / "runs"),
                      "--model_path", model_path, "--phasea_root", str(phasea_root)],
                     out / f"full_{kind}.log")
    for sample in mech:
        key = sample["prompt_key"]
        for kind in ("rescue", "poison"):
            judged = []
            for path in sorted((out / "runs" / key).glob(f"{kind}_k*_full.json")):
                judged.append(json.loads(path.read_text())["result"])
            if judged:
                summaries.append({"sample_key": key, "kind": kind, **summarize_frontier(judged)})
    write_jsonl(out / "frontier_summaries.jsonl", summaries)
    summary = {"n_screen": len(screen_requests), "n_full": len(full_requests), "n_summaries": len(summaries)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
