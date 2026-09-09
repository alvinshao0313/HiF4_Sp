#!/usr/bin/env python3
"""Single Go entry: resume from artifact digests; formal mismatch continues via matched-prompt mechanism."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability import gpu_pool
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PHASEA_ROOT,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import read_jsonl, write_jsonl
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion import VARIANTS

HERE = Path(__file__).resolve().parent
HOOKS = HERE.parent
RESULTS = REPO_ROOT / "Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability"
DEFAULT_RUN_ROOT = RESULTS / "format_conversion_lcb_mechanism_go_20260907"
MAX_NEW_TOKENS = 38912


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class ArtifactTask:
    def __init__(self, root: Path, name: str, inputs: list[Path], parameters: dict):
        self.path = root / "manifests" / (name + ".json")
        self.expected = {
            "schema_version": 1, "name": name,
            "inputs": {str(p.resolve()): digest(p) for p in inputs if p.is_file()},
            "parameters": parameters,
        }

    def reusable(self) -> bool:
        if not self.path.exists():
            return False
        old = json.loads(self.path.read_text())
        if old.get("status") != "PASS" or any(old.get(k) != v for k, v in self.expected.items()):
            return False
        outputs = old.get("outputs", {})
        return bool(outputs) and all(Path(p).is_file() and digest(Path(p)) == h for p, h in outputs.items())

    def finish(self, outputs: list[Path], evidence: dict | None = None) -> None:
        missing = [str(p) for p in outputs if not p.is_file()]
        if not outputs or missing:
            raise RuntimeError(f"incomplete artifacts for {self.expected['name']}: {missing}")
        write_json(self.path, {
            **self.expected, "status": "PASS", "evidence": evidence or {},
            "outputs": {str(p.resolve()): digest(p) for p in outputs},
        })


def configure_gpu_pool() -> None:
    inventory = gpu_pool.query_gpus()
    os.environ["PROJECT_GPU_POOL"] = ",".join(str(g.index) for g in inventory)
    os.environ.pop("GPU_POOL", None)
    os.environ["GPU_MIN_FREE_RATIO"] = "0.90"
    os.environ["GPU_MAX_UTIL"] = "10"


def eligible_pairs(needed: int = 1) -> list[list[int]]:
    configure_gpu_pool()
    ids = gpu_pool.available_gpus()
    if len(ids) < 2 * needed:
        raise RuntimeError(
            f"HARDWARE_BLOCKED: need {needed} TP2 pair(s); available idle GPUs={ids}"
        )
    return [ids[i:i + 2] for i in range(0, 2 * needed, 2)]


def run_command(command: list[str], log: Path, *, gpu_ids: list[int] | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    env = gpu_pool.cuda_env(gpu_ids) if gpu_ids is not None else dict(os.environ)
    cmd = ["conda", "run", "-n", "hif4", "--no-capture-output", *command]
    write_json(log.with_suffix(".command.json"), {"command": cmd, "gpu_ids": gpu_ids})
    print(f"[Go] start {log.stem}; GPUs={gpu_ids}", flush=True)
    with log.open("w") as out:
        process = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(15)
        if process.returncode:
            raise RuntimeError(f"command failed ({process.returncode}); inspect {log}")
    print(f"[Go] PASS {log.stem}", flush=True)


def gpu_command(command: list[str], log: Path) -> None:
    pair = eligible_pairs(1)[0]
    run_command(command, log, gpu_ids=pair)


def parallel_gpu_commands(jobs: list[tuple[list[str], Path]]) -> None:
    """Run independent TP2 jobs on distinct idle pairs."""
    if not jobs:
        return
    pairs = eligible_pairs(len(jobs))
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(run_command, cmd, log, gpu_ids=pair)
                   for (cmd, log), pair in zip(jobs, pairs)]
        for future in as_completed(futures):
            future.result()


def update_audit(root: Path, phasea_root: Path) -> dict:
    audit_json = root / "00_audit/CURRENT_REPO_AUDIT.json"
    audit_md = root / "00_audit/CURRENT_REPO_AUDIT.md"
    mismatch = root / "00_audit/FORMAL_PROTOCOL_MISMATCH.json"
    formal_summary = root / "01_formal_matrix/formal_task_summary.json"
    if not mismatch.exists() or not formal_summary.exists():
        raise RuntimeError("formal protocol audit artifacts missing; rerun Task1/2 first")
    mismatch_payload = json.loads(mismatch.read_text())
    formal = json.loads(formal_summary.read_text())
    payload = json.loads(audit_json.read_text()) if audit_json.exists() else {}
    payload.update({
        "timestamp_updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "REAL_VLLM_CORE_READY": True,
        "FORMAL_LCB_E0_E3_REUSABLE": False,
        "FORMAL_LCB_E0_E3_REUSABLE_AS_FORMAT_ONLY_COMPARISON": False,
        "FORMAL_LABELS_REUSABLE_AS_OBSERVED_FACTS": True,
        "FORMAL_PROMPT_PROTOCOL_ALIGNED": bool(formal.get("FORMAL_PROMPT_PROTOCOL_ALIGNED")),
        "OLD_MMLU_RESULTS_REUSABLE": True,
        "old_lcb_isolated_reusable": False,
        "mechanism_continuation": "interest_seed_under_formal_prompt_mismatch_with_matched_E0_prompts",
        "formal_mismatch_status": mismatch_payload.get("status"),
        "phasea_root": str(Path(phasea_root).resolve()),
    })
    write_json(audit_json, payload)
    audit_md.write_text(
        "# 当前仓库审计\n\n"
        f"- REAL_VLLM_CORE_READY = true\n"
        f"- FORMAL_LCB_E0_E3_REUSABLE (format-only) = false\n"
        f"- FORMAL_LABELS_REUSABLE_AS_OBSERVED_FACTS = true\n"
        f"- FORMAL_PROMPT_PROTOCOL_ALIGNED = {formal.get('FORMAL_PROMPT_PROTOCOL_ALIGNED')}\n"
        f"- OLD_MMLU_RESULTS_REUSABLE = true\n"
        "- 继续路径：E0 chat-wrapper 统一 prompt 的 isolated greedy 机制链；"
        "正式 61 observed difference 仅作 interest seed，不作格式因果 regression。\n"
    )
    return payload


def task_formal_and_cohort(root: Path, phasea_root: Path, model_path: str) -> tuple[Path, Path]:
    matrix = root / "01_formal_matrix/formal_task_matrix.jsonl"
    summary = root / "01_formal_matrix/formal_task_summary.json"
    if not matrix.exists() or not summary.exists():
        raise RuntimeError("formal matrix missing")
    summary_payload = json.loads(summary.read_text())
    cohort_path = root / "02_cohort/benchmark_cohort.jsonl"
    manifest_path = root / "02_cohort/benchmark_cohort_manifest.json"
    task = ArtifactTask(root, "02_cohort", [matrix, summary], {
        "count": 16, "role": "interest_or_aligned",
        "aligned": bool(summary_payload.get("FORMAL_PROMPT_PROTOCOL_ALIGNED")),
    })
    if task.reusable():
        return cohort_path, manifest_path
    from transformers import AutoTokenizer
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.build_cohort import (
        build_benchmark_cohort,
        build_interest_cohort_under_prompt_mismatch,
        load_formal_artifacts,
    )
    details, _ = load_formal_artifacts(phasea_root)
    matrix_rows = read_jsonl(matrix)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    existing = read_jsonl(cohort_path) if cohort_path.exists() else []
    if summary_payload.get("FORMAL_PROMPT_PROTOCOL_ALIGNED"):
        rows, manifest = build_benchmark_cohort(matrix_rows, details["E0"], tokenizer, existing_rows=existing)
    else:
        rows, manifest = build_interest_cohort_under_prompt_mismatch(
            matrix_rows, details, tokenizer, existing_rows=existing)
    write_jsonl(cohort_path, rows)
    write_json(manifest_path, manifest)
    task.finish([cohort_path, manifest_path], {"n_samples": len(rows), "role": manifest["cohort_role"]})
    return cohort_path, manifest_path


def expand_cohort_if_needed(root: Path, phasea_root: Path, model_path: str,
                            target_regression: int = 8) -> None:
    judge_summary = root / "04_greedy_judge/greedy_task_summary.json"
    if not judge_summary.exists():
        return
    summary = json.loads(judge_summary.read_text())
    n_reg = summary.get("group_counts", {}).get("mechanism_regression", 0)
    if n_reg >= target_regression:
        return
    matrix = read_jsonl(root / "01_formal_matrix/formal_task_matrix.jsonl")
    cohort = read_jsonl(root / "02_cohort/benchmark_cohort.jsonl")
    used = {str(row["doc_id"]) for row in cohort}
    from transformers import AutoTokenizer
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.build_cohort import (
        build_interest_cohort_under_prompt_mismatch,
        load_formal_artifacts,
        select_regressions,
    )
    details, _ = load_formal_artifacts(phasea_root)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    # Add 8 more formal interest regressions each expansion round.
    extra, manifest = build_interest_cohort_under_prompt_mismatch(
        matrix, details, tokenizer, count=len([r for r in cohort if r["formal_group"] == "formal_regression"]) + 8,
        existing_rows=cohort, excluded_ids=())
    # Keep previous rows stable; only append new doc_ids.
    old_ids = used
    new_rows = [row for row in extra if str(row["doc_id"]) not in old_ids]
    if not new_rows:
        write_json(root / "04_greedy_judge/MECHANISM_LABEL_BRIDGE_INSUFFICIENT.json",
                   {"status": True, "mechanism_regression": n_reg, "exhausted_formal_interest": True})
        return
    merged = cohort + new_rows
    write_jsonl(root / "02_cohort/benchmark_cohort.jsonl", merged)
    write_json(root / "02_cohort/benchmark_cohort_manifest.json", {**manifest, "n_samples": len(merged),
                                                                  "expanded": True, "previous_n": len(cohort)})


def task_isolated(root: Path, phasea_root: Path, model_path: str, variants: list[str]) -> None:
    cohort = root / "02_cohort/benchmark_cohort.jsonl"
    out_dir = root / "03_isolated"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for variant in variants:
        out = out_dir / f"{variant}.jsonl"
        meta = out.with_suffix(".meta.json")
        task = ArtifactTask(root, f"03_isolated_{variant}", [cohort], {
            "variant": variant, "max_new_tokens": MAX_NEW_TOKENS, "temperature": 0,
            "top_p": 1, "top_k": 0, "min_p": 0, "tp": 2,
        })
        if task.reusable() and meta.exists():
            continue
        # Resume partial jsonl via runner --resume; finish only when meta exists.
        cmd = ["python", str(HOOKS / "run_isolated_free_run.py"),
               "--variant", variant, "--output", str(out),
               "--prompt_source", str(cohort), "--phasea_root", str(phasea_root),
               "--model_path", model_path, "--max_new_tokens", str(MAX_NEW_TOKENS),
               "--max_samples", "64", "--resume"]
        jobs.append((variant, cmd, out_dir / f"{variant}.log", out, meta, task))
    # Launch as many TP2 pairs as free GPUs allow.
    while jobs:
        n = max(1, len(gpu_pool.available_gpus()) // 2)
        n = min(n, len(jobs))
        if n < 1:
            raise RuntimeError("HARDWARE_BLOCKED: no idle TP2 pair for isolated free-run")
        batch, jobs = jobs[:n], jobs[n:]
        parallel_gpu_commands([(cmd, log) for _, cmd, log, _, _, _ in batch])
        for variant, _cmd, _log, out, meta, task in batch:
            if not meta.exists():
                raise RuntimeError(f"isolated {variant} finished without completion metadata")
            task.finish([out, meta], {"variant": variant})


def task_judge(root: Path, model_path: str, variants: list[str]) -> dict:
    cohort = root / "02_cohort/benchmark_cohort.jsonl"
    isolated = [root / f"03_isolated/{v}.jsonl" for v in variants]
    out_matrix = root / "04_greedy_judge/greedy_task_matrix.jsonl"
    out_summary = root / "04_greedy_judge/greedy_task_summary.json"
    task = ArtifactTask(root, "04_greedy_judge", [cohort, *isolated], {"variants": variants})
    if not task.reusable():
        run_command(["python", str(HERE / "judge_isolated_lcb.py"),
                     "--run_root", str(root), "--model_path", model_path,
                     "--variants", *variants],
                    root / "04_greedy_judge/judge.log")
        task.finish([out_matrix, out_summary])
    return json.loads(out_summary.read_text())


def select_mechanism_cohort(root: Path, target: int = 8) -> Path:
    matrix = read_jsonl(root / "04_greedy_judge/greedy_task_matrix.jsonl")
    cohort = read_jsonl(root / "02_cohort/benchmark_cohort.jsonl")
    by_key = {str(r["prompt_key"]): r for r in cohort}
    regressions = [r for r in matrix if r["mechanism_group"] == "mechanism_regression"]
    robust = [r for r in matrix if r["mechanism_group"] == "mechanism_robust"]
    selected = []
    used_robust = set()
    for reg in regressions[:target]:
        match = next((x for x in robust if x["doc_id"] == by_key[reg["prompt_key"]].get("matched_doc_id")
                      and x["prompt_key"] not in used_robust), None)
        if match is None:
            match = next((x for x in robust if x["prompt_key"] not in used_robust), None)
        if match is None:
            break
        used_robust.add(match["prompt_key"])
        selected.extend([
            {**by_key[reg["prompt_key"]], "mechanism_group": "mechanism_regression",
             "matched_regression_key": reg["prompt_key"], "t_lex_doc": reg["doc_id"]},
            {**by_key[match["prompt_key"]], "mechanism_group": "mechanism_robust",
             "matched_regression_key": reg["prompt_key"]},
        ])
    out = root / "02_cohort/mechanism_cohort.jsonl"
    write_jsonl(out, selected)
    write_json(root / "02_cohort/mechanism_cohort_manifest.json", {
        "n_samples": len(selected),
        "n_mechanism_regression": sum(1 for r in selected if r["mechanism_group"] == "mechanism_regression"),
        "n_mechanism_robust": sum(1 for r in selected if r["mechanism_group"] == "mechanism_robust"),
        "MECHANISM_LABEL_BRIDGE_INSUFFICIENT": sum(1 for r in selected if r["mechanism_group"] == "mechanism_regression") < 4,
    })
    return out


def task_probe_plan(root: Path, model_path: str) -> tuple[Path, Path]:
    from transformers import AutoTokenizer
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.build_probe_plan import build_plans
    mech = root / "02_cohort/mechanism_cohort.jsonl"
    e0 = read_jsonl(root / "03_isolated/E0.jsonl")
    e1 = read_jsonl(root / "03_isolated/E1.jsonl")
    cohort = read_jsonl(mech)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    heavy, feature, events, transitions = build_plans(cohort, e0, e1, tokenizer)
    out = root / "05_probe_plan"
    out.mkdir(parents=True, exist_ok=True)
    heavy_path, feature_path = out / "probe_plan.json", out / "feature_probe_plan.json"
    write_json(heavy_path, heavy)
    write_json(feature_path, feature)
    write_jsonl(out / "divergence_events.jsonl", events)
    write_jsonl(out / "think_transition_events.jsonl", transitions)
    return heavy_path, feature_path


def smoke_neutrality(root: Path) -> None:
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.validate_neutrality import (
        compare_forced_logits,
        compare_greedy,
    )
    folder = root / "00_audit/gpu_smoke"
    folder.mkdir(parents=True, exist_ok=True)
    source = root / "03_isolated/E0.jsonl"
    if not source.exists():
        # Fall back to legacy prompts only for pre-isolated smoke.
        source = RESULTS / "trajectory_stability_real_vllm_lcb/isolated/E0.jsonl"
    plan_path = folder / "smoke_probe_plan.json"
    samples = []
    for row in read_jsonl(source)[:2]:
        samples.append({**row, "positions": [{"decode_index": j} for j in range(min(128, max(row.get("output_len", 128), 1)))],
                        "max_required_decode_index": min(127, max(row.get("output_len", 128), 1) - 1)})
    write_json(plan_path, {"samples": samples, "purpose": "two_sample_runtime_neutrality_only"})
    runtime_sources = [HOOKS / n for n in ("worker_hooks.py", "hook_state.py", "build_llm.py",
                                          "forced_trajectory.py", "run_variant_hook_capture.py")]
    for variant in ("E0", "E1"):
        modes = ("greedy_none", "greedy_feature_scan", "forced_logits_only", "forced_core_qkv")
        manifests, roots = {}, {}
        for mode in modes:
            target = folder / variant / mode
            path = target / f"{variant}_{mode}_manifest.json"
            task = ArtifactTask(root, f"smoke_{variant}_{mode}", [plan_path, *runtime_sources],
                                {"variant": variant, "mode": mode})
            if not task.reusable():
                gpu_command(["python", str(HOOKS / "run_variant_hook_capture.py"),
                             "--variant", variant, "--mode", mode, "--probe_plan", str(plan_path),
                             "--output_root", str(target)], folder / f"{variant}_{mode}.log")
                task.finish([path, *sorted(target.rglob("*.pt"))])
            manifests[mode] = json.loads(path.read_text())
            roots[mode] = target
        evidence = {
            "status": "PASS", "variant": variant,
            "feature_greedy_neutrality": compare_greedy(manifests["greedy_none"], manifests["greedy_feature_scan"]),
            "qkv_raw_logit_neutrality": compare_forced_logits(
                roots["forced_logits_only"], roots["forced_core_qkv"],
                manifests["forced_logits_only"], manifests["forced_core_qkv"]),
        }
        if any(row.get("first_divergence") is not None for row in evidence["feature_greedy_neutrality"]):
            raise RuntimeError(f"FEATURE_SCAN_NEUTRALITY_FAILED for {variant}")
        write_json(folder / f"{variant}_neutrality.json", evidence)


def task_capture(root: Path, plan: Path, variants: list[str], mode: str, out_name: str) -> None:
    for variant in variants:
        target = root / out_name / variant
        manifest = target / f"{variant}_{mode}_manifest.json"
        task = ArtifactTask(root, f"{out_name}_{variant}_{mode}", [plan], {"variant": variant, "mode": mode})
        if task.reusable():
            continue
        gpu_command(["python", str(HOOKS / "run_variant_hook_capture.py"),
                     "--variant", variant, "--mode", mode, "--probe_plan", str(plan),
                     "--output_root", str(target)],
                    root / out_name / f"{variant}_{mode}.log")
        task.finish([manifest, *sorted(target.rglob("*.pt"))])


def task_quant_features(root: Path) -> None:
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.quant_features import (
        analyze_feature_capture,
    )
    out = root / "06_feature_capture/quant_features.jsonl"
    canonical = root / "03_isolated/E0.jsonl"
    for variant in ("E0", "E1"):
        capture_root = root / "06_feature_capture" / variant
        manifest = capture_root / f"{variant}_forced_feature_scan_manifest.json"
        if not manifest.exists():
            raise RuntimeError(f"missing feature capture for {variant}")
        analyze_feature_capture(capture_root / "hooks" / variant if (capture_root / "hooks").exists() else capture_root,
                                root / "06_feature_capture" / f"{variant}_features.jsonl",
                                manifest_path=manifest, canonical_path=canonical)
    write_json(root / "06_feature_capture/quant_features_done.json", {"status": "PASS"})


def task_compare_core(root: Path) -> None:
    out = root / "07_core_capture/e0_e1_compare.jsonl"
    task = ArtifactTask(root, "07_compare_e0_e1", [
        root / "07_core_capture/E0/E0_forced_core_manifest.json",
        root / "07_core_capture/E1/E1_forced_core_manifest.json",
    ], {"reference": "E0", "variant": "E1", "mode": "forced_core"})
    if task.reusable():
        return
    run_command(["python", str(HOOKS / "compare_hook_states.py"),
                 "--reference_root", str(root / "07_core_capture/E0"),
                 "--variant_root", str(root / "07_core_capture/E1"),
                 "--variant", "E1", "--mode", "forced_core",
                 "--output", str(out)],
                root / "07_core_capture/compare.log")
    task.finish([out])


def task_analyze(root: Path) -> None:
    run_command(["python", str(HERE / "analyze_mechanism.py"), "--run_root", str(root)],
                root / "analysis/analyze.log")


def write_stop(root: Path, reason: str, details: dict) -> None:
    path = root / "analysis/STOP_REASON.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# STOP_REASON\n\n{reason}\n\n```json\n{json.dumps(details, indent=2, ensure_ascii=False)}\n```\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--phasea_root", type=Path, default=DEFAULT_PHASEA_ROOT)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--through", default="report",
                        choices=["audit", "cohort", "isolated", "judge", "smoke", "feature",
                                 "core", "puncture", "frontier", "state", "e2e3", "report"])
    parser.add_argument("--skip_smoke", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "hif4":
        # conda run sets this inconsistently; require explicit python from env via conda run.
        pass
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    configure_gpu_pool()
    stages = ["audit", "cohort", "isolated", "judge", "smoke", "feature", "core",
              "puncture", "frontier", "state", "e2e3", "report"]
    stop_at = stages.index(args.through)

    replan_stop = (
        (root / "REPLAN_20260908_STOP_AFTER_JUDGE").exists()
        or os.environ.get("HIF4_REPLAN_STOP_AFTER_JUDGE") == "1"
    )
    if replan_stop:
        # 2026-09-08 replan: reuse completed matched-prompt isolated artifacts;
        # do not rebuild formal cohort (phaseA E1–E7 LCB paths may be quarantined)
        # and do not continue the superseded heavy mechanism chain.
        print("[Go] REPLAN mode: matched-prompt greedy judge only", flush=True)
        e0 = root / "03_isolated/E0.jsonl"
        e1 = root / "03_isolated/E1.jsonl"
        e0_meta = root / "03_isolated/E0.meta.json"
        e1_meta = root / "03_isolated/E1.meta.json"
        cohort = root / "02_cohort/benchmark_cohort.jsonl"
        if not (e0.exists() and e1.exists() and e0_meta.exists() and e1_meta.exists() and cohort.exists()):
            raise RuntimeError(
                "REPLAN judge-only requires complete E0/E1 isolated meta + cohort; "
                f"e0={e0.exists()} e1={e1.exists()} e0_meta={e0_meta.exists()} "
                f"e1_meta={e1_meta.exists()} cohort={cohort.exists()}"
            )
        n0 = len(read_jsonl(e0))
        n1 = len(read_jsonl(e1))
        nc = len(read_jsonl(cohort))
        if not (n0 == n1 == nc):
            raise RuntimeError(f"REPLAN isolated/cohort size mismatch: E0={n0} E1={n1} cohort={nc}")
        (root / "manifests/04_greedy_judge.json").unlink(missing_ok=True)
        summary = task_judge(root, args.model_path, ["E0", "E1"])
        write_json(
            root / "04_greedy_judge/REPLAN_STOP_GATE.json",
            {
                "status": "STOPPED_AFTER_MATCHED_PROMPT_GREEDY_JUDGE",
                "reason": (
                    "2026-09-08 replan: matched-prompt greedy judge only; "
                    "do not continue expand/feature/core/puncture/frontier/state/E2/E3"
                ),
                "plan": (
                    "2026-09-08-qwen3-30b-hif4-error-localization-replan-after-lcb-prompt-audit-cn.md"
                ),
                "summary": summary,
            },
        )
        from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.write_matched_prompt_greedy_audit import (
            build_audit,
        )
        payload, markdown = build_audit(root)
        out_dir = root / "04_greedy_judge"
        (out_dir / "MATCHED_PROMPT_GREEDY_AUDIT.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        )
        (out_dir / "MATCHED_PROMPT_GREEDY_AUDIT.md").write_text(markdown)
        print(
            "[Go] REPLAN STOP GATE: MATCHED_PROMPT_GREEDY_AUDIT written; "
            "heavy mechanism chain cancelled",
            flush=True,
        )
        return

    print("[Go] Task1 audit", flush=True)
    update_audit(root, args.phasea_root)
    if stop_at == 0:
        return

    print("[Go] Task2/3 formal matrix + interest/aligned cohort", flush=True)
    task_formal_and_cohort(root, args.phasea_root, args.model_path)
    if stop_at <= 1:
        return

    print("[Go] Task4 isolated greedy E0/E1", flush=True)
    task_isolated(root, args.phasea_root, args.model_path, ["E0", "E1"])
    if stop_at <= 2:
        return

    print("[Go] Task5 official greedy judge", flush=True)
    summary = task_judge(root, args.model_path, ["E0", "E1"])
    # Adaptive expansion until mechanism-regression >=8 or interest exhausted.
    rounds = 0
    while summary.get("group_counts", {}).get("mechanism_regression", 0) < 8 and rounds < 8:
        before = len(read_jsonl(root / "02_cohort/benchmark_cohort.jsonl"))
        expand_cohort_if_needed(root, args.phasea_root, args.model_path)
        after = len(read_jsonl(root / "02_cohort/benchmark_cohort.jsonl"))
        if after <= before:
            break
        task_isolated(root, args.phasea_root, args.model_path, ["E0", "E1"])
        # Invalidate judge manifest by rewriting with new inputs.
        (root / "manifests/04_greedy_judge.json").unlink(missing_ok=True)
        summary = task_judge(root, args.model_path, ["E0", "E1"])
        rounds += 1
    if summary.get("group_counts", {}).get("mechanism_regression", 0) < 4:
        write_json(root / "04_greedy_judge/MECHANISM_LABEL_BRIDGE_INSUFFICIENT.json",
                   {"status": True, **summary})
    select_mechanism_cohort(root)
    if stop_at <= 3:
        return

    heavy, feature = task_probe_plan(root, args.model_path)
    if not args.skip_smoke and stop_at >= 4:
        print("[Go] GPU neutrality smoke", flush=True)
        smoke_neutrality(root)
    if stop_at <= 4:
        return

    print("[Go] Task7/8 feature scan + quant features", flush=True)
    task_capture(root, feature, ["E0", "E1"], "forced_feature_scan", "06_feature_capture")
    task_quant_features(root)
    if stop_at <= 5:
        return

    print("[Go] Task9 forced_core", flush=True)
    task_capture(root, heavy, ["E0", "E1"], "forced_core", "07_core_capture")
    task_compare_core(root)
    if stop_at <= 6:
        return

    print("[Go] Task10/11/12 observational analysis before heavy causal modules", flush=True)
    task_analyze(root)

    print("[Go] Task10 production puncture driver", flush=True)
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion import run_puncture_go
    try:
        run_puncture_go.run(root, args.phasea_root, args.model_path, variants=["E0", "E1"])
    except Exception as exc:
        write_stop(root, f"CAUSAL_PUNCTURE_BLOCKED: {exc}", {"exc": str(exc)})
        print(f"[Go] puncture blocked: {exc}", flush=True)
    if stop_at <= 7:
        return

    print("[Go] Task13 semantic frontier driver", flush=True)
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion import run_frontier_go
    try:
        run_frontier_go.run(root, args.phasea_root, args.model_path)
    except Exception as exc:
        write_stop(root, f"CAUSAL_SEMANTIC_FRONTIER_BLOCKED: {exc}", {"exc": str(exc)})
        print(f"[Go] frontier blocked: {exc}", flush=True)
    if stop_at <= 8:
        return

    print("[Go] Task14/15 state intervention driver", flush=True)
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion import run_state_go
    try:
        run_state_go.run(root, args.phasea_root, args.model_path)
    except Exception as exc:
        write_stop(root, f"CAUSAL_STATE_INTERVENTION_BLOCKED: {exc}", {"exc": str(exc)})
        print(f"[Go] state intervention blocked: {exc}", flush=True)
    if stop_at <= 9:
        return

    print("[Go] Task16/17 E2/E3 same-cohort controls", flush=True)
    task_isolated(root, args.phasea_root, args.model_path, ["E2", "E3"])
    (root / "manifests/04_greedy_judge.json").unlink(missing_ok=True)
    task_judge(root, args.model_path, ["E0", "E1", "E2", "E3"])
    task_capture(root, heavy, ["E2", "E3"], "forced_core", "07_core_capture")
    try:
        run_puncture_go.run(root, args.phasea_root, args.model_path, variants=["E2", "E3"])
    except Exception as exc:
        write_stop(root, f"CAUSAL_PUNCTURE_E2E3_BLOCKED: {exc}", {"exc": str(exc)})
    if stop_at <= 10:
        return

    print("[Go] Task18/19 final report", flush=True)
    task_analyze(root)
    report = root / "analysis/FORMAT_CONVERSION_LCB_MECHANISM_REPORT.md"
    if not report.exists():
        raise RuntimeError("final report missing")
    print(f"[Go] COMPLETE {report}", flush=True)


if __name__ == "__main__":
    main()
