#!/usr/bin/env python3
"""Official judge + paired stats for clean controlled sampled LCB E0/E1."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.judge_isolated_lcb import (
    judge_completion,
    official_checker_manifest,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import (
    read_jsonl,
    write_jsonl,
)


def exact_mcnemar_p(n01: int, n10: int) -> float:
    """Two-sided exact McNemar (binomial) on discordant pairs."""
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    lower = sum(math.comb(n, i) for i in range(0, k + 1))
    return min(1.0, (2.0 * lower) / (2**n))


def paired_bootstrap_delta(e0_pass: list[bool], e1_pass: list[bool], *, n_boot: int = 10000, seed: int = 20260908):
    import random

    n = len(e0_pass)
    if n != len(e1_pass) or n == 0:
        raise ValueError("bootstrap requires equal nonempty paired vectors")
    rng = random.Random(seed)
    observed = (sum(e1_pass) - sum(e0_pass)) / n
    samples = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        d = (sum(e1_pass[i] for i in idx) - sum(e0_pass[i] for i in idx)) / n
        samples.append(d)
    samples.sort()
    lo = samples[int(0.025 * (n_boot - 1))]
    hi = samples[int(0.975 * (n_boot - 1))]
    return {
        "delta_accuracy": observed,
        "delta_accuracy_pp": observed * 100.0,
        "ci95_low": lo,
        "ci95_high": hi,
        "ci95_low_pp": lo * 100.0,
        "ci95_high_pp": hi * 100.0,
        "n_boot": n_boot,
        "seed": seed,
    }


def decide_gate(delta_pp: float, ci_high_pp: float, mcnemar_p: float) -> dict:
    material = delta_pp <= -5.0
    statistically_supported = (ci_high_pp < 0.0) or (mcnemar_p < 0.05)
    if material and statistically_supported:
        gate = "A"
        confirmed = True
        label = "CLEAN_FORMAT_LOSS_CONFIRMED"
    elif (not material and statistically_supported and delta_pp < 0) or (material and not statistically_supported):
        gate = "B"
        confirmed = False
        label = "CLEAN_FORMAT_LOSS_SMALL_OR_UNCERTAIN"
    else:
        gate = "C"
        confirmed = False
        label = "CLEAN_FORMAT_LOSS_NOT_CONFIRMED"
    return {
        "gate": gate,
        "MATERIAL": material,
        "STATISTICALLY_SUPPORTED": statistically_supported,
        "CLEAN_FORMAT_LOSS_CONFIRMED": confirmed,
        "CLEAN_FORMAT_LOSS": label,
        "delta_pp": delta_pp,
        "ci95_high_pp": ci_high_pp,
        "mcnemar_p": mcnemar_p,
    }


def _pair_row(source: dict, t0: dict, t1: dict, j0: dict, j1: dict) -> dict:
    return {
        "doc_id": str(source["doc_id"]),
        "prompt_key": str(source["prompt_key"]),
        "seed": t0["seed"],
        "E0_pass": j0["pass"],
        "E1_pass": j1["pass"],
        "E0_generation_len": j0["generation_len"],
        "E1_generation_len": j1["generation_len"],
        "E0_finished_thinking": j0["finished_thinking"],
        "E1_finished_thinking": j1["finished_thinking"],
        "E0_failure_class": j0["failure_class"],
        "E1_failure_class": j1["failure_class"],
        "E0_reused_formal": bool(t0.get("reused_formal_phasea")),
        "pair_cell": (
            "n10"
            if j0["pass"] and not j1["pass"]
            else "n01"
            if (not j0["pass"] and j1["pass"])
            else "both_pass"
            if j0["pass"] and j1["pass"]
            else "both_fail"
        ),
    }


def _judge_one(
    source: dict,
    t0: dict,
    t1: dict,
    *,
    reused_e0: bool,
    think_end: list[int],
) -> dict:
    key = str(source["prompt_key"])
    if t0["input_ids"] != source["input_ids"] or t1["input_ids"] != source["input_ids"]:
        raise RuntimeError(f"input_ids mismatch for {key}")
    if t0.get("seed") != t1.get("seed"):
        raise RuntimeError(f"seed mismatch for {key}: {t0.get('seed')} vs {t1.get('seed')}")
    if reused_e0 or t0.get("E0_pass_authoritative") is not None:
        e0_pass = bool(t0["E0_pass_authoritative"])
        j0 = {
            "pass": e0_pass,
            "generation_len": int(t0.get("output_len") or 0),
            "finished_thinking": None,
            "failure_class": "reused_formal_phasea" if e0_pass else "reused_formal_phasea_fail",
        }
    else:
        j0 = judge_completion(source, t0["raw_text"], t0["output_ids"], think_end_ids=think_end)
    j1 = judge_completion(source, t1["raw_text"], t1["output_ids"], think_end_ids=think_end)
    return _pair_row(source, t0, t1, j0, j1)


def main() -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompt_manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.run_root
    out_dir = root / "10_clean_lcb"
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / "clean_lcb_pair_matrix.partial.jsonl"
    progress_path = out_dir / "clean_lcb_judge.progress.json"

    manifest = read_jsonl(args.prompt_manifest)
    e0 = {str(r["prompt_key"]): r for r in read_jsonl(out_dir / "clean_lcb_e0.jsonl")}
    e1 = {str(r["prompt_key"]): r for r in read_jsonl(out_dir / "clean_lcb_e1.jsonl")}
    if len(manifest) != 175:
        raise RuntimeError(f"expected 175 manifest rows, got {len(manifest)}")
    missing0 = [r["prompt_key"] for r in manifest if r["prompt_key"] not in e0]
    missing1 = [r["prompt_key"] for r in manifest if r["prompt_key"] not in e1]
    if missing0 or missing1:
        raise RuntimeError(f"incomplete clean outputs missing_e0={len(missing0)} missing_e1={len(missing1)}")

    reused_e0 = all(bool(e0[str(r["prompt_key"])].get("reused_formal_phasea")) for r in manifest)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    think_end = tokenizer.encode("</think>", add_special_tokens=False)

    done: dict[str, dict] = {}
    if partial_path.exists():
        for row in read_jsonl(partial_path):
            done[str(row["prompt_key"])] = row
        print(json.dumps({"resume_partial": len(done)}, ensure_ascii=False), flush=True)

    pending = [s for s in manifest if str(s["prompt_key"]) not in done]
    workers = max(1, int(args.workers))

    def _run(source: dict) -> dict:
        key = str(source["prompt_key"])
        return _judge_one(source, e0[key], e1[key], reused_e0=reused_e0, think_end=think_end)

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool, partial_path.open("a", encoding="utf-8") as pf:
            futures = {pool.submit(_run, s): s for s in pending}
            finished = len(done)
            for fut in as_completed(futures):
                row = fut.result()
                key = str(row["prompt_key"])
                done[key] = row
                pf.write(json.dumps(row, ensure_ascii=False) + "\n")
                pf.flush()
                finished += 1
                if finished % 5 == 0 or finished == len(manifest):
                    progress_path.write_text(
                        json.dumps({"done": finished, "total": len(manifest), "workers": workers}) + "\n"
                    )
                    print(json.dumps({"judge_progress": finished, "total": len(manifest)}, ensure_ascii=False), flush=True)

    pair_rows = [done[str(s["prompt_key"])] for s in manifest]
    if len(pair_rows) != len(manifest):
        raise RuntimeError(f"pair_rows incomplete {len(pair_rows)}/{len(manifest)}")
    write_jsonl(out_dir / "clean_lcb_pair_matrix.jsonl", pair_rows)
    e0_pass = [bool(r["E0_pass"]) for r in pair_rows]
    e1_pass = [bool(r["E1_pass"]) for r in pair_rows]
    n = len(pair_rows)
    n10 = sum(r["pair_cell"] == "n10" for r in pair_rows)
    n01 = sum(r["pair_cell"] == "n01" for r in pair_rows)
    both_pass = sum(r["pair_cell"] == "both_pass" for r in pair_rows)
    both_fail = sum(r["pair_cell"] == "both_fail" for r in pair_rows)
    boot = paired_bootstrap_delta(e0_pass, e1_pass)
    mcnemar_p = exact_mcnemar_p(n01, n10)
    delta_pp = boot["delta_accuracy_pp"]
    gate = decide_gate(delta_pp, boot["ci95_high_pp"], mcnemar_p)
    summary = {
        "schema_version": 1,
        "n_tasks": n,
        "pass_counts": {"E0": sum(e0_pass), "E1": sum(e1_pass)},
        "accuracy": {"E0": sum(e0_pass) / n, "E1": sum(e1_pass) / n},
        "pair_counts": {"n10": n10, "n01": n01, "both_pass": both_pass, "both_fail": both_fail},
        "delta_accuracy": boot["delta_accuracy"],
        "delta_accuracy_pp": delta_pp,
        "paired_bootstrap_95ci": boot,
        "exact_mcnemar_p": mcnemar_p,
        "gate": gate,
        "checker": official_checker_manifest(),
        "e0_policy": "reused_audited_phasea_formal" if reused_e0 else "fresh_benchmark",
        "execution_shape": "max_num_seqs=128",
        "historical_formal_observed_only": {"E0": 95, "E1": 34, "note": "prompt_mismatch_not_format_causal"},
    }
    (out_dir / "CLEAN_LCB_E0_E1_REPORT.json").write_text(json.dumps(summary, indent=2) + "\n")
    md = "\n".join(
        [
            "# CLEAN_LCB_E0_E1_REPORT",
            "",
            "协议：E0=复用已审计 Phase-A formal；E1=exact same `input_ids` + controlled sampled + `max_num_seqs=128` + 官方 LCB judge。",
            "",
            f"- E0 pass = **{summary['pass_counts']['E0']}/{n}**"
            + ("（reused formal）" if reused_e0 else ""),
            f"- E1 pass = **{summary['pass_counts']['E1']}/{n}**",
            f"- n10 (E0 Pass / E1 Fail) = **{n10}**",
            f"- n01 (E0 Fail / E1 Pass) = **{n01}**",
            f"- both pass / both fail = **{both_pass}** / **{both_fail}**",
            f"- delta accuracy (E1-E0) = **{delta_pp:.3f} pp**",
            f"- paired bootstrap 95% CI = **[{boot['ci95_low_pp']:.3f}, {boot['ci95_high_pp']:.3f}] pp**",
            f"- exact McNemar p = **{mcnemar_p:.6g}**",
            "",
            f"## Gate = **{gate['gate']}** (`{gate['CLEAN_FORMAT_LOSS']}`)",
            "",
            f"- MATERIAL (delta <= -5pp): {gate['MATERIAL']}",
            f"- STATISTICALLY_SUPPORTED: {gate['STATISTICALLY_SUPPORTED']}",
            "",
            "历史 formal 95/175 vs 34/175 仅作 observed facts（prompt mismatch），不可归因。",
            "",
        ]
    )
    (out_dir / "CLEAN_LCB_E0_E1_REPORT.md").write_text(md)
    print(json.dumps({"gate": gate, "pass_counts": summary["pass_counts"], "delta_pp": delta_pp}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
