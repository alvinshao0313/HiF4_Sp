#!/usr/bin/env python3
"""Official judge + paired stats for clean LCB E2/E3/E4 vs reused E0 (and vs E1).

Same control variables as clean E1: exact E0 input_ids, seed=1234,
max_num_seqs=128 protocol, official LCB checker. Does not rewrite E0/E1 Gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.judge_clean_sampled_lcb import (
    exact_mcnemar_p,
    paired_bootstrap_delta,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.judge_isolated_lcb import (
    judge_completion,
    official_checker_manifest,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import (
    read_jsonl,
    write_jsonl,
)

TARGET_VARIANTS = ("E2", "E3", "E4")
VARIANT_LABEL = {
    "E2": "E2_r64_only",
    "E3": "E3_fusable_adopted",
    "E4": "E4_fusable_r64_adopted",
}


def _pair_cell(base_pass: bool, other_pass: bool) -> str:
    if base_pass and not other_pass:
        return "n10"
    if (not base_pass) and other_pass:
        return "n01"
    if base_pass and other_pass:
        return "both_pass"
    return "both_fail"


def _summarize_pair(base_name: str, other_name: str, base_pass: list[bool], other_pass: list[bool]) -> dict:
    n = len(base_pass)
    if n != len(other_pass) or n == 0:
        raise ValueError(f"paired vectors mismatch for {base_name}/{other_name}")
    cells = [_pair_cell(a, b) for a, b in zip(base_pass, other_pass)]
    n10 = sum(c == "n10" for c in cells)
    n01 = sum(c == "n01" for c in cells)
    boot = paired_bootstrap_delta(base_pass, other_pass)
    mcnemar_p = exact_mcnemar_p(n01, n10)
    return {
        "base": base_name,
        "other": other_name,
        "n_tasks": n,
        "pass_counts": {base_name: sum(base_pass), other_name: sum(other_pass)},
        "accuracy": {base_name: sum(base_pass) / n, other_name: sum(other_pass) / n},
        "pair_counts": {
            "n10": n10,
            "n01": n01,
            "both_pass": sum(c == "both_pass" for c in cells),
            "both_fail": sum(c == "both_fail" for c in cells),
        },
        "delta_accuracy": boot["delta_accuracy"],
        "delta_accuracy_pp": boot["delta_accuracy_pp"],
        "paired_bootstrap_95ci": boot,
        "exact_mcnemar_p": mcnemar_p,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompt_manifest", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=list(TARGET_VARIANTS), choices=list(TARGET_VARIANTS))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    out_dir = args.run_root / "10_clean_lcb"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_jsonl(args.prompt_manifest)
    if len(manifest) != 175:
        raise RuntimeError(f"expected 175 manifest rows, got {len(manifest)}")

    e0 = {str(r["prompt_key"]): r for r in read_jsonl(out_dir / "clean_lcb_e0.jsonl")}
    e1_path = out_dir / "clean_lcb_e1.jsonl"
    e1 = {str(r["prompt_key"]): r for r in read_jsonl(e1_path)} if e1_path.exists() else {}
    missing0 = [r["prompt_key"] for r in manifest if r["prompt_key"] not in e0]
    if missing0:
        raise RuntimeError(f"incomplete clean E0 missing={len(missing0)}")

    reused_e0 = all(bool(e0[str(r["prompt_key"])].get("reused_formal_phasea")) for r in manifest)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    think_end = tokenizer.encode("</think>", add_special_tokens=False)

    # Prefer already-judged E0/E1 from existing pair matrix when present.
    e0_pass_by_key: dict[str, bool] = {}
    e1_pass_by_key: dict[str, bool] = {}
    pair_e0e1 = out_dir / "clean_lcb_pair_matrix.jsonl"
    if pair_e0e1.exists():
        for row in read_jsonl(pair_e0e1):
            e0_pass_by_key[str(row["prompt_key"])] = bool(row["E0_pass"])
            e1_pass_by_key[str(row["prompt_key"])] = bool(row["E1_pass"])
    else:
        for source in manifest:
            key = str(source["prompt_key"])
            t0 = e0[key]
            if reused_e0 or t0.get("E0_pass_authoritative") is not None:
                e0_pass_by_key[key] = bool(t0["E0_pass_authoritative"])
            else:
                j0 = judge_completion(source, t0["raw_text"], t0["output_ids"], think_end_ids=think_end)
                e0_pass_by_key[key] = bool(j0["pass"])
            if key in e1:
                j1 = judge_completion(source, e1[key]["raw_text"], e1[key]["output_ids"], think_end_ids=think_end)
                e1_pass_by_key[key] = bool(j1["pass"])

    variant_pass: dict[str, dict[str, bool]] = {}
    variant_meta: dict[str, dict] = {}
    for variant in args.variants:
        traj_path = out_dir / f"clean_lcb_{variant.lower()}.jsonl"
        meta_path = traj_path.with_suffix(".meta.json")
        if not traj_path.exists() or not meta_path.exists():
            raise RuntimeError(f"missing clean trajectories/meta for {variant}: {traj_path}")
        meta = json.loads(meta_path.read_text())
        protocol = meta.get("clean_protocol") or {}
        runtime = meta.get("runtime") or {}
        if int(protocol.get("max_num_seqs", -1)) != 128 or int(runtime.get("max_num_seqs", -1)) != 128:
            raise RuntimeError(f"{variant} execution shape lock failed: {meta_path}")
        if int(protocol.get("seed", -1)) != 1234 or int(runtime.get("seed", -1)) != 1234:
            raise RuntimeError(f"{variant} seed lock failed: {meta_path}")
        if meta.get("num_trajectories") != 175:
            raise RuntimeError(f"{variant} incomplete trajectories: {meta.get('num_trajectories')}")
        rows = {str(r["prompt_key"]): r for r in read_jsonl(traj_path)}
        missing = [r["prompt_key"] for r in manifest if r["prompt_key"] not in rows]
        if missing:
            raise RuntimeError(f"{variant} missing {len(missing)} trajectories")
        for source in manifest:
            key = str(source["prompt_key"])
            if rows[key]["input_ids"] != source["input_ids"]:
                raise RuntimeError(f"{variant} input_ids mismatch: {key}")
            if int(rows[key].get("seed", -1)) != 1234:
                raise RuntimeError(f"{variant} per-row seed mismatch: {key}")

        judge_partial = out_dir / f"clean_lcb_{variant.lower()}_judge.partial.jsonl"
        judge_done: dict[str, dict] = {}
        if judge_partial.exists():
            for row in read_jsonl(judge_partial):
                judge_done[str(row["prompt_key"])] = row
        pending = [s for s in manifest if str(s["prompt_key"]) not in judge_done]

        def _run(source: dict, *, _rows=rows, _variant=variant) -> dict:
            key = str(source["prompt_key"])
            traj = _rows[key]
            j = judge_completion(source, traj["raw_text"], traj["output_ids"], think_end_ids=think_end)
            return {
                "prompt_key": key,
                "doc_id": str(source["doc_id"]),
                "variant": _variant,
                "pass": bool(j["pass"]),
                "generation_len": int(j["generation_len"]),
                "finished_thinking": j.get("finished_thinking"),
                "failure_class": j.get("failure_class"),
                "seed": int(traj["seed"]),
            }

        workers = max(1, int(args.workers))
        if pending:
            with ThreadPoolExecutor(max_workers=workers) as pool, judge_partial.open("a", encoding="utf-8") as pf:
                futures = {pool.submit(_run, s): s for s in pending}
                finished = len(judge_done)
                for fut in as_completed(futures):
                    row = fut.result()
                    judge_done[str(row["prompt_key"])] = row
                    pf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    pf.flush()
                    finished += 1
                    if finished % 5 == 0 or finished == len(manifest):
                        print(
                            json.dumps(
                                {"judge_progress": finished, "total": len(manifest), "variant": variant},
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )

        judged = [judge_done[str(s["prompt_key"])] for s in manifest]
        write_jsonl(out_dir / f"clean_lcb_{variant.lower()}_judge.jsonl", judged)
        variant_pass[variant] = {str(r["prompt_key"]): bool(r["pass"]) for r in judged}
        variant_meta[variant] = {
            "label": VARIANT_LABEL[variant],
            "runtime": runtime,
            "pass_count": sum(bool(r["pass"]) for r in judged),
        }

    # Build multi-variant matrix vs E0 (and E1 if available).
    matrix_rows = []
    for source in manifest:
        key = str(source["prompt_key"])
        row = {
            "doc_id": str(source["doc_id"]),
            "prompt_key": key,
            "E0_pass": bool(e0_pass_by_key[key]),
            "E1_pass": e1_pass_by_key.get(key),
        }
        for variant in args.variants:
            row[f"{variant}_pass"] = bool(variant_pass[variant][key])
            row[f"E0_{variant}_cell"] = _pair_cell(row["E0_pass"], row[f"{variant}_pass"])
            if key in e1_pass_by_key:
                row[f"E1_{variant}_cell"] = _pair_cell(bool(e1_pass_by_key[key]), row[f"{variant}_pass"])
        matrix_rows.append(row)
    write_jsonl(out_dir / "clean_lcb_e2_e4_matrix.jsonl", matrix_rows)

    e0_vec = [bool(e0_pass_by_key[str(s["prompt_key"])]) for s in manifest]
    e1_vec = (
        [bool(e1_pass_by_key[str(s["prompt_key"])]) for s in manifest]
        if len(e1_pass_by_key) == len(manifest)
        else None
    )
    comparisons = {}
    for variant in args.variants:
        other = [bool(variant_pass[variant][str(s["prompt_key"])]) for s in manifest]
        comparisons[f"E0_{variant}"] = _summarize_pair("E0", variant, e0_vec, other)
        if e1_vec is not None:
            comparisons[f"E1_{variant}"] = _summarize_pair("E1", variant, e1_vec, other)

    summary = {
        "schema_version": 1,
        "n_tasks": len(manifest),
        "policy": {
            "e0": "reused_audited_phasea_formal" if reused_e0 else "fresh_benchmark",
            "prompt": "exact_E0_raw_cache_input_ids",
            "execution_shape": "max_num_seqs=128",
            "seed": 1234,
            "e3_e4_artifact": "adopted",
            "note": "supplementary clean LCB E2-E4 under same controls as E1; does not rewrite Gate C",
        },
        "pass_counts": {
            "E0": sum(e0_vec),
            **({ "E1": sum(e1_vec) } if e1_vec is not None else {}),
            **{v: variant_meta[v]["pass_count"] for v in args.variants},
        },
        "accuracy": {
            "E0": sum(e0_vec) / len(manifest),
            **({ "E1": sum(e1_vec) / len(manifest) } if e1_vec is not None else {}),
            **{v: variant_meta[v]["pass_count"] / len(manifest) for v in args.variants},
        },
        "variants": variant_meta,
        "comparisons": comparisons,
        "checker": official_checker_manifest(),
    }
    (out_dir / "CLEAN_LCB_E2_E4_REPORT.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# CLEAN_LCB_E2_E4_REPORT",
        "",
        "补测协议（与 clean E1 控变量一致）：exact E0 `input_ids` + controlled sampled + `max_num_seqs=128` + seed=1234 + 官方 LCB judge；E3/E4=adopted。",
        "不改写既有 E0/E1 Gate。",
        "",
        "| Variant | pass | acc | delta vs E0 (pp) | McNemar p vs E0 |",
        "|---|---:|---:|---:|---:|",
        f"| E0 | {summary['pass_counts']['E0']}/175 | {100*summary['accuracy']['E0']:.2f}% | — | — |",
    ]
    if e1_vec is not None:
        lines.append(
            f"| E1 | {summary['pass_counts']['E1']}/175 | {100*summary['accuracy']['E1']:.2f}% | "
            f"{(summary['accuracy']['E1']-summary['accuracy']['E0'])*100:.3f} | (existing) |"
        )
    for variant in args.variants:
        cmp0 = comparisons[f"E0_{variant}"]
        lines.append(
            f"| {variant} ({VARIANT_LABEL[variant]}) | {summary['pass_counts'][variant]}/175 | "
            f"{100*summary['accuracy'][variant]:.2f}% | {cmp0['delta_accuracy_pp']:.3f} | {cmp0['exact_mcnemar_p']:.6g} |"
        )
    lines.extend(["", "## Paired vs E0", ""])
    for variant in args.variants:
        c = comparisons[f"E0_{variant}"]
        pc = c["pair_counts"]
        boot = c["paired_bootstrap_95ci"]
        lines.extend(
            [
                f"### E0 vs {variant}",
                f"- n10/n01/both_pass/both_fail = {pc['n10']}/{pc['n01']}/{pc['both_pass']}/{pc['both_fail']}",
                f"- delta = {c['delta_accuracy_pp']:.3f} pp; 95% CI [{boot['ci95_low_pp']:.3f}, {boot['ci95_high_pp']:.3f}] pp",
                f"- exact McNemar p = {c['exact_mcnemar_p']:.6g}",
                "",
            ]
        )
    if e1_vec is not None:
        lines.extend(["## Paired vs E1", ""])
        for variant in args.variants:
            c = comparisons[f"E1_{variant}"]
            pc = c["pair_counts"]
            boot = c["paired_bootstrap_95ci"]
            lines.extend(
                [
                    f"### E1 vs {variant}",
                    f"- n10/n01/both_pass/both_fail = {pc['n10']}/{pc['n01']}/{pc['both_pass']}/{pc['both_fail']}",
                    f"- delta = {c['delta_accuracy_pp']:.3f} pp; 95% CI [{boot['ci95_low_pp']:.3f}, {boot['ci95_high_pp']:.3f}] pp",
                    f"- exact McNemar p = {c['exact_mcnemar_p']:.6g}",
                    "",
                ]
            )
    (out_dir / "CLEAN_LCB_E2_E4_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"pass_counts": summary["pass_counts"], "report": str(out_dir / "CLEAN_LCB_E2_E4_REPORT.md")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
