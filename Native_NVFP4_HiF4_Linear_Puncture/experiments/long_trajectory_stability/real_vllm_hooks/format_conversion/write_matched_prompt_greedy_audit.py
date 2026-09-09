#!/usr/bin/env python3
"""Write MATCHED_PROMPT_GREEDY_AUDIT after official E0/E1 greedy judge (replan Task1)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import read_jsonl


def _mechanism_bucket(row: dict) -> str:
    if row.get("mechanism_group"):
        return str(row["mechanism_group"])
    if row["E0_pass"] and not row["E1_pass"]:
        return "mechanism_regression"
    if row["E0_pass"] and row["E1_pass"]:
        return "mechanism_robust"
    return "mechanism_e0_fail"


def _greedy_cell(row: dict) -> str:
    if row["E0_pass"] and not row["E1_pass"]:
        return "greedy_e0_pass_e1_fail"
    if row["E0_pass"] and row["E1_pass"]:
        return "greedy_both_pass"
    return "greedy_e0_fail"


def build_audit(run_root: Path) -> tuple[dict, str]:
    matrix = read_jsonl(run_root / "04_greedy_judge/greedy_task_matrix.jsonl")
    formal = {
        str(row["doc_id"]): row
        for row in read_jsonl(run_root / "01_formal_matrix/formal_task_matrix.jsonl")
    }
    cohort = {
        str(row["doc_id"]): row
        for row in read_jsonl(run_root / "02_cohort/benchmark_cohort.jsonl")
    }
    if len(matrix) != len(cohort):
        raise RuntimeError(f"judge matrix n={len(matrix)} != cohort n={len(cohort)}")

    group_counts = Counter(_mechanism_bucket(row) for row in matrix)
    transfer = {
        "formal_regression": Counter(),
        "formal_robust": Counter(),
        "other_formal": Counter(),
    }
    rows_out = []
    for row in matrix:
        doc_id = str(row["doc_id"])
        formal_row = formal.get(doc_id)
        formal_group = formal_row["formal_group"] if formal_row else "missing_formal"
        cell = _greedy_cell(row)
        bucket = "other_formal"
        if formal_group == "formal_regression":
            bucket = "formal_regression"
        elif formal_group == "formal_robust":
            bucket = "formal_robust"
        transfer[bucket][cell] += 1
        rows_out.append(
            {
                "doc_id": doc_id,
                "prompt_key": row["prompt_key"],
                "formal_group": formal_group,
                "cohort_formal_group": cohort[doc_id].get("formal_group"),
                "mechanism_group": _mechanism_bucket(row),
                "greedy_cell": cell,
                "E0_pass": row["E0_pass"],
                "E1_pass": row["E1_pass"],
                "E0_generation_len": row.get("E0_generation_len"),
                "E1_generation_len": row.get("E1_generation_len"),
                "E0_finished_thinking": row.get("E0_finished_thinking"),
                "E1_finished_thinking": row.get("E1_finished_thinking"),
                "E0_failure_class": row.get("E0_failure_class"),
                "E1_failure_class": row.get("E1_failure_class"),
            }
        )

    payload = {
        "schema_version": 1,
        "status": "PASS",
        "n_tasks": len(matrix),
        "protocol": "matched_prompt_isolated_greedy_official_LCB_judge",
        "warning": (
            "55-task interest cohort under old formal prompt mismatch; "
            "do not extrapolate full LCB accuracy"
        ),
        "group_counts": dict(group_counts),
        "old_formal_to_matched_greedy_transfer": {
            key: dict(counter) for key, counter in transfer.items()
        },
        "rows": rows_out,
        "stop_gate": "matched_prompt_greedy_audit_complete; next=clean_controlled_sampled_E0_E1_175",
    }

    md_lines = [
        "# MATCHED_PROMPT_GREEDY_AUDIT",
        "",
        "协议：同一 E0 chat-wrapped `input_ids` 上的 isolated greedy + 官方 LCB judge。",
        "",
        f"- n_tasks = **{payload['n_tasks']}**（有偏 interest cohort，不可外推全 175）",
        f"- mechanism_regression = **{group_counts.get('mechanism_regression', 0)}**",
        f"- mechanism_robust = **{group_counts.get('mechanism_robust', 0)}**",
        f"- mechanism_e0_fail = **{group_counts.get('mechanism_e0_fail', 0)}**",
        "",
        "## old formal → matched greedy 转移矩阵",
        "",
        "| old formal label | greedy E0 Pass/E1 Fail | greedy both Pass | greedy E0 Fail |",
        "|---|---:|---:|---:|",
    ]
    for label in ("formal_regression", "formal_robust"):
        c = transfer[label]
        md_lines.append(
            f"| old {label.replace('formal_', '')} | "
            f"{c.get('greedy_e0_pass_e1_fail', 0)} | "
            f"{c.get('greedy_both_pass', 0)} | "
            f"{c.get('greedy_e0_fail', 0)} |"
        )
    md_lines.extend(
        [
            "",
            "## Stop Gate",
            "",
            "本文件写完后，旧 heavy mechanism chain（feature/core/puncture/state/E2/E3）不得继续。",
            "下一主任务：clean controlled sampled LCB E0/E1 全 175。",
            "",
        ]
    )
    return payload, "\n".join(md_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    args = parser.parse_args()
    out_dir = args.run_root / "04_greedy_judge"
    payload, markdown = build_audit(args.run_root)
    (out_dir / "MATCHED_PROMPT_GREEDY_AUDIT.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    (out_dir / "MATCHED_PROMPT_GREEDY_AUDIT.md").write_text(markdown)
    print(json.dumps(payload["group_counts"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
