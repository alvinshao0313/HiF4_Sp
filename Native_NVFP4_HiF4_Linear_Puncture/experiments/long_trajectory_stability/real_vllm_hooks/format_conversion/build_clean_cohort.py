#!/usr/bin/env python3
"""Build clean_regression/clean_robust cohort from clean paired LCB matrix (replan Task3)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import (
    read_jsonl,
    write_jsonl,
)


def match_robust(regressions: list[dict], robust: list[dict], prompt_by_doc: dict) -> list[tuple[dict, dict]]:
    available = {str(r["doc_id"]): r for r in robust}
    pairs = []
    for reg in regressions:
        if not available:
            break
        reg_id = str(reg["doc_id"])
        reg_plen = len(prompt_by_doc[reg_id]["input_ids"])
        reg_elen = int(reg["E0_generation_len"])
        choice = min(
            available,
            key=lambda doc_id: (
                abs(int(available[doc_id]["E0_generation_len"]) - reg_elen),
                abs(len(prompt_by_doc[doc_id]["input_ids"]) - reg_plen),
                int(doc_id),
            ),
        )
        pairs.append((reg, available.pop(choice)))
    return pairs


def build(run_root: Path, max_pairs: int) -> dict:
    pair = read_jsonl(run_root / "10_clean_lcb/clean_lcb_pair_matrix.jsonl")
    manifest = {
        str(r["doc_id"]): r
        for r in read_jsonl(run_root / "10_clean_lcb/clean_lcb_prompt_manifest.jsonl")
    }
    regressions = [r for r in pair if r["pair_cell"] == "n10"]
    robust = [r for r in pair if r["pair_cell"] == "both_pass"]
    regressions = sorted(regressions, key=lambda r: (int(r["E0_generation_len"]), int(r["doc_id"])))
    selected_reg = regressions[:max_pairs]
    pairs = match_robust(selected_reg, robust, manifest)
    cohort = []
    for pair_index, (reg, rob) in enumerate(pairs):
        reg_src = manifest[str(reg["doc_id"])]
        rob_src = manifest[str(rob["doc_id"])]
        for row, src, group, other_src in (
            (reg, reg_src, "clean_regression", rob_src),
            (rob, rob_src, "clean_robust", reg_src),
        ):
            cohort.append(
                {
                    **{k: src[k] for k in (
                        "doc_id", "prompt_key", "input_ids", "prompt_text_sha256",
                        "raw_input_ids_sha256", "prompt_source", "gold", "specific",
                    )},
                    "clean_group": group,
                    "mechanism_group": (
                        "mechanism_regression" if group == "clean_regression" else "mechanism_robust"
                    ),
                    "matched_doc_id": other_src["doc_id"],
                    "matched_regression_key": reg_src["prompt_key"],
                    "matched_prompt_key": other_src["prompt_key"],
                    "pair_index": pair_index,
                    "E0_generation_len": row["E0_generation_len"],
                    "E1_generation_len": row["E1_generation_len"],
                    "formal_group": "from_clean_pair_matrix",
                    "cohort_role": "clean_controlled_cohort",
                }
            )
    out_dir = run_root / "11_clean_cohort"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "clean_cohort.jsonl", cohort)
    # Alias expected by older puncture helpers.
    write_jsonl(out_dir / "mechanism_cohort.jsonl", cohort)
    write_jsonl(run_root / "02_cohort/mechanism_cohort.jsonl", cohort)
    meta = {
        "schema_version": 1,
        "n_samples": len(cohort),
        "n_pairs": len(pairs),
        "max_pairs_requested": max_pairs,
        "n_clean_regression_available": len(regressions),
        "n_clean_robust_available": len(robust),
        "doc_ids": [r["doc_id"] for r in cohort],
        "selection": "clean_n10_plus_matched_both_pass",
        "matching": ["E0_generation_len_distance", "prompt_token_length_distance", "doc_id"],
    }
    (out_dir / "clean_cohort_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--max_pairs", type=int, default=8)
    args = parser.parse_args()
    meta = build(args.run_root, args.max_pairs)
    print(json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
