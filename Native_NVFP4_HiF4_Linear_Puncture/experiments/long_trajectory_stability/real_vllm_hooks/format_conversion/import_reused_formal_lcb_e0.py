#!/usr/bin/env python3
"""Import audited Phase-A formal LCB E0 pass labels for clean E0/E1 pairing."""
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

DEFAULT_DETAILS = (
    Path(__file__).resolve().parents[4]
    / "results/e2e_diag_reconstruction/phaseA_refactor_20260825T035730Z"
    / "E0_native_nvfp4/eval/livecodebench/vllm_run"
    / "2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3/details"
    / "2026-08-31T18-56-08.973761"
    / "details_lcb:codegeneration_v6|0_2026-08-31T18-56-08.973761.json"
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt_manifest", type=Path, required=True)
    p.add_argument("--e0_details", type=Path, default=DEFAULT_DETAILS)
    p.add_argument("--output_dir", type=Path, required=True)
    args = p.parse_args()
    manifest = read_jsonl(args.prompt_manifest)
    details = json.loads(args.e0_details.read_text())
    by_inputs = {}
    for row in details:
        key = json.dumps(row["doc"]["specific"].get("inputs"), ensure_ascii=False)
        if key in by_inputs:
            raise RuntimeError("duplicate specific.inputs in E0 details")
        by_inputs[key] = row
    reused = []
    e0_rows = []
    for source in manifest:
        key = json.dumps(source["specific"].get("inputs"), ensure_ascii=False)
        if key not in by_inputs:
            raise RuntimeError(f"no Phase-A E0 detail for doc_id={source['doc_id']}")
        det = by_inputs[key]
        passed = float((det.get("metric") or {}).get("codegen_pass@1:16") or 0.0) >= 0.5
        texts = det.get("model_response", {}).get("text_post_processed") or []
        raw_text = texts[0] if texts else ""
        reused.append(
            {
                "doc_id": str(source["doc_id"]),
                "prompt_key": str(source["prompt_key"]),
                "E0_pass": passed,
                "metric": det.get("metric"),
                "details_doc_id": det["doc"].get("id"),
                "source_details": str(args.e0_details.resolve()),
            }
        )
        # Minimal trajectory-shaped row so judge can re-check if needed; pass is authoritative.
        e0_rows.append(
            {
                "prompt_key": str(source["prompt_key"]),
                "doc_id": str(source["doc_id"]),
                "input_ids": source["input_ids"],
                "output_ids": [],  # unused when --reused_e0_pass_table is provided
                "output_len": 0,
                "raw_text": raw_text,
                "seed": 1234,
                "variant": "E0",
                "reused_formal_phasea": True,
                "E0_pass_authoritative": passed,
            }
        )
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "reused_formal_lcb_e0_pass_table.jsonl", reused)
    write_jsonl(out / "clean_lcb_e0.jsonl", e0_rows)
    meta = {
        "schema_version": 1,
        "policy": "reuse_audited_phasea_formal_e0",
        "n": len(reused),
        "pass_count": sum(1 for r in reused if r["E0_pass"]),
        "details_path": str(args.e0_details.resolve()),
        "seed": 1234,
        "max_num_seqs": 128,
        "NOT_A_NEW_BENCHMARK_RUN": True,
    }
    (out / "reused_formal_lcb_e0_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
