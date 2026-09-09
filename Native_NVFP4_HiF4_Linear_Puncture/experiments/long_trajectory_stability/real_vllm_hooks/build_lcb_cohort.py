#!/usr/bin/env python3
"""Build LiveCodeBench prompt cohort for Task 15 (E0-pass / E1-fail)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.config import (
    DEFAULT_MODEL_PATH,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import (
    prompt_key,
    write_jsonl,
)

DEFAULT_PHASEA = (
    REPO_ROOT
    / "Native_NVFP4_HiF4_Linear_Puncture/results/e2e_diag_reconstruction/"
    "phaseA_refactor_20260825T035730Z"
)

# Documented + diverse Core-A flips; 58 is the only completed WA.
DEFAULT_DOC_IDS = [
    "15",
    "36",
    "46",
    "77",
    "58",
    "65",
    "59",
    "30",
    "70",
    "50",
    "39",
    "26",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phasea_root", default=str(DEFAULT_PHASEA))
    p.add_argument("--classification", default=None)
    p.add_argument("--e0_details", default=None)
    p.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--doc_ids", nargs="*", default=DEFAULT_DOC_IDS)
    p.add_argument("--output", required=True)
    p.add_argument("--manifest", required=True)
    return p.parse_args()


def find_e0_details(phasea: Path) -> Path:
    matches = sorted(
        phasea.glob(
            "E0_native_nvfp4/eval/livecodebench/**/details_lcb:codegeneration_v6|0_*.json"
        )
    )
    if not matches:
        raise RuntimeError(f"no E0 LCB details under {phasea}")
    return matches[-1]


def main() -> None:
    args = parse_args()
    phasea = Path(args.phasea_root).resolve()
    classification_path = (
        Path(args.classification).resolve()
        if args.classification
        else phasea / "lcb_e0_pass_e1_fail_classification.json"
    )
    details_path = (
        Path(args.e0_details).resolve() if args.e0_details else find_e0_details(phasea)
    )
    cls = json.loads(classification_path.read_text(encoding="utf-8"))
    flip_ids = {str(r["id"]) for r in cls["records"]}
    wanted = [str(x) for x in args.doc_ids]
    missing_flip = [x for x in wanted if x not in flip_ids]
    if missing_flip:
        raise RuntimeError(f"doc ids not in E0-pass/E1-fail set: {missing_flip}")

    details = json.loads(details_path.read_text(encoding="utf-8"))
    by_id = {str(item["doc"]["id"]): item for item in details}
    missing_details = [x for x in wanted if x not in by_id]
    if missing_details:
        raise RuntimeError(f"doc ids missing from E0 details: {missing_details}")

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    flip_meta = {str(r["id"]): r for r in cls["records"]}
    rows = []
    for doc_id in wanted:
        item = by_id[doc_id]
        passed = float(item["metric"]["codegen_pass@1:16"]) == 1.0
        if not passed:
            raise RuntimeError(f"E0 details for doc {doc_id} is not pass")
        prompt_text = item["model_response"]["input"]
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise RuntimeError(f"empty prompt text for doc {doc_id}")
        input_ids = [int(x) for x in tok.encode(prompt_text, add_special_tokens=False)]
        if not input_ids:
            raise RuntimeError(f"empty input_ids for doc {doc_id}")
        key = prompt_key(input_ids)
        rows.append(
            {
                "prompt_key": key,
                "doc_id": doc_id,
                "input_ids": input_ids,
                "output_ids": [],
                "output_len": 0,
                "raw_text": "",
                "metric": item.get("metric"),
                "gold": item.get("gold"),
                "specific": {
                    "task": "livecodebench",
                    "phasea_e1_fail_meta": flip_meta[doc_id],
                    "prompt_source": "e0_details_model_response.input_tokenized",
                    "formal_lcb_pass_label_protocol": "sampled_thinking_temp0.6",
                    "mechanism_protocol": "greedy_isolated_free_run",
                },
            }
        )

    out = Path(args.output).resolve()
    man = Path(args.manifest).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, rows)
    payload = {
        "schema_version": 1,
        "task": "livecodebench",
        "n_samples": len(rows),
        "doc_ids": wanted,
        "classification": str(classification_path),
        "e0_details": str(details_path),
        "model_path": args.model_path,
        "samples": [
            {
                "doc_id": r["doc_id"],
                "prompt_key": r["prompt_key"],
                "prompt_len": len(r["input_ids"]),
                "e1_fail_cls": r["specific"]["phasea_e1_fail_meta"].get("cls"),
                "e1_n_tokens_formal": r["specific"]["phasea_e1_fail_meta"].get("n_tokens"),
            }
            for r in rows
        ],
    }
    man.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out)
    print(man)
    print(json.dumps({"n": len(rows), "doc_ids": wanted}, indent=2))


if __name__ == "__main__":
    main()
