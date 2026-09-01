#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import read_jsonl, write_jsonl

THRESHOLDS = (128, 512, 2048, 8192)


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if int(x) != int(y):
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def by_key(rows: list[dict]) -> dict[str, dict]:
    out = {str(row["prompt_key"]): row for row in rows}
    if len(out) != len(rows):
        raise ValueError("duplicate prompt_key in normalized trajectories")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--normalized_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--variants", default="E1,E2,E3,E4")
    args = p.parse_args()

    normalized_dir = Path(args.normalized_dir)
    output_dir = Path(args.output_dir)
    variants = [x.strip().upper() for x in args.variants.split(",") if x.strip()]
    e0_rows = read_jsonl(normalized_dir / "E0.jsonl")
    e0 = by_key(e0_rows)
    events: list[dict] = []
    summary: dict[str, dict] = {}

    for variant in variants:
        current = by_key(read_jsonl(normalized_dir / f"{variant}.jsonl"))
        if set(current) != set(e0):
            missing = sorted(set(e0) - set(current))[:5]
            extra = sorted(set(current) - set(e0))[:5]
            raise RuntimeError(f"{variant} prompt set differs from E0; missing={missing}, extra={extra}")
        for key, ref in e0.items():
            if [int(x) for x in current[key]["input_ids"]] != [int(x) for x in ref["input_ids"]]:
                raise RuntimeError(f"{variant} exact prompt token ids differ from E0 for {key}")
        divs: list[int] = []
        exact = 0
        for key, ref in e0.items():
            cur = current[key]
            a = [int(x) for x in ref["output_ids"]]
            b = [int(x) for x in cur["output_ids"]]
            d = first_divergence(a, b)
            if d is None:
                exact += 1
            else:
                divs.append(d)
            events.append(
                {
                    "variant": variant,
                    "prompt_key": key,
                    "doc_id": ref.get("doc_id"),
                    "e0_output_len": len(a),
                    "variant_output_len": len(b),
                    "first_divergence": d,
                    "identical_to_end": d is None,
                    "e0_token_at_divergence": a[d] if d is not None and d < len(a) else None,
                    "variant_token_at_divergence": b[d] if d is not None and d < len(b) else None,
                }
            )
        n = len(e0)
        survived = {
            str(t): sum(1 for key in e0 if (first_divergence(e0[key]["output_ids"], current[key]["output_ids"]) is None or first_divergence(e0[key]["output_ids"], current[key]["output_ids"]) >= t)) / n
            for t in THRESHOLDS
        }
        divs_sorted = sorted(divs)
        median = None if not divs_sorted else divs_sorted[len(divs_sorted) // 2]
        summary[variant] = {
            "num_samples": n,
            "exact_trajectory_fraction": exact / n,
            "diverged_fraction": 1.0 - exact / n,
            "median_first_divergence": median,
            "survival_fraction": survived,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "divergence_events.jsonl", events)
    (output_dir / "free_run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
