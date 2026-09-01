#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.config import (
    DEFAULT_CAUSAL_REPLAY_SAMPLES,
    DEFAULT_DIVERGENCE_OFFSETS,
    DEFAULT_MAX_PROBE_DECODE_INDEX,
    DEFAULT_PROBES_PER_BIN,
    POSITION_BINS,
    decode_bin,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import read_jsonl


def evenly_spaced(lo: int, hi: int, count: int) -> list[int]:
    if hi <= lo or count <= 0:
        return []
    span = hi - lo
    n = min(count, span)
    return sorted({lo + min(span - 1, ((2 * i + 1) * span) // (2 * n)) for i in range(n)})


def select_samples(rows: list[dict], count: int) -> list[dict]:
    if count >= len(rows):
        return list(rows)
    ranked = sorted(rows, key=lambda x: (int(x["output_len"]), str(x["prompt_key"])))
    n_long = count // 2
    selected = ranked[-n_long:] if n_long else []
    remaining = ranked[:-n_long] if n_long else ranked
    needed = count - len(selected)
    if needed > 0 and remaining:
        indices = evenly_spaced(0, len(remaining), needed)
        selected.extend(remaining[i] for i in indices)
    return sorted(selected, key=lambda x: str(x["prompt_key"]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--e0_trajectories", required=True)
    p.add_argument("--divergence_events", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num_samples", type=int, default=DEFAULT_CAUSAL_REPLAY_SAMPLES)
    p.add_argument("--probes_per_bin", type=int, default=DEFAULT_PROBES_PER_BIN)
    p.add_argument("--max_decode_index", type=int, default=DEFAULT_MAX_PROBE_DECODE_INDEX)
    args = p.parse_args()

    e0_rows = read_jsonl(Path(args.e0_trajectories))
    selected = select_samples(e0_rows, args.num_samples)
    events = read_jsonl(Path(args.divergence_events))
    events_by_key: dict[str, list[dict]] = {}
    for event in events:
        events_by_key.setdefault(str(event["prompt_key"]), []).append(event)

    samples: list[dict] = []
    for row in selected:
        output_len = len(row["output_ids"])
        reasons: dict[int, list[str]] = {}
        cap = min(output_len, args.max_decode_index + 1)
        for bin_name, lo, hi in POSITION_BINS:
            upper = cap if hi is None else min(cap, hi)
            for index in evenly_spaced(lo, upper, args.probes_per_bin):
                reasons.setdefault(index, []).append(f"uniform:{bin_name}")
        for event in events_by_key.get(str(row["prompt_key"]), []):
            d = event.get("first_divergence")
            if d is None:
                continue
            for offset in DEFAULT_DIVERGENCE_OFFSETS:
                index = int(d) + int(offset)
                if 0 <= index < cap:
                    reasons.setdefault(index, []).append(
                        f"divergence:{event['variant']}:{offset:+d}"
                    )
        positions = [
            {
                "decode_index": index,
                "bin": decode_bin(index),
                "reasons": sorted(set(reasons[index])),
            }
            for index in sorted(reasons)
        ]
        if not positions:
            raise RuntimeError(f"no probe positions for {row['prompt_key']}")
        samples.append(
            {
                "prompt_key": row["prompt_key"],
                "doc_id": row.get("doc_id"),
                "input_ids": row["input_ids"],
                "output_ids": row["output_ids"],
                "output_len": output_len,
                "max_required_decode_index": positions[-1]["decode_index"],
                "positions": positions,
            }
        )

    payload = {
        "schema_version": 1,
        "selection_policy": "half-longest + half-length-quantile representatives",
        "num_samples": len(samples),
        "probes_per_bin": args.probes_per_bin,
        "max_decode_index": args.max_decode_index,
        "samples": samples,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
