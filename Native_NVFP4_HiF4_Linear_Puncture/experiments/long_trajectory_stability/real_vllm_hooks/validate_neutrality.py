#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.validate_capture import (
    load_manifest,
    load_raw_logits,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--greedy_no_hook_root", required=True)
    p.add_argument("--greedy_core_root", required=True)
    p.add_argument("--forced_logits_root", required=True)
    p.add_argument("--forced_core_root", required=True)
    p.add_argument("--variant", default="E0")
    p.add_argument("--output", required=True)
    return p.parse_args()


def sample_map(manifest: dict) -> dict[str, dict]:
    return {str(row["sample_key"]): row for row in manifest["samples"]}


def compare_greedy(no_hook: dict, core: dict) -> list[dict]:
    a = sample_map(no_hook)
    b = sample_map(core)
    if set(a) != set(b):
        raise RuntimeError("greedy hook-on/off sample sets differ")
    rows = []
    for key in sorted(a):
        ids_a = [int(x) for x in a[key]["generated_ids"]]
        ids_b = [int(x) for x in b[key]["generated_ids"]]
        if ids_a != ids_b:
            first = next(
                (i for i, (x, y) in enumerate(zip(ids_a, ids_b)) if x != y),
                min(len(ids_a), len(ids_b)),
            )
            raise RuntimeError(
                f"core hook changes greedy E0 trajectory for {key}: first_divergence={first}"
            )
        rows.append({"sample_key": key, "num_tokens": len(ids_a), "first_divergence": None})
    return rows


def compare_forced_logits(
    logits_root: Path, core_root: Path, logits_manifest: dict, core_manifest: dict
) -> list[dict]:
    a = sample_map(logits_manifest)
    b = sample_map(core_manifest)
    if set(a) != set(b):
        raise RuntimeError("forced minimal/core sample sets differ")
    rows = []
    for key in sorted(a):
        if a[key]["generated_ids"] != b[key]["generated_ids"]:
            raise RuntimeError(f"forced output differs between minimal/core for {key}")
        probes = [int(x) for x in a[key]["probe_decode_indices"]]
        if probes != [int(x) for x in b[key]["probe_decode_indices"]]:
            raise RuntimeError(f"forced probe lists differ for {key}")
        for decode_index in probes:
            ca = load_raw_logits(logits_root, logits_manifest["variant"], key, decode_index)
            cb = load_raw_logits(core_root, core_manifest["variant"], key, decode_index)
            by_rank_a = {int(row["tp_rank"]): row["logits"] for row in ca}
            by_rank_b = {int(row["tp_rank"]): row["logits"] for row in cb}
            common = sorted(set(by_rank_a) & set(by_rank_b))
            if not common:
                raise RuntimeError(f"no common logits rank for {key}/{decode_index}")
            for rank in common:
                ta = by_rank_a[rank]
                tb = by_rank_b[rank]
                if not torch.equal(ta, tb):
                    max_abs = float((ta.float() - tb.float()).abs().max().item())
                    raise RuntimeError(
                        f"core hook changes raw logits for {key}/{decode_index}/rank{rank}: "
                        f"max_abs={max_abs}"
                    )
                rows.append(
                    {
                        "sample_key": key,
                        "decode_index": decode_index,
                        "rank": rank,
                        "exact": True,
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    gn_root = Path(args.greedy_no_hook_root).resolve()
    gc_root = Path(args.greedy_core_root).resolve()
    fl_root = Path(args.forced_logits_root).resolve()
    fc_root = Path(args.forced_core_root).resolve()
    gn = load_manifest(gn_root, args.variant, "greedy_none")
    gc = load_manifest(gc_root, args.variant, "greedy_core")
    fl = load_manifest(fl_root, args.variant, "forced_logits_only")
    fc = load_manifest(fc_root, args.variant, "forced_core")

    payload = {
        "status": "PASS",
        "variant": args.variant,
        "greedy_hook_neutrality": compare_greedy(gn, gc),
        "forced_raw_logit_neutrality": compare_forced_logits(fl_root, fc_root, fl, fc),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
