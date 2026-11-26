#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch


REPLICATED_BOUNDARIES = {
    "layer_in",
    "input_norm",
    "o_proj",
    "post_attn_norm",
    "router_logits",
    "moe_out",
    "layer_out",
    "final_norm",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_root", required=True)
    p.add_argument("--variant", default="E0")
    p.add_argument("--mode", default="forced_core")
    p.add_argument("--reference_root", default=None)
    p.add_argument("--require_e0_target_top1", action="store_true")
    p.add_argument("--require_legacy_focus", action="store_true")
    return p.parse_args()


def load_manifest(root: Path, variant: str, mode: str) -> dict:
    path = root / f"{variant}_{mode}_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def index_records(payload: dict) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    for row in payload["records"]:
        key = (
            int(row["decode_index"]),
            row["boundary"],
            row["role"],
            row["layer"],
        )
        if key in out:
            raise RuntimeError(f"duplicate hook record key: {key}")
        out[key] = row
    return out


def validate_hook_rank_pair(run_root: Path, variant: str, sample: dict) -> dict:
    sample_key = sample["sample_key"]
    base = run_root / "hooks" / variant / sample_key
    p0 = base / "rank0.pt"
    p1 = base / "rank1.pt"
    if not p0.is_file() or not p1.is_file():
        raise RuntimeError(f"missing TP hook files for {sample_key}: {p0}, {p1}")
    r0 = torch.load(p0, map_location="cpu", weights_only=False)
    r1 = torch.load(p1, map_location="cpu", weights_only=False)
    if int(r0["world_size"]) != 2 or int(r1["world_size"]) != 2:
        raise RuntimeError(f"invalid TP world size for {sample_key}")
    expected_probes = {int(x) for x in sample["probe_decode_indices"]}
    for rank_payload in (r0, r1):
        seen = {int(row["decode_index"]) for row in rank_payload["records"]}
        if seen != expected_probes:
            raise RuntimeError(
                f"probe coverage mismatch for {sample_key} rank={rank_payload['rank']}: "
                f"seen={sorted(seen)} expected={sorted(expected_probes)}"
            )
        by_probe: dict[int, set[str]] = defaultdict(set)
        for row in rank_payload["records"]:
            by_probe[int(row["decode_index"])].add(str(row["boundary"]))
        required = {
            "layer_in",
            "input_norm",
            "attention_core",
            "o_proj",
            "post_attn_norm",
            "router_logits",
            "moe_out",
            "layer_out",
            "final_norm",
        }
        for decode_index in expected_probes:
            missing = required - by_probe[decode_index]
            if missing:
                raise RuntimeError(
                    f"missing boundaries for {sample_key} j={decode_index} "
                    f"rank={rank_payload['rank']}: {sorted(missing)}"
                )

    i0 = index_records(r0)
    i1 = index_records(r1)
    shared = set(i0) & set(i1)
    mismatched_replicated = []
    checked_replicated = 0
    for key in sorted(shared, key=str):
        boundary = key[1]
        if boundary not in REPLICATED_BOUNDARIES:
            continue
        checked_replicated += 1
        if not torch.equal(i0[key]["tensor"], i1[key]["tensor"]):
            mismatched_replicated.append(key)
    if mismatched_replicated:
        raise RuntimeError(
            f"replicated TP tensors differ for {sample_key}; first={mismatched_replicated[:5]}"
        )
    return {
        "sample_key": sample_key,
        "rank0_records": len(r0["records"]),
        "rank1_records": len(r1["records"]),
        "checked_replicated": checked_replicated,
    }


def load_raw_logits(run_root: Path, variant: str, sample_key: str, decode_index: int) -> list[dict]:
    base = run_root / "raw_logits" / variant / sample_key
    files = sorted(base.glob(f"rank*_decode{decode_index}.pt"))
    if not files:
        raise RuntimeError(f"no raw logits captured for {variant}/{sample_key}/{decode_index}")
    return [torch.load(path, map_location="cpu", weights_only=False) for path in files]


def validate_raw_logits(
    run_root: Path, manifest: dict, *, require_e0_target_top1: bool = False
) -> list[dict]:
    rows = []
    for sample in manifest["samples"]:
        expected_ids = sample.get("expected_forced_ids")
        if expected_ids is None:
            continue
        key = sample["sample_key"]
        for decode_index in sample["probe_decode_indices"]:
            captures = load_raw_logits(run_root, manifest["variant"], key, int(decode_index))
            top1s = []
            expected = int(expected_ids[int(decode_index)])
            for capture in captures:
                logits = capture["logits"]
                if logits.ndim != 1:
                    raise RuntimeError(f"captured logits are not 1D: {capture['shape']}")
                top1s.append(int(logits.argmax().item()))
            if len(set(top1s)) != 1:
                raise RuntimeError(
                    f"raw logits TP captures disagree for {key}/{decode_index}: {top1s}"
                )
            top1 = top1s[0]
            if require_e0_target_top1 and manifest["variant"] == "E0" and top1 != expected:
                raise RuntimeError(
                    f"E0 raw logits mismatch at {key}/{decode_index}: top1={top1} expected={expected}"
                )
            rows.append(
                {
                    "sample_key": key,
                    "decode_index": int(decode_index),
                    "top1": top1,
                    "expected": expected,
                    "ranks_captured": [int(x["tp_rank"]) for x in captures],
                }
            )
    return rows


def compare_reference_logits(run_root: Path, reference_root: Path, manifest: dict) -> dict:
    reference_manifest = load_manifest(reference_root, manifest["variant"], "forced_logits_only")
    ref_samples = {row["sample_key"]: row for row in reference_manifest["samples"]}
    comparisons = 0
    for sample in manifest["samples"]:
        key = sample["sample_key"]
        if key not in ref_samples:
            raise RuntimeError(f"reference root missing sample {key}")
        for decode_index in sample["probe_decode_indices"]:
            current = load_raw_logits(run_root, manifest["variant"], key, int(decode_index))
            reference = load_raw_logits(reference_root, manifest["variant"], key, int(decode_index))
            cur_by_rank = {int(x["tp_rank"]): x["logits"] for x in current}
            ref_by_rank = {int(x["tp_rank"]): x["logits"] for x in reference}
            common = sorted(set(cur_by_rank) & set(ref_by_rank))
            if not common:
                raise RuntimeError(f"no common raw-logit rank for {key}/{decode_index}")
            for rank in common:
                if not torch.equal(cur_by_rank[rank], ref_by_rank[rank]):
                    diff = (cur_by_rank[rank].float() - ref_by_rank[rank].float()).abs().max().item()
                    raise RuntimeError(
                        f"hook changes raw logits at {key}/{decode_index}/rank{rank}: max_abs={diff}"
                    )
                comparisons += 1
    return {"exact_logit_comparisons": comparisons}


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    manifest = load_manifest(run_root, args.variant, args.mode)
    forced_samples = [row for row in manifest["samples"] if row.get("forced_exact") is not None]
    for sample in forced_samples:
        if sample["forced_exact"] is not True:
            raise RuntimeError(f"forced trajectory not exact: {sample['sample_key']}")
    hook_rows = []
    if args.mode.endswith("core"):
        for sample in manifest["samples"]:
            hook_rows.append(validate_hook_rank_pair(run_root, args.variant, sample))
    raw_rows = (
        validate_raw_logits(
            run_root,
            manifest,
            require_e0_target_top1=bool(args.require_e0_target_top1),
        )
        if args.mode.startswith("forced_")
        else []
    )
    reference = None
    if args.reference_root is not None:
        reference = compare_reference_logits(run_root, Path(args.reference_root).resolve(), manifest)

    focus = {(row["sample_key"], row["decode_index"]): row for row in raw_rows}
    required_focus = {
        ("n159_c94220492", 10): 279,
        ("n346_c216392826", 12): 3019,
        ("n346_c216392826", 24): 24301,
        ("n346_c216392826", 35): 3405,
    }
    if args.require_legacy_focus:
        for key, expected in required_focus.items():
            if key in focus and int(focus[key]["top1"]) != expected:
                raise RuntimeError(
                    f"legacy focus E0 top1 mismatch {key}: {focus[key]['top1']} != {expected}"
                )

    payload = {
        "status": "PASS",
        "variant": args.variant,
        "mode": args.mode,
        "forced_exact_samples": len(forced_samples),
        "hook_samples": hook_rows,
        "raw_logit_probes": raw_rows,
        "reference_comparison": reference,
    }
    out = run_root / f"{args.variant}_{args.mode}_validation.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
