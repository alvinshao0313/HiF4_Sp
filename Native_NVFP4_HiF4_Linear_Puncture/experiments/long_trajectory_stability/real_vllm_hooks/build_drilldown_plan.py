#!/usr/bin/env python3
"""Build MoE/Attention frontier drill-down plan from E0↔variant core compare."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import (
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--compare_jsonl", required=True)
    p.add_argument("--divergence_events", required=True)
    p.add_argument("--probe_plan", required=True)
    p.add_argument("--variant", default="E1")
    p.add_argument("--output", required=True)
    p.add_argument("--n_earliest", type=int, default=4)
    p.add_argument("--n_stable", type=int, default=4)
    p.add_argument("--max_tokens_per_sample", type=int, default=2)
    p.add_argument("--max_layers_per_token", type=int, default=3)
    p.add_argument("--tp_rank", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    plan = json.loads(Path(args.probe_plan).read_text(encoding="utf-8"))
    sample_keys = [str(s["prompt_key"]) for s in plan["samples"]]
    events = {
        str(e["prompt_key"]): e
        for e in read_jsonl(Path(args.divergence_events))
        if str(e.get("variant")) == args.variant
    }
    ranked = []
    for key in sample_keys:
        ev = events.get(key)
        if ev is None or ev.get("first_divergence") is None:
            raise RuntimeError(f"missing first_divergence for {key}/{args.variant}")
        ranked.append((key, int(ev["first_divergence"])))
    earliest = sorted(ranked, key=lambda x: x[1])[: args.n_earliest]
    stable = sorted(ranked, key=lambda x: -x[1])[: args.n_stable]
    focus = {k for k, _ in earliest + stable}

    div_pos: dict[str, list[int]] = {}
    for sample in plan["samples"]:
        key = str(sample["prompt_key"])
        idxs = []
        for pos in sample["positions"]:
            if any(str(r).startswith(f"divergence:{args.variant}") for r in pos.get("reasons", [])):
                idxs.append(int(pos["decode_index"]))
        div_pos[key] = sorted(set(idxs))

    # sample -> decode -> boundary/layer metrics
    layer_out: dict[str, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    attn: dict[str, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    moe: dict[str, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    logits: dict[tuple[str, int], dict] = {}

    with Path(args.compare_jsonl).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sk = str(row["sample_key"])
            if sk not in focus:
                continue
            if row.get("boundary") != "raw_logits":
                if row.get("tp_rank") is not None and int(row["tp_rank"]) != args.tp_rank:
                    continue
            di = int(row["decode_index"])
            b = row["boundary"]
            if b == "raw_logits":
                logits[(sk, di)] = row
            elif b == "layer_out" and row.get("role") == "branch":
                layer_out[sk][di][int(row["layer"])] = float(row["rel_l2"])
            elif b == "attention_core":
                attn[sk][di][int(row["layer"])] = float(row["rel_l2"])
            elif b == "moe_out":
                moe[sk][di][int(row["layer"])] = float(row["rel_l2"])

    frontiers = []
    branch_votes: Counter[str] = Counter()
    for group, items in (("earliest_divergence", earliest), ("longest_stable", stable)):
        for sk, fd in items:
            cands = sorted(div_pos.get(sk, []), key=lambda x: abs(x - fd))[: args.max_tokens_per_sample]
            if not cands:
                raise RuntimeError(f"no divergence probes for {sk}")
            for di in cands:
                lo = layer_out[sk][di]
                jumps = []
                for layer in range(48):
                    if layer not in lo:
                        continue
                    prev = lo.get(layer - 1, 0.0)
                    jumps.append(
                        {
                            "layer": layer,
                            "delta_rel_l2": float(lo[layer] - prev),
                            "rel_l2": float(lo[layer]),
                            "attention_core_rel_l2": attn[sk][di].get(layer),
                            "moe_out_rel_l2": moe[sk][di].get(layer),
                        }
                    )
                jumps.sort(key=lambda x: -x["delta_rel_l2"])
                top = jumps[: args.max_layers_per_token]
                moe_gt = 0
                attn_gt = 0
                for j in top:
                    a = j["attention_core_rel_l2"]
                    m = j["moe_out_rel_l2"]
                    if a is None or m is None:
                        continue
                    if m >= a:
                        moe_gt += 1
                    else:
                        attn_gt += 1
                preferred = "moe" if moe_gt >= attn_gt else "attention"
                branch_votes[preferred] += 1
                lg = logits.get((sk, di), {})
                frontiers.append(
                    {
                        "group": group,
                        "sample_key": sk,
                        "first_divergence": fd,
                        "decode_index": di,
                        "candidate_layers": [j["layer"] for j in top],
                        "layer_diagnostics": top,
                        "preferred_branch": preferred,
                        "e0_margin": lg.get("e0_margin"),
                        "target_rank_variant": lg.get("target_rank_variant"),
                        "top1_agree": lg.get("top1_agree"),
                        "logit_kl_e0_to_variant": lg.get("logit_kl_e0_to_variant"),
                    }
                )

    recommended = "moe" if branch_votes["moe"] >= branch_votes["attention"] else "attention"
    payload = {
        "schema_version": 1,
        "variant": args.variant,
        "selection": {
            "earliest_divergence": [
                {"sample_key": k, "first_divergence": fd} for k, fd in earliest
            ],
            "longest_stable": [{"sample_key": k, "first_divergence": fd} for k, fd in stable],
        },
        "max_tokens_per_sample": args.max_tokens_per_sample,
        "max_layers_per_token": args.max_layers_per_token,
        "branch_votes": dict(branch_votes),
        "recommended_drilldown": recommended,
        "frontiers": frontiers,
        "notes": (
            "Selected from core-boundary compare only. "
            "Deep Q/K/V or per-expert capture must use this plan and must not expand to full model."
        ),
    }
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out)
    print(
        json.dumps(
            {
                "recommended_drilldown": recommended,
                "n_frontiers": len(frontiers),
                "branch_votes": dict(branch_votes),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
