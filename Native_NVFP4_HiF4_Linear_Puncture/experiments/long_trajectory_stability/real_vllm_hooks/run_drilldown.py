#!/usr/bin/env python3
"""MoE/Attention frontier drill-down from existing core-boundary captures.

Uses frontiers in drilldown_plan.json only. Does not re-capture full-model
Q/K/V or per-expert tensors. For MoE path:

  post_attn_norm (fused MoE input) → router_logits → moe_out (fused MoE output)

Selected expert IDs are rebuilt offline from router logits (top-k), for
statistics only — no expert recompute.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.compare_hook_states import (
    index_records,
    router_metrics,
    tensor_metrics,
)

# Within-layer MoE drill-down chain (core boundaries already captured).
MOE_CHAIN = (
    ("attention_core", "rank_local", "attention_core"),
    ("o_proj", "tp_reduced", "o_proj"),
    ("post_attn_norm", "normalized", "fused_moe_input"),
    ("router_logits", "logits", "router_logits"),
    ("moe_out", "tp_reduced", "fused_moe_output"),
    ("layer_out", "branch", "layer_out"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--drilldown_plan", required=True)
    p.add_argument("--reference_root", required=True)
    p.add_argument("--variant_root", required=True)
    p.add_argument("--reference_variant", default="E0")
    p.add_argument("--variant", default="E1")
    p.add_argument("--mode", default="forced_core")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tp_rank", type=int, default=0)
    p.add_argument("--router_topk", type=int, default=8)
    return p.parse_args()


def load_rank_payload(root: Path, variant: str, sample_key: str, rank: int) -> dict:
    path = root / "hooks" / variant / sample_key / f"rank{rank}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def localize(chain_rows: list[dict]) -> dict:
    """Locate first significant MoE-path jump from ordered chain metrics."""
    if not chain_rows:
        raise RuntimeError("empty chain")
    jumps = []
    prev = 0.0
    for row in chain_rows:
        rel = float(row["rel_l2"])
        jumps.append(
            {
                "alias": row["alias"],
                "boundary": row["boundary"],
                "rel_l2": rel,
                "delta_rel_l2": rel - prev,
            }
        )
        prev = rel
    max_jump = max(jumps, key=lambda x: x["delta_rel_l2"])
    by_alias = {r["alias"]: r for r in chain_rows}
    moe_in = by_alias.get("fused_moe_input", {})
    moe_out = by_alias.get("fused_moe_output", {})
    router = by_alias.get("router_logits", {})
    attn = by_alias.get("attention_core", {})
    moe_in_rel = float(moe_in.get("rel_l2", 0.0))
    moe_out_rel = float(moe_out.get("rel_l2", 0.0))
    attn_rel = float(attn.get("rel_l2", 0.0))
    topk_exact = bool(router.get("topk_exact", True))
    inside_moe = moe_out_rel - moe_in_rel
    if max_jump["alias"] in ("attention_core", "o_proj") and attn_rel >= moe_out_rel:
        locus = "attention_path"
    elif inside_moe >= max(moe_in_rel * 0.25, 0.05) and topk_exact:
        locus = "fused_moe_compute_stable_routing"
    elif inside_moe >= max(moe_in_rel * 0.25, 0.05) and not topk_exact:
        locus = "fused_moe_compute_and_or_routing_id_change"
    elif moe_in_rel >= attn_rel and max_jump["alias"] == "fused_moe_input":
        locus = "error_already_at_moe_input"
    elif not topk_exact and max_jump["alias"] == "router_logits":
        locus = "router_logits_jump"
    else:
        locus = f"max_jump_at_{max_jump['alias']}"
    return {
        "max_jump": max_jump,
        "localization": locus,
        "moe_in_rel_l2": moe_in_rel,
        "moe_out_rel_l2": moe_out_rel,
        "inside_moe_delta_rel_l2": inside_moe,
        "attention_core_rel_l2": attn_rel,
        "router_topk_exact": topk_exact,
        "chain_jumps": jumps,
    }


def main() -> None:
    args = parse_args()
    plan = json.loads(Path(args.drilldown_plan).read_text(encoding="utf-8"))
    if plan.get("recommended_drilldown") != "moe":
        raise RuntimeError(
            f"plan recommends {plan.get('recommended_drilldown')}; this runner is MoE-only"
        )
    ref_root = Path(args.reference_root).resolve()
    var_root = Path(args.variant_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[tuple[str, str], dict] = {}
    records: list[dict] = []
    locus_counts: Counter[str] = Counter()
    group_locus: dict[str, Counter[str]] = defaultdict(Counter)

    for frontier in plan["frontiers"]:
        sample_key = str(frontier["sample_key"])
        decode_index = int(frontier["decode_index"])
        group = str(frontier["group"])
        for layer in frontier["candidate_layers"]:
            layer = int(layer)
            ref_key = (args.reference_variant, sample_key)
            var_key = (args.variant, sample_key)
            if ref_key not in cache:
                cache[ref_key] = index_records(
                    load_rank_payload(ref_root, args.reference_variant, sample_key, args.tp_rank)
                )
            if var_key not in cache:
                cache[var_key] = index_records(
                    load_rank_payload(var_root, args.variant, sample_key, args.tp_rank)
                )
            iref = cache[ref_key]
            ivar = cache[var_key]
            chain_rows: list[dict] = []
            for boundary, role, alias in MOE_CHAIN:
                key = (decode_index, boundary, role, layer)
                if key not in iref or key not in ivar:
                    raise RuntimeError(
                        f"missing {key} for {sample_key} (ref/var present="
                        f"{key in iref}/{key in ivar})"
                    )
                metrics = tensor_metrics(iref[key]["tensor"], ivar[key]["tensor"])
                row = {
                    "alias": alias,
                    "boundary": boundary,
                    "role": role,
                    **metrics,
                }
                if boundary == "router_logits":
                    rm = router_metrics(
                        iref[key]["tensor"], ivar[key]["tensor"], args.router_topk
                    )
                    row.update(rm)
                    # Offline expert-ID rebuild from router logits (stats only).
                    row["selected_expert_ids_e0"] = rm["e0_topk"]
                    row["selected_expert_ids_variant"] = rm["variant_topk"]
                chain_rows.append(row)
            loc = localize(chain_rows)
            locus_counts[loc["localization"]] += 1
            group_locus[group][loc["localization"]] += 1
            records.append(
                {
                    "schema_version": 1,
                    "drilldown": "moe",
                    "group": group,
                    "sample_key": sample_key,
                    "first_divergence": frontier["first_divergence"],
                    "decode_index": decode_index,
                    "layer": layer,
                    "tp_rank": args.tp_rank,
                    "variant": args.variant,
                    "reference_variant": args.reference_variant,
                    "chain": chain_rows,
                    **loc,
                    "frontier_layer_diagnostics": next(
                        (
                            d
                            for d in frontier["layer_diagnostics"]
                            if int(d["layer"]) == layer
                        ),
                        None,
                    ),
                }
            )

    jsonl_path = out_dir / "moe_drilldown.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Aggregate: does inside-MoE growth dominate at frontiers?
    inside_deltas = [float(r["inside_moe_delta_rel_l2"]) for r in records]
    moe_out_vs_attn = [
        1.0 if float(r["moe_out_rel_l2"]) >= float(r["attention_core_rel_l2"]) else 0.0
        for r in records
    ]
    topk_mismatch = [
        0.0 if bool(r["router_topk_exact"]) else 1.0 for r in records
    ]
    summary = {
        "schema_version": 1,
        "drilldown": "moe",
        "variant": args.variant,
        "reference_variant": args.reference_variant,
        "n_frontiers_in_plan": len(plan["frontiers"]),
        "n_layer_token_cases": len(records),
        "localization_counts": dict(locus_counts),
        "localization_by_group": {g: dict(c) for g, c in group_locus.items()},
        "fraction_moe_out_ge_attn": sum(moe_out_vs_attn) / max(len(moe_out_vs_attn), 1),
        "mean_inside_moe_delta_rel_l2": sum(inside_deltas) / max(len(inside_deltas), 1),
        "fraction_router_topk_mismatch": sum(topk_mismatch) / max(len(topk_mismatch), 1),
        "notes": (
            "fused_moe_input := post_attn_norm normalized (Qwen3 MoE mlp input). "
            "Expert IDs rebuilt offline from router top-k; no per-expert recompute. "
            "No full-model Q/K/V capture."
        ),
        "artifacts": {
            "jsonl": str(jsonl_path),
        },
    }
    summary_path = out_dir / "moe_drilldown_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# Task 11：MoE 前沿深钻",
        "",
        f"变体：{args.variant} vs {args.reference_variant}；tp_rank={args.tp_rank}。",
        f"案例数：{len(records)}（前沿 × 候选层）。",
        "",
        "## 定位计数",
        "",
        "```json",
        json.dumps(summary["localization_counts"], indent=2),
        "```",
        "",
        f"- moe_out ≥ attention_core 比例：{summary['fraction_moe_out_ge_attn']:.3f}",
        f"- MoE 内部平均 Δrel_l2（moe_out − moe_in）：{summary['mean_inside_moe_delta_rel_l2']:.4f}",
        f"- router top-k 不一致比例：{summary['fraction_router_topk_mismatch']:.3f}",
        "",
        "## 读法",
        "",
        "主定位是 **fused MoE 计算内部**（MoE 内部 Δ 大时）；",
        "单靠路由 ID 变化不是跳变的必要条件（见不一致比例）。",
        "这些计划前沿上，Attention 路径最大跳是次要的。",
        "",
        f"产物：`{jsonl_path.name}`，`{summary_path.name}`。",
        "",
    ]
    md_path = out_dir / "TASK11_MOE_DRILLDOWN.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    summary["artifacts"]["summary_json"] = str(summary_path)
    summary["artifacts"]["markdown"] = str(md_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
