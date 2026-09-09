#!/usr/bin/env python3
"""Summarize real-vLLM E0↔variant hook metrics into H1/H2/H3 judgments."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.config import (
    POSITION_BINS,
    decode_bin,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import (
    read_jsonl,
)

BIN_ORDER = [name for name, _, _ in POSITION_BINS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--compare_jsonl", required=True)
    p.add_argument("--divergence_events", required=True)
    p.add_argument("--probe_plan", required=True)
    p.add_argument("--variant", default="E1")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tp_rank", type=int, default=0)
    return p.parse_args()


def mean(xs: list[float]) -> float | None:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return None if not vals else float(sum(vals) / len(vals))


def median(xs: list[float]) -> float | None:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return None if not vals else float(statistics.median(vals))


def load_compare(path: Path, tp_rank: int) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rank = row.get("tp_rank")
        if row.get("boundary") == "raw_logits":
            rows.append(row)
            continue
        if rank is None or int(rank) == int(tp_rank):
            rows.append(row)
    return rows


def probe_tags(plan: dict, variant: str) -> dict[tuple[str, int], set[str]]:
    out: dict[tuple[str, int], set[str]] = {}
    div_prefix = f"divergence:{variant}"
    for sample in plan["samples"]:
        key = str(sample["prompt_key"])
        for pos in sample["positions"]:
            idx = int(pos["decode_index"])
            tags = set()
            for reason in pos.get("reasons", []):
                reason = str(reason)
                if reason.startswith(div_prefix):
                    tags.add("divergence")
                if reason.startswith("uniform:"):
                    tags.add("uniform")
            out[(key, idx)] = tags
    return out


def bin_curves(rows: list[dict]) -> dict:
    by_bin: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    sample_late_minus_early: dict[str, list[float]] = defaultdict(list)

    layer_out = [
        r
        for r in rows
        if r.get("boundary") == "layer_out" and r.get("role") == "branch" and r.get("layer") == 47
    ]
    logits = [r for r in rows if r.get("boundary") == "raw_logits"]

    for row in layer_out:
        b = decode_bin(int(row["decode_index"]))
        by_bin[b]["final_hidden_rel_l2"].append(float(row["rel_l2"]))
    for row in logits:
        b = decode_bin(int(row["decode_index"]))
        by_bin[b]["logit_kl"].append(float(row["logit_kl_e0_to_variant"]))
        by_bin[b]["target_rank"].append(float(row["target_rank_variant"]))
        by_bin[b]["top1_agree"].append(1.0 if row.get("top1_agree") else 0.0)

    # per-sample late-early for final hidden
    by_sample: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in layer_out:
        b = decode_bin(int(row["decode_index"]))
        by_sample[str(row["sample_key"])][b].append(float(row["rel_l2"]))
    early = BIN_ORDER[0]
    late_candidates = [b for b in BIN_ORDER if b != early]
    for sample_key, bins in by_sample.items():
        if early not in bins:
            continue
        early_m = mean(bins[early])
        if early_m is None:
            continue
        for late in reversed(late_candidates):
            if late in bins:
                late_m = mean(bins[late])
                if late_m is not None:
                    sample_late_minus_early[sample_key].append(late_m - early_m)
                break

    curve = {}
    for b in BIN_ORDER:
        vals = by_bin.get(b, {})
        if not vals:
            continue
        curve[b] = {
            "final_hidden_rel_l2": mean(vals.get("final_hidden_rel_l2", [])),
            "logit_kl": mean(vals.get("logit_kl", [])),
            "target_rank": mean(vals.get("target_rank", [])),
            "top1_agree": mean(vals.get("top1_agree", [])),
            "n_final_hidden": len(vals.get("final_hidden_rel_l2", [])),
            "n_logits": len(vals.get("logit_kl", [])),
        }

    deltas = [v[0] for v in sample_late_minus_early.values() if v]
    return {
        "by_bin": curve,
        "late_minus_early_final_hidden": {
            "n_samples": len(deltas),
            "mean": mean(deltas),
            "median": median(deltas),
            "fraction_positive": None
            if not deltas
            else float(sum(1 for x in deltas if x > 0) / len(deltas)),
        },
    }


def h2_tables(rows: list[dict], tags: dict[tuple[str, int], set[str]], variant: str) -> dict:
    logits = [r for r in rows if r.get("boundary") == "raw_logits"]
    layer47 = [
        r
        for r in rows
        if r.get("boundary") == "layer_out" and r.get("role") == "branch" and r.get("layer") == 47
    ]
    div_margin, uni_margin = [], []
    div_rank, uni_rank = [], []
    div_hidden, uni_hidden = [], []
    for row in logits:
        key = (str(row["sample_key"]), int(row["decode_index"]))
        tag = tags.get(key, set())
        if "divergence" in tag:
            div_margin.append(float(row["e0_margin"]))
            div_rank.append(float(row["target_rank_variant"]))
        if "uniform" in tag:
            uni_margin.append(float(row["e0_margin"]))
            uni_rank.append(float(row["target_rank_variant"]))
    for row in layer47:
        key = (str(row["sample_key"]), int(row["decode_index"]))
        tag = tags.get(key, set())
        if "divergence" in tag:
            div_hidden.append(float(row["rel_l2"]))
        if "uniform" in tag:
            uni_hidden.append(float(row["rel_l2"]))
    return {
        "variant": variant,
        "divergence_e0_margin_median": median(div_margin),
        "uniform_e0_margin_median": median(uni_margin),
        "divergence_target_rank_mean": mean(div_rank),
        "uniform_target_rank_mean": mean(uni_rank),
        "divergence_final_hidden_rel_l2_median": median(div_hidden),
        "uniform_final_hidden_rel_l2_median": median(uni_hidden),
        "n_divergence_logit_probes": len(div_margin),
        "n_uniform_logit_probes": len(uni_margin),
        "prefix_rescue_poison_done": False,
    }


def h3_tables(rows: list[dict], tags: dict[tuple[str, int], set[str]]) -> dict:
    routers = [
        r for r in rows if r.get("boundary") == "router_logits" and r.get("role") == "logits"
    ]
    div_change = uni_change = 0
    div_n = uni_n = 0
    by_layer_div = defaultdict(lambda: [0, 0])
    for row in routers:
        key = (str(row["sample_key"]), int(row["decode_index"]))
        tag = tags.get(key, set())
        changed = not bool(row.get("topk_exact", True))
        if "divergence" in tag:
            div_n += 1
            div_change += int(changed)
            layer = int(row["layer"])
            by_layer_div[layer][1] += 1
            by_layer_div[layer][0] += int(changed)
        if "uniform" in tag:
            uni_n += 1
            uni_change += int(changed)
    layer_rates = [
        {
            "layer": layer,
            "topk_change_rate": float(c / n) if n else None,
            "n": n,
        }
        for layer, (c, n) in sorted(by_layer_div.items())
        if n >= 4
    ]
    layer_rates.sort(key=lambda x: (-(x["topk_change_rate"] or -1), x["layer"]))
    return {
        "divergence_router_topk_change_rate": None if div_n == 0 else float(div_change / div_n),
        "uniform_router_topk_change_rate": None if uni_n == 0 else float(uni_change / uni_n),
        "n_divergence_router": div_n,
        "n_uniform_router": uni_n,
        "top_layers_by_divergence_topk_change": layer_rates[:10],
    }


def judge_h1(curves: dict) -> dict:
    by_bin = curves["by_bin"]
    present = [b for b in BIN_ORDER if b in by_bin]
    if len(present) < 2:
        return {"verdict": "INCONCLUSIVE", "reason": "fewer than two decode bins present"}
    early, late = present[0], present[-1]
    hidden_early = by_bin[early].get("final_hidden_rel_l2")
    hidden_late = by_bin[late].get("final_hidden_rel_l2")
    kl_early = by_bin[early].get("logit_kl")
    kl_late = by_bin[late].get("logit_kl")
    rank_early = by_bin[early].get("target_rank")
    rank_late = by_bin[late].get("target_rank")
    late_early = curves["late_minus_early_final_hidden"]
    growth = (
        hidden_early is not None
        and hidden_late is not None
        and hidden_late > hidden_early * 1.25
        and (late_early.get("fraction_positive") or 0) >= 0.6
    )
    logit_worsen = (
        kl_early is not None
        and kl_late is not None
        and kl_late > kl_early * 1.25
        and rank_early is not None
        and rank_late is not None
        and rank_late >= rank_early
    )
    if growth and logit_worsen:
        verdict = "SUPPORT"
        reason = "多数样本上 final-hidden 与 logits 都从早 bin 恶化到晚 bin"
    elif growth or logit_worsen:
        verdict = "WEAK_PARTIAL"
        reason = "状态 / logits 里只有一侧出现早→晚变差"
    else:
        verdict = "NOT_SUPPORTED"
        reason = "没有清楚的、随长度增长的固定历史漂移"
    return {
        "verdict": verdict,
        "reason": reason,
        "early_bin": early,
        "late_bin": late,
        "hidden_early": hidden_early,
        "hidden_late": hidden_late,
        "kl_early": kl_early,
        "kl_late": kl_late,
        "rank_early": rank_early,
        "rank_late": rank_late,
        "fraction_samples_late_gt_early": late_early.get("fraction_positive"),
    }


def judge_h2(h2: dict) -> dict:
    dm = h2.get("divergence_e0_margin_median")
    um = h2.get("uniform_e0_margin_median")
    margin_ok = dm is not None and um is not None and dm < um * 0.75
    # criterion 3 requires causal intervention not yet run
    if margin_ok and not h2["prefix_rescue_poison_done"]:
        verdict = "PARTIAL_SUPPORT"
        reason = (
            "分叉 probe 的 E0 裕度低于均匀对照，"
            "但前缀救援/投毒尚未跑"
        )
    elif margin_ok:
        verdict = "SUPPORT"
        reason = "分叉邻域 E0 裕度更低；因果检验待做或已有"
    else:
        verdict = "NOT_SUPPORTED"
        reason = "分叉 probe 的裕度并不明显低于均匀对照"
    return {"verdict": verdict, "reason": reason, **h2}


def judge_h3(h3: dict) -> dict:
    d = h3.get("divergence_router_topk_change_rate")
    u = h3.get("uniform_router_topk_change_rate")
    enrich = d is not None and u is not None and d > u + 0.05 and d > 0.02
    top = h3.get("top_layers_by_divergence_topk_change") or []
    has_layer = bool(top) and (top[0].get("topk_change_rate") or 0) >= 0.1
    if enrich and has_layer:
        verdict = "SUPPORT"
        reason = "分叉 probe 上 router top-k 变化富集，且能指出具体层"
    elif enrich or has_layer:
        verdict = "WEAK_PARTIAL"
        reason = "只有部分 router 边界信号"
    else:
        verdict = "NOT_SUPPORTED"
        reason = "分叉附近没有清楚的 router top-k 富集"
    return {"verdict": verdict, "reason": reason, **h3}


def write_report(path: Path, payload: dict) -> None:
    h1 = payload["H1"]
    h2 = payload["H2"]
    h3 = payload["H3"]
    lines = [
        f"# Real-vLLM 长轨迹汇总（{payload['variant']}）",
        "",
        "## 事实",
        f"- 使用的 compare 行数：{payload['num_compare_rows']}",
        f"- 自由生成首次分叉中位数（{payload['variant']}）："
        f"{payload['free_run']['median_first_divergence']}",
        f"- 自由生成存活率@128：{payload['free_run']['survival_128']}",
        "",
        "## H1（固定历史漂移）",
        f"- 判定：**{h1['verdict']}**",
        f"- 理由：{h1['reason']}",
        "",
        "## H2（低裕度分叉）",
        f"- 判定：**{h2['verdict']}**",
        f"- 理由：{h2['reason']}",
        f"- 分叉处 E0 裕度中位数：{h2.get('divergence_e0_margin_median')}",
        f"- 均匀对照 E0 裕度中位数：{h2.get('uniform_e0_margin_median')}",
        "",
        "## H3（router 边界放大）",
        f"- 判定：**{h3['verdict']}**",
        f"- 理由：{h3['reason']}",
        f"- 分叉处 router top-k 变化率：{h3.get('divergence_router_topk_change_rate')}",
        f"- 均匀对照 router top-k 变化率：{h3.get('uniform_router_topk_change_rate')}",
        "",
        "否定 / 部分结论是刻意的，不要硬写成 H1 叙事。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    compare_path = Path(args.compare_jsonl).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_compare(compare_path, args.tp_rank)
    plan = json.loads(Path(args.probe_plan).read_text(encoding="utf-8"))
    tags = probe_tags(plan, args.variant)
    events = [
        e
        for e in read_jsonl(Path(args.divergence_events))
        if str(e.get("variant")) == args.variant
    ]
    firsts = [int(e["first_divergence"]) for e in events if e.get("first_divergence") is not None]
    surv128 = None if not firsts else float(sum(1 for x in firsts if x >= 128) / len(firsts))

    curves = bin_curves(rows)
    h1 = judge_h1(curves)
    h2 = judge_h2(h2_tables(rows, tags, args.variant))
    h3 = judge_h3(h3_tables(rows, tags))

    payload = {
        "schema_version": 1,
        "variant": args.variant,
        "tp_rank_consumed": int(args.tp_rank),
        "num_compare_rows": len(rows),
        "free_run": {
            "median_first_divergence": median([float(x) for x in firsts]),
            "survival_128": surv128,
            "n_events": len(events),
        },
        "curves": curves,
        "H1": h1,
        "H2": h2,
        "H3": h3,
    }
    json_path = out_dir / f"{args.variant}_h1h2h3_summary.json"
    md_path = out_dir / f"{args.variant}_H1H2H3_REPORT.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(md_path, payload)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
