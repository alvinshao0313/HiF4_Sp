#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.config import POSITION_BINS
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import read_jsonl

BIN_ORDER = [x[0] for x in POSITION_BINS]


def finite(values) -> list[float]:
    out = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def mean(values) -> float | None:
    vals = finite(values)
    return None if not vals else float(sum(vals) / len(vals))


def median(values) -> float | None:
    vals = finite(values)
    return None if not vals else float(statistics.median(vals))


def frac(values) -> float | None:
    vals = list(values)
    return None if not vals else float(sum(bool(x) for x in vals) / len(vals))


def aggregate_variant(rows: list[dict], variant: str) -> dict:
    logits = [x for x in rows if x.get("scope") == "logit"]
    layers = [x for x in rows if x.get("scope") == "layer"]
    by_bin: dict[str, dict] = {}
    for bin_name in BIN_ORDER:
        subset = [x for x in logits if x.get("bin") == bin_name]
        layer_subset = [x for x in layers if x.get("bin") == bin_name]
        final_hidden = [x for x in layer_subset if int(x.get("layer_id", -1)) == 47]
        if not subset:
            continue
        by_bin[bin_name] = {
            "count": len(subset),
            "logit_kl": mean(x.get("logit_kl_e0_to_variant") for x in subset),
            "logit_js": mean(x.get("logit_js") for x in subset),
            "centered_cosine": mean(x.get("logit_centered_cosine") for x in subset),
            "top1_agreement": frac(x.get("top1_agreement") for x in subset),
            "target_nll_delta": mean(x.get("target_nll_delta") for x in subset),
            "e0_margin_median": median(x.get("e0_top1_top2_margin") for x in subset),
            "final_hidden_rel_l2": mean(x.get("hidden_rel_l2") for x in final_hidden),
            "router_topk_exact_all_layers": frac(x.get("router_topk_exact") for x in layer_subset),
        }

    divergence_logits = [
        x for x in logits if any(reason == f"divergence:{variant}:+0" for reason in x.get("reasons", []))
    ]
    uniform_logits = [x for x in logits if any(str(r).startswith("uniform:") for r in x.get("reasons", []))]
    divergence_layers = [
        x for x in layers if any(reason == f"divergence:{variant}:+0" for reason in x.get("reasons", []))
    ]
    uniform_layers = [x for x in layers if any(str(r).startswith("uniform:") for r in x.get("reasons", []))]

    router_by_layer: dict[int, dict] = {}
    grouped_div = defaultdict(list)
    grouped_uniform = defaultdict(list)
    for row in divergence_layers:
        grouped_div[int(row["layer_id"])].append(row)
    for row in uniform_layers:
        grouped_uniform[int(row["layer_id"])].append(row)
    for layer_id in range(48):
        d = grouped_div.get(layer_id, [])
        u = grouped_uniform.get(layer_id, [])
        router_by_layer[layer_id] = {
            "divergence_router_exact": frac(x.get("router_topk_exact") for x in d),
            "uniform_router_exact": frac(x.get("router_topk_exact") for x in u),
            "divergence_router_overlap": mean(x.get("router_topk_overlap") for x in d),
            "uniform_router_overlap": mean(x.get("router_topk_overlap") for x in u),
            "divergence_hidden_rel_l2": mean(x.get("hidden_rel_l2") for x in d),
            "uniform_hidden_rel_l2": mean(x.get("hidden_rel_l2") for x in u),
        }

    nonempty_bins = [b for b in BIN_ORDER if b in by_bin]
    early = by_bin[nonempty_bins[0]] if nonempty_bins else None
    late = by_bin[nonempty_bins[-1]] if nonempty_bins else None
    kl_growth = None
    if early and late and early["logit_kl"] is not None and late["logit_kl"] is not None:
        kl_growth = late["logit_kl"] / max(early["logit_kl"], 1e-12)

    div_margin = median(x.get("e0_top1_top2_margin") for x in divergence_logits)
    uniform_margin = median(x.get("e0_top1_top2_margin") for x in uniform_logits)
    return {
        "by_decode_bin": by_bin,
        "teacher_forcing_kl_late_over_early": kl_growth,
        "divergence_points": {
            "count": len(divergence_logits),
            "e0_margin_median": div_margin,
            "uniform_e0_margin_median": uniform_margin,
            "top1_agreement": frac(x.get("top1_agreement") for x in divergence_logits),
            "logit_kl": mean(x.get("logit_kl_e0_to_variant") for x in divergence_logits),
            "target_nll_delta": mean(x.get("target_nll_delta") for x in divergence_logits),
        },
        "router_by_layer": router_by_layer,
        "evidence_flags": {
            "teacher_forcing_drift_grows_with_length": bool(kl_growth is not None and kl_growth >= 2.0),
            "divergence_occurs_at_smaller_logit_margin": bool(
                div_margin is not None and uniform_margin is not None and div_margin < 0.75 * uniform_margin
            ),
            "router_mismatch_enriched_at_divergence": bool(
                any(
                    value["divergence_router_exact"] is not None
                    and value["uniform_router_exact"] is not None
                    and value["divergence_router_exact"] + 0.10 < value["uniform_router_exact"]
                    for value in router_by_layer.values()
                )
            ),
        },
    }


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{float(value):.{digits}f}"


def write_report(path: Path, free_summary: dict, semantic: dict) -> None:
    lines = [
        "# NVFP4 → HiF4 长轨迹稳定性诊断报告",
        "",
        "## 1. Free-run 轨迹分叉",
        "",
        "| 方案 | 完全一致率 | 首次分叉中位位置 | 相同前缀≥128 | ≥512 | ≥2048 | ≥8192 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in ("E1", "E2", "E3", "E4"):
        row = free_summary.get(variant, {})
        survival = row.get("survival_fraction", {})
        lines.append(
            f"| {variant} | {fmt(row.get('exact_trajectory_fraction'))} | {row.get('median_first_divergence', '-')} | "
            f"{fmt(survival.get('128'))} | {fmt(survival.get('512'))} | {fmt(survival.get('2048'))} | {fmt(survival.get('8192'))} |"
        )
    lines.extend(["", "## 2. Teacher-forcing：误差是否随 decode position 增长", ""])
    for variant in ("E1", "E2", "E3", "E4"):
        result = semantic.get(variant, {})
        lines.append(f"### {variant}")
        lines.append("")
        lines.append("| decode 区间 | logit KL(E0||V) | final hidden rel-L2 | router top-k exact | centered cosine | top1 agreement | target NLL Δ |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for bin_name in BIN_ORDER:
            row = result.get("by_decode_bin", {}).get(bin_name)
            if row:
                lines.append(
                    f"| {bin_name} | {fmt(row.get('logit_kl'))} | {fmt(row.get('final_hidden_rel_l2'))} | "
                    f"{fmt(row.get('router_topk_exact_all_layers'))} | {fmt(row.get('centered_cosine'))} | "
                    f"{fmt(row.get('top1_agreement'))} | {fmt(row.get('target_nll_delta'))} |"
                )
        flags = result.get("evidence_flags", {})
        div = result.get("divergence_points", {})
        lines.extend(
            [
                "",
                f"- late/early KL ratio: **{fmt(result.get('teacher_forcing_kl_late_over_early'))}**",
                f"- 首次分叉位置 E0 logit margin 中位数: **{fmt(div.get('e0_margin_median'))}**；统一探针中位数: **{fmt(div.get('uniform_e0_margin_median'))}**",
                f"- 长度相关 teacher-forcing drift: **{flags.get('teacher_forcing_drift_grows_with_length')}**",
                f"- 小 margin 分叉富集: **{flags.get('divergence_occurs_at_smaller_logit_margin')}**",
                f"- router mismatch 在分叉点富集: **{flags.get('router_mismatch_enriched_at_divergence')}**",
                "",
            ]
        )
    lines.extend(
        [
            "## 3. 解释门禁",
            "",
            "- 若 E0 `e0_semantic_parity.json` 的 greedy top-1 parity < 0.99，本报告中的 hidden/router/logit semantic replay 不得用于机制结论。",
            "- 若 free-run 很早分叉，但 teacher-forcing KL 不随位置增长，同时分叉点 E0 margin 显著更小，优先解释为 **trajectory bifurcation / decision-margin sensitivity**，而不是随机噪声线性累积。",
            "- 若 teacher-forcing KL、hidden rel-L2 随 decode position 明显增长，即使固定 E0 token 轨迹也越来越偏离，则支持 **state-distribution / long-context drift**。",
            "- 若 router top-k mismatch 在首次分叉附近显著富集，应继续做 router-aware DIAG / margin gate，而不是仅优化 block NMSE。",
            "- E4 是否‘组合失败’必须同时看 free-run、teacher-forcing 和 router 三条证据；不得只凭 MMLU-Pro300 的 1–2 道题差异下结论。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_root", required=True)
    args = p.parse_args()
    root = Path(args.run_root)
    free_summary = json.loads((root / "analysis/free_run_summary.json").read_text(encoding="utf-8"))
    semantic: dict[str, dict] = {}
    for variant in ("E1", "E2", "E3", "E4"):
        rows = read_jsonl(root / f"semantic/{variant}/semantic_metrics.jsonl")
        semantic[variant] = aggregate_variant(rows, variant)
    payload = {"free_run": free_summary, "semantic": semantic}
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "long_trajectory_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(out / "LONG_TRAJECTORY_STABILITY_REPORT.md", free_summary, semantic)
    print(out / "LONG_TRAJECTORY_STABILITY_REPORT.md")


if __name__ == "__main__":
    main()
