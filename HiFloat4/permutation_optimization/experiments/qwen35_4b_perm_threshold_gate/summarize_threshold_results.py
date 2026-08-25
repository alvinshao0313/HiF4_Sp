#!/usr/bin/env python3
"""Summarize fast/full eval results for the threshold-gate experiment.

Fast stage: ARC-Easy / ARC-Challenge / PIQA deltas vs identity, eligibility
filtering and top-2 threshold selection. Full stage: validation tasks
(BoolQ/HellaSwag/WinoGrande/MMLU) final verdict. Selection and validation
task sets are strictly disjoint by construction here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FAST_TASKS = ("arc_easy", "arc_challenge", "piqa")
VALIDATION_TASKS = ("boolq", "hellaswag", "winogrande", "mmlu")

MIN_ARC_CHALLENGE_DELTA_PP = -0.2
MIN_TASKS_NOT_WORSE_FAST = 2
MIN_TASKS_NOT_WORSE_FULL = 3
MIN_MMLU_DELTA_PP = -0.2


def extract_accuracy_scores(payload: dict) -> dict[str, float]:
    """Pull per-task accuracy from a raw lm-eval payload or a compact one."""
    out: dict[str, float] = {}
    results = payload.get("results")
    if isinstance(results, dict):
        for task in FAST_TASKS + VALIDATION_TASKS:
            trez = results.get(task)
            if not isinstance(trez, dict):
                continue
            for key in ("acc,none", "acc", "acc_norm,none", "acc_norm"):
                if isinstance(trez.get(key), (int, float)):
                    out[task] = float(trez[key])
                    break
        return out
    scores = payload.get("scores")
    if isinstance(scores, dict):
        for task in FAST_TASKS + VALIDATION_TASKS:
            if isinstance(scores.get(task), (int, float)):
                out[task] = float(scores[task])
    return out


def _macro(scores: dict[str, float], tasks: tuple[str, ...]) -> float | None:
    vals = [scores[t] for t in tasks if t in scores]
    if len(vals) != len(tasks):
        return None
    return sum(vals) / len(vals)


def summarize_fast_variants(
    variant_payloads: dict[str, dict],
    threshold_metadata: dict[str, dict],
) -> list[dict]:
    """Per-variant fast-task deltas vs identity (identity row excluded)."""
    if "identity" not in variant_payloads:
        raise KeyError("identity baseline payload is required")
    identity_scores = extract_accuracy_scores(variant_payloads["identity"])
    identity_macro = _macro(identity_scores, FAST_TASKS)
    if identity_macro is None:
        raise KeyError(f"identity payload missing fast tasks: {identity_scores}")

    rows: list[dict] = []
    for variant, payload in sorted(variant_payloads.items()):
        if variant == "identity":
            continue
        scores = extract_accuracy_scores(payload)
        macro = _macro(scores, FAST_TASKS)
        if macro is None:
            raise KeyError(f"{variant} payload missing fast tasks: {scores}")
        deltas = {t: (scores[t] - identity_scores[t]) * 100.0 for t in FAST_TASKS}
        meta = threshold_metadata.get(variant, {})
        n_not_worse = sum(1 for t in FAST_TASKS if deltas[t] >= 0.0)
        rows.append(
            {
                "variant": variant,
                "kind": "threshold" if variant.startswith("tau_") else "control",
                "scores": scores,
                "macro_accuracy": macro,
                "task_deltas_pp": deltas,
                "macro_delta_pp": (macro - identity_macro) * 100.0,
                "n_tasks_not_worse": n_not_worse,
                "threshold_pct": meta.get("threshold_pct"),
                "n_reordered": meta.get("n_reordered"),
                "eligible": bool(
                    variant.startswith("tau_")
                    and (macro - identity_macro) * 100.0 > 0.0
                    and deltas["arc_challenge"] >= MIN_ARC_CHALLENGE_DELTA_PP
                    and n_not_worse >= MIN_TASKS_NOT_WORSE_FAST
                ),
            }
        )
    return rows


def select_fast_thresholds(rows: list[dict], max_candidates: int = 2) -> list[str]:
    """Rank eligible thresholds: macro, then ARC-C, then fewer layers, then tau."""
    eligible = [r for r in rows if r.get("eligible")]
    eligible.sort(
        key=lambda r: (
            -r["macro_delta_pp"],
            -r["task_deltas_pp"]["arc_challenge"],
            r["n_reordered"] if r["n_reordered"] is not None else 1 << 30,
            -(r["threshold_pct"] or 0.0),
        )
    )
    return [r["variant"] for r in eligible[:max_candidates]]


def summarize_full_variants(
    variant_payloads: dict[str, dict],
    threshold_metadata: dict[str, dict],
) -> list[dict]:
    """Per-variant validation-task deltas vs identity on the held-out tasks."""
    if "identity" not in variant_payloads:
        raise KeyError("identity baseline payload is required")
    identity_scores = extract_accuracy_scores(variant_payloads["identity"])
    identity_macro = _macro(identity_scores, VALIDATION_TASKS)
    if identity_macro is None:
        raise KeyError(f"identity payload missing validation tasks: {identity_scores}")

    rows: list[dict] = []
    for variant, payload in sorted(variant_payloads.items()):
        if variant == "identity":
            continue
        scores = extract_accuracy_scores(payload)
        macro = _macro(scores, VALIDATION_TASKS)
        if macro is None:
            raise KeyError(f"{variant} payload missing validation tasks: {scores}")
        deltas = {t: (scores[t] - identity_scores[t]) * 100.0 for t in VALIDATION_TASKS}
        meta = threshold_metadata.get(variant, {})
        rows.append(
            {
                "variant": variant,
                "scores": scores,
                "macro_accuracy": macro,
                "task_deltas_pp": deltas,
                "macro_delta_pp": (macro - identity_macro) * 100.0,
                "n_tasks_not_worse": sum(1 for t in VALIDATION_TASKS if deltas[t] >= 0.0),
                "threshold_pct": meta.get("threshold_pct"),
                "n_reordered": meta.get("n_reordered"),
            }
        )
    return rows


def final_verdict(rows: list[dict]) -> dict:
    """Final 'does permutation help' judgement on validation tasks."""
    by_variant = {r["variant"]: r for r in rows}
    default = by_variant.get("selected_default")
    tau_rows = [r for r in rows if r["variant"].startswith("tau_")]
    best = max(tau_rows, key=lambda r: (r["macro_delta_pp"], -abs(r["task_deltas_pp"]["mmlu"])), default=None)
    if best is None:
        return {"useful": False, "reason": "no threshold variant evaluated"}
    default_macro = default["macro_delta_pp"] if default else 0.0
    conds = {
        "macro_positive": best["macro_delta_pp"] > 0.0,
        "tasks_not_worse": best["n_tasks_not_worse"] >= MIN_TASKS_NOT_WORSE_FULL,
        "mmlu_not_worse": best["task_deltas_pp"]["mmlu"] >= MIN_MMLU_DELTA_PP,
        "not_below_selected_default": best["macro_delta_pp"] >= default_macro,
    }
    return {
        "useful": bool(all(conds.values())),
        "best_variant": best["variant"],
        "conditions": conds,
        "best_macro_delta_pp": best["macro_delta_pp"],
    }


def _load_payloads(eval_dir: Path) -> dict[str, dict]:
    payloads: dict[str, dict] = {}
    for p in sorted(eval_dir.glob("*.json")):
        if p.stem in {"summary"}:
            continue
        payloads[p.stem] = json.loads(p.read_text())
    return payloads


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["fast", "full"], required=True)
    ap.add_argument("--eval-dir", type=str, required=True)
    ap.add_argument("--threshold-metadata", type=str, default="")
    ap.add_argument("--output-dir", type=str, required=True)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    payloads = _load_payloads(eval_dir)
    meta: dict[str, dict] = {}
    if args.threshold_metadata:
        meta = json.loads(Path(args.threshold_metadata).read_text())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "fast":
        rows = summarize_fast_variants(payloads, meta)
        selected = select_fast_thresholds(rows)
        payload = {
            "rows": rows,
            "selected_thresholds": selected,
            "stop": not selected,
            "conclusion": (
                "fast stage: no threshold shows stable positive gain; "
                "当前候选排序在阈值门控后仍未显示稳定下游收益"
                if not selected
                else f"fast stage winners: {selected}"
            ),
        }
        (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
        lines = [
            "# Fast Eval Summary (ARC-Easy / ARC-Challenge / PIQA, 0-shot)",
            "",
            "| variant | n_reordered | ARC-E Δpp | ARC-C Δpp | PIQA Δpp | macro Δpp | eligible |",
            "|---|---:|---:|---:|---:|---:|---|",
            "| identity | — | 0.000 | 0.000 | 0.000 | 0.000 | — |",
        ]
        for r in rows:
            d = r["task_deltas_pp"]
            lines.append(
                f"| {r['variant']} | {r['n_reordered'] if r['n_reordered'] is not None else '—'} "
                f"| {d['arc_easy']:+.3f} | {d['arc_challenge']:+.3f} | {d['piqa']:+.3f} "
                f"| {r['macro_delta_pp']:+.3f} | {r['eligible']} |"
            )
        lines += [
            "",
            f"selected for full eval: {selected if selected else '无 — 停止完整评测'}",
            "",
        ]
        (out_dir / "summary.md").write_text("\n".join(lines))
        print(json.dumps({"selected_thresholds": selected, "stop": not selected}, indent=2))
        return

    rows = summarize_full_variants(payloads, meta)
    verdict = final_verdict(rows)
    payload = {"rows": rows, "verdict": verdict}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    lines = [
        "# Full Eval Summary (BoolQ / HellaSwag / WinoGrande / MMLU, 0-shot)",
        "",
        "| variant | BoolQ Δpp | HellaSwag Δpp | WinoGrande Δpp | MMLU Δpp | macro Δpp |",
        "|---|---:|---:|---:|---:|---:|",
        "| identity | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |",
    ]
    for r in rows:
        d = r["task_deltas_pp"]
        lines.append(
            f"| {r['variant']} | {d['boolq']:+.3f} | {d['hellaswag']:+.3f} "
            f"| {d['winogrande']:+.3f} | {d['mmlu']:+.3f} | {r['macro_delta_pp']:+.3f} |"
        )
    if verdict["useful"]:
        lines += ["", "结论：当前排序候选有用，但需要逐层收益阈值门控。", ""]
    else:
        lines += [
            "",
            "结论：简单阈值无法挽救当前全局排序候选，下一步应优化候选生成，"
            "例如原始 G64 内 Wanda-aware G4/G8 分组，而不是继续调 gate 阈值。",
            "",
        ]
    lines.append(f"verdict: {json.dumps(verdict, ensure_ascii=False)}")
    (out_dir / "summary.md").write_text("\n".join(lines))
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
