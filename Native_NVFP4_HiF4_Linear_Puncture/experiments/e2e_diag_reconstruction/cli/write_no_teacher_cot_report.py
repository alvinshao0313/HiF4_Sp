"""Write the no-Teacher-CoT ablation report from stage summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import read_json

COLS = (
    "method",
    "calib_source",
    "accepted",
    "rollback",
    "would_rollback_count",
    "mean_recovery",
    "median_recovery",
    "arc_easy",
    "arc_challenge",
    "mmlu_pro_300",
    "aime25_avg5",
    "delta_mmlu_pro_300_vs_nvfp4_pp",
)


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _table(rows: list[dict[str, Any]]) -> list[str]:
    header = "| " + " | ".join(COLS) + " |"
    sep = "|" + "|".join("---" for _ in COLS) + "|"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in COLS) + " |")
    return lines


def _rows(summary_path: Path) -> list[dict[str, Any]]:
    blob = read_json(summary_path)
    return list(blob.get("rows") or [])


def _load_stats(cache_root: Path) -> dict[str, Any] | None:
    p = cache_root / "calibration" / "stats.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage_map", required=True, help="JSON mapping stage name -> directory")
    p.add_argument("--output_md", required=True)
    args = p.parse_args(argv)
    stages = json.loads(Path(args.stage_map).read_text(encoding="utf-8"))
    phase_a = Path(stages["phase_a"])
    lines = [
        "# NVFP4→HiF4 E2E 消融报告（暂不使用 Teacher-CoT）",
        "",
        "研究问题：在不依赖 Teacher 自回归 CoT 的前提下，DIAG/R64 结构、训练机制和 calibration source 对 Qwen3-8B NVFP4→HiF4 转换的 E2E 影响。",
        "",
        "实验设置：默认 `s1k_original` 128/32 seed=42；fake HiF4 QDQ；fast eval 为 ARC-E/C + MMLU-Pro 300；AIME25 avg@5 只跑 finalist。",
        "",
        "## 1. 结构消融 E0–E7",
        "",
    ]
    lines.extend(_table(_rows(phase_a / "summary.json")))
    a_sum = read_json(phase_a / "summary.json")
    lines += [
        "",
        f"best_fusable_preset = {a_sum.get('best_fusable_preset')}",
        f"best_online_preset = {a_sum.get('best_online_preset')}",
        f"best_overall_preset = {a_sum.get('best_overall_preset')}",
        "",
        "## 2. Fusable DIAG component",
        "",
    ]
    lines.extend(_table(_rows(Path(stages["phase_b"]) / "summary.json")))
    lines += ["", "## 3. 训练机制：joint / input mode / loss", ""]
    for key, title in (
        ("train_scope", "joint vs linear_independent"),
        ("input_mode", "progressive_student vs teacher input"),
        ("loss", "reconstruction loss"),
    ):
        lines += [f"### {title}", ""]
        lines.extend(_table(_rows(Path(stages[key]) / "summary.json")))
        lines.append("")
    lines += ["## 4. 稳定性：rollback / clamp", ""]
    for key, title in (("rollback", "rollback on/off"), ("clamp", "clamp -4,4 vs none")):
        lines += [f"### {title}", ""]
        lines.extend(_table(_rows(Path(stages[key]) / "summary.json")))
        lines.append("")
    lines += ["## 5. Calibration source + Finalist E2E", ""]
    d_dir = Path(stages["phase_d"])
    lines.extend(_table(_rows(d_dir / "summary.json")))
    lines += ["", "### Calibration length stats（sample 数固定，不是等 token budget）", ""]
    stats_map = stages.get("calib_stats") or {}
    for name, root in stats_map.items():
        blob = _load_stats(Path(root))
        if blob is None:
            lines.append(f"- `{name}`: stats.json 缺失")
            continue
        tr = blob["train"]
        va = blob["val"]
        lines.append(
            f"- `{name}` train n={tr['n_samples']} tokens={tr['total_tokens']} "
            f"max={tr['max_seqlen']} p50={tr['p50_seqlen']} p95={tr['p95_seqlen']}; "
            f"val n={va['n_samples']} tokens={va['total_tokens']} "
            f"max={va['max_seqlen']} p50={va['p50_seqlen']} p95={va['p95_seqlen']}"
        )
    d_sum = read_json(d_dir / "summary.json")
    lines += [
        "",
        "### Finalist",
        "",
        f"Phase D best_fusable calib_source={((d_sum.get('best_fusable') or {}).get('calib_source'))}",
        f"Phase D best_online calib_source={((d_sum.get('best_online') or {}).get('calib_source'))}",
        "",
        "可支持结论写在各阶段 report.md 观察之后；本文件只汇总表。不能把 sample-count 对照解释成等 token-budget 对照。当前没有 Teacher-CoT 实验。",
        "",
    ]
    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
