#!/usr/bin/env python3
"""Summarize unified clean MMLU-Pro300 E0–E7 results for the replan report."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LABELS = [
    ("E0", "E0_native_nvfp4"),
    ("E1", "E1_direct_hif4"),
    ("E2", "E2_r64_only"),
    ("E3", "E3_fusable"),
    ("E4", "E4_fusable_r64"),
    ("E5", "E5_online"),
    ("E6", "E6_online_diag_then_r64"),
    ("E7", "E7_online_r64_then_diag"),
]


def _pick_score(metrics: dict) -> tuple[float, str]:
    results = metrics.get("results", metrics)
    task = results.get("mmlu_pro|0") or results.get("mmlu_pro") or {}
    for key in ("extractive_match", "exact_match", "acc", "acc,none"):
        if key in task and isinstance(task[key], (int, float)):
            return float(task[key]), key
    raise RuntimeError(f"no extractive_match-like metric in {sorted(task)}")


def exact_mcnemar_p(n01: int, n10: int) -> float:
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    lower = sum(math.comb(n, i) for i in range(0, k + 1))
    return min(1.0, (2.0 * lower) / (2**n))


def paired_bootstrap_delta(e0: list[bool], e1: list[bool], *, n_boot: int = 10000, seed: int = 20260908):
    import random

    n = len(e0)
    rng = random.Random(seed)
    observed = (sum(e1) - sum(e0)) / n
    samples = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        samples.append((sum(e1[i] for i in idx) - sum(e0[i] for i in idx)) / n)
    samples.sort()
    return {
        "delta": observed,
        "delta_pp": observed * 100.0,
        "ci95_low_pp": samples[int(0.025 * (n_boot - 1))] * 100.0,
        "ci95_high_pp": samples[int(0.975 * (n_boot - 1))] * 100.0,
        "n_boot": n_boot,
        "seed": seed,
    }


def load_item_correctness(details_path: Path) -> dict[str, bool]:
    rows = json.loads(details_path.read_text())
    out = {}
    for row in rows:
        doc_id = str(row["doc"]["id"])
        metric = row.get("metric") or {}
        # lighteval mmlu_pro uses extractive_match
        val = metric.get("extractive_match")
        if val is None:
            # fallback common keys
            for k, v in metric.items():
                if "extractive" in k or k in ("exact_match", "acc"):
                    val = v
                    break
        if val is None:
            raise RuntimeError(f"missing item metric for doc {doc_id} in {details_path}")
        out[doc_id] = float(val) >= 0.5 if isinstance(val, (int, float)) else bool(val)
    return out


def find_details(variant_dir: Path) -> Path | None:
    paths = sorted(variant_dir.glob("**/details_mmlu_pro|0_*.json"))
    return paths[-1] if paths else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root / "20_mmlu_pro_300"
    scores = {}
    metric_key = None
    protocols = {}
    for short, dirname in LABELS:
        metrics_path = root / dirname / "eval/mmlu_pro/metrics.json"
        if not metrics_path.exists():
            raise RuntimeError(f"missing metrics for {short}: {metrics_path}")
        metrics = json.loads(metrics_path.read_text())
        score, key = _pick_score(metrics)
        metric_key = metric_key or key
        scores[short] = score
        protocols[short] = {
            "metrics_path": str(metrics_path.resolve()),
            "batch_size_max_num_seqs": metrics.get("batch_size_max_num_seqs"),
            "tensor_parallel_size": metrics.get("tensor_parallel_size"),
            "kv_cache_dtype": metrics.get("kv_cache_dtype"),
            "enforce_eager": metrics.get("enforce_eager"),
            "enable_thinking": metrics.get("enable_thinking"),
            "max_samples": metrics.get("max_samples"),
            "max_new_tokens": metrics.get("max_new_tokens"),
            "temperature": metrics.get("temperature"),
        }

    # Paired flips if details available.
    details = {}
    for short, dirname in LABELS:
        path = find_details(root / dirname)
        if path is not None:
            details[short] = load_item_correctness(path)
    flips = {}
    boot = None
    mcnemar = None
    if "E0" in details and "E1" in details and set(details["E0"]) == set(details["E1"]):
        ids = sorted(details["E0"], key=lambda x: int(x) if x.isdigit() else x)
        e0 = [details["E0"][i] for i in ids]
        e1 = [details["E1"][i] for i in ids]
        n10 = sum(a and not b for a, b in zip(e0, e1))
        n01 = sum((not a) and b for a, b in zip(e0, e1))
        flips["E0_E1"] = {
            "n10_E0pass_E1fail": n10,
            "n01_E0fail_E1pass": n01,
            "both_pass": sum(a and b for a, b in zip(e0, e1)),
            "both_fail": sum((not a) and (not b) for a, b in zip(e0, e1)),
            "n": len(ids),
        }
        boot = paired_bootstrap_delta(e0, e1)
        mcnemar = exact_mcnemar_p(n01, n10)
        for short, _ in LABELS[2:]:
            if short not in details or set(details[short]) != set(details["E0"]):
                continue
            cur = [details[short][i] for i in ids]
            flips[f"E0_{short}"] = {
                "n10": sum(a and not b for a, b in zip(e0, cur)),
                "n01": sum((not a) and b for a, b in zip(e0, cur)),
            }
            flips[f"E1_{short}"] = {
                "n10": sum(a and not b for a, b in zip(e1, cur)),
                "n01": sum((not a) and b for a, b in zip(e1, cur)),
            }

    e0 = scores["E0"]
    summary = {
        "schema_version": 1,
        "task": "mmlu_pro|0",
        "max_samples": 300,
        "metric_key": metric_key,
        "scores": scores,
        "delta_pp_vs_E0": {k: (v - e0) * 100.0 for k, v in scores.items()},
        "protocol": protocols,
        "paired_flips": flips,
        "E0_E1_bootstrap": boot,
        "E0_E1_mcnemar_p": mcnemar,
        "note": "E3/E4 use adopted artifacts; candidate DIAG is excluded from this ranking table",
    }
    (root / "CLEAN_MMLU_PRO_E0_E7_REPORT.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# CLEAN_MMLU_PRO_E0_E7_REPORT",
        "",
        "统一协议：`mmlu_pro|0` max_samples=300, thinking=true, T=0.6, top_p=0.95, top_k=20, TP=2, KV=BF16, enforce_eager, batch_size/max_num_seqs=128。",
        "E0 = 复用已审计 Phase-A formal；E1–E7 = 本轮统一重跑。E3/E4 = adopted artifact；candidate 不进入本表。",
        "",
        "| Variant | score | delta vs E0 (pp) |",
        "|---|---:|---:|",
    ]
    for short, _ in LABELS:
        lines.append(
            f"| {short} | {scores[short]*100:.2f}% | {summary['delta_pp_vs_E0'][short]:+.2f} |"
        )
    if boot is not None:
        lines.extend(
            [
                "",
                "## E0 vs E1 paired",
                "",
                f"- flips n10/n01 = {flips['E0_E1']['n10_E0pass_E1fail']}/{flips['E0_E1']['n01_E0fail_E1pass']}",
                f"- delta = {boot['delta_pp']:.3f} pp; 95% CI [{boot['ci95_low_pp']:.3f}, {boot['ci95_high_pp']:.3f}]",
                f"- exact McNemar p = {mcnemar:.6g}",
            ]
        )
    lines.append("")
    (root / "CLEAN_MMLU_PRO_E0_E7_REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"scores": scores, "report": str(root / "CLEAN_MMLU_PRO_E0_E7_REPORT.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
