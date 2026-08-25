#!/usr/bin/env python3
"""Summarize a dynamic-input-sparse e2e run directory into csv/json/report.md."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

METHOD_DIRS = [
    ("dense", "DENSE", 1.0),
    ("m8_keep075", "M8", 0.75),
    ("m8_keep050", "M8", 0.50),
    ("m8_keep025", "M8", 0.25),
    ("m1_keep075", "M1", 0.75),
    ("m1_keep050", "M1", 0.50),
    ("m1_keep025", "M1", 0.25),
]


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _arc_scores(method_dir: Path) -> tuple[float | None, float | None]:
    data = _load_json(method_dir / "arc" / "lm_eval.json")
    if not data:
        return None, None
    return data.get("arc_easy"), data.get("arc_challenge")


def _mmlu_pro(method_dir: Path) -> float | None:
    # Prefer compact marker written by harness; else scan lighteval results.
    compact = _load_json(method_dir / "mmlu_pro" / "score.json")
    if compact and "extractive_match" in compact:
        return float(compact["extractive_match"])
    results_root = method_dir / "mmlu_pro"
    if not results_root.is_dir():
        return None
    candidates = sorted(results_root.rglob("results_*.json"))
    if not candidates:
        return None
    data = json.loads(candidates[-1].read_text(encoding="utf-8"))
    # lighteval nested structure varies; try common keys
    try:
        results = data.get("results") or data
        for key, val in results.items():
            if "mmlu_pro" in str(key) and isinstance(val, dict):
                for mk in (
                    "extractive_match",
                    "extractive_match,(truncation=False / fewshots=0)",
                ):
                    if mk in val and isinstance(val[mk], (int, float)):
                        return float(val[mk])
                for mk, mv in val.items():
                    if "extractive" in str(mk) and isinstance(mv, (int, float)):
                        return float(mv)
    except Exception:
        return None
    return None


def _aime(method_dir: Path) -> float | None:
    compact = _load_json(method_dir / "aime25" / "score.json")
    if compact and "avg5" in compact:
        return float(compact["avg5"])
    results_root = method_dir / "aime25"
    if not results_root.is_dir():
        return None
    candidates = sorted(results_root.rglob("results_*.json"))
    if not candidates:
        return None
    data = json.loads(candidates[-1].read_text(encoding="utf-8"))
    try:
        results = data.get("results") or data
        for key, val in results.items():
            if "aime" in str(key).lower() and isinstance(val, dict):
                for mk, mv in val.items():
                    if isinstance(mv, (int, float)) and (
                        "pass" in str(mk) or "qem" in str(mk) or "extract" in str(mk)
                    ):
                        return float(mv)
    except Exception:
        return None
    return None


def _fmt(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.6f}"


def summarize(run_dir: Path) -> None:
    rows = []
    dense = None
    by_key: dict[tuple[str, float], dict] = {}
    for dirname, method, keep in METHOD_DIRS:
        d = run_dir / dirname
        if not (d / "DONE").is_file():
            continue
        arc_e, arc_c = _arc_scores(d)
        mmlu = _mmlu_pro(d)
        aime = _aime(d)
        row = {
            "method": method,
            "keep_ratio": keep,
            "k_sparsity": round(1.0 - keep, 4) if method != "DENSE" else 0.0,
            "arc_easy": arc_e,
            "arc_challenge": arc_c,
            "mmlu_pro_300": mmlu,
            "aime25_avg5": aime,
        }
        by_key[(method, keep)] = row
        if method == "DENSE":
            dense = row
        rows.append(row)

    if dense is None:
        raise SystemExit(f"dense baseline incomplete under {run_dir}")

    for row in rows:
        for task in ("arc_easy", "arc_challenge", "mmlu_pro_300", "aime25_avg5"):
            drop_key = f"{task}_drop_vs_dense" if task != "mmlu_pro_300" else "mmlu_pro_drop_vs_dense"
            if task == "aime25_avg5":
                drop_key = "aime25_drop_vs_dense"
            if task == "arc_easy":
                drop_key = "arc_easy_drop_vs_dense"
            if task == "arc_challenge":
                drop_key = "arc_challenge_drop_vs_dense"
            base = dense[task]
            cur = row[task]
            row[drop_key] = None if base is None or cur is None else float(base) - float(cur)
        drops = [
            row["arc_easy_drop_vs_dense"],
            row["arc_challenge_drop_vs_dense"],
            row["mmlu_pro_drop_vs_dense"],
        ]
        valid = [x for x in drops if x is not None]
        row["mean_drop_non_aime"] = sum(valid) / len(valid) if valid else None

    fieldnames = [
        "method",
        "keep_ratio",
        "k_sparsity",
        "arc_easy",
        "arc_challenge",
        "mmlu_pro_300",
        "aime25_avg5",
        "arc_easy_drop_vs_dense",
        "arc_challenge_drop_vs_dense",
        "mmlu_pro_drop_vs_dense",
        "aime25_drop_vs_dense",
        "mean_drop_non_aime",
    ]
    csv_path = run_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: _fmt(row[k]) if k != "method" else row[k] for k in fieldnames})

    gaps = {"M8_minus_M1": {}, "M1_minus_dense": {}, "M8_minus_dense": {}}
    for keep in (0.75, 0.5, 0.25):
        m1 = by_key.get(("M1", keep))
        m8 = by_key.get(("M8", keep))
        for task in ("arc_easy", "arc_challenge", "mmlu_pro_300", "aime25_avg5"):
            key = f"keep_{keep}_{task}"
            if m1 and m1[task] is not None and dense[task] is not None:
                gaps["M1_minus_dense"][key] = float(m1[task]) - float(dense[task])
            if m8 and m8[task] is not None and dense[task] is not None:
                gaps["M8_minus_dense"][key] = float(m8[task]) - float(dense[task])
            if m1 and m8 and m1[task] is not None and m8[task] is not None:
                gaps["M8_minus_M1"][key] = float(m8[task]) - float(m1[task])

    summary = {"rows": rows, "gaps": gaps}
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Dynamic Input-Only Block Sparse E2E Report",
        "",
        "## 1. Experiment Question",
        "",
        "When only MLP input K-blocks are dynamically selected, how much end-to-end accuracy is lost under M1-input-only oracle vs M8 block-energy predictor?",
        "",
        "## 2. Input-Only Algorithm Definition",
        "",
        "Y = (X ⊙ MX_expanded) @ W.T ; no output mask MY.",
        "",
        "## 3. M1 Full-Output Oracle",
        "",
        "Backward-greedy + 1-swap exact recovery minimizing ||Y - Yhat(S)||_2^2 with full output.",
        "",
        "## 4. M8 Block-Energy Predictor",
        "",
        "Score = mean(Xp_block^2) * G_W[k], Top-K stable.",
        "",
        "## 5. Why Gate/Up Share a Mask",
        "",
        "Same MLP input; vLLM fuses gate_up_proj.",
        "",
        "## 6. Why Down Uses a Separate Dynamic Mask",
        "",
        "Down consumes sparse-path H = SiLU(gate)*up.",
        "",
        "## 7. Evaluation Protocol",
        "",
        "ARC via lm_eval/HF; MMLU-Pro-300 and AIME25-avg5 via vLLM+lighteval; TP=1; enforce_eager; BF16.",
        "",
        "## 8. Primary Accuracy Table",
        "",
        "| method | keep | ARC-E | ARC-C | MMLU-Pro-300 | AIME25-avg5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['keep_ratio']} | {_fmt(row['arc_easy'])} | "
            f"{_fmt(row['arc_challenge'])} | {_fmt(row['mmlu_pro_300'])} | "
            f"{_fmt(row['aime25_avg5'])} |"
        )
    lines += [
        "",
        "## 9. Accuracy vs Input Keep Ratio",
        "",
        "See summary.csv drops vs same-run dense.",
        "",
        "## 10. M8 Gap to M1 Oracle",
        "",
        "```json",
        json.dumps(gaps["M8_minus_M1"], indent=2),
        "```",
        "",
        "## 11. Per-Task Observations",
        "",
        "_Fill after inspecting raw logs._",
        "",
        "## 12. M1 Runtime Cost",
        "",
        "See `m1_runtime.json` if present; otherwise smoke timing logs.",
        "",
        "## 13. vLLM Compatibility Status",
        "",
        "TP=1 eager path with UnquantizedLinearMethod input masking.",
        "",
        "## 14. Limitations",
        "",
        "Dense-mask GEMM is an accuracy reference, not a sparse-kernel speed result. No TP>1.",
        "",
        "## 15. Decision / Next Step",
        "",
        "Compare Dense-M1 (intrinsic input sparsity cost) vs M1-M8 (predictor gap). Choose A/B/C/D from the plan.",
        "",
    ]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {run_dir / 'summary.json'}")
    print(f"wrote {run_dir / 'report.md'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=str)
    args = p.parse_args()
    summarize(Path(args.run_dir).resolve())


if __name__ == "__main__":
    main()
