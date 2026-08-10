from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    tmp.replace(path)


def atomic_torch_save(path: Path, obj: Any) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(obj, tmp)
    tmp.replace(path)


CONDITION_SUMMARY_FIELDS = [
    "method_id",
    "output_keep_ratio",
    "input_keep_ratio",
    "output_overlap_to_ref_mean",
    "output_overlap_to_ref_median",
    "output_overlap_to_ref_p10",
    "output_overlap_to_ref_p90",
    "input_overlap_to_m1_mean",
    "input_overlap_to_m1_median",
    "input_overlap_to_m1_p10",
    "input_overlap_to_m1_p90",
    "input_overlap_to_conditional_oracle_mean",
    "input_overlap_to_conditional_oracle_median",
    "real_output_nrmse_mean",
    "real_output_nrmse_median",
    "nrmse_regret_vs_m1_mean",
    "nrmse_regret_vs_m1_median",
    "joint_sparse_output_nrmse_mean",
    "joint_sparse_output_nrmse_median",
    "num_output_blocks_kept",
    "num_input_blocks_kept",
    "num_compute_blocks",
    "compute_block_ratio",
    "spearman_mean",
    "kendall_mean",
]

LATENCY_FIELDS = [
    "method_id",
    "output_keep_ratio",
    "input_keep_ratio",
    "timing_scope",
    "median_ms",
    "p10_ms",
    "p90_ms",
    "repeats",
    "peak_memory_bytes",
]


def write_run_artifacts(
    output_dir: Path,
    *,
    manifest: dict[str, Any],
    config: dict[str, Any],
    masks: dict[str, Any],
    per_block_metrics: list[dict[str, Any]],
    condition_summary: list[dict[str, Any]],
    latency_rows: list[dict[str, Any]],
    aggregate_summary: dict[str, Any] | None,
    report_md: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    atomic_write_json(output_dir / "manifest.json", manifest)
    atomic_write_json(output_dir / "config.json", config)
    atomic_torch_save(output_dir / "masks.pt", masks)

    # jsonl
    lines = [json.dumps(r, ensure_ascii=False) for r in per_block_metrics]
    atomic_write_text(output_dir / "per_block_metrics.jsonl", "\n".join(lines) + ("\n" if lines else ""))

    atomic_write_csv(
        output_dir / "condition_summary.csv",
        condition_summary,
        CONDITION_SUMMARY_FIELDS,
    )
    atomic_write_csv(output_dir / "latency.csv", latency_rows, LATENCY_FIELDS)

    if aggregate_summary is not None:
        atomic_write_json(output_dir / "aggregate_summary.json", aggregate_summary)
    if report_md is not None:
        atomic_write_text(output_dir / "report.md", report_md)
