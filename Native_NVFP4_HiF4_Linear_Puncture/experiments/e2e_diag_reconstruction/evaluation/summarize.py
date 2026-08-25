"""Summarize reconstruction runs vs native NVFP4."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

BASELINE_METHOD_PRESETS = {
    "E0_native_nvfp4": "native_nvfp4",
    "E1_direct_hif4": "direct_hif4",
    "E2_r64_only": "r64_only",
}
FUSABLE_PRESETS = ("fusable", "fusable_r64")
ONLINE_PRESETS = ("online", "online_diag_then_r64", "online_r64_then_diag")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _arc_score(eval_dir: Path, name: str) -> float | None:
    p = eval_dir / "arc" / "metrics.json"
    if not p.is_file():
        return None
    scores = _load_json(p).get("scores", {})
    return scores.get(name)


def _metric_from_lighteval(blob: dict[str, Any]) -> float | None:
    results = blob.get("results") or blob
    if not isinstance(results, dict):
        return None
    for _task, trez in results.items():
        if not isinstance(trez, dict):
            continue
        for key in (
            "extractive_match",
            "extractive_match,none",
            "acc",
            "acc,none",
            "qem",
        ):
            if key in trez and isinstance(trez[key], (int, float)):
                return float(trez[key])
    return None


def _lighteval_score(eval_dir: Path, subdir: str) -> float | None:
    p = eval_dir / subdir / "metrics.json"
    if not p.is_file():
        return None
    return _metric_from_lighteval(_load_json(p))


def structure_preset(row: dict[str, Any]) -> str:
    method = str(row.get("method") or "")
    if method in BASELINE_METHOD_PRESETS:
        return BASELINE_METHOD_PRESETS[method]
    mode = row.get("diag_mode")
    use_r64 = bool(row.get("use_r64"))
    rot = row.get("rot_order")
    if mode == "fusable":
        return "fusable_r64" if use_r64 else "fusable"
    if mode == "online":
        if not use_r64:
            return "online"
        if rot == "rot_then_diag":
            return "online_r64_then_diag"
        return "online_diag_then_r64"
    raise ValueError(f"cannot map structure_preset for method={method!r} diag_mode={mode!r}")


def layer_stats(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return {}
    blob = _load_json(summary_path)
    layers = blob.get("layers", [])
    recoveries = [float(x.get("recovery_vs_identity", 0.0)) for x in layers]
    would = sum(1 for x in layers if x.get("would_rollback"))
    return {
        "accepted": int(blob.get("accepted", 0)),
        "rollback": int(blob.get("rollback", 0)),
        "would_rollback_count": would,
        "n_layers": len(layers),
        "recoveries": recoveries,
        "mean_recovery": (sum(recoveries) / len(recoveries)) if recoveries else None,
        "median_recovery": statistics.median(recoveries) if recoveries else None,
    }


def collect_run_row(run_dir: Path, method: str, nvfp4: dict[str, float | None] | None = None) -> dict[str, Any]:
    eval_dir = run_dir / "eval"
    cfg_path = run_dir / "config.json"
    cfg = _load_json(cfg_path) if cfg_path.is_file() else {}
    row = {
        "method": method,
        "run_dir": str(run_dir),
        "diag_mode": cfg.get("diag_mode"),
        "use_r64": cfg.get("use_r64"),
        "rot_order": cfg.get("rot_order"),
        "diag_train_scope": cfg.get("diag_train_scope"),
        "recon_loss": cfg.get("recon_loss"),
        "diag_log2_clamp": cfg.get("diag_log2_clamp"),
        "fusable_diag_components": cfg.get("fusable_diag_components"),
        "calib_input_mode": cfg.get("calib_input_mode"),
        "layer_rollback": cfg.get("layer_rollback"),
        "diag_lr": cfg.get("diag_lr"),
        "calib_source": cfg.get("calib_source"),
        "teacher_trace_policy": cfg.get("teacher_trace_policy"),
        "arc_easy": _arc_score(eval_dir, "arc_easy"),
        "arc_challenge": _arc_score(eval_dir, "arc_challenge"),
        "mmlu_pro_300": _lighteval_score(eval_dir, "mmlu_pro"),
        "aime25_avg5": _lighteval_score(eval_dir, "aime25"),
    }
    row["structure_preset"] = (
        BASELINE_METHOD_PRESETS[method]
        if method in BASELINE_METHOD_PRESETS
        else structure_preset(row)
    )
    row.update(layer_stats(run_dir))
    if nvfp4 is not None:
        for key in ("arc_easy", "arc_challenge", "mmlu_pro_300", "aime25_avg5"):
            cur = row.get(key)
            base = nvfp4.get(key)
            row[f"delta_{key}_vs_nvfp4_pp"] = (
                None if cur is None or base is None else (cur - base) * 100.0
            )
    return row


def choose_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Rank by MMLU-Pro300, then ARC-Challenge, then ARC-Easy."""

    def key(r: dict[str, Any]):
        return (
            r.get("mmlu_pro_300") if r.get("mmlu_pro_300") is not None else -1.0,
            r.get("arc_challenge") if r.get("arc_challenge") is not None else -1.0,
            r.get("arc_easy") if r.get("arc_easy") is not None else -1.0,
        )

    return max(rows, key=key)


def _best_preset_among(rows: list[dict[str, Any]], allowed: tuple[str, ...]) -> str | None:
    subset = [r for r in rows if r.get("structure_preset") in allowed]
    if not subset:
        return None
    return str(choose_best(subset)["structure_preset"])


def summarize_runs(run_map: dict[str, str], nvfp4_method: str = "E0_native_nvfp4") -> dict[str, Any]:
    parsed = {name: collect_run_row(Path(path), name) for name, path in run_map.items()}
    nv = parsed.get(nvfp4_method)
    nv_scores = None
    if nv is not None:
        nv_scores = {
            k: nv.get(k) for k in ("arc_easy", "arc_challenge", "mmlu_pro_300", "aime25_avg5")
        }
    rows = []
    for name, path in run_map.items():
        rows.append(collect_run_row(Path(path), name, nv_scores))
    fusable = [r for r in rows if r.get("diag_mode") == "fusable"]
    online = [r for r in rows if r.get("diag_mode") == "online"]
    best_fusable_preset = _best_preset_among(rows, FUSABLE_PRESETS)
    best_online_preset = _best_preset_among(rows, ONLINE_PRESETS)
    overall_candidates = [
        r
        for r in rows
        if r.get("structure_preset") in {best_fusable_preset, best_online_preset}
        and r.get("structure_preset") is not None
    ]
    best_overall_preset = (
        str(choose_best(overall_candidates)["structure_preset"]) if overall_candidates else None
    )
    return {
        "rows": rows,
        "best_fusable": choose_best(fusable) if fusable else None,
        "best_online": choose_best(online) if online else None,
        "best_fusable_preset": best_fusable_preset,
        "best_online_preset": best_online_preset,
        "best_overall_preset": best_overall_preset,
    }


def write_chinese_report(summary: dict[str, Any], out_path: Path) -> None:
    lines = ["# E2E DIAG 重建结果总表", ""]
    lines.append(
        "| 方法 | preset | source | policy | accepted | rollback | would_rollback | "
        "mean_rec | ARC-E | ARC-C | MMLU-Pro300 | AIME25 |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary["rows"]:
        lines.append(
            "| {method} | {structure_preset} | {calib_source} | {teacher_trace_policy} | "
            "{accepted} | {rollback} | {would_rollback_count} | {mean_recovery} | "
            "{arc_easy} | {arc_challenge} | {mmlu_pro_300} | {aime25_avg5} |".format(
                **{k: r.get(k) for k in (
                    "method",
                    "structure_preset",
                    "calib_source",
                    "teacher_trace_policy",
                    "accepted",
                    "rollback",
                    "would_rollback_count",
                    "mean_recovery",
                    "arc_easy",
                    "arc_challenge",
                    "mmlu_pro_300",
                    "aime25_avg5",
                )}
            )
        )
    lines.append("")
    if summary.get("best_fusable_preset"):
        lines.append(f"最佳 fusable preset：{summary['best_fusable_preset']}")
    if summary.get("best_online_preset"):
        lines.append(f"最佳 online preset：{summary['best_online_preset']}")
    if summary.get("best_overall_preset"):
        lines.append(f"最佳 overall preset：{summary['best_overall_preset']}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
