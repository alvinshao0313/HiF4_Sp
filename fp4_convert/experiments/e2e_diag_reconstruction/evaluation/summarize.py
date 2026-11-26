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


def _runtime_metric_allowed(eval_dir: Path, payload: dict[str, Any]) -> bool:
    guard_path = eval_dir / "runtime_abi_guard.json"
    if not guard_path.is_file():
        return True
    guard = _load_json(guard_path)
    if not guard.get("runtime_abi_required"):
        return True
    expected = int(guard.get("runtime_abi_version", -1))
    try:
        got = int(payload.get("runtime_abi_version", -2))
    except (TypeError, ValueError):
        return False
    return got == expected


def _arc_score(eval_dir: Path, name: str) -> float | None:
    p = eval_dir / "arc" / "metrics.json"
    if not p.is_file():
        return None
    payload = _load_json(p)
    if not _runtime_metric_allowed(eval_dir, payload):
        return None
    scores = payload.get("scores", {})
    return scores.get(name)


def _metric_from_lighteval(blob: dict[str, Any]) -> float | None:
    results = blob.get("results") or blob
    if not isinstance(results, dict):
        return None
    for _task, trez in results.items():
        if not isinstance(trez, dict):
            continue
        for key in (
            "codegen_pass@1:16",
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
    payload = _load_json(p)
    if not _runtime_metric_allowed(eval_dir, payload):
        return None
    return _metric_from_lighteval(payload)


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


def _layer_recovery(layer: dict[str, Any]) -> float | None:
    if "recovery_vs_identity" in layer:
        return float(layer["recovery_vs_identity"])
    identity = layer.get("identity_val_loss")
    adopted = layer.get("adopted_val_loss", layer.get("best_val_loss"))
    if identity is None or adopted is None:
        return None
    identity_f = float(identity)
    if identity_f <= 0:
        return 0.0
    return max(0.0, 1.0 - float(adopted) / identity_f)


def _mean_or_none(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def layer_stats(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return {}
    blob = _load_json(summary_path)
    layers = blob.get("layers", [])
    recoveries = [v for v in (_layer_recovery(x) for x in layers) if v is not None]
    candidate_losses = [
        float(x["candidate_best_val_loss"])
        for x in layers
        if x.get("candidate_best_val_loss") is not None
    ]
    adopted_losses = [
        float(x["adopted_val_loss"])
        for x in layers
        if x.get("adopted_val_loss") is not None
    ]
    candidate_kls = [
        float(x["candidate_best_router_kl"])
        for x in layers
        if x.get("candidate_best_router_kl") is not None
    ]
    adopted_kls = [
        float(x["adopted_router_kl"])
        for x in layers
        if x.get("adopted_router_kl") is not None
    ]
    accepted_count = sum(1 for x in layers if x.get("accepted"))
    rollback_count = sum(1 for x in layers if x.get("rollback"))
    if not layers and ("accepted" in blob or "rollback" in blob):
        accepted_count = int(blob.get("accepted", 0))
        rollback_count = int(blob.get("rollback", 0))
    would_legacy = sum(1 for x in layers if x.get("would_rollback"))
    return {
        "accepted": accepted_count,
        "rollback": rollback_count,
        "accepted_count": accepted_count,
        "rollback_count": rollback_count,
        "would_rollback_count": would_legacy,
        "loss_would_rollback_count": sum(1 for x in layers if x.get("loss_would_rollback")),
        "loss_rollback_applied_count": sum(1 for x in layers if x.get("loss_rollback_applied")),
        "router_would_rollback_count": sum(1 for x in layers if x.get("router_would_rollback")),
        "router_rollback_applied_count": sum(1 for x in layers if x.get("router_rollback_applied")),
        "candidate_best_val_loss_mean": _mean_or_none(candidate_losses),
        "candidate_best_val_loss_median": _median_or_none(candidate_losses),
        "adopted_val_loss_mean": _mean_or_none(adopted_losses),
        "adopted_val_loss_median": _median_or_none(adopted_losses),
        "candidate_best_router_kl_mean": _mean_or_none(candidate_kls),
        "candidate_best_router_kl_median": _median_or_none(candidate_kls),
        "adopted_router_kl_mean": _mean_or_none(adopted_kls),
        "adopted_router_kl_median": _median_or_none(adopted_kls),
        "candidate_router_topk_mismatch_tokens": sum(
            int(x.get("candidate_router_topk_mismatch_tokens", 0)) for x in layers
        ),
        "candidate_router_topk_mismatch_ratio_mean": _mean_or_none(
            [
                float(x["candidate_router_topk_mismatch_ratio"])
                for x in layers
                if x.get("candidate_router_topk_mismatch_ratio") is not None
            ]
        ),
        "router_topk_mismatch_tokens": sum(
            int(x.get("router_topk_mismatch_tokens", 0)) for x in layers
        ),
        "router_topk_mismatch_ratio_mean": _mean_or_none(
            [
                float(x["router_topk_mismatch_ratio"])
                for x in layers
                if x.get("router_topk_mismatch_ratio") is not None
            ]
        ),
        "n_layers": len(layers),
        "recoveries": recoveries,
        "mean_recovery": _mean_or_none(recoveries),
        "median_recovery": _median_or_none(recoveries),
    }


def is_candidate_diagnostic(row: dict[str, Any]) -> bool:
    if row.get("is_candidate_diagnostic") is True:
        return True
    return str(row.get("artifact_diag_variant") or "adopted") == "candidate"


def collect_run_row(run_dir: Path, method: str, nvfp4: dict[str, float | None] | None = None) -> dict[str, Any]:
    eval_dir = run_dir / "eval"
    cfg_path = run_dir / "config.json"
    cfg = _load_json(cfg_path) if cfg_path.is_file() else {}
    artifact_diag_variant = str(cfg.get("artifact_diag_variant") or "adopted")
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
        "loss_rollback": cfg.get("loss_rollback"),
        "router_rollback": cfg.get("router_rollback"),
        "router_align_type": cfg.get("router_align_type"),
        "router_align_temperature": cfg.get("router_align_temperature"),
        "router_align_loss_weight": cfg.get("router_align_loss_weight"),
        "artifact_diag_variant": artifact_diag_variant,
        "is_candidate_diagnostic": artifact_diag_variant == "candidate",
        "diag_lr": cfg.get("diag_lr"),
        "optimizer": cfg.get("optimizer"),
        "weight_decay": cfg.get("weight_decay"),
        "calib_source": cfg.get("calib_source"),
        "teacher_trace_policy": cfg.get("teacher_trace_policy"),
        "arc_easy": _arc_score(eval_dir, "arc_easy"),
        "arc_challenge": _arc_score(eval_dir, "arc_challenge"),
        "mmlu_pro_300": _lighteval_score(eval_dir, "mmlu_pro"),
        "aime25_avg5": _lighteval_score(eval_dir, "aime25"),
        "livecodebench": _lighteval_score(eval_dir, "livecodebench"),
    }
    row["structure_preset"] = (
        BASELINE_METHOD_PRESETS[method]
        if method in BASELINE_METHOD_PRESETS
        else structure_preset(row)
    )
    row.update(layer_stats(run_dir))
    if nvfp4 is not None:
        for key in ("arc_easy", "arc_challenge", "mmlu_pro_300", "aime25_avg5", "livecodebench"):
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


def _ranking_eligible(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if not is_candidate_diagnostic(r)]


def _best_preset_among(rows: list[dict[str, Any]], allowed: tuple[str, ...]) -> str | None:
    subset = [r for r in _ranking_eligible(rows) if r.get("structure_preset") in allowed]
    if not subset:
        return None
    return str(choose_best(subset)["structure_preset"])


def summarize_runs(run_map: dict[str, str], nvfp4_method: str = "E0_native_nvfp4") -> dict[str, Any]:
    parsed = {name: collect_run_row(Path(path), name) for name, path in run_map.items()}
    nv = parsed.get(nvfp4_method)
    nv_scores = None
    if nv is not None:
        nv_scores = {
            k: nv.get(k) for k in ("arc_easy", "arc_challenge", "mmlu_pro_300", "aime25_avg5", "livecodebench")
        }
    rows = []
    for name, path in run_map.items():
        rows.append(collect_run_row(Path(path), name, nv_scores))
    eligible = _ranking_eligible(rows)
    fusable = [r for r in eligible if r.get("diag_mode") == "fusable"]
    online = [r for r in eligible if r.get("diag_mode") == "online"]
    best_fusable_preset = _best_preset_among(rows, FUSABLE_PRESETS)
    best_online_preset = _best_preset_among(rows, ONLINE_PRESETS)
    overall_candidates = [
        r
        for r in eligible
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
        "| 方法 | preset | variant | source | loss_rb | router_rb | accepted | rollback | "
        "loss_rb_applied | router_rb_applied | mean_rec | ARC-E | ARC-C | MMLU-Pro300 | AIME25 | LiveCodeBench |"
    )
    lines.append(
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in summary["rows"]:
        lines.append(
            "| {method} | {structure_preset} | {artifact_diag_variant} | {calib_source} | "
            "{loss_rollback} | {router_rollback} | {accepted_count} | {rollback_count} | "
            "{loss_rollback_applied_count} | {router_rollback_applied_count} | {mean_recovery} | "
            "{arc_easy} | {arc_challenge} | {mmlu_pro_300} | {aime25_avg5} | {livecodebench} |".format(
                **{
                    k: r.get(k)
                    for k in (
                        "method",
                        "structure_preset",
                        "artifact_diag_variant",
                        "calib_source",
                        "loss_rollback",
                        "router_rollback",
                        "accepted_count",
                        "rollback_count",
                        "loss_rollback_applied_count",
                        "router_rollback_applied_count",
                        "mean_recovery",
                        "arc_easy",
                        "arc_challenge",
                        "mmlu_pro_300",
                        "aime25_avg5",
                        "livecodebench",
                    )
                }
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
