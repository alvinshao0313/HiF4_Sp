"""AX5-R: low-cost S0 rule selection from discovery oracle."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Literal

SKIP_THRESHOLD = 0.05
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def run_rule_selection(
    discovery_rows: list[dict[str, Any]],
    *,
    split: Literal["discovery", "validation"] = "discovery",
) -> dict[str, Any]:
    """R0/R1/R2 rule search on AX1 discovery rows (metadata only)."""
    rows = [r for r in discovery_rows if r.get("split", "discovery") == split]
    if not rows:
        return {"status": "no_rows", "candidate_for_e2e": False}

    out_recoveries = [float(r["output_recovery"]) for r in rows if r.get("output_recovery") not in (None, "")]
    mean_out = statistics.mean(out_recoveries) if out_recoveries else 0.0
    if mean_out < SKIP_THRESHOLD:
        return {
            "status": "skipped_due_to_low_s0_recovery",
            "mean_output_recovery": mean_out,
            "candidate_for_e2e": False,
        }

    oracle_alphas = [float(r["alpha_oracle_nvfp4"]) for r in rows if r.get("alpha_oracle_nvfp4") not in (None, "")]
    r0_alpha = statistics.median(oracle_alphas) if oracle_alphas else 7.0

    by_proj: dict[str, list[float]] = {p: [] for p in PROJECTIONS}
    for r in rows:
        p = r.get("projection")
        if p in by_proj and r.get("alpha_oracle_nvfp4") not in (None, ""):
            by_proj[p].append(float(r["alpha_oracle_nvfp4"]))
    r1 = {p: (statistics.median(v) if v else r0_alpha) for p, v in by_proj.items()}

    # R2: choose alpha bins by max/RMS using oracle alphas as labels.
    # Score = mean closeness of bin-alpha to per-row oracle alpha, mapped to R_Y proxy.
    t1_candidates = [2.0, 3.0, 4.0, 5.0]
    t2_candidates = [4.0, 5.0, 6.0, 8.0]
    alpha_candidates = [5.0, 6.0, 7.0, 8.0, 9.0]
    best_r2: dict[str, Any] = {
        "t1": 3.0,
        "t2": 5.0,
        "alpha1": 8.0,
        "alpha2": 7.0,
        "alpha3": 6.0,
        "mean_R_Y": -1.0,
        "mean_alpha_abs_err": 1e9,
    }
    for t1 in t1_candidates:
        for t2 in t2_candidates:
            if t2 <= t1:
                continue
            for a1 in alpha_candidates:
                for a2 in alpha_candidates:
                    for a3 in alpha_candidates:
                        errs: list[float] = []
                        rys: list[float] = []
                        for r in rows:
                            mr = float(r.get("max_over_rms", 3.0) or 3.0)
                            alpha = a3 if mr >= t2 else (a2 if mr >= t1 else a1)
                            a_star = float(r.get("alpha_oracle_nvfp4", 7.0))
                            errs.append(abs(alpha - a_star))
                            # Proxy: scale oracle R_Y by how close rule alpha is to oracle.
                            r_y = float(r.get("output_recovery", 0.0) or 0.0)
                            scale = max(0.0, 1.0 - abs(alpha - a_star) / 3.0)
                            rys.append(r_y * scale)
                        if not rys:
                            continue
                        mean_err = statistics.mean(errs)
                        m = statistics.mean(rys)
                        if mean_err < best_r2["mean_alpha_abs_err"] - 1e-12 or (
                            abs(mean_err - best_r2["mean_alpha_abs_err"]) <= 1e-12 and m > best_r2["mean_R_Y"]
                        ):
                            best_r2 = {
                                "t1": t1,
                                "t2": t2,
                                "alpha1": a1,
                                "alpha2": a2,
                                "alpha3": a3,
                                "mean_R_Y": m,
                                "mean_alpha_abs_err": mean_err,
                            }

    oracle_mean_ry = statistics.mean(out_recoveries)
    # current alpha=7 has R_Y=0 by definition vs itself; gain is oracle recovery.
    oracle_gain = oracle_mean_ry
    rule_gain = float(best_r2["mean_R_Y"])
    candidate = rule_gain >= 0.5 * oracle_gain if oracle_gain > 0 else False

    return {
        "status": "completed",
        "mean_output_recovery": mean_out,
        "R0_global_alpha": r0_alpha,
        "R1_projection_alpha": r1,
        "R2_three_bin": best_r2,
        "baseline_output_recovery": 0.0,
        "oracle_output_recovery": oracle_mean_ry,
        "candidate_for_e2e": candidate,
    }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def build_root_cause_ranking(
    ax_rows: dict[str, list[dict[str, Any]]],
    a2_csv_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate AX1–AX4 + selected A2 variants into mechanism-level ranking."""
    buckets: dict[str, dict[str, Any]] = {}

    def _acc(name: str, source: str, r_y: float, *, note: str = "") -> None:
        b = buckets.setdefault(
            name,
            {"root_cause": name, "evidence_source": source, "values": [], "note": note},
        )
        b["values"].append(r_y)
        if note and not b.get("note"):
            b["note"] = note

    for r in ax_rows.get("ax1", []):
        if r.get("output_recovery") in (None, ""):
            continue
        _acc("S0 位置", "AX1", float(r["output_recovery"]))

    # AX2: G16/G32 recovery vs G64→A_N baseline
    for r in ax_rows.get("ax2", []):
        if r.get("R_Y") in (None, ""):
            continue
        gs = str(r.get("group_size", ""))
        if gs == "16":
            _acc("64-group 共享粒度(G16)", "AX2", float(r["R_Y"]))
        elif gs == "32":
            _acc("64-group 共享粒度(G32)", "AX2", float(r["R_Y"]))

    for r in ax_rows.get("ax4", []):
        if r.get("R_Y") in (None, ""):
            continue
        hybrid = str(r.get("hybrid", r.get("variant", ""))).upper()
        match = str(r.get("match_kind", "raw"))
        legal = str(r.get("is_valid_hardware_format", "")).lower() in {"true", "1"}
        if legal:
            continue
        if hybrid == "NH":
            _acc(f"NVFP4 Scale + HiF4 Payload (NH/{match})", "AX4", float(r["R_Y"]))
        elif hybrid == "HN":
            _acc(f"HiF4 Scale + NVFP4 Payload (HN/{match})", "AX4", float(r["R_Y"]))
        else:
            _acc(f"Scale-Payload 交互({hybrid}/{match})", "AX4", float(r["R_Y"]))

    if a2_csv_rows:
        by_var: dict[str, list[float]] = defaultdict(list)
        for r in a2_csv_rows:
            if r.get("exclude_from_main_rcf") in ("True", "true", True, "1"):
                continue
            key = f"{r.get('format')}:{r.get('variant')}"
            if r.get("R_cf_output") not in (None, ""):
                by_var[key].append(float(r["R_cf_output"]))
        mapping = {
            "hif4:continuous_s0": ("S0 表示精度", "A2_R_cf_vs_X"),
            "hif4:bf16_s0_no_e6m2": ("S0 E6M2 表示", "A2_R_cf_vs_X"),
            "hif4:oracle_e8": ("层级指数 e8", "A2_R_cf_vs_X"),
            "hif4:oracle_e4": ("层级指数 e4", "A2_R_cf_vs_X"),
            "hif4:oracle_e8_e4_joint": ("层级指数 e8+e4", "A2_R_cf_vs_X"),
            "hif4:continuous_payload_clipped": ("Payload/Clipping", "A2_R_cf_vs_X"),
            "nvfp4:nv_oracle_global_scale": ("NVFP4 global-scale Oracle", "A2_R_cf_vs_X"),
            "nvfp4:nv_continuous_local_scale": ("NVFP4 local-scale 连续化", "A2_R_cf_vs_X"),
            "nvfp4:nv_continuous_payload": ("NVFP4 payload 连续化", "A2_R_cf_vs_X"),
        }
        for key, vals in by_var.items():
            if key in mapping:
                name, note = mapping[key]
                for v in vals:
                    _acc(name, "A2", v, note=note)

    ranking = []
    for name, b in buckets.items():
        ranking.append(
            {
                "root_cause": name,
                "evidence_source": b["evidence_source"],
                "R_Y": _mean(b["values"]),
                "n": len(b["values"]),
                "note": b.get("note", ""),
            }
        )
    ranking.sort(key=lambda x: x["R_Y"], reverse=True)
    for i, row in enumerate(ranking, start=1):
        row["rank"] = i
    return ranking
