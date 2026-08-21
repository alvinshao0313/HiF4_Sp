"""Read-only reporting for HiF4 deployment-equivalent scaling experiments."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

REPO_ROOT = Path("/home/shaoyuantian/program/HiF4_Sp")
AX_ROOT = REPO_ROOT / "Inference_Paradigm_Conversion" / "results" / "20260811T_ax_final_consolidated"


def _load_pt(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"expected dict artifact at {path}")
    return value


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise TypeError(f"expected dict JSON at {path}")
    return value


def _recover(num: float, den: float) -> float | None:
    if den < 1e-12:
        return None
    return 1.0 - num / den


def _mean(values: Iterable[float]) -> float | None:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else None


def _sum_numeric(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                out[key] += float(value)
    return dict(out)


def _recipe_meta(candidates: dict[str, Any] | None, rid: str) -> dict[str, Any]:
    if not candidates:
        return {}
    rec = candidates.get("recipes", {}).get(rid, {})
    return rec if isinstance(rec, dict) else {}


def _aggregate_activation(
    artifact: dict[str, Any] | None,
    candidates: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not artifact:
        return []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in artifact.get("records", {}).values():
        if isinstance(rec, dict) and "domain" in rec and "recipe_id" in rec:
            groups[(str(rec["domain"]), str(rec["recipe_id"]))].append(rec)
    rows: list[dict[str, Any]] = []
    for (domain, rid), recs in sorted(groups.items()):
        s = _sum_numeric(recs)
        meta = _recipe_meta(candidates, rid)
        numel = s.get("activation_numel", 0.0)
        group_count = s.get("dispersion_group_count", 0.0)
        payload_counts = [s.get(f"payload_bin_{i}_count", 0.0) for i in range(8)]
        payload_total = sum(payload_counts)
        probs = [x / payload_total for x in payload_counts if x > 0 and payload_total > 0]
        entropy = -sum(p * math.log2(p) for p in probs) if probs else 0.0
        row: dict[str, Any] = {
            "domain": domain,
            "recipe_id": rid,
            "kind": meta.get("kind", ""),
            "granularity": meta.get("granularity", ""),
            "alpha": meta.get("alpha", ""),
            "deployable": bool(meta.get("deployable", True)),
            "diagnostic": meta.get("diagnostic", ""),
            "activation_R_Y_conv": _recover(
                s.get("activation_conv_error_sum", math.inf),
                s.get("baseline_activation_conv_error_sum", 0.0),
            ),
            "activation_R_Y_local": _recover(
                s.get("activation_local_error_sum", math.inf),
                s.get("baseline_activation_local_error_sum", 0.0),
            ),
            "zero_rate": s.get("hif4_zero_count", 0.0) / numel if numel else None,
            "nv_nonzero_to_hif4_zero_rate": s.get("nv_nonzero_to_hif4_zero_count", 0.0) / numel if numel else None,
            "boundary_rate": s.get("hif4_boundary_count", 0.0) / numel if numel else None,
            "payload_entropy_bits": entropy,
            "effective_codes": sum(1 for x in payload_counts if x > 0),
            "num_activation_values": int(numel),
        }
        for sub in (16, 8, 4):
            row[f"before_sub{sub}_log2_amax_range"] = (
                s.get(f"before_sub{sub}_log2_amax_range_sum", 0.0) / group_count
                if group_count
                else None
            )
            row[f"after_sub{sub}_log2_amax_range"] = (
                s.get(f"after_sub{sub}_log2_amax_range_sum", 0.0) / group_count
                if group_count
                else None
            )
        rows.append(row)
    return rows


def _aggregate_full(
    artifact: dict[str, Any] | None,
    candidates: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not artifact:
        return []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in artifact.get("records", {}).values():
        if isinstance(rec, dict) and "domain" in rec and "recipe_id" in rec:
            groups[(str(rec["domain"]), str(rec["recipe_id"]))].append(rec)
    rows: list[dict[str, Any]] = []
    for (domain, rid), recs in sorted(groups.items()):
        s = _sum_numeric(recs)
        meta = _recipe_meta(candidates, rid)
        wf_den = s.get("baseline_weight_local_error_sum", 0.0)
        weight_energy = s.get("weight_ref_energy_once", 0.0)
        weight_error = s.get("weight_error_sum_once", 0.0)
        weight_nmse = weight_error / weight_energy if weight_energy >= 1e-12 else None
        weight_numel = s.get("weight_numel_once", 0.0)
        row = {
            "domain": domain,
            "recipe_id": rid,
            "kind": meta.get("kind", ""),
            "granularity": meta.get("granularity", ""),
            "alpha": meta.get("alpha", ""),
            "deployable": bool(meta.get("deployable", True)),
            "diagnostic": meta.get("diagnostic", ""),
            "joint_R_Y_conv": _recover(
                s.get("joint_conv_error_sum", math.inf), s.get("baseline_conv_error_sum", 0.0)
            ),
            "joint_R_Y_local": _recover(
                s.get("joint_local_error_sum", math.inf), s.get("baseline_local_error_sum", 0.0)
            ),
            "weight_functional_R_Y_local": _recover(
                s.get("weight_local_error_sum", math.inf), wf_den
            ),
            "weight_functional_error_ratio": (
                s.get("weight_local_error_sum", math.inf) / wf_den if wf_den >= 1e-12 else None
            ),
            "weight_nmse": weight_nmse,
            "weight_sqnr_db": (-10.0 * math.log10(weight_nmse)) if weight_nmse is not None and weight_nmse > 0 else None,
            "weight_zero_rate": s.get("weight_zero_count_once", 0.0) / weight_numel if weight_numel else None,
            "weight_boundary_rate": s.get("weight_boundary_count_once", 0.0) / weight_numel if weight_numel else None,
            "joint_numel": int(s.get("joint_numel", 0.0)),
        }
        rows.append(row)
    return rows


def _candidate_scale_rows(candidates: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not candidates:
        return []
    rows: list[dict[str, Any]] = []
    accum: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for layer, domains in candidates.get("scales", {}).items():
        for domain, recipes in domains.items():
            for rid, d in recipes.items():
                x = d.detach().float().reshape(-1)
                meta = _recipe_meta(candidates, rid)
                lo = float(meta.get("min_scale", 0.5))
                hi = float(meta.get("max_scale", 2.0))
                dst = accum[(domain, rid)]
                dst["count"] += float(x.numel())
                dst["clamp_low"] += float(torch.isclose(x, torch.tensor(lo)).sum().item())
                dst["clamp_high"] += float(torch.isclose(x, torch.tensor(hi)).sum().item())
                dst["sum_log2"] += float(torch.log2(x).sum().item())
                dst["sum_sq_log2"] += float(torch.log2(x).pow(2).sum().item())
                dst["min"] = min(float(dst.get("min", float("inf"))), float(x.min().item()))
                dst["max"] = max(float(dst.get("max", float("-inf"))), float(x.max().item()))
    for (domain, rid), rec in sorted(accum.items()):
        n = rec["count"]
        mean_log2 = rec["sum_log2"] / n
        var = max(0.0, rec["sum_sq_log2"] / n - mean_log2 * mean_log2)
        rows.append(
            {
                "domain": domain,
                "recipe_id": rid,
                "scale_min_observed": rec["min"],
                "scale_max_observed": rec["max"],
                "mean_log2_scale": mean_log2,
                "std_log2_scale": math.sqrt(var),
                "clamp_low_rate": rec["clamp_low"] / n,
                "clamp_high_rate": rec["clamp_high"] / n,
                "clamp_total_rate": (rec["clamp_low"] + rec["clamp_high"]) / n,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _old_ax_summary() -> dict[str, Any]:
    ax1 = _read_csv(AX_ROOT / "ax1_s0_divisor_oracle.csv")
    ax2 = _read_csv(AX_ROOT / "ax2_group_size_ablation.csv")
    ax1_vals = [float(r["output_recovery"]) for r in ax1 if r.get("output_recovery")]
    g16 = [float(r["R_Y"]) for r in ax2 if r.get("variant") == "G16" and r.get("R_Y")]
    g32 = [float(r["R_Y"]) for r in ax2 if r.get("variant") == "G32" and r.get("R_Y")]
    return {
        "AX1_mean_output_recovery_activation_only": _mean(ax1_vals),
        "AX2_G16_mean_R_Y": _mean(g16),
        "AX2_G32_mean_R_Y": _mean(g32),
    }


def _best(rows: list[dict[str, Any]], *, domain: str | None = None, kind: str | None = None, field: str) -> dict[str, Any] | None:
    options = []
    for row in rows:
        if domain is not None and row.get("domain") != domain:
            continue
        if kind is not None and row.get("kind") != kind:
            continue
        value = row.get(field)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            options.append(row)
    return max(options, key=lambda r: (float(r[field]), str(r.get("recipe_id", "")))) if options else None


def _safe_name(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_")


def _plot(run_dir: Path, name: str, draw) -> str:
    import matplotlib.pyplot as plt

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    draw(ax)
    fig.tight_layout()
    path = fig_dir / name
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path.relative_to(run_dir))


def _make_figures(
    run_dir: Path,
    activation_rows: list[dict[str, Any]],
    full_rows: list[dict[str, Any]],
    scale_rows: list[dict[str, Any]],
    policy: dict[str, Any] | None,
    validation: dict[str, Any] | None,
    all_policy: dict[str, Any] | None,
    trajectory: dict[str, Any] | None,
    e2e: dict[str, Any] | None,
    ax_old: dict[str, Any],
) -> list[str]:
    figures: list[str] = []

    def fig1(ax):
        labels: list[str] = []
        values: list[float] = []
        ax1 = ax_old.get("AX1_mean_output_recovery_activation_only")
        if isinstance(ax1, (int, float)):
            labels.append("AX1 act-only")
            values.append(float(ax1))
        for kind, label in (("pts_layer", "PTS-L act"), ("phase_g64", "G64 phase act")):
            row = _best(activation_rows, kind=kind, field="activation_R_Y_conv")
            if row:
                labels.append(label)
                values.append(float(row["activation_R_Y_conv"]))
        for kind, label in (("pts_layer", "PTS-L joint"), ("phase_g64", "G64 phase joint")):
            row = _best(full_rows, kind=kind, field="joint_R_Y_conv")
            if row:
                labels.append(label)
                values.append(float(row["joint_R_Y_conv"]))
        ax.bar(labels, values)
        ax.axhline(0.0, linewidth=1)
        ax.set_ylabel("R_Y_conv")
        ax.set_title("PTS / scale-phase recovery")
        ax.tick_params(axis="x", rotation=25)
    figures.append(_plot(run_dir, "fig_es1_pts_recovery.png", fig1))

    ordinary = [r for r in activation_rows if r.get("kind") == "equalize" and r.get("deployable") and not r.get("diagnostic")]
    if ordinary:
        def fig2(ax):
            for domain in sorted({r["domain"] for r in ordinary}):
                xs, ys = [], []
                for gran in (1, 4, 8, 16):
                    opts = [r for r in ordinary if r["domain"] == domain and int(r["granularity"]) == gran and r.get("activation_R_Y_conv") is not None]
                    if opts:
                        best = max(opts, key=lambda r: float(r["activation_R_Y_conv"]))
                        xs.append(gran)
                        ys.append(float(best["activation_R_Y_conv"]))
                if xs:
                    ax.plot(xs, ys, marker="o", label=domain)
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 4, 8, 16], labels=["1", "4", "8", "16"])
            ax.set_xlabel("Equalization granularity")
            ax.set_ylabel("Best activation-only R_Y_conv")
            ax.legend()
            ax.set_title("Recovery vs granularity")
        figures.append(_plot(run_dir, "fig_es2_recovery_vs_granularity.png", fig2))

        def fig3(ax):
            by_gran = []
            for gran in (1, 4, 8, 16):
                opts = [r for r in ordinary if int(r["granularity"]) == gran and r.get("zero_rate") is not None]
                if opts:
                    by_gran.append((gran, _mean(float(r["zero_rate"]) for r in opts)))
            ax.plot([x for x, _ in by_gran], [y for _, y in by_gran], marker="o")
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 4, 8, 16], labels=["1", "4", "8", "16"])
            ax.set_xlabel("Granularity")
            ax.set_ylabel("HiF4 zero rate")
            ax.set_title("Zeroing vs granularity")
        figures.append(_plot(run_dir, "fig_es2_zero_rate_vs_granularity.png", fig3))

    selected_recipes = (policy or {}).get("domain_recipes", {})
    selected_act = [r for r in activation_rows if selected_recipes.get(r["domain"]) == r["recipe_id"]]
    if selected_act:
        def fig4(ax):
            labels = ["sub16", "sub8", "sub4"]
            before = []
            after = []
            for sub in (16, 8, 4):
                before.append(_mean(float(r[f"before_sub{sub}_log2_amax_range"]) for r in selected_act if r.get(f"before_sub{sub}_log2_amax_range") is not None) or 0.0)
                after.append(_mean(float(r[f"after_sub{sub}_log2_amax_range"]) for r in selected_act if r.get(f"after_sub{sub}_log2_amax_range") is not None) or 0.0)
            x = list(range(len(labels)))
            ax.plot(x, before, marker="o", label="before")
            ax.plot(x, after, marker="o", label="after")
            ax.set_xticks(x, labels=labels)
            ax.set_ylabel("mean log2 amax range")
            ax.legend()
            ax.set_title("64-group internal dispersion")
        figures.append(_plot(run_dir, "fig_es2_dispersion_before_after.png", fig4))

    act_map = {(r["domain"], r["recipe_id"]): r for r in activation_rows}
    joined = [(r, act_map.get((r["domain"], r["recipe_id"]))) for r in full_rows]
    joined = [(f, a) for f, a in joined if a and f.get("joint_R_Y_conv") is not None and f.get("weight_functional_error_ratio") is not None and a.get("activation_R_Y_conv") is not None]
    if joined:
        def fig5(ax):
            for f, a in joined:
                ax.scatter(float(a["activation_R_Y_conv"]), float(f["weight_functional_error_ratio"]))
            ax.axhline(1.0, linewidth=1)
            ax.set_xlabel("activation-only R_Y_conv")
            ax.set_ylabel("weight functional error / standard")
            ax.set_title("Activation gain vs weight penalty")
        figures.append(_plot(run_dir, "fig_es3_activation_weight_tradeoff.png", fig5))

        def fig6(ax):
            groups = sorted(
                {
                    (f["domain"], int(f["granularity"]))
                    for f, _ in joined
                    if isinstance(f.get("alpha"), (int, float))
                    and isinstance(f.get("granularity"), (int, float))
                    and f.get("kind") == "equalize"
                }
            )
            for domain, granularity in groups:
                pts = sorted(
                    (float(f["alpha"]), float(f["joint_R_Y_conv"]))
                    for f, _ in joined
                    if f["domain"] == domain
                    and int(f["granularity"]) == granularity
                    and isinstance(f.get("alpha"), (int, float))
                    and f.get("kind") == "equalize"
                )
                if pts:
                    ax.plot(
                        [p[0] for p in pts],
                        [p[1] for p in pts],
                        marker="o",
                        label=f"{domain}-g{granularity}",
                    )
            ax.set_xlabel("alpha")
            ax.set_ylabel("joint W4A4 R_Y_conv")
            ax.legend()
            ax.set_title("Full W4A4 recovery vs alpha")
        figures.append(_plot(run_dir, "fig_es3_full_output_vs_alpha.png", fig6))

    if scale_rows:
        def fig7(ax):
            chosen = [r for r in scale_rows if selected_recipes.get(r["domain"]) == r["recipe_id"]]
            if not chosen:
                chosen = sorted(scale_rows, key=lambda r: float(r["clamp_total_rate"]), reverse=True)[:12]
            labels = [f"{r['domain']}:{r['recipe_id']}" for r in chosen]
            ax.bar(labels, [float(r["clamp_total_rate"]) for r in chosen])
            ax.set_ylabel("scale clamp rate")
            ax.tick_params(axis="x", rotation=70)
            ax.set_title("Scale bound saturation")
        figures.append(_plot(run_dir, "fig_es3_scale_clamp_rate.png", fig7))

    o_free = [r for r in activation_rows if r.get("domain") == "o_in" and r.get("diagnostic") == "o_free_oracle"]
    o_tied = [r for r in ordinary if r.get("domain") == "o_in"] if ordinary else []
    if o_free and o_tied:
        def fig8(ax):
            xs = [1, 4, 8, 16]
            free_y, tied_y = [], []
            for gran in xs:
                fopts = [r for r in o_free if int(r["granularity"]) == gran and r.get("activation_R_Y_conv") is not None]
                topts = [r for r in o_tied if int(r["granularity"]) == gran and r.get("activation_R_Y_conv") is not None]
                free_y.append(max(float(r["activation_R_Y_conv"]) for r in fopts) if fopts else float("nan"))
                tied_y.append(max(float(r["activation_R_Y_conv"]) for r in topts) if topts else float("nan"))
            ax.plot(xs, free_y, marker="o", label="O free oracle")
            ax.plot(xs, tied_y, marker="o", label="GQA tied")
            ax.set_xscale("log", base=2)
            ax.set_xticks(xs, labels=[str(x) for x in xs])
            ax.set_ylabel("activation-only R_Y_conv")
            ax.legend()
            ax.set_title("O free oracle vs GQA-tied deployable")
        figures.append(_plot(run_dir, "fig_es4_o_free_vs_gqa_tied.png", fig8))

    if validation and policy:
        def fig9(ax):
            discovery = []
            valid = []
            labels = []
            for li, block in sorted((policy.get("block_summary") or {}).items(), key=lambda kv: int(kv[0])):
                vm = (validation.get("combined_by_layer") or {}).get(li)
                if vm:
                    labels.append(li)
                    discovery.append(float(block["final_metrics"]["mean_R_Y_conv"]))
                    valid.append(float(vm["mean_R_Y_conv"]))
            x = list(range(len(labels)))
            ax.plot(x, discovery, marker="o", label="discovery")
            ax.plot(x, valid, marker="o", label="validation")
            ax.set_xticks(x, labels=labels)
            ax.set_xlabel("layer")
            ax.set_ylabel("block R_Y_conv")
            ax.legend()
            ax.set_title("Discovery vs validation")
        figures.append(_plot(run_dir, "fig_es6_discovery_validation.png", fig9))

    if full_rows and selected_recipes:
        def fig10(ax):
            full_art = _load_pt(run_dir / "es_full_eval_merged.pt") or {}
            rows = full_art.get("records", {})
            sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
            for rec in rows.values():
                if not isinstance(rec, dict):
                    continue
                if selected_recipes.get(rec.get("domain")) != rec.get("recipe_id"):
                    continue
                p = str(rec.get("projection"))
                for key in ("joint_conv_error_sum", "baseline_conv_error_sum"):
                    if isinstance(rec.get(key), (int, float)):
                        sums[p][key] += float(rec[key])
            labels, vals = [], []
            for p, s in sorted(sums.items()):
                r = _recover(s["joint_conv_error_sum"], s["baseline_conv_error_sum"])
                if r is not None:
                    labels.append(p)
                    vals.append(r)
            ax.bar(labels, vals)
            ax.axhline(0, linewidth=1)
            ax.set_ylabel("joint R_Y_conv")
            ax.tick_params(axis="x", rotation=30)
            ax.set_title("Recovery by projection")
        figures.append(_plot(run_dir, "fig_es6_recovery_by_projection.png", fig10))

    if all_policy:
        layer_checks = all_policy.get("layer_checks", {})
        if layer_checks:
            def fig11(ax):
                xs, ys = [], []
                for li, rec in sorted(layer_checks.items(), key=lambda kv: int(kv[0])):
                    xs.append(int(li))
                    ys.append(float(rec["final_metrics"]["mean_R_Y_conv"]))
                ax.plot(xs, ys, marker="o")
                ax.axhline(-0.05, linewidth=1)
                ax.axhline(0.0, linewidth=1)
                ax.set_xlabel("decoder layer")
                ax.set_ylabel("final block R_Y_conv")
                ax.set_title("All-layer block recovery")
            figures.append(_plot(run_dir, "fig_es65_all_layer_block_recovery.png", fig11))

            def fig12(ax):
                enabled = all_policy.get("enabled_by_layer", {})
                domains = ["attn_in", "mlp_in", "down_in", "o_in"]
                kept = [sum(d in vals for vals in enabled.values()) for d in domains]
                total = len(enabled)
                rolled = [total - k for k in kept]
                x = list(range(len(domains)))
                ax.bar(x, kept, label="kept")
                ax.bar(x, rolled, bottom=kept, label="rolled back")
                ax.set_xticks(x, labels=domains, rotation=25)
                ax.set_ylabel("number of layers")
                ax.legend()
                ax.set_title("Domain retention across layers")
            figures.append(_plot(run_dir, "fig_es65_domain_rollback_counts.png", fig12))

    if trajectory:
        def fig13(ax):
            labels = ["standard HiF4", "optimized HiF4"]
            vals = [float(trajectory["mean_standard_kl_last"]), float(trajectory["mean_optimized_kl_last"])]
            ax.bar(labels, vals)
            ax.set_ylabel("mean last-token KL to NVFP4 source")
            ax.set_title("Target trajectory KL")
        figures.append(_plot(run_dir, "fig_es65_target_trajectory_kl.png", fig13))

    if e2e:
        def fig14(ax):
            labels, vals = [], []
            for task, rec in e2e.get("deltas", {}).items():
                for variant in ("source_nvfp4", "standard_hif4", "optimized_hif4"):
                    if rec.get(variant) is not None:
                        labels.append(f"{task}\n{variant}")
                        vals.append(float(rec[variant]))
            ax.bar(labels, vals)
            ax.set_ylabel("accuracy")
            ax.tick_params(axis="x", rotation=30)
            ax.set_title("ARC semantic E2E")
        figures.append(_plot(run_dir, "fig_es7_arc_accuracy.png", fig14))
    return figures


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return "N/A"
        return f"{float(value):.{digits}f}"
    return str(value)


def build_hif4_scaling_report(run_dir: Path) -> dict[str, Any]:
    """Build tables, figures, summary.json and a Chinese report without model loading."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    candidates = _load_pt(run_dir / "candidate_scales.pt")
    activation_art = _load_pt(run_dir / "es_eval_merged.pt")
    full_art = _load_pt(run_dir / "es_full_eval_merged.pt")
    refined_candidates = _load_pt(run_dir / "es5_refined_candidate_scales.pt")
    refined_eval = _load_pt(run_dir / "es5_refine_eval_merged.pt")
    if candidates and refined_candidates:
        candidates = dict(candidates)
        candidates["recipes"] = dict(candidates.get("recipes", {}))
        candidates["recipes"].update(refined_candidates.get("recipes", {}))
        candidates["scales"] = {
            layer: {domain: dict(recipes) for domain, recipes in domains.items()}
            for layer, domains in candidates.get("scales", {}).items()
        }
        for layer, domains in refined_candidates.get("scales", {}).items():
            candidates["scales"].setdefault(layer, {})
            for domain, recipes in domains.items():
                candidates["scales"][layer].setdefault(domain, {}).update(recipes)
    if refined_eval:
        if activation_art is None:
            activation_art = {"records": {}}
        else:
            activation_art = dict(activation_art)
            activation_art["records"] = dict(activation_art.get("records", {}))
        if full_art is None:
            full_art = {"records": {}}
        else:
            full_art = dict(full_art)
            full_art["records"] = dict(full_art.get("records", {}))
        overlap_a = set(activation_art["records"]).intersection(refined_eval.get("records", {}))
        overlap_f = set(full_art["records"]).intersection(refined_eval.get("records", {}))
        if overlap_a or overlap_f:
            raise ValueError("coarse/refined report record key collision")
        activation_art["records"].update(refined_eval.get("records", {}))
        full_art["records"].update(refined_eval.get("records", {}))
    policy = _load_json(run_dir / "best_scaling_policy.json")
    validation = _load_json(run_dir / "es6_validation.json")
    all_policy = _load_json(run_dir / "best_scaling_policy_all_layers.json")
    trajectory = _load_json(run_dir / "es65_target_trajectory.json")
    e2e = _load_json(run_dir / "es7_e2e_summary.json")
    activation_rows = _aggregate_activation(activation_art, candidates)
    full_rows = _aggregate_full(full_art, candidates)
    scale_rows = _candidate_scale_rows(candidates)
    _write_csv(run_dir / "es2_activation_only_candidates.csv", activation_rows)
    _write_csv(run_dir / "es3_full_w4a4_candidates.csv", full_rows)
    _write_csv(run_dir / "es3_scale_stats.csv", scale_rows)
    ax_old = _old_ax_summary()
    figures = _make_figures(
        run_dir,
        activation_rows,
        full_rows,
        scale_rows,
        policy,
        validation,
        all_policy,
        trajectory,
        e2e,
        ax_old,
    )

    selected = dict((policy or {}).get("domain_recipes", {}))
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "old_ax": ax_old,
        "selected_recipes": selected,
        "representative_validation": (validation or {}).get("combined"),
        "representative_validation_pass": (validation or {}).get("validation_pass"),
        "paired_validation_win_rate": (validation or {}).get("paired_win_rate"),
        "target_trajectory_regression": (trajectory or {}).get("target_trajectory_regression"),
        "e2e_candidate_for_extended": (e2e or {}).get("candidate_for_extended_e2e"),
        "figures": figures,
    }

    deployable_activation_rows = [
        r for r in activation_rows if r.get("deployable", True) and not r.get("diagnostic")
    ]
    for domain in ("attn_in", "mlp_in", "down_in", "o_in"):
        summary[f"best_activation_{domain}"] = _best(
            deployable_activation_rows, domain=domain, field="activation_R_Y_conv"
        )
        summary[f"best_joint_{domain}"] = _best(
            full_rows, domain=domain, field="joint_R_Y_conv"
        )

    report_lines = [
        "# HiF4 部署等价缩放实验报告",
        "",
        "> 本报告只读取已经生成的实验 artifact，不在报告阶段重新加载模型、重新搜索 scale 或修改 policy。",
        "",
        "## 1. 实验目标",
        "",
        "验证 Layer PTS、G64 scale-phase、16/8/4/per-channel 静态等价缩放，以及固定 β 的 weight-aware balance，能否在不修改 HiF4 格式和 kernel、且不增加在线乘除法的前提下，减少 NVFP4 W4A4 → HiF4 W4A4 的转换误差。",
        "",
        "## 2. 旧 AX 基线",
        "",
        f"- AX1 S0/divisor activation-only mean output recovery：{_fmt(ax_old.get('AX1_mean_output_recovery_activation_only'))}",
        f"- AX2 G16 mean R_Y：{_fmt(ax_old.get('AX2_G16_mean_R_Y'))}",
        f"- AX2 G32 mean R_Y：{_fmt(ax_old.get('AX2_G32_mean_R_Y'))}",
        "",
        "注意：AX1 没有 HiF4 weight QDQ，因此只能与本实验 activation-only 栏比较，不能与 joint-W4A4 结果直接排名。",
        "",
        "## 3. Activation-only 搜索",
        "",
    ]
    for domain in ("attn_in", "mlp_in", "down_in", "o_in"):
        row = summary.get(f"best_activation_{domain}")
        if row:
            report_lines.append(
                f"- **{domain}**：最佳 `{row['recipe_id']}`，activation-only R_Y_conv={_fmt(row.get('activation_R_Y_conv'))}，zero rate={_fmt(row.get('zero_rate'))}。"
            )
    report_lines += ["", "## 4. Joint W4A4 与 weight penalty", ""]
    for domain in ("attn_in", "mlp_in", "down_in", "o_in"):
        row = summary.get(f"best_joint_{domain}")
        if row:
            report_lines.append(
                f"- **{domain}**：最佳 joint 候选 `{row['recipe_id']}`，R_Y_conv={_fmt(row.get('joint_R_Y_conv'))}，weight functional error ratio={_fmt(row.get('weight_functional_error_ratio'))}，weight NMSE={_fmt(row.get('weight_nmse'))}，SQNR={_fmt(row.get('weight_sqnr_db'))} dB。"
            )
    report_lines += ["", "## 5. 冻结 recipe 与代表层 block closure", ""]
    if selected:
        for domain, rid in selected.items():
            diag = (policy or {}).get("domain_diagnostics", {}).get(domain, {})
            conflict = diag.get("cross_layer", {}).get("recipe_conflict")
            report_lines.append(f"- {domain}: `{rid}`；跨代表层 recipe_conflict={conflict}。")
    else:
        report_lines.append("- 尚未生成代表层 policy。")

    report_lines += ["", "## 6. 独立 Validation", ""]
    if validation:
        c = validation.get("combined", {})
        report_lines += [
            f"- mean R_Y_conv={_fmt(c.get('mean_R_Y_conv'))}",
            f"- median R_Y_conv={_fmt(c.get('median_R_Y_conv'))}",
            f"- mean R_Y_local={_fmt(c.get('mean_R_Y_local'))}",
            f"- paired win-rate={_fmt(validation.get('paired_win_rate'))}",
            f"- validation_pass={validation.get('validation_pass')}",
        ]
    else:
        report_lines.append("- 尚未运行 validation。")

    report_lines += ["", "## 7. 36 层实例化与 target trajectory", ""]
    if all_policy:
        enabled = all_policy.get("enabled_by_layer", {})
        for domain in ("attn_in", "mlp_in", "down_in", "o_in"):
            kept = sum(domain in values for values in enabled.values())
            report_lines.append(f"- {domain}: 36 层中保留 {kept}/{len(enabled)} 层。")
    else:
        report_lines.append("- 尚未生成 all-layer policy。")
    if trajectory:
        report_lines += [
            f"- standard HiF4 mean KL={_fmt(trajectory.get('mean_standard_kl_last'))}",
            f"- optimized HiF4 mean KL={_fmt(trajectory.get('mean_optimized_kl_last'))}",
            f"- target_trajectory_regression={trajectory.get('target_trajectory_regression')}",
        ]

    report_lines += ["", "## 8. ARC-Easy / ARC-Challenge E2E", ""]
    if e2e:
        for task, rec in e2e.get("deltas", {}).items():
            report_lines.append(
                f"- {task}: NVFP4={_fmt(rec.get('source_nvfp4'))}, standard HiF4={_fmt(rec.get('standard_hif4'))}, optimized HiF4={_fmt(rec.get('optimized_hif4'))}, Δopt-std={_fmt(rec.get('optimized_minus_standard'))}。"
            )
        report_lines.append(f"- candidate_for_extended_e2e={e2e.get('candidate_for_extended_e2e')}")
        if e2e.get("o_in_enabled"):
            report_lines.append(f"- cache consistency passed={(e2e.get('cache_consistency') or {}).get('passed')}")
    else:
        report_lines.append("- 尚未运行 ARC E2E。")

    report_lines += ["", "## 9. 部署结论", ""]
    if all_policy:
        report_lines.append(
            f"- runtime_extra_ops={all_policy.get('runtime_extra_ops')}；hif4_format_changed={all_policy.get('hif4_format_changed')}。"
        )
    if trajectory and e2e:
        success = (
            not bool(trajectory.get("target_trajectory_regression"))
            and bool(e2e.get("candidate_for_extended_e2e"))
        )
        report_lines.append(f"- 当前完整闭环成功判定：**{success}**。")
    report_lines += ["", "## 10. 图表", ""]
    for fig in figures:
        report_lines.append(f"- `{fig}`")

    (run_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return summary
