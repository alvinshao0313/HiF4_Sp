"""Build Root Cause Ledger from structured experiment results (no hand-copied numbers)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from Inference_Paradigm_Conversion.ipc_analysis.io_utils import atomic_write_json, write_text
from Inference_Paradigm_Conversion.ipc_analysis.records import RootCauseRecord


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_csv(path: Path, key: str) -> float | None:
    if not path.is_file():
        return None
    rows = list(csv.DictReader(path.open()))
    xs = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return sum(xs) / len(xs) if xs else None


def build_ledger(results_root: Path, out_dir: Path) -> dict[str, Any]:
    results_root = Path(results_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weight_run = results_root / "latest_weight_run_id.txt"
    al_run = results_root / "latest_repr_al_run_id.txt"
    w3_run = results_root / "latest_w3_run_id.txt"
    w4_run = results_root / "latest_w4_run_id.txt"
    l3_run = results_root / "latest_l3_run_id.txt"
    mlp_run = results_root / "latest_mlp_run_id.txt"
    attn_run = results_root / "latest_attn_run_id.txt"
    gemm_run = results_root / "latest_gemm_run_id.txt"
    inj_n1 = results_root / "latest_inject_n1_n2_run_id.txt"
    inj_ps = results_root / "latest_inject_prefix_suffix_run_id.txt"
    inj_or = results_root / "latest_inject_oracle_run_id.txt"
    syn_run = results_root / "latest_synthetic_run_id.txt"
    a2_run = results_root / "latest_a2_run_id.txt"
    a5_run = results_root / "latest_a5_run_id.txt"
    l2_run = results_root / "latest_l2_run_id.txt"

    records: list[RootCauseRecord] = []

    if weight_run.is_file():
        wr = results_root / weight_run.read_text().strip()
        ws = _read_json(wr / "weight_summary.json") if (wr / "weight_summary.json").is_file() else {}
        records.append(
            RootCauseRecord(
                cause_id="C_W_BASE",
                hypothesis_id="H1a",
                mechanism="nvfp4_qat_to_hif4_weight_format_error",
                path_id="P1_semantic",
                evidence_class="observational_correlation",
                recoverable_error_fraction=0.0,
                affected_scope="all_linear_weights",
                metric_name="global_weight_nmse",
                metric_value=float(ws.get("global_nmse", 0.0)),
                notes=f"source={wr.name}",
            )
        )

    if w3_run.is_file():
        w3 = results_root / w3_run.read_text().strip()
        if (w3 / "w3_summary.json").is_file():
            w3s = _read_json(w3 / "w3_summary.json")
            ranking = w3s.get("ranking_by_mean_R_cf_output") or w3s.get("ranking", [])
            for item in ranking:
                if item.get("variant") == "full":
                    continue
                records.append(
                    RootCauseRecord(
                        cause_id=f"C_W3_{item['variant']}",
                        hypothesis_id="W3",
                        mechanism=f"hif4_idealize_{item['variant']}",
                        path_id="P1_semantic",
                        evidence_class="controlled_causal_evidence",
                        recoverable_error_fraction=float(item.get("mean_R_cf_output", 0.0)),
                        affected_scope="representative_linear_weights",
                        metric_name="mean_R_cf_output",
                        metric_value=float(item.get("mean_R_cf_output", 0.0)),
                        notes="illegal_probe" if not item.get("legal_hif4", True) else "",
                    )
                )

    if al_run.is_file():
        al = results_root / al_run.read_text().strip()
        als = _read_json(al / "repr_al_summary.json") if (al / "repr_al_summary.json").is_file() else {}
        records.append(
            RootCauseRecord(
                cause_id="C_A_DELTA",
                hypothesis_id="H2",
                mechanism="nvfp4_to_hif4_activation_delta",
                path_id="P2_matched_semantic",
                evidence_class="observational_correlation",
                recoverable_error_fraction=0.0,
                affected_scope="representative_layers_activation",
                metric_name="mean_nmse_hif4_vs_nvfp4",
                metric_value=float(als.get("mean_nmse_hif4_vs_nvfp4", 0.0)),
                notes=f"source={al.name}",
            )
        )
        # Linear decomposition dominance
        lin_path = al / "linear_decomp.csv"
        if lin_path.is_file():
            rows = [r for r in csv.DictReader(lin_path.open()) if r["path_id"] == "P2_matched_semantic"]
            if rows:
                e_dw = sum(float(r["energy_delta_w_an"]) for r in rows) / len(rows)
                e_da = sum(float(r["energy_wn_delta_a"]) for r in rows) / len(rows)
                e_x = sum(float(r["energy_delta_w_delta_a"]) for r in rows) / len(rows)
                records.append(
                    RootCauseRecord(
                        cause_id="C_L_CROSS",
                        hypothesis_id="H3",
                        mechanism="linear_deltaA_dominates_deltaW_under_P2",
                        path_id="P2_matched_semantic",
                        evidence_class="controlled_causal_evidence",
                        recoverable_error_fraction=0.0,
                        affected_scope="representative_linears",
                        metric_name="energy_wn_da_over_dw_an",
                        metric_value=e_da / e_dw if e_dw > 0 else 0.0,
                        extras={
                            "mean_energy_delta_w_an": e_dw,
                            "mean_energy_wn_delta_a": e_da,
                            "mean_energy_delta_w_delta_a": e_x,
                        },
                    )
                )

    if w4_run.is_file():
        w4 = results_root / w4_run.read_text().strip()
        if (w4 / "w4_summary.json").is_file():
            s = _read_json(w4 / "w4_summary.json")
            r16 = float(s.get("group16_recoverable_vs_group64_energy", 0.0))
            records.append(
                RootCauseRecord(
                    cause_id="C_H1_W4",
                    hypothesis_id="H1",
                    mechanism="16_to_64_group_size_counterfactual",
                    path_id="P1_semantic",
                    evidence_class="controlled_causal_evidence",
                    recoverable_error_fraction=r16,
                    affected_scope="representative_weight_groups",
                    metric_name="group16_recoverable_vs_group64_energy",
                    metric_value=r16,
                    notes=s.get("h1_status", ""),
                    extras={
                        "group_size_mean_output_error": s.get("group_size_mean_output_error"),
                        "group32_recoverable_vs_group64_energy": s.get(
                            "group32_recoverable_vs_group64_energy"
                        ),
                        "equalize_mean_recoverable": s.get("equalize_mean_recoverable"),
                        "dispersion_dose_spearman": s.get(
                            "dispersion_dose_spearman_vs_output_error"
                        ),
                    },
                )
            )

    if l3_run.is_file():
        l3 = results_root / l3_run.read_text().strip()
        if (l3 / "l3_summary.json").is_file():
            s = _read_json(l3 / "l3_summary.json")
            corr = s.get("correlations", {})
            records.append(
                RootCauseRecord(
                    cause_id="C_H4_L3",
                    hypothesis_id="H4",
                    mechanism="raw_nmse_vs_output_aware_predictors",
                    path_id="P1_semantic",
                    evidence_class="observational_correlation",
                    recoverable_error_fraction=0.0,
                    affected_scope="representative_weights",
                    metric_name="spearman_diag_act_weighted",
                    metric_value=float(
                        corr.get("diag_act_weighted_error", {})
                        .get("spearman", {})
                        .get("estimate", 0.0)
                    ),
                    extras=corr,
                )
            )

    if mlp_run.is_file():
        mp = results_root / mlp_run.read_text().strip()
        if (mp / "mlp_summary.json").is_file():
            s = _read_json(mp / "mlp_summary.json")
            records.append(
                RootCauseRecord(
                    cause_id="C_M_PRODUCT",
                    hypothesis_id="H5-MLP",
                    mechanism="mlp_product_cross_term_share",
                    path_id="P2_matched_semantic",
                    evidence_class="controlled_causal_evidence",
                    recoverable_error_fraction=0.0,
                    affected_scope="representative_mlp",
                    metric_name="mean_product_cross_share",
                    metric_value=float(s.get("mean_product_cross_share", 0.0)),
                    notes=f"source={mp.name}",
                )
            )

    if attn_run.is_file():
        ap = results_root / attn_run.read_text().strip()
        if (ap / "attention_summary.json").is_file():
            s = _read_json(ap / "attention_summary.json")
            records.append(
                RootCauseRecord(
                    cause_id="C_T_ATTN",
                    hypothesis_id="H6-Attention",
                    mechanism="full_attention_logits_kl_and_flip",
                    path_id="P1_semantic",
                    evidence_class="controlled_causal_evidence",
                    recoverable_error_fraction=0.0,
                    affected_scope="representative_attention",
                    metric_name="mean_kl_st",
                    metric_value=float(s.get("mean_kl_st", 0.0)),
                    extras={
                        "mean_flip": s.get("mean_flip"),
                        "mean_logits_gain": s.get("mean_logits_gain"),
                        "mean_nmse_residual": s.get("mean_nmse_residual"),
                        "linear_attn_present": s.get("linear_attn_present"),
                    },
                    notes=f"source={ap.name}",
                )
            )

    if gemm_run.is_file():
        gp = results_root / gemm_run.read_text().strip()
        if (gp / "gemm_summary.json").is_file():
            s = _read_json(gp / "gemm_summary.json")
            records.append(
                RootCauseRecord(
                    cause_id="C_G_GEMM",
                    hypothesis_id="G",
                    mechanism="format_semantic_gemm_oracle",
                    path_id="P1_semantic",
                    evidence_class="controlled_causal_evidence",
                    recoverable_error_fraction=0.0,
                    affected_scope="synthetic_gemm",
                    metric_name="P1_output_nmse",
                    metric_value=float(s.get("P1_output_nmse", 0.0)),
                    extras={"P2_output_nmse": s.get("P2_output_nmse")},
                    notes=f"source={gp.name}",
                )
            )

    if inj_n1.is_file():
        ip = results_root / inj_n1.read_text().strip()
        sp = ip / "injection_n1_n2_summary.json"
        if sp.is_file():
            s = _read_json(sp)
            records.append(
                RootCauseRecord(
                    cause_id="C_N_INJECT",
                    hypothesis_id="N1-N2",
                    mechanism="single_linear_and_layer_weight_injection",
                    path_id="P1_semantic",
                    evidence_class="controlled_causal_evidence",
                    recoverable_error_fraction=0.0,
                    affected_scope="representative_injection",
                    metric_name="mean_kl_last",
                    metric_value=float(s.get("mean_kl_last", 0.0)),
                    extras={"mean_kl_by_mask": s.get("mean_kl_by_mask")},
                    notes=f"source={ip.name}",
                )
            )

    if inj_ps.is_file():
        ip = results_root / inj_ps.read_text().strip()
        sp = ip / "injection_prefix_suffix_summary.json"
        if sp.is_file():
            s = _read_json(sp)
            records.append(
                RootCauseRecord(
                    cause_id="C_N_PREFIX_SUFFIX",
                    hypothesis_id="N4-N5",
                    mechanism="prefix_suffix_layer_conversion_curve",
                    path_id="P1_semantic",
                    evidence_class="controlled_causal_evidence",
                    recoverable_error_fraction=0.0,
                    affected_scope="layer_boundaries",
                    metric_name="num_rows",
                    metric_value=float(s.get("num_rows", 0.0)),
                    extras={"mean_kl_by_boundary": s.get("mean_kl_by_boundary")},
                    notes=f"source={ip.name}",
                )
            )

    if inj_or.is_file():
        ip = results_root / inj_or.read_text().strip()
        sp = ip / "injection_oracle_summary.json"
        if sp.is_file():
            s = _read_json(sp)
            rec = s.get("mean_recoverable_kl_by_frac", {})
            best = max((float(v) for v in rec.values()), default=0.0)
            records.append(
                RootCauseRecord(
                    cause_id="C_N_ORACLE",
                    hypothesis_id="N7",
                    mechanism="oracle_restore_high_dispersion_groups",
                    path_id="P1_semantic",
                    evidence_class="controlled_causal_evidence",
                    recoverable_error_fraction=best,
                    affected_scope="middle_down_proj",
                    metric_name="best_mean_recoverable_kl",
                    metric_value=best,
                    extras={
                        "mean_recoverable_kl_by_frac": rec,
                        "mean_random_recoverable_kl_by_frac": s.get(
                            "mean_random_recoverable_kl_by_frac"
                        ),
                    },
                    notes=f"source={ip.name}",
                )
            )

    if syn_run.is_file():
        sp = results_root / syn_run.read_text().strip()
        if (sp / "synthetic_summary.json").is_file():
            s = _read_json(sp / "synthetic_summary.json")
            for key in ("S1", "S2", "S3", "S4", "S5", "S6", "S7"):
                block = s.get(key)
                if not isinstance(block, dict):
                    continue
                records.append(
                    RootCauseRecord(
                        cause_id=f"C_SYN_{key}",
                        hypothesis_id=str(block.get("hypothesis_id", key)),
                        mechanism=f"synthetic_{key.lower()}",
                        path_id="synthetic",
                        evidence_class="controlled_causal_evidence",
                        recoverable_error_fraction=0.0,
                        affected_scope="synthetic_probe",
                        metric_name="supports_mechanism",
                        metric_value=1.0 if block.get("supports_mechanism") else 0.0,
                        notes=f"source={sp.name}",
                        extras={k: v for k, v in block.items() if k != "hypothesis_id"},
                    )
                )

    if a2_run.is_file():
        ap = results_root / a2_run.read_text().strip()
        if (ap / "a2_summary.json").is_file():
            s = _read_json(ap / "a2_summary.json")
            ranking = s.get("ranking_by_mean_R_cf_output") or s.get("ranking_by_mean_R_cf") or []
            # pick top legal non-full variants per format
            for item in ranking:
                if item.get("variant") in ("full", "nv_full"):
                    continue
                fmt = item.get("format", "")
                var = item.get("variant", "")
                rcf = float(item.get("mean_R_cf_output", item.get("mean_R_cf", 0.0)))
                records.append(
                    RootCauseRecord(
                        cause_id=f"C_A2_{fmt}_{var}",
                        hypothesis_id="H2",
                        mechanism=f"activation_idealize_{fmt}_{var}",
                        path_id="P2_matched_semantic",
                        evidence_class="controlled_causal_evidence",
                        recoverable_error_fraction=rcf,
                        affected_scope="representative_activations",
                        metric_name="mean_R_cf_output",
                        metric_value=rcf,
                        notes=f"source={ap.name}; n={item.get('n')}",
                        extras={
                            "mean_R_cf": item.get("mean_R_cf"),
                            "mean_nmse": item.get("mean_nmse"),
                        },
                    )
                )

    if a5_run.is_file():
        ap = results_root / a5_run.read_text().strip()
        if (ap / "a5_summary.json").is_file():
            s = _read_json(ap / "a5_summary.json")
            records.append(
                RootCauseRecord(
                    cause_id="C_A5_H2",
                    hypothesis_id="H2",
                    mechanism="activation_distribution_interventions",
                    path_id="P2_matched_semantic",
                    evidence_class="controlled_causal_evidence",
                    recoverable_error_fraction=0.0,
                    affected_scope="representative_activation_groups",
                    metric_name="baseline_mean_nmse_h_vs_n",
                    metric_value=float(s.get("baseline_mean_nmse_h_vs_n", 0.0)),
                    notes=f"source={ap.name}",
                    extras={"top_ranking": (s.get("ranking") or [])[:8]},
                )
            )

    if l2_run.is_file():
        lp = results_root / l2_run.read_text().strip()
        if (lp / "l2_summary.json").is_file():
            s = _read_json(lp / "l2_summary.json")
            records.append(
                RootCauseRecord(
                    cause_id="C_L2_SHAPLEY",
                    hypothesis_id="H3",
                    mechanism="shapley_phi_a_over_phi_w",
                    path_id="P2_matched_semantic",
                    evidence_class="controlled_causal_evidence",
                    recoverable_error_fraction=0.0,
                    affected_scope="representative_linears",
                    metric_name="ratio_phi_a_over_phi_w",
                    metric_value=float(s.get("ratio_phi_a_over_phi_w", 0.0)),
                    notes=f"source={lp.name}; fp64_fail={s.get('fp64_audit_fail')}",
                    extras=s,
                )
            )

    ledger = {
        "schema_version": 1,
        "num_records": len(records),
        "records": [r.to_dict() for r in records],
    }
    atomic_write_json(out_dir / "root_cause_ledger.json", ledger)

    # Markdown auto report (numbers only from ledger)
    lines = [
        "# Root Cause Ledger",
        "",
        f"records: {len(records)}",
        "",
        "| cause_id | hypothesis | path_id | evidence | metric | value | R_cf |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| {r.cause_id} | {r.hypothesis_id} | {r.path_id} | {r.evidence_class} | "
            f"{r.metric_name} | {r.metric_value:.6g} | {r.recoverable_error_fraction:.4f} |"
        )
    write_text(out_dir / "root_cause_ledger.md", "\n".join(lines) + "\n")
    return ledger
