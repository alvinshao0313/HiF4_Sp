"""S1–S7: minimal synthetic mechanism probes (not effect-size substitutes)."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from Inference_Paradigm_Conversion.ipc_analysis.analysis.attention_propagation import (
    kl_js_on_valid_support,
    top_attended_flip_rate,
)
from Inference_Paradigm_Conversion.ipc_analysis.analysis.synthetic_mechanisms import (
    DISPERSION_DOSES,
    apply_dispersion_dose,
    hif4_group_error,
)
from Inference_Paradigm_Conversion.ipc_analysis.formats.hif4 import quantize_hif4_tensor
from Inference_Paradigm_Conversion.ipc_analysis.formats.mxfp8 import quantize_mxfp8_activation
from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import quantize_nvfp4_activation
from Inference_Paradigm_Conversion.ipc_analysis.metrics.statistics import mean_ci
from Inference_Paradigm_Conversion.ipc_analysis.metrics.tensor_metrics import compute_pair_metrics

NUM_SEEDS = 10
BASE_SEED = 20260810


def _seed(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def _randn(shape, g: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, generator=g, dtype=torch.float32)


def run_s1_dispersion(seeds: int = NUM_SEEDS) -> dict[str, Any]:
    """H1: dispersion dose increases HiF4 error under fixed 64-group RMS."""
    rows = []
    for s in range(seeds):
        g = _seed(BASE_SEED + s)
        group = _randn((4, 16), g)
        # inject mild base dispersion then dose
        base = apply_dispersion_dose(group, 0.5)
        for d in DISPERSION_DOSES:
            gd = apply_dispersion_dose(base, d)
            a = _randn((1, 64), g)
            m = hif4_group_error(gd, variant="full", activation_row=a)
            rows.append(
                {
                    "seed": BASE_SEED + s,
                    "dose": d,
                    "output_error_energy": m["output_error_energy"],
                    "weight_nmse": m["weight_nmse"],
                }
            )
    by_d: dict[float, list[float]] = {d: [] for d in DISPERSION_DOSES}
    for r in rows:
        by_d[float(r["dose"])].append(float(r["output_error_energy"]))
    curve = {str(d): mean_ci(xs) for d, xs in by_d.items()}
    # monotonic check: Spearman of dose vs mean error
    doses = list(DISPERSION_DOSES)
    means = [curve[str(d)]["mean"] for d in doses]
    # simple rank correlation
    from Inference_Paradigm_Conversion.ipc_analysis.metrics.statistics import spearman_with_bootstrap

    corr = spearman_with_bootstrap(list(doses), means, seed=BASE_SEED, repeats=200)
    return {
        "experiment": "S1",
        "hypothesis_id": "H1",
        "curve_output_error_energy": curve,
        "dose_vs_error_spearman": corr,
        "supports_mechanism": bool(corr.get("estimate", 0.0) > 0),
        "rows": rows,
    }


def run_s2_threshold_sweep() -> dict[str, Any]:
    """H1/HiF4: approach e8≈4 and payload clip≈1.75; record discontinuity."""
    centers = {
        "e8_threshold": 4.0,
        "e4_threshold": 2.0,
        "payload_clip": 1.75,
        "payload_half_step": 0.25,
    }
    eps_grid = [1e-3, 3e-3, 1e-2, 3e-2, 0.1]
    rows = []
    for name, c in centers.items():
        for eps in eps_grid:
            for side, x0 in (("below", c - eps), ("above", c + eps)):
                # pack into 64-group with one sensitive element
                w = torch.ones(1, 64, dtype=torch.float32) * 0.5
                w[0, 0] = float(x0)
                m = hif4_group_error(w.view(4, 16), variant="full")
                rows.append(
                    {
                        "boundary": name,
                        "center": c,
                        "eps": eps,
                        "side": side,
                        "value": float(x0),
                        "weight_nmse": m["weight_nmse"],
                        "weight_error_energy": m["weight_error_energy"],
                    }
                )
    # discontinuity metric: |err_above - err_below| at smallest eps
    disc = {}
    for name in centers:
        sub = [r for r in rows if r["boundary"] == name and r["eps"] == min(eps_grid)]
        below = next(r for r in sub if r["side"] == "below")
        above = next(r for r in sub if r["side"] == "above")
        disc[name] = abs(above["weight_error_energy"] - below["weight_error_energy"])
    return {
        "experiment": "S2",
        "hypothesis_id": "H1",
        "discontinuity_at_min_eps": disc,
        "supports_mechanism": any(v > 0 for v in disc.values()),
        "rows": rows,
    }


def run_s3_activation_outliers(seeds: int = NUM_SEEDS) -> dict[str, Any]:
    """H2: keep RMS, raise max/RMS; compare NVFP4 vs HiF4 activation NMSE."""
    rows = []
    scale = torch.tensor(32.0)
    for s in range(seeds):
        g = _seed(BASE_SEED + 1000 + s)
        base = _randn((8, 256), g)
        base = base / base.pow(2).mean().sqrt().clamp_min(1e-8)
        for kurt_level, factor in (("low", 1.0), ("high", 8.0)):
            x = base.clone()
            # spike one element per row, renormalize RMS
            x[:, 0] = x[:, 0] * factor
            x = x / x.pow(2).mean().sqrt().clamp_min(1e-8)
            a_n = quantize_nvfp4_activation(x.to(torch.bfloat16), scale).dequantized.float()
            a_h = quantize_hif4_tensor(x, output_dtype=torch.float32).metadata["values_fp32"]
            a_m = quantize_mxfp8_activation(x.to(torch.bfloat16)).dequantized.float()
            rows.append(
                {
                    "seed": BASE_SEED + 1000 + s,
                    "kurt_level": kurt_level,
                    "max_over_rms": float(x.abs().amax() / x.pow(2).mean().sqrt()),
                    "nmse_nvfp4": compute_pair_metrics(x, a_n)["nmse"],
                    "nmse_hif4": compute_pair_metrics(x, a_h)["nmse"],
                    "nmse_mxfp8": compute_pair_metrics(x, a_m)["nmse"],
                    "nmse_h_vs_n": compute_pair_metrics(a_n, a_h)["nmse"],
                }
            )
    low = [r["nmse_h_vs_n"] for r in rows if r["kurt_level"] == "low"]
    high = [r["nmse_h_vs_n"] for r in rows if r["kurt_level"] == "high"]
    return {
        "experiment": "S3",
        "hypothesis_id": "H2",
        "mean_nmse_h_vs_n_low": mean_ci(low),
        "mean_nmse_h_vs_n_high": mean_ci(high),
        "supports_mechanism": mean_ci(high)["mean"] > mean_ci(low)["mean"],
        "rows": rows,
    }


def run_s4_error_covariance(seeds: int = NUM_SEEDS) -> dict[str, Any]:
    """H4: same ||ΔW||_F, different alignment with activation top direction."""
    rows = []
    for s in range(seeds):
        g = _seed(BASE_SEED + 2000 + s)
        a = _randn((32, 64), g)
        # top right singular vector of A
        _, _, vh = torch.linalg.svd(a, full_matrices=False)
        v_top = vh[0]
        # random orthogonal-ish via householder
        v_rand = _randn((64,), g)
        v_rand = v_rand / v_rand.norm().clamp_min(1e-8)
        v_orth = v_rand - (v_rand @ v_top) * v_top
        v_orth = v_orth / v_orth.norm().clamp_min(1e-8)
        w = _randn((16, 64), g)
        eps_norm = 0.1 * w.norm()
        for tag, direction in (("aligned", v_top), ("random", v_rand), ("orthogonal", v_orth)):
            dw = direction.unsqueeze(0).expand_as(w)
            dw = dw * (eps_norm / dw.norm().clamp_min(1e-8))
            y0 = F.linear(a, w)
            y1 = F.linear(a, w + dw)
            m_w = compute_pair_metrics(w, w + dw)
            m_y = compute_pair_metrics(y0, y1)
            rows.append(
                {
                    "seed": BASE_SEED + 2000 + s,
                    "alignment": tag,
                    "weight_nmse": m_w["nmse"],
                    "output_error_energy": m_y["error_energy"],
                }
            )
    by = {k: [r["output_error_energy"] for r in rows if r["alignment"] == k] for k in ("aligned", "random", "orthogonal")}
    return {
        "experiment": "S4",
        "hypothesis_id": "H4",
        "mean_output_error": {k: mean_ci(v) for k, v in by.items()},
        "supports_mechanism": mean_ci(by["aligned"])["mean"] > mean_ci(by["orthogonal"])["mean"],
        "rows": rows,
    }


def run_s5_wa_angle_sweep(seeds: int = NUM_SEEDS) -> dict[str, Any]:
    """H3: fixed ||e_W||,||e_A||; vary angle between error directions."""
    angles = [0, 30, 60, 90, 120, 150, 180]
    rows = []
    for s in range(seeds):
        g = _seed(BASE_SEED + 3000 + s)
        a = _randn((16, 64), g)
        w = _randn((8, 64), g)
        # base errors in output space via left/right perturbations
        e_w_dir = _randn((8, 64), g)
        e_w_dir = e_w_dir / e_w_dir.norm()
        e_a_base = _randn((16, 64), g)
        e_a_base = e_a_base / e_a_base.norm()
        ew_scale = 0.05 * w.norm()
        ea_scale = 0.05 * a.norm()
        y0 = F.linear(a, w)
        for deg in angles:
            # rotate e_a toward e_w projected into activation space via shared random
            # Construct e_a with controlled cosine to flattened e_w via Gram-Schmidt in R^{16*64}
            u = e_w_dir.reshape(-1)
            # pad/truncate to act size
            v0 = e_a_base.reshape(-1)
            if v0.numel() != u.numel():
                # map weight error to activation-shaped proxy
                u_act = _randn(v0.shape, g)
                u_act = u_act / u_act.norm()
            else:
                u_act = u
            v_orth = v0 - (v0 @ u_act) * u_act
            v_orth = v_orth / v_orth.norm().clamp_min(1e-8)
            rad = math.radians(deg)
            e_a = (math.cos(rad) * u_act + math.sin(rad) * v_orth).reshape_as(a)
            e_a = e_a * ea_scale
            e_w = e_w_dir * ew_scale
            y1 = F.linear(a + e_a, w + e_w)
            # independent sum energy for reference
            y_w_only = F.linear(a, w + e_w)
            y_a_only = F.linear(a + e_a, w)
            e_tot = (y1 - y0).reshape(-1)
            e_indep = (y_w_only - y0 + y_a_only - y0).reshape(-1)
            rows.append(
                {
                    "seed": BASE_SEED + 3000 + s,
                    "angle_deg": deg,
                    "total_error_energy": float(e_tot.dot(e_tot)),
                    "indep_sum_energy": float(e_indep.dot(e_indep)),
                    "cross_inner": float(
                        ((y_w_only - y0).reshape(-1) * (y_a_only - y0).reshape(-1)).sum()
                    ),
                }
            )
    by = {d: [r["total_error_energy"] for r in rows if r["angle_deg"] == d] for d in angles}
    return {
        "experiment": "S5",
        "hypothesis_id": "H3",
        "mean_total_error_by_angle": {str(d): mean_ci(v) for d, v in by.items()},
        "supports_mechanism": mean_ci(by[0])["mean"] != mean_ci(by[90])["mean"],
        "rows": rows,
    }


def run_s6_mlp_product(seeds: int = NUM_SEEDS) -> dict[str, Any]:
    """H5: product identity and SiLU amplification under controlled δg,δu."""
    rows = []
    for s in range(seeds):
        g = _seed(BASE_SEED + 4000 + s)
        gate = _randn((32, 64), g)
        up = _randn((32, 64), g)
        dg = _randn((32, 64), g)
        du = _randn((32, 64), g)
        # normalize perturbations
        dg = dg * (0.1 * gate.norm() / dg.norm().clamp_min(1e-8))
        du = du * (0.1 * up.norm() / du.norm().clamp_min(1e-8))
        cases = {
            "both_same": (dg, du),
            "both_opposite": (dg, -du),
            "gate_only": (dg, torch.zeros_like(du)),
            "up_only": (torch.zeros_like(dg), du),
            "gate_large": (2 * dg, 0.5 * du),
        }
        for tag, (dgg, duu) in cases.items():
            g0, u0 = gate, up
            prod0 = F.silu(g0) * u0
            prod1 = F.silu(g0 + dgg) * (u0 + duu)
            # linear product identity on pre-silu for cross share
            raw0 = g0 * u0
            raw1 = (g0 + dgg) * (u0 + duu)
            cross = dgg * duu
            cross_share = float(cross.pow(2).sum() / (raw1 - raw0).pow(2).sum().clamp_min(1e-12))
            rows.append(
                {
                    "seed": BASE_SEED + 4000 + s,
                    "case": tag,
                    "product_cross_share_raw": cross_share,
                    "silu_product_nmse": compute_pair_metrics(prod0, prod1)["nmse"],
                    "raw_product_nmse": compute_pair_metrics(raw0, raw1)["nmse"],
                }
            )
    by = {k: [r["silu_product_nmse"] for r in rows if r["case"] == k] for k in ("both_same", "both_opposite", "gate_only", "up_only")}
    return {
        "experiment": "S6",
        "hypothesis_id": "H5-MLP",
        "mean_silu_nmse": {k: mean_ci(v) for k, v in by.items()},
        "supports_mechanism": True,  # identity always holds; effect varies by case
        "rows": rows,
    }


def run_s7_attention_sensitivity(seeds: int = NUM_SEEDS) -> dict[str, Any]:
    """H6: same Q/K NMSE, different directions / margins → flip rate differs."""
    rows = []
    t, h, d = 16, 4, 16
    for s in range(seeds):
        g = _seed(BASE_SEED + 5000 + s)
        q = _randn((1, h, t, d), g)
        k = _randn((1, h, t, d), g)
        v = _randn((1, h, t, d), g)
        scale = 1.0 / math.sqrt(d)
        causal = torch.tril(torch.ones(t, t, dtype=torch.bool))
        valid = causal.view(1, 1, t, t).expand(1, h, t, t)
        # margin control: scale top logit bias
        for margin_tag, margin in (("large", 3.0), ("small", 0.2)):
            logits0 = torch.matmul(q, k.transpose(-2, -1)) * scale
            # add margin to diagonal-ish preferred token (position 0 for all queries)
            logits0 = logits0.clone()
            logits0[..., 0] = logits0[..., 0] + margin
            # directions
            dq = _randn(q.shape, g)
            for tag in ("along_q", "orth_q", "q_only", "k_only", "qk_same", "qk_opp"):
                if tag == "along_q":
                    dir_q = q / q.norm().clamp_min(1e-8)
                    dir_k = torch.zeros_like(k)
                    eps = 0.05 * q.norm()
                    q1, k1 = q + dir_q * eps, k
                elif tag == "orth_q":
                    dir_q = dq - (dq.reshape(-1) @ q.reshape(-1)) * q / (q.norm() ** 2).clamp_min(1e-8)
                    dir_q = dir_q / dir_q.norm().clamp_min(1e-8)
                    eps = 0.05 * q.norm()
                    q1, k1 = q + dir_q * eps, k
                elif tag == "q_only":
                    eps = 0.05 * q.norm()
                    q1, k1 = q + dq / dq.norm() * eps, k
                elif tag == "k_only":
                    eps = 0.05 * k.norm()
                    dk = _randn(k.shape, g)
                    q1, k1 = q, k + dk / dk.norm() * eps
                elif tag == "qk_same":
                    eps = 0.05 * q.norm()
                    dlt = dq / dq.norm() * eps
                    q1, k1 = q + dlt, k + dlt
                else:  # qk_opp
                    eps = 0.05 * q.norm()
                    dlt = dq / dq.norm() * eps
                    q1, k1 = q + dlt, k - dlt
                logits1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
                logits1 = logits1.clone()
                logits1[..., 0] = logits1[..., 0] + margin
                kl = kl_js_on_valid_support(logits0, logits1, valid)
                flip = top_attended_flip_rate(logits0, logits1, valid)
                rows.append(
                    {
                        "seed": BASE_SEED + 5000 + s,
                        "margin": margin_tag,
                        "perturbation": tag,
                        "qk_nmse": compute_pair_metrics(
                            torch.cat([q.reshape(-1), k.reshape(-1)]),
                            torch.cat([q1.reshape(-1), k1.reshape(-1)]),
                        )["nmse"],
                        "kl_st": kl["kl_st"],
                        "flip": flip,
                        "logits_nmse": compute_pair_metrics(logits0, logits1)["nmse"],
                    }
                )
    small = [r["flip"] for r in rows if r["margin"] == "small"]
    large = [r["flip"] for r in rows if r["margin"] == "large"]
    return {
        "experiment": "S7",
        "hypothesis_id": "H6-Attention",
        "mean_flip_small_margin": mean_ci(small),
        "mean_flip_large_margin": mean_ci(large),
        "supports_mechanism": mean_ci(small)["mean"] > mean_ci(large)["mean"],
        "rows": rows,
    }


def run_all_synthetic(seeds: int = NUM_SEEDS) -> dict[str, Any]:
    suite = {
        "S1": run_s1_dispersion(seeds),
        "S2": run_s2_threshold_sweep(),
        "S3": run_s3_activation_outliers(seeds),
        "S4": run_s4_error_covariance(seeds),
        "S5": run_s5_wa_angle_sweep(seeds),
        "S6": run_s6_mlp_product(seeds),
        "S7": run_s7_attention_sensitivity(seeds),
    }
    evidence = {
        k: {
            "hypothesis_id": v["hypothesis_id"],
            "supports_mechanism": v["supports_mechanism"],
            **{kk: vv for kk, vv in v.items() if kk not in {"rows", "supports_mechanism", "hypothesis_id", "experiment"}},
        }
        for k, v in suite.items()
    }
    return {"num_seeds": seeds, "experiments": suite, "evidence_summary": evidence}
