"""Offline Linear case runner: BASE / E1-E6 / C0-C3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import (
    load_packed_linear_state,
    resolve_local_snapshot,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.config import AppConfig, load_config, results_dir
from Native_NVFP4_HiF4_Linear_Puncture.src.diagonal_search import (
    config_dict,
    diagonal_result_to_row,
    search_channelwise_diagonal,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.formats import (
    qdq_hif4_direct,
    qdq_mxfp8_post_rotation,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import (
    ensure_dir,
    load_pt,
    module_capture_stem,
    save_pt,
    write_json,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.metrics import (
    compare_tensors,
    nmse,
    recovery_ratio,
    zero_rate,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.native_nvfp4 import (
    PackedNVFP4LinearState,
    dequantize_packed_weight,
    qdq_nvfp4_post_rotation,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.weight_variants import (
    build_hif4_direct_weight,
    build_hif4_greedy_weight,
    weight_variant_row,
    write_weight_variants_csv,
)

VARIANT_SPECS = [
    # variant_id, weight_format, activation_format, weight_opt, diag
    ("BASE_WN_AN", "W_N", "A_N", "none", False),
    ("E1_WN_AM", "W_N", "A_M", "none", False),
    ("E2_WH_AM_RTN", "W_H", "A_M", "rtn", False),
    ("E3_WH_AM_GREEDY", "W_Hg", "A_M", "greedy", False),
    ("E4_WH_AH_RTN", "W_H", "A_H", "rtn", False),
    ("E5_WH_AH_DIAG", "W_H_D", "A_H_D", "rtn+diag", True),
    ("E6_WH_AH_GREEDY", "W_Hg", "A_H", "greedy", False),
    ("C0_FP", "W_N", "X_rot", "none", False),
    ("C1_WN_AH", "W_N", "A_H", "none", False),
    ("C2_WH_AN_RTN", "W_H", "A_N", "rtn", False),
    ("C3_WH_AN_GREEDY", "W_Hg", "A_N", "greedy", False),
]

HEADLINE_VARIANT_IDS = (
    "E1_WN_AM",
    "E2_WH_AM_RTN",
    "E3_WH_AM_GREEDY",
    "E4_WH_AH_RTN",
    "E5_WH_AH_DIAG",
    "E6_WH_AH_GREEDY",
)


def diagonal_validation_output(
    x_rot_val: torch.Tensor,
    w_n: torch.Tensor,
    d: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """E5: Q_H(X/d) @ Q_H(W*d)^T with fixed calibration D."""
    x_d = x_rot_val.to(torch.float32) / d.to(torch.float32)
    w_d = w_n.to(torch.float32) * d.to(torch.float32)
    a_h_d = qdq_hif4_direct(x_d)
    w_h_d = qdq_hif4_direct(w_d)
    return _linear(a_h_d, w_h_d, bias)


def run_module_linear_cases(
    *,
    module_name: str,
    x_rot_cal: torch.Tensor,
    x_rot_val: torch.Tensor,
    input_global_scale: torch.Tensor,
    w_n: torch.Tensor,
    w_h_rtn: torch.Tensor,
    w_h_greedy: torch.Tensor,
    bias: torch.Tensor | None,
    d: torch.Tensor,
) -> dict[str, Any]:
    """Compute headline + control outputs for one module (validation rows)."""
    scale = input_global_scale.to(torch.float32)
    a_n_val = qdq_nvfp4_post_rotation(x_rot_val, scale)
    a_m_val = qdq_mxfp8_post_rotation(x_rot_val)
    a_h_val = qdq_hif4_direct(x_rot_val)

    y_nn = _linear(a_n_val, w_n, bias)
    headline = {
        "E1_WN_AM": _linear(a_m_val, w_n, bias),
        "E2_WH_AM_RTN": _linear(a_m_val, w_h_rtn, bias),
        "E3_WH_AM_GREEDY": _linear(a_m_val, w_h_greedy, bias),
        "E4_WH_AH_RTN": _linear(a_h_val, w_h_rtn, bias),
        "E5_WH_AH_DIAG": diagonal_validation_output(x_rot_val, w_n, d, bias),
        "E6_WH_AH_GREEDY": _linear(a_h_val, w_h_greedy, bias),
    }
    controls = {
        "C0_FP": _linear(x_rot_val, w_n, bias),
        "C1_WN_AH": _linear(a_h_val, w_n, bias),
        "C2_WH_AN_RTN": _linear(a_n_val, w_h_rtn, bias),
        "C3_WH_AN_GREEDY": _linear(a_n_val, w_h_greedy, bias),
    }
    return {
        "module_name": module_name,
        "Y_NN": y_nn,
        "headline": headline,
        "controls": controls,
        "x_rot_cal": x_rot_cal,
        "x_rot_val": x_rot_val,
    }


def _load_capture(run_dir: Path, module_name: str, split: str) -> dict[str, Any]:
    path = run_dir / "captures" / f"{module_capture_stem(module_name)}_{split}.pt"
    return load_pt(path, map_location="cpu")


def _packed_state_from_snapshot(
    snapshot: Path, weight_map: dict[str, str], module_name: str
) -> PackedNVFP4LinearState:
    packed = load_packed_linear_state(snapshot, weight_map, module_name)
    return PackedNVFP4LinearState(
        module_name=module_name,
        weight_packed=packed["weight_packed"],  # type: ignore[arg-type]
        weight_scale=packed["weight_scale"],  # type: ignore[arg-type]
        weight_global_scale=packed["weight_global_scale"].to(torch.float32),  # type: ignore[union-attr]
        input_global_scale=packed["input_global_scale"].to(torch.float32),  # type: ignore[union-attr]
        rotation_matrix=packed["rotation_matrix"].to(torch.bfloat16),  # type: ignore[union-attr]
        bias=packed["bias"],
    )


def _linear(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    dtype = w.dtype
    return F.linear(
        x.to(dtype=dtype),
        w,
        None if bias is None else bias.to(dtype=dtype),
    )


def run_linear_cases(
    config: AppConfig,
    run_id: str,
    *,
    device: str = "cuda",
    modules: list[str] | None = None,
) -> dict[str, Any]:
    run_dir = results_dir(run_id)
    snapshot = resolve_local_snapshot(config.model.model_id)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]

    module_names = modules or config.formal_module_names
    weight_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    linear_rows: list[dict[str, Any]] = []
    diag_dir = ensure_dir(run_dir / "diagonal_scales")

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)

    for module_name in module_names:
        print(f"[linear_cases] {module_name}", flush=True)
        cal = _load_capture(run_dir, module_name, "cal")
        val = _load_capture(run_dir, module_name, "val")
        x_cal = cal["x_rot_bf16"].to(device=torch_device)
        x_val = val["x_rot_bf16"].to(device=torch_device)
        scale = cal["input_global_scale_fp32"].to(device=torch_device, dtype=torch.float32)

        state = _packed_state_from_snapshot(snapshot, weight_map, module_name)
        w_n = dequantize_packed_weight(state).to(device=torch_device, dtype=torch.float32)
        bias = (
            state.bias.to(device=torch_device, dtype=torch.float32)
            if state.bias is not None
            else None
        )

        # Activations from the same saved X_rot.
        a_n_cal = qdq_nvfp4_post_rotation(x_cal, scale).to(torch.float32)
        a_n_val = qdq_nvfp4_post_rotation(x_val, scale).to(torch.float32)
        a_m_val = qdq_mxfp8_post_rotation(x_val).to(torch.float32)
        a_h_val = qdq_hif4_direct(x_val, output_dtype=torch.float32)

        direct = build_hif4_direct_weight(w_n)
        greedy = build_hif4_greedy_weight(
            w_n,
            device=torch_device,
            memory_budget_fraction=config.weight_greedy.memory_budget_fraction,
        )
        w_h = direct.reconstruction.to(device=torch_device, dtype=torch.float32)
        w_hg = greedy.reconstruction.to(device=torch_device, dtype=torch.float32)
        weight_rows.append(
            weight_variant_row(module_name, w_n.detach().cpu(), direct, greedy)
        )

        # Diagonal search on cal only (GPU).
        diag = search_channelwise_diagonal(
            x_cal.to(torch.float32),
            a_n_cal,
            w_n,
            config.diagonal_search,
        )
        save_pt(
            diag_dir / f"{module_capture_stem(module_name)}.pt",
            {
                "d": diag.d,
                "log2_d": diag.log2_d,
                "group_kept_mask": diag.group_kept_mask,
                "config": config_dict(config.diagonal_search),
            },
        )
        diag_rows.append(diagonal_result_to_row(module_name, diag, k_dim=w_n.shape[1]))

        d = diag.d.to(device=torch_device, dtype=torch.float32)
        x_d_val = x_val.to(torch.float32) / d
        w_d = w_n * d
        a_h_d_val = qdq_hif4_direct(x_d_val, output_dtype=torch.float32)
        w_h_d = qdq_hif4_direct(w_d, output_dtype=torch.float32)

        y_nn = _linear(a_n_val, w_n, bias)
        outputs = {
            "BASE_WN_AN": y_nn,
            "E1_WN_AM": _linear(a_m_val, w_n, bias),
            "E2_WH_AM_RTN": _linear(a_m_val, w_h, bias),
            "E3_WH_AM_GREEDY": _linear(a_m_val, w_hg, bias),
            "E4_WH_AH_RTN": _linear(a_h_val, w_h, bias),
            "E5_WH_AH_DIAG": _linear(a_h_d_val, w_h_d, bias),
            "E6_WH_AH_GREEDY": _linear(a_h_val, w_hg, bias),
            "C0_FP": _linear(x_val.to(torch.float32), w_n, bias),
            "C1_WN_AH": _linear(a_h_val, w_n, bias),
            "C2_WH_AN_RTN": _linear(a_n_val, w_h, bias),
            "C3_WH_AN_GREEDY": _linear(a_n_val, w_hg, bias),
        }

        layer_idx = int(cal["layer_idx"])
        projection = str(cal["projection"])
        act_nmse_m = nmse(a_m_val, a_n_val)
        act_nmse_h = nmse(a_h_val, a_n_val)
        w_nmse_rtn = nmse(w_h, w_n)
        w_nmse_g = nmse(w_hg, w_n)

        for vid, wfmt, afmt, wopt, diag_on in VARIANT_SPECS:
            metrics = compare_tensors(outputs[vid], y_nn)
            row = {
                "run_id": run_id,
                "split": "val",
                "layer_idx": layer_idx,
                "projection": projection,
                "module_name": module_name,
                "variant_id": vid,
                "weight_format": wfmt,
                "activation_format": afmt,
                "weight_optimization": wopt,
                "diagonal_search_enabled": diag_on,
                "diagonal_scale_min": float(d.min()) if diag_on else float("nan"),
                "diagonal_scale_median": float(torch.median(d)) if diag_on else float("nan"),
                "diagonal_scale_max": float(d.max()) if diag_on else float("nan"),
                "num_rows": int(x_val.shape[0]),
                "act_nmse_am_vs_an": act_nmse_m,
                "act_nmse_ah_vs_an": act_nmse_h,
                "zero_rate_an": zero_rate(a_n_val),
                "zero_rate_am": zero_rate(a_m_val),
                "zero_rate_ah": zero_rate(a_h_val),
                "weight_nmse_rtn": w_nmse_rtn,
                "weight_nmse_greedy": w_nmse_g,
                **{k: metrics[k] for k in metrics if k != "num_output_elements"},
                "num_output_elements": int(metrics["num_output_elements"]),
            }
            # BASE vs itself: force exact zeros for stability
            if vid == "BASE_WN_AN":
                row["error_energy"] = 0.0
                row["nmse"] = 0.0
                row["sqnr_db"] = float("inf")
                row["relative_l2"] = 0.0
                row["mae"] = 0.0
                row["max_abs_error"] = 0.0
                row["cosine"] = 1.0
                row["bias_mean"] = 0.0
            linear_rows.append(row)

    write_weight_variants_csv(weight_rows, run_dir / "weight_variants.csv")
    pd.DataFrame(diag_rows).to_csv(run_dir / "diagonal_search.csv", index=False)
    df = pd.DataFrame(linear_rows)
    df.to_csv(run_dir / "linear_results.csv", index=False)

    # Global energy-weighted summary.
    g_rows = []
    for vid, *_ in VARIANT_SPECS:
        sub = df[df["variant_id"] == vid]
        ref = float(sub["reference_energy"].sum())
        err = float(sub["error_energy"].sum())
        g_nmse = 0.0 if vid == "BASE_WN_AN" else (err / ref if ref > 0 else float("nan"))
        module_nmses = sub["nmse"].astype(float)
        worst_idx = module_nmses.idxmax() if len(sub) else None
        g_rows.append(
            {
                "variant_id": vid,
                "global_nmse": g_nmse,
                "global_sqnr_db": float("inf")
                if g_nmse == 0
                else (-10.0 * torch.log10(torch.tensor(max(g_nmse, 1e-30))).item()),
                "median_module_nmse": float(module_nmses.median()) if len(sub) else float("nan"),
                "q90_module_nmse": float(module_nmses.quantile(0.9)) if len(sub) else float("nan"),
                "max_module_nmse": float(module_nmses.max()) if len(sub) else float("nan"),
                "worst_module": str(sub.loc[worst_idx, "module_name"]) if worst_idx is not None else "",
                "total_reference_energy": ref,
                "total_error_energy": err,
            }
        )
    gdf = pd.DataFrame(g_rows)
    gdf.to_csv(run_dir / "global_summary.csv", index=False)

    def _g(vid: str) -> dict[str, Any]:
        r = gdf[gdf["variant_id"] == vid].iloc[0].to_dict()
        return {
            "global_nmse": r["global_nmse"],
            "sqnr_db": r["global_sqnr_db"],
            "median_module_nmse": r["median_module_nmse"],
            "q90_module_nmse": r["q90_module_nmse"],
            "worst_module": r["worst_module"],
        }

    e2 = float(gdf.loc[gdf["variant_id"] == "E2_WH_AM_RTN", "total_error_energy"].iloc[0])
    e3 = float(gdf.loc[gdf["variant_id"] == "E3_WH_AM_GREEDY", "total_error_energy"].iloc[0])
    e4 = float(gdf.loc[gdf["variant_id"] == "E4_WH_AH_RTN", "total_error_energy"].iloc[0])
    e5 = float(gdf.loc[gdf["variant_id"] == "E5_WH_AH_DIAG", "total_error_energy"].iloc[0])
    e6 = float(gdf.loc[gdf["variant_id"] == "E6_WH_AH_GREEDY", "total_error_energy"].iloc[0])
    c2 = float(gdf.loc[gdf["variant_id"] == "C2_WH_AN_RTN", "total_error_energy"].iloc[0])
    c3 = float(gdf.loc[gdf["variant_id"] == "C3_WH_AN_GREEDY", "total_error_energy"].iloc[0])

    q_greedy_hm = _g("E3_WH_AM_GREEDY")
    q_greedy_hm["recovery"] = recovery_ratio(e2, e3)
    q_greedy_hm["direct_error_zero"] = e2 == 0.0

    q_diag = _g("E5_WH_AH_DIAG")
    q_diag["recovery"] = recovery_ratio(e4, e5)
    q_diag["direct_error_zero"] = e4 == 0.0

    q_greedy_hh = _g("E6_WH_AH_GREEDY")
    q_greedy_hh["recovery"] = recovery_ratio(e4, e6)
    q_greedy_hh["direct_error_zero"] = e4 == 0.0

    summary = {
        "run_id": run_id,
        "num_modules": len(module_names),
        "questions": {
            "wn_am_loss_vs_wn_an": _g("E1_WN_AM"),
            "wh_am_rtn_loss_vs_wn_an": _g("E2_WH_AM_RTN"),
            "wh_am_greedy_recovery": q_greedy_hm,
            "wh_ah_rtn_loss_vs_wn_an": _g("E4_WH_AH_RTN"),
            "wh_ah_diagonal_recovery": q_diag,
            "wh_ah_greedy_recovery": q_greedy_hh,
        },
        "weight_only_recovery_c3_vs_c2": recovery_ratio(c2, c3),
        "notes": {
            "baseline": "Y_NN = Linear(A_N, W_N)",
            "aggregate": "energy-weighted global NMSE",
            "diagonal_search": "calibration-only; validation only evaluates fixed D",
            "scope": "Linear-local semantic oracle; online rotation retained; not end-to-end deploy accuracy",
        },
    }
    write_json(run_dir / "summary.json", summary)
    print(f"LINEAR CASES DONE -> {run_dir}")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run offline Linear puncture cases")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    run_linear_cases(config, args.run_id, device=args.device)


if __name__ == "__main__":
    main()
