"""Offline activation (d,t8,t4) calibration — no online search."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Literal

import torch

from .metrics import nmse
from .model_hooks import LINEAR_SUFFIXES, module_type_of
from .quantizer import HiF4QuantConfig, quantize_hif4

Granularity = Literal["global", "per_module_type", "per_layer"]


def _grid() -> tuple[list[float], list[float], list[float]]:
    def vals(start: float, stop: float, step: float) -> list[float]:
        out: list[float] = []
        x = start
        while x <= stop + 1e-9:
            out.append(round(x, 10))
            x += step
        return out

    return vals(5.5, 7.5, 0.25), vals(3.4, 4.1, 0.1), vals(1.70, 2.05, 0.05)


@dataclass
class LayerCalibResult:
    name: str
    module_type: str
    best_config: HiF4QuantConfig
    cal_output_mse: float
    cal_act_nmse: float
    val_output_mse: float
    val_act_nmse: float
    standard_val_output_mse: float


def _split_rows(x: torch.Tensor, val_fraction: float = 0.25) -> tuple[torch.Tensor, torch.Tensor]:
    n = x.shape[0]
    n_val = max(1, int(n * val_fraction))
    n_cal = max(1, n - n_val)
    return x[:n_cal], x[n_cal : n_cal + n_val]


def scored_config(
    x: torch.Tensor,
    w_col_energy: torch.Tensor,
    config: HiF4QuantConfig,
    device: torch.device,
) -> tuple[float, float]:
    """Return (output_mse_approx, act_nmse)."""
    # Ensure last dim divisible by 64
    if x.shape[-1] % 64 != 0:
        raise ValueError(f"activation last dim must % 64 == 0, got {x.shape[-1]}")
    xd = x.to(device=device, dtype=torch.float32)
    xq = quantize_hif4(xd, config=config).reconstruction
    act_n = nmse(xd, xq)
    # Diagonal approx: sum_j ||W[:,j]||^2 * (x_j - xq_j)^2
    energy = w_col_energy.to(device=device, dtype=torch.float32)
    if energy.numel() != xd.shape[-1]:
        raise ValueError("weight_col_energy length mismatch")
    diff2 = (xd - xq).pow(2)
    out_mse = float((diff2 * energy.view(1, -1)).sum().item() / max(xd.shape[0], 1))
    return out_mse, act_n


def search_layer_params(
    x_cal: torch.Tensor,
    x_val: torch.Tensor,
    w_col_energy: torch.Tensor,
    *,
    device: torch.device,
    name: str,
) -> LayerCalibResult:
    ds, t8s, t4s = _grid()
    best_cfg = HiF4QuantConfig()
    best_cal = float("inf")
    best_act = float("inf")
    for d, t8, t4 in product(ds, t8s, t4s):
        cfg = HiF4QuantConfig(s0_divisor=d, e8_threshold=t8, e4_threshold=t4)
        out_mse, act_n = scored_config(x_cal, w_col_energy, cfg, device)
        if out_mse < best_cal:
            best_cal = out_mse
            best_act = act_n
            best_cfg = cfg
    val_out, val_act = scored_config(x_val, w_col_energy, best_cfg, device)
    std_out, _ = scored_config(x_val, w_col_energy, HiF4QuantConfig(), device)
    mt = module_type_of(name) or "other"
    return LayerCalibResult(
        name=name,
        module_type=mt,
        best_config=best_cfg,
        cal_output_mse=best_cal,
        cal_act_nmse=best_act,
        val_output_mse=val_out,
        val_act_nmse=val_act,
        standard_val_output_mse=std_out,
    )


def calibrate_activations(
    inputs: dict[str, torch.Tensor],
    weight_col_energy: dict[str, torch.Tensor],
    *,
    granularity: Granularity = "per_layer",
    device: str | torch.device = "cuda",
    val_fraction: float = 0.25,
) -> dict[str, Any]:
    device = torch.device(device)
    layer_results: dict[str, LayerCalibResult] = {}
    for name, x in inputs.items():
        if name not in weight_col_energy:
            continue
        if x.shape[-1] % 64 != 0:
            continue
        x_cal, x_val = _split_rows(x.float(), val_fraction)
        layer_results[name] = search_layer_params(
            x_cal, x_val, weight_col_energy[name], device=device, name=name
        )

    # Aggregate by granularity
    param_map: dict[str, HiF4QuantConfig] = {}
    if granularity == "per_layer":
        for name, r in layer_results.items():
            param_map[name] = r.best_config
    elif granularity == "per_module_type":
        by_type: dict[str, list[LayerCalibResult]] = {}
        for r in layer_results.values():
            by_type.setdefault(r.module_type, []).append(r)
        type_cfg: dict[str, HiF4QuantConfig] = {}
        for mt, rows in by_type.items():
            # Re-search on concatenated cal rows would be ideal; use median params of layer optima.
            ds = sorted(r.best_config.s0_divisor for r in rows)
            t8s = sorted(r.best_config.e8_threshold for r in rows)
            t4s = sorted(r.best_config.e4_threshold for r in rows)
            mid = len(rows) // 2
            type_cfg[mt] = HiF4QuantConfig(
                s0_divisor=ds[mid], e8_threshold=t8s[mid], e4_threshold=t4s[mid]
            )
        for name, r in layer_results.items():
            param_map[name] = type_cfg[r.module_type]
    elif granularity == "global":
        # Median across layers
        rows = list(layer_results.values())
        if not rows:
            raise ValueError("no layers to calibrate")
        ds = sorted(r.best_config.s0_divisor for r in rows)
        t8s = sorted(r.best_config.e8_threshold for r in rows)
        t4s = sorted(r.best_config.e4_threshold for r in rows)
        mid = len(rows) // 2
        gcfg = HiF4QuantConfig(
            s0_divisor=ds[mid], e8_threshold=t8s[mid], e4_threshold=t4s[mid]
        )
        for name in layer_results:
            param_map[name] = gcfg
    else:
        raise ValueError(f"unknown granularity {granularity}")

    summary = {
        "granularity": granularity,
        "layers": {
            n: {
                "module_type": r.module_type,
                "best": {
                    "s0_divisor": r.best_config.s0_divisor,
                    "e8_threshold": r.best_config.e8_threshold,
                    "e4_threshold": r.best_config.e4_threshold,
                },
                "cal_output_mse": r.cal_output_mse,
                "val_output_mse": r.val_output_mse,
                "standard_val_output_mse": r.standard_val_output_mse,
                "val_improvement": r.standard_val_output_mse - r.val_output_mse,
                "cal_act_nmse": r.cal_act_nmse,
                "val_act_nmse": r.val_act_nmse,
            }
            for n, r in layer_results.items()
        },
        "param_map": {
            n: {
                "s0_divisor": c.s0_divisor,
                "e8_threshold": c.e8_threshold,
                "e4_threshold": c.e4_threshold,
            }
            for n, c in param_map.items()
        },
    }
    return {"summary": summary, "param_map": param_map, "layer_results": layer_results}
