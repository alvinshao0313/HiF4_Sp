"""N0–N7: conversion mask injection + Shapley format/shift + oracle repair."""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any, Iterator, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from Inference_Paradigm_Conversion.ipc_analysis.formats.hif4 import quantize_hif4_tensor
from Inference_Paradigm_Conversion.ipc_analysis.formats.mxfp8 import quantize_mxfp8_activation
from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import quantize_nvfp4_activation
from Inference_Paradigm_Conversion.ipc_analysis.metrics.tensor_metrics import compute_pair_metrics

MaskKind = Literal[
    "single_linear",
    "single_layer",
    "module_class",
    "prefix_layers",
    "suffix_layers",
    "full",
]


@dataclass(frozen=True)
class MaskSpec:
    kind: MaskKind
    layer_idx: int | None = None
    projection: str | None = None
    prefix_k: int | None = None
    suffix_k: int | None = None


def prefix_suffix_boundaries(num_layers: int) -> list[int]:
    """Preregistered N4/N5 boundaries; dedup sorted."""
    raw = [
        0,
        1,
        2,
        4,
        round(num_layers / 8),
        round(num_layers / 4),
        round(3 * num_layers / 8),
        round(num_layers / 2),
        round(5 * num_layers / 8),
        round(3 * num_layers / 4),
        round(7 * num_layers / 8),
        num_layers,
    ]
    return sorted({int(x) for x in raw if 0 <= int(x) <= num_layers})


def _match_module(name: str, spec: MaskSpec) -> bool:
    if "lm_head" in name:
        return False
    if not isinstance(name, str):
        return False
    # parse layer
    layer_idx = None
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            layer_idx = int(parts[i + 1])
            break
    proj = None
    for cand in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if name.endswith(cand):
            proj = cand
            break
    if proj is None:
        return False

    if spec.kind == "full":
        return True
    if spec.kind == "single_linear":
        return layer_idx == spec.layer_idx and proj == spec.projection
    if spec.kind == "single_layer":
        return layer_idx == spec.layer_idx
    if spec.kind == "module_class":
        return proj == spec.projection
    if spec.kind == "prefix_layers":
        assert spec.prefix_k is not None
        return layer_idx is not None and layer_idx < spec.prefix_k
    if spec.kind == "suffix_layers":
        assert spec.suffix_k is not None
        # convert last suffix_k layers: indices >= L-suffix_k
        # caller passes suffix_k as count from end; need L — store as absolute start via layer_idx
        assert spec.layer_idx is not None  # absolute start index
        return layer_idx is not None and layer_idx >= spec.layer_idx
    raise ValueError(f"unknown mask kind {spec.kind}")


def convert_linear_weight_inplace(linear: nn.Linear) -> torch.Tensor:
    """HiF4 QDQ BF16 in-place; returns backup CPU tensor for restore."""
    backup = linear.weight.detach().cpu().clone()
    w = linear.weight.detach().float()
    view = quantize_hif4_tensor(w, group_dim=-1, output_dtype=linear.weight.dtype)
    with torch.no_grad():
        linear.weight.copy_(view.dequantized.to(device=linear.weight.device))
    return backup


@contextlib.contextmanager
def with_conversion_mask(
    model: nn.Module,
    mask_spec: MaskSpec,
    path_id: str = "P1_semantic",
) -> Iterator[list[str]]:
    """Temporarily convert matching Linear weights to HiF4; restore on exit.

    Only one model resides on GPU. path_id recorded for callers; activation
    path is applied separately via hooks when needed.
    """
    del path_id  # activation handling is outside for P2
    backups: list[tuple[nn.Linear, torch.Tensor]] = []
    converted: list[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if not _match_module(name, mask_spec):
            continue
        backups.append((mod, convert_linear_weight_inplace(mod)))
        converted.append(name)
    try:
        yield converted
    finally:
        with torch.no_grad():
            for mod, bak in backups:
                mod.weight.copy_(bak.to(device=mod.weight.device, dtype=mod.weight.dtype))
        del backups


def shapley_format_shift(
    f_s,
    f_t,
    x_s: torch.Tensor,
    x_t: torch.Tensor,
) -> dict[str, Any]:
    """N0 two-factor Shapley: format vs upstream shift. f_* : Tensor -> Tensor."""
    if x_s.shape != x_t.shape:
        raise ValueError("source/target input shapes must match for Shapley")
    y_ss = f_s(x_s)
    y_ts = f_t(x_s)
    y_st = f_s(x_t)
    y_tt = f_t(x_t)
    phi_format = 0.5 * ((y_ts - y_ss) + (y_tt - y_st))
    phi_shift = 0.5 * ((y_st - y_ss) + (y_tt - y_ts))
    total = y_tt - y_ss
    recon = phi_format + phi_shift
    resid = total - recon
    te = float((total * total).sum().item())
    return {
        "energy_total": te,
        "energy_phi_format": float((phi_format * phi_format).sum().item()),
        "energy_phi_shift": float((phi_shift * phi_shift).sum().item()),
        "residual_rel": float((resid * resid).sum().item()) / te if te > 0 else 0.0,
        "format_share": float((phi_format * phi_format).sum().item()) / te if te > 0 else 0.0,
        "shift_share": float((phi_shift * phi_shift).sum().item()) / te if te > 0 else 0.0,
    }


def logits_distance(logits_a: torch.Tensor, logits_b: torch.Tensor) -> dict[str, float]:
    """Teacher-forced logits distance (NMSE + mean KL on last token softmax)."""
    m = compute_pair_metrics(logits_a.float(), logits_b.float())
    # KL on last position distribution
    tiny = torch.finfo(torch.float32).tiny
    p = torch.softmax(logits_a.float()[..., -1, :], dim=-1).clamp_min(tiny)
    q = torch.softmax(logits_b.float()[..., -1, :], dim=-1).clamp_min(tiny)
    kl = float((p * (p.log() - q.log())).sum(dim=-1).mean().item())
    return {"logits_nmse": m["nmse"], "logits_error_energy": m["error_energy"], "kl_last": kl}


@torch.no_grad()
def oracle_repair_groups(
    w_n: torch.Tensor,
    w_h: torch.Tensor,
    risk_score: torch.Tensor,
    *,
    top_frac: float,
    mode: str = "restore_source",
) -> torch.Tensor:
    """Replace top-risk 64-groups in W_H with source (or keep H).

    w: [O, K], risk_score: [O, K//64]
    """
    if w_n.shape != w_h.shape:
        raise ValueError("shape mismatch")
    o, k = w_n.shape
    if k % 64 != 0:
        raise ValueError("K must be divisible by 64")
    n64 = k // 64
    if risk_score.shape != (o, n64):
        raise ValueError(f"risk_score shape {tuple(risk_score.shape)} != {(o, n64)}")
    flat = risk_score.reshape(-1)
    n = flat.numel()
    n_top = max(1, int(math.ceil(top_frac * n)))
    thresh = torch.topk(flat, n_top).values.min()
    mask = (risk_score >= thresh).unsqueeze(-1).expand(o, n64, 64)
    w_n_g = w_n.reshape(o, n64, 64)
    w_h_g = w_h.reshape(o, n64, 64)
    if mode == "restore_source":
        out = torch.where(mask, w_n_g, w_h_g)
    elif mode == "random":
        # random same cardinality
        rand = torch.rand_like(flat)
        thr = torch.topk(rand, n_top).values.min()
        rmask = (rand.reshape(o, n64) >= thr).unsqueeze(-1).expand(o, n64, 64)
        out = torch.where(rmask, w_n_g, w_h_g)
    else:
        raise ValueError(mode)
    return out.reshape(o, k)


def sub16_dispersion_risk(w_n: torch.Tensor) -> torch.Tensor:
    """H1 risk: log2 amax range of 4×16 subblocks per 64-group. [O, K//64]."""
    o, k = w_n.shape
    g = w_n.float().reshape(o, k // 64, 4, 16)
    amax = g.abs().amax(dim=-1)
    amax_pos_max = torch.where(amax > 0, amax, torch.zeros_like(amax)).amax(dim=-1)
    amax_pos_min = torch.where(amax > 0, amax, torch.full_like(amax, float("inf"))).amin(dim=-1)
    return torch.where(
        torch.isfinite(amax_pos_min) & (amax_pos_min > 0) & (amax_pos_max > 0),
        torch.log2(amax_pos_max) - torch.log2(amax_pos_min),
        torch.zeros_like(amax_pos_max),
    )


def install_activation_qdq_hooks(
    model: nn.Module,
    *,
    path_id: str,
    scales: dict[str, torch.Tensor] | None,
    module_names: set[str] | None = None,
) -> list[Any]:
    """P1: MXFP8 on Linear inputs; P2_matched: NVFP4/HiF4 on matched modules."""
    handles = []

    def make_hook(name: str):
        def hook(_mod, inputs):
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            if module_names is not None and name not in module_names:
                return
            xb = x.to(torch.bfloat16)
            if path_id == "P1_semantic":
                y = quantize_mxfp8_activation(xb).dequantized.to(dtype=x.dtype)
            elif path_id == "P2_matched_semantic":
                if scales is None:
                    raise ValueError("P2 requires scales")
                from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import (
                    resolve_nvfp4_scale_for_module,
                )

                # Target path uses HiF4; source path uses NVFP4 — caller sets which
                # by converting weights; for activation, if weight converted use HiF4.
                # Here we always apply format based on whether module weight was converted:
                # simplified: NVFP4 if name not in converted_set is handled by caller.
                raise RuntimeError("use install_p2_hooks with converted set")
            else:
                raise ValueError(path_id)
            return (y,) + tuple(inputs[1:])

        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "lm_head" not in name:
            if module_names is None or name in module_names:
                handles.append(mod.register_forward_pre_hook(make_hook(name)))
    return handles


def install_p2_activation_hooks(
    model: nn.Module,
    *,
    scales: dict[str, torch.Tensor],
    converted_names: set[str],
    matched_names: set[str],
) -> list[Any]:
    """Matched coverage: converted modules get HiF4 act; others in matched get NVFP4."""
    from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import (
        resolve_nvfp4_scale_for_module,
    )

    handles = []

    def make_hook(name: str):
        def hook(_mod, inputs):
            x = inputs[0]
            if not torch.is_tensor(x) or name not in matched_names:
                return
            xb = x.to(torch.bfloat16)
            if name in converted_names:
                y = quantize_hif4_tensor(xb.float(), output_dtype=x.dtype).dequantized
            else:
                scale = resolve_nvfp4_scale_for_module(scales, name).to(x.device)
                y = quantize_nvfp4_activation(xb, scale, output_dtype=x.dtype).dequantized
            return (y,) + tuple(inputs[1:])

        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name in matched_names:
            handles.append(mod.register_forward_pre_hook(make_hook(name)))
    return handles
