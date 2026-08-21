"""W0–W2: NVFP4-QAT BF16 source weight → HiF4 conversion analysis."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

from Inference_Paradigm_Conversion.ipc_analysis.config import LINEAR_PROJECTIONS
from Inference_Paradigm_Conversion.ipc_analysis.formats.fingerprint import (
    list_safetensor_keys,
    load_nvfp4_qat_dequant_weight,
)
from Inference_Paradigm_Conversion.ipc_analysis.formats.hif4 import quantize_hif4_tensor
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import (
    GzipJsonlWriter,
    atomic_write_json,
    ensure_dir,
    write_csv,
)
from Inference_Paradigm_Conversion.ipc_analysis.metrics.statistics import (
    spearman_with_bootstrap,
)
from Inference_Paradigm_Conversion.ipc_analysis.metrics.streaming import ErrorAccumulator
from Inference_Paradigm_Conversion.ipc_analysis.metrics.tensor_metrics import (
    compute_pair_metrics,
)

_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


@dataclass
class Sub16Dispersion:
    sub16_amax_ratio: float
    sub16_rms_ratio: float
    sub16_log2_amax_range: float
    sub16_energy_share_max: float
    sub16_cv_amax: float
    sub16_cv_rms: float


def split_64_to_sub16_along_k(weight: torch.Tensor) -> torch.Tensor:
    """Reshape [..., K] with K%64==0 into groups of 64, each split to 4×16 on K."""
    if weight.ndim != 2:
        raise ValueError(f"expected 2D weight [out,in], got {tuple(weight.shape)}")
    out_f, in_f = weight.shape
    if in_f % 64 != 0:
        raise ValueError(f"K={in_f} not divisible by 64")
    # [out, n64, 4, 16]
    return weight.reshape(out_f, in_f // 64, 4, 16)


def compute_sub16_dispersion(group64: torch.Tensor) -> Sub16Dispersion:
    """group64: [4, 16] float tensor for one HiF4 group."""
    if group64.shape != (4, 16):
        raise ValueError(f"expected (4,16), got {tuple(group64.shape)}")
    g = group64.to(torch.float32)
    amax = g.abs().amax(dim=-1)  # [4]
    rms = torch.sqrt((g * g).mean(dim=-1).clamp_min(0))
    energy = (g * g).sum(dim=-1)
    eps = 1e-12

    def _ratio(v: torch.Tensor) -> float:
        vmin = float(v[v > 0].min().item()) if bool((v > 0).any()) else 0.0
        vmax = float(v.max().item())
        if vmin <= 0:
            return float("inf") if vmax > 0 else 1.0
        return vmax / vmin

    amax_ratio = _ratio(amax)
    rms_ratio = _ratio(rms)
    amax_pos = amax[amax > 0]
    if amax_pos.numel() == 0:
        log2_range = 0.0
    else:
        log2_range = float(
            (torch.log2(amax_pos.max()) - torch.log2(amax_pos.min())).item()
        )
    e_sum = float(energy.sum().item())
    energy_share_max = float(energy.max().item() / e_sum) if e_sum > 0 else 0.0

    def _cv(v: torch.Tensor) -> float:
        mean = float(v.mean().item())
        if mean == 0:
            return 0.0
        return float(v.std(unbiased=False).item() / abs(mean))

    return Sub16Dispersion(
        sub16_amax_ratio=amax_ratio if math.isfinite(amax_ratio) else 1.0e300,
        sub16_rms_ratio=rms_ratio if math.isfinite(rms_ratio) else 1.0e300,
        sub16_log2_amax_range=log2_range,
        sub16_energy_share_max=energy_share_max,
        sub16_cv_amax=_cv(amax),
        sub16_cv_rms=_cv(rms),
    )


def convert_weight_w0(
    w_n_fp32: torch.Tensor,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor | dict[str, float]]:
    """W0 path: W_N(fp32) -> HiF4 FP32 QDQ -> BF16 carrier; report both errors."""
    x = w_n_fp32.to(device=device, dtype=torch.float32)
    view = quantize_hif4_tensor(x, group_dim=-1, output_dtype=torch.bfloat16)
    w_h_fp32 = view.metadata["values_fp32"].to(torch.float32)
    w_h_bf16 = view.dequantized.to(torch.bfloat16)
    format_err = compute_pair_metrics(x.cpu(), w_h_fp32.cpu())
    storage_err = compute_pair_metrics(w_h_fp32.cpu(), w_h_bf16.float().cpu())
    return {
        "W_N": x.cpu(),
        "W_H_FP32": w_h_fp32.cpu(),
        "W_H_BF16": w_h_bf16.cpu(),
        "E_hif4_format": format_err,
        "E_target_storage": storage_err,
        "metadata": {k: v for k, v in view.metadata.items() if k not in {"values_fp32"}},
    }


def iter_linear_weight_names(
    checkpoint: Path,
    *,
    layer_indices: list[int] | None = None,
    projections: tuple[str, ...] = LINEAR_PROJECTIONS,
) -> Iterator[tuple[str, int, str]]:
    weight_map = list_safetensor_keys(checkpoint)
    for name in sorted(weight_map.keys()):
        if not name.endswith(".weight"):
            continue
        m = _LAYER_RE.search(name)
        if not m:
            continue
        layer = int(m.group(1))
        if layer_indices is not None and layer not in layer_indices:
            continue
        proj = None
        for p in projections:
            if name.endswith(f".{p}.weight"):
                proj = p
                break
        if proj is None:
            continue
        yield name, layer, proj


def _vectorized_group_stats(
    w_n: torch.Tensor,
    w_h: torch.Tensor,
) -> dict[str, Any]:
    """Vectorized W2 stats over all HiF4 64-groups along K.

    Returns per-group nmse / dispersion tensors with shape [out, n64].
    """
    g_n = split_64_to_sub16_along_k(w_n.to(torch.float32))  # [O, G, 4, 16]
    g_h = split_64_to_sub16_along_k(w_h.to(torch.float32))
    ref = g_n.reshape(g_n.shape[0], g_n.shape[1], -1)
    tgt = g_h.reshape(g_h.shape[0], g_h.shape[1], -1)
    err = tgt - ref
    ref_e = (ref * ref).sum(dim=-1)
    err_e = (err * err).sum(dim=-1)
    nmse = torch.where(ref_e > 0, err_e / ref_e, torch.zeros_like(ref_e))

    amax = g_n.abs().amax(dim=-1)  # [O,G,4]
    rms = torch.sqrt((g_n * g_n).mean(dim=-1).clamp_min(0))
    energy = (g_n * g_n).sum(dim=-1)
    amax_max = amax.amax(dim=-1)
    amax_min_pos = torch.where(amax > 0, amax, torch.full_like(amax, float("inf"))).amin(dim=-1)
    amax_ratio = torch.where(
        torch.isfinite(amax_min_pos) & (amax_min_pos > 0),
        amax_max / amax_min_pos,
        torch.ones_like(amax_max),
    )
    amax_pos_max = torch.where(amax > 0, amax, torch.zeros_like(amax)).amax(dim=-1)
    amax_pos_min = torch.where(amax > 0, amax, torch.full_like(amax, float("inf"))).amin(dim=-1)
    log2_range = torch.where(
        torch.isfinite(amax_pos_min) & (amax_pos_min > 0) & (amax_pos_max > 0),
        torch.log2(amax_pos_max) - torch.log2(amax_pos_min),
        torch.zeros_like(amax_pos_max),
    )
    e_sum = energy.sum(dim=-1).clamp_min(1e-12)
    energy_share_max = energy.amax(dim=-1) / e_sum
    amax_mean = amax.mean(dim=-1).clamp_min(1e-12)
    amax_cv = amax.std(dim=-1, unbiased=False) / amax_mean
    rms_mean = rms.mean(dim=-1).clamp_min(1e-12)
    rms_cv = rms.std(dim=-1, unbiased=False) / rms_mean

    # Energy-weighted global group summary
    group_summary = {
        "nmse": float(err_e.sum().item() / ref_e.sum().item()) if float(ref_e.sum()) > 0 else 0.0,
        "reference_energy": float(ref_e.sum().item()),
        "error_energy": float(err_e.sum().item()),
        "numel": float(ref.numel()),
        "error_p50": float(nmse.flatten().quantile(0.50).item()),
        "error_p90": float(nmse.flatten().quantile(0.90).item()),
        "error_p99": float(nmse.flatten().quantile(0.99).item()),
        "error_p99_9": float(nmse.flatten().quantile(0.999).item()),
    }
    return {
        "nmse": nmse,
        "error_energy": err_e,
        "reference_energy": ref_e,
        "sub16_amax_ratio": amax_ratio,
        "sub16_log2_amax_range": log2_range,
        "sub16_energy_share_max": energy_share_max,
        "sub16_cv_amax": amax_cv,
        "sub16_cv_rms": rms_cv,
        "group_summary": group_summary,
        "out_f": g_n.shape[0],
        "n64": g_n.shape[1],
    }


def analyze_weight_tensor(
    checkpoint: Path,
    tensor_name: str,
    layer_idx: int,
    projection: str,
    device: torch.device | str,
    *,
    emit_group_records: bool,
    group_writer: GzipJsonlWriter | None,
    max_group_records: int | None = None,
) -> dict[str, Any]:
    view = load_nvfp4_qat_dequant_weight(checkpoint, tensor_name, device=device)
    w_n = view.dequantized
    conv = convert_weight_w0(w_n, device=device)
    w_h_fp32 = conv["W_H_FP32"]
    assert isinstance(w_h_fp32, torch.Tensor)

    format_metrics = conv["E_hif4_format"]
    storage_metrics = conv["E_target_storage"]
    assert isinstance(format_metrics, dict)
    assert isinstance(storage_metrics, dict)

    stats = _vectorized_group_stats(w_n.cpu(), w_h_fp32.cpu())
    nmse = stats["nmse"]
    log2_range = stats["sub16_log2_amax_range"]
    disp_x = log2_range.reshape(-1)
    err_y = nmse.reshape(-1)

    # Deterministic subsample for within-tensor Spearman (avoid millions of points)
    n_groups = int(disp_x.numel())
    max_corr = 8192
    if n_groups > max_corr:
        idx = torch.linspace(0, n_groups - 1, max_corr).round().long()
        xs = disp_x[idx].tolist()
        ys = err_y[idx].tolist()
        cluster_ids = (idx // max(1, stats["n64"])).tolist()  # cluster by out-row
    else:
        xs = disp_x.tolist()
        ys = err_y.tolist()
        cluster_ids = [i // max(1, stats["n64"]) for i in range(n_groups)]

    if len(xs) >= 2:
        corr = spearman_with_bootstrap(
            xs,
            ys,
            seed=20260810 + layer_idx,
            repeats=200,
            cluster_ids=cluster_ids,
        )
        within_spearman = corr["estimate"]
    else:
        within_spearman = 0.0

    emitted = 0
    if emit_group_records and group_writer is not None:
        flat_n = n_groups
        take = flat_n if max_group_records is None else min(flat_n, max_group_records)
        # Deterministic stride sample over groups
        sel = torch.linspace(0, flat_n - 1, take).round().long() if take > 0 else torch.zeros(0, dtype=torch.long)
        out_f = stats["out_f"]
        n64 = stats["n64"]
        for flat_i in sel.tolist():
            o = int(flat_i) // n64
            g = int(flat_i) % n64
            group_writer.write(
                {
                    "tensor_name": tensor_name,
                    "layer_idx": layer_idx,
                    "projection": projection,
                    "out_row": o,
                    "group64_index": g,
                    "hif4_weight_group_nmse": float(nmse[o, g].item()),
                    "error_energy": float(stats["error_energy"][o, g].item()),
                    "reference_energy": float(stats["reference_energy"][o, g].item()),
                    "sub16_amax_ratio": float(stats["sub16_amax_ratio"][o, g].item()),
                    "sub16_log2_amax_range": float(log2_range[o, g].item()),
                    "sub16_energy_share_max": float(stats["sub16_energy_share_max"][o, g].item()),
                    "sub16_cv_amax": float(stats["sub16_cv_amax"][o, g].item()),
                    "sub16_cv_rms": float(stats["sub16_cv_rms"][o, g].item()),
                    "metadata_source": "observable_sub16_dispersion",
                }
            )
            emitted += 1

    return {
        "tensor_name": tensor_name,
        "layer_idx": layer_idx,
        "projection": projection,
        "shape": list(w_n.shape),
        "path_id_format": "P1_semantic",
        "path_id_storage": "W_storage_probe",
        "E_hif4_format": format_metrics,
        "E_target_storage": storage_metrics,
        "group_summary": stats["group_summary"],
        "within_tensor_spearman_log2range_vs_nmse": within_spearman,
        "num_groups": n_groups,
        "group_records_emitted": emitted,
    }


def run_weight_analysis(
    checkpoint: Path,
    out_dir: Path,
    *,
    device: str = "cuda:0",
    layer_indices: list[int] | None = None,
    projections: tuple[str, ...] = LINEAR_PROJECTIONS,
    emit_all_group_records: bool = False,
    max_group_records_per_tensor: int | None = 4096,
    shard_id: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    """W1 full/partial scan + W2 streaming group records."""
    out_dir = ensure_dir(out_dir)
    device_t = torch.device(device)
    if device_t.type == "cuda":
        torch.cuda.set_device(device_t)

    names = list(iter_linear_weight_names(checkpoint, layer_indices=layer_indices, projections=projections))
    # Shard by tensor for multi-GPU
    names = [t for i, t in enumerate(names) if i % num_shards == shard_id]

    global_acc = ErrorAccumulator()
    tensor_rows: list[dict[str, Any]] = []
    corr_points: list[dict[str, Any]] = []

    group_path = out_dir / f"weight_group_records_shard{shard_id}.jsonl.gz"
    with GzipJsonlWriter(group_path, flush_every=512) as writer:
        for tensor_name, layer_idx, projection in names:
            rec = analyze_weight_tensor(
                checkpoint,
                tensor_name,
                layer_idx,
                projection,
                device_t,
                emit_group_records=True,
                group_writer=writer,
                max_group_records=(
                    None if emit_all_group_records else max_group_records_per_tensor
                ),
            )
            fmt = rec["E_hif4_format"]
            # Merge into global energy-weighted accumulator via reconstructing
            # from energies is not enough for cosine; reload small? Better:
            # accumulate from the stored energies carefully.
            # For exact global NMSE we need sum of energies — update from metrics.
            global_acc.numel += int(fmt["numel"])
            global_acc.reference_energy += float(fmt["reference_energy"])
            global_acc.target_energy += float(fmt["target_energy"])
            global_acc.error_energy += float(fmt["error_energy"])
            # Approximate remaining fields from tensor metrics (energy-weighted later)
            global_acc.abs_error_sum += float(fmt["mae"]) * float(fmt["numel"])
            global_acc.signed_error_sum += float(fmt["mean_signed_error"]) * float(fmt["numel"])
            global_acc.max_abs_error = max(global_acc.max_abs_error, float(fmt["max_abs_error"]))

            row = {
                "tensor_name": tensor_name,
                "layer_idx": layer_idx,
                "projection": projection,
                "shape0": rec["shape"][0],
                "shape1": rec["shape"][1],
                "nmse": fmt["nmse"],
                "sqnr_db": fmt["sqnr_db"],
                "cosine": fmt["cosine"],
                "mae": fmt["mae"],
                "error_p99": fmt["error_p99"],
                "error_p99_9": fmt["error_p99_9"],
                "top1pct_error_energy_share": fmt["top1pct_error_energy_share"],
                "reference_energy": fmt["reference_energy"],
                "error_energy": fmt["error_energy"],
                "storage_nmse": rec["E_target_storage"]["nmse"],
                "within_tensor_spearman_log2range_vs_nmse": rec[
                    "within_tensor_spearman_log2range_vs_nmse"
                ],
                "num_groups": rec["num_groups"],
            }
            tensor_rows.append(row)
            corr_points.append(
                {
                    "layer_idx": layer_idx,
                    "projection": projection,
                    "tensor_name": tensor_name,
                    "spearman": rec["within_tensor_spearman_log2range_vs_nmse"],
                    "nmse": fmt["nmse"],
                    "error_energy": fmt["error_energy"],
                }
            )
            print(
                f"[W] shard{shard_id} L{layer_idx}.{projection} nmse={fmt['nmse']:.6e}",
                flush=True,
            )

    # Global NMSE from accumulated energies
    global_metrics = {
        "nmse": (
            global_acc.error_energy / global_acc.reference_energy
            if global_acc.reference_energy > 0
            else 0.0
        ),
        "reference_energy": global_acc.reference_energy,
        "error_energy": global_acc.error_energy,
        "numel": global_acc.numel,
        "max_abs_error": global_acc.max_abs_error,
        "mae": (
            global_acc.abs_error_sum / global_acc.numel if global_acc.numel else 0.0
        ),
    }

    # Cross-tensor Spearman of per-tensor mean dispersion proxy vs nmse:
    # use within-tensor spearman as observational summary point + nmse
    if len(corr_points) >= 2:
        cross = spearman_with_bootstrap(
            [c["spearman"] for c in corr_points],
            [c["nmse"] for c in corr_points],
            seed=20260810,
            repeats=1000,
            cluster_ids=[c["layer_idx"] for c in corr_points],
        )
    else:
        cross = {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": float(len(corr_points))}

    summary = {
        "shard_id": shard_id,
        "num_shards": num_shards,
        "device": str(device_t),
        "num_tensors": len(tensor_rows),
        "global_format_metrics": global_metrics,
        "cross_tensor_spearman_withinCorr_vs_nmse": cross,
        "hypothesis_H1a": (
            "observational_correlation: larger sub16 dynamic-range dispersion "
            "tends to co-occur with larger HiF4 group NMSE; not causal yet"
        ),
        "evidence_class": "observational_correlation",
    }
    write_csv(out_dir / f"weight_tensor_summary_shard{shard_id}.csv", tensor_rows)
    atomic_write_json(out_dir / f"weight_summary_shard{shard_id}.json", summary)
    atomic_write_json(out_dir / f"weight_corr_points_shard{shard_id}.json", corr_points)
    return summary
