"""Global-barrier experiment pipeline for HiF4 deployment-equivalent scaling.

The pipeline is intentionally split into raw-stat collection, global stat merge,
fixed candidate construction, fixed-candidate evaluation, and raw-error merge.
No shard is allowed to derive its own deployable D.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

import torch
import torch.nn as nn
import yaml

from Inference_Paradigm_Conversion.ipc_analysis.analysis.hif4_equivalent_scaling import (
    build_equalization_scale,
    build_weight_aware_equalization_scale,
    candidate_pts_scales,
    collapse_gqa_o_amplitude,
    expand_gqa_o_scale,
    expand_group_scales,
    finalize_channel_amplitude,
    shared_input_weight_stat,
    update_channel_stats,
)
from Inference_Paradigm_Conversion.ipc_analysis.config import load_experiment_config
from Inference_Paradigm_Conversion.ipc_analysis.formats.fingerprint import (
    list_safetensor_keys,
    load_nvfp4_qat_dequant_weight,
)
from Inference_Paradigm_Conversion.ipc_analysis.formats.hif4 import quantize_hif4_tensor
from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import (
    load_nvfp4_activation_scales,
    quantize_nvfp4_activation,
    resolve_activation_scale_path,
    resolve_nvfp4_scale_for_module,
)

SCHEMA_VERSION = 1
DOMAINS = ("attn_in", "mlp_in", "down_in", "o_in")
DOMAIN_PROJECTIONS: dict[str, tuple[str, ...]] = {
    "attn_in": ("q_proj", "k_proj", "v_proj"),
    "mlp_in": ("gate_proj", "up_proj"),
    "down_in": ("down_proj",),
    "o_in": ("o_proj",),
}
CANONICAL_PROJECTION = {
    "attn_in": "q_proj",
    "mlp_in": "gate_proj",
    "down_in": "down_proj",
    "o_in": "o_proj",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_jsonable(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _safe_torch_load(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"expected dict artifact at {path}")
    return obj


def _alpha_slug(alpha: float) -> str:
    x = float(alpha)
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    return (f"{x:.4f}".rstrip("0").rstrip(".")).replace("-", "m").replace(".", "p")


def _key(layer: int, domain: str, phase: str) -> str:
    return f"{int(layer)}:{domain}:{phase}"


def _split_key(key: str) -> tuple[int, str, str]:
    a, b, c = key.split(":", 2)
    return int(a), b, c


def _projection_to_domain(projection: str) -> str:
    for domain, projections in DOMAIN_PROJECTIONS.items():
        if projection in projections:
            return domain
    raise ValueError(f"unsupported projection {projection!r}")


def _layer_idx(name: str) -> int | None:
    parts = name.split(".")
    for i, part in enumerate(parts[:-1]):
        if part == "layers" and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def _projection(name: str) -> str | None:
    for projection in (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ):
        if name.endswith(projection):
            return projection
    return None


def load_scaling_experiment_config(path: Path | str) -> tuple[Any, dict[str, Any], str]:
    """Load the scaling overlay plus its referenced formal IPC configuration."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or not isinstance(raw.get("experiment"), dict):
        raise TypeError("scaling config must contain an experiment mapping")
    base = Path(str(raw.get("base_config", "")))
    if not base.is_absolute():
        base = Path("/home/shaoyuantian/program/HiF4_Sp") / base
    formal = load_experiment_config(base)
    experiment = dict(raw["experiment"])
    experiment["checkpoint"] = str(formal.source_checkpoint_path())
    experiment["base_config"] = str(base.resolve())
    config_hash = _sha256_jsonable({"base_config": str(base.resolve()), "experiment": experiment})
    return formal, experiment, config_hash


def _validate_shard_common(
    artifacts: list[dict[str, Any]],
    *,
    expected_num_shards: int,
    hash_fields: tuple[str, ...],
) -> None:
    if len(artifacts) != expected_num_shards:
        raise ValueError(
            f"expected {expected_num_shards} shard artifacts, found {len(artifacts)}"
        )
    ids = sorted(int(x["shard_id"]) for x in artifacts)
    if ids != list(range(expected_num_shards)):
        raise ValueError(f"shard ids must be 0..{expected_num_shards - 1}, got {ids}")
    if any(int(x.get("num_shards", -1)) != expected_num_shards for x in artifacts):
        raise ValueError("num_shards metadata mismatch")
    splits = {str(x.get("split")) for x in artifacts}
    if len(splits) != 1:
        raise ValueError(f"split mismatch across shards: {sorted(splits)}")
    for field in hash_fields:
        vals = {str(x.get(field)) for x in artifacts}
        if len(vals) != 1:
            raise ValueError(f"{field} mismatch across shards: {sorted(vals)}")
    sample_ids: list[str] = []
    for art in artifacts:
        sample_ids.extend(str(x) for x in art.get("sample_ids", []))
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample ids overlap across shards")


def merge_scaling_stats(
    run_dir: Path,
    *,
    expected_num_shards: int,
) -> Path:
    """Merge raw sufficient statistics; never average finalized amplitudes."""
    run_dir = Path(run_dir)
    paths = [run_dir / f"es_stats_shard{i}.pt" for i in range(expected_num_shards)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing stats shards: {missing}")
    shards = [_safe_torch_load(p) for p in paths]
    _validate_shard_common(
        shards,
        expected_num_shards=expected_num_shards,
        hash_fields=("config_sha256",),
    )

    model_meta = shards[0].get("model_meta", {})
    if any(x.get("model_meta", {}) != model_meta for x in shards[1:]):
        raise ValueError("model_meta mismatch across stats shards")

    merged_stats: dict[str, dict[str, Any]] = {}
    stat_keys = sorted({k for shard in shards for k in shard.get("stats", {})})
    for key in stat_keys:
        parts = [s["stats"][key] for s in shards if key in s.get("stats", {})]
        if len(parts) != len(shards):
            raise ValueError(f"stats key {key!r} missing from one or more shards")
        sum_sq = sum((p["sum_sq"].to(torch.float64) for p in parts), torch.zeros_like(parts[0]["sum_sq"], dtype=torch.float64))
        max_abs = parts[0]["max_abs"].to(torch.float64).clone()
        for p in parts[1:]:
            max_abs = torch.maximum(max_abs, p["max_abs"].to(torch.float64))
        merged_stats[key] = {
            "sum_sq": sum_sq,
            "max_abs": max_abs,
            "count": int(sum(int(p["count"]) for p in parts)),
        }

    merged_phase: dict[str, dict[str, Any]] = {}
    phase_keys = sorted({k for shard in shards for k in shard.get("phase_g64", {})})
    for key in phase_keys:
        parts = [s["phase_g64"][key] for s in shards if key in s.get("phase_g64", {})]
        if len(parts) != len(shards):
            raise ValueError(f"phase_g64 key {key!r} missing from one or more shards")
        shape = tuple(parts[0]["error_sum"].shape)
        if any(tuple(p["error_sum"].shape) != shape for p in parts):
            raise ValueError(f"phase_g64 shape mismatch for {key}")
        err = sum((p["error_sum"].to(torch.float64) for p in parts), torch.zeros_like(parts[0]["error_sum"], dtype=torch.float64))
        merged_phase[key] = {
            "error_sum": err,
            "count": int(sum(int(p["count"]) for p in parts)),
        }

    out = {
        "schema_version": SCHEMA_VERSION,
        "split": shards[0]["split"],
        "shard_id": None,
        "num_shards": expected_num_shards,
        "config_sha256": shards[0]["config_sha256"],
        "sample_ids": sorted(
            str(x) for shard in shards for x in shard.get("sample_ids", [])
        ),
        "model_meta": model_meta,
        "stats": merged_stats,
        "phase_g64": merged_phase,
    }
    out_path = run_dir / "es_stats_merged.pt"
    torch.save(out, out_path)
    return out_path


def _deploy_amplitude(
    merged: dict[str, Any],
    *,
    layer: int,
    domain: str,
) -> torch.Tensor:
    phase_amps: list[torch.Tensor] = []
    for phase in ("prefill", "decode"):
        rec = merged.get("stats", {}).get(_key(layer, domain, phase))
        if rec is None:
            continue
        phase_amps.append(
            finalize_channel_amplitude(rec["sum_sq"], rec["max_abs"], int(rec["count"]))
            .to(torch.float32)
        )
    if not phase_amps:
        raise KeyError(f"no activation stats for layer={layer} domain={domain}")
    width = phase_amps[0].numel()
    if any(x.numel() != width for x in phase_amps):
        raise ValueError(f"phase amplitude width mismatch for layer={layer} domain={domain}")
    out = phase_amps[0]
    for x in phase_amps[1:]:
        out = torch.maximum(out, x)
    return out


def _model_meta_int(meta: dict[str, Any], key: str) -> int:
    if key not in meta:
        raise KeyError(f"model_meta missing {key}")
    return int(meta[key])


def _o_unique_amplitude(amplitude: torch.Tensor, model_meta: dict[str, Any]) -> torch.Tensor:
    return collapse_gqa_o_amplitude(
        amplitude,
        num_attention_heads=_model_meta_int(model_meta, "num_attention_heads"),
        num_key_value_heads=_model_meta_int(model_meta, "num_key_value_heads"),
        head_dim=_model_meta_int(model_meta, "head_dim"),
        reduction="max",
    ).reshape(-1)


def _expand_o_unique(d_unique: torch.Tensor, model_meta: dict[str, Any]) -> torch.Tensor:
    return expand_gqa_o_scale(
        d_unique,
        num_attention_heads=_model_meta_int(model_meta, "num_attention_heads"),
        num_key_value_heads=_model_meta_int(model_meta, "num_key_value_heads"),
        head_dim=_model_meta_int(model_meta, "head_dim"),
    )


def _collapse_o_phase_error(
    error: torch.Tensor,
    model_meta: dict[str, Any],
    *,
    group_size: int,
) -> torch.Tensor:
    """Tie per-query-head G64 phase errors into the unique KV-head scale domain."""
    hq = _model_meta_int(model_meta, "num_attention_heads")
    hkv = _model_meta_int(model_meta, "num_key_value_heads")
    hd = _model_meta_int(model_meta, "head_dim")
    repeat = hq // hkv
    if hd % group_size != 0:
        raise ValueError("head_dim must be divisible by phase_g64 group_size")
    unique_groups = hkv * (hd // group_size)
    if error.shape[0] == unique_groups:
        return error
    full_groups = hq * (hd // group_size)
    if error.shape[0] != full_groups:
        raise ValueError(
            f"o_in phase error has {error.shape[0]} groups; expected {unique_groups} or {full_groups}"
        )
    return error.reshape(hkv, repeat, hd // group_size, error.shape[1]).sum(dim=1).reshape(unique_groups, error.shape[1])


def _phase_g64_scale(
    merged: dict[str, Any],
    *,
    layer: int,
    domain: str,
    width: int,
    pts_grid: torch.Tensor,
    group_size: int,
    model_meta: dict[str, Any],
) -> tuple[torch.Tensor, list[int]]:
    normalized_phase_errors: list[torch.Tensor] = []
    for phase in ("prefill", "decode"):
        rec = merged.get("phase_g64", {}).get(_key(layer, domain, phase))
        if rec is None:
            continue
        count = int(rec["count"])
        if count <= 0:
            continue
        err = rec["error_sum"].to(torch.float64) / float(count)
        if domain == "o_in":
            err = _collapse_o_phase_error(err, model_meta, group_size=group_size)
        normalized_phase_errors.append(err)
    if not normalized_phase_errors:
        raise KeyError(f"no phase_g64 errors for layer={layer} domain={domain}")
    shape = tuple(normalized_phase_errors[0].shape)
    if shape[1] != pts_grid.numel():
        raise ValueError("phase_g64 candidate count does not match PTS grid")
    if any(tuple(x.shape) != shape for x in normalized_phase_errors):
        raise ValueError("phase_g64 prefill/decode shapes differ")
    objective = sum(normalized_phase_errors) / float(len(normalized_phase_errors))
    best = objective.argmin(dim=1).to(torch.long)
    group_scales = pts_grid[best].to(torch.float32)
    if domain == "o_in":
        unique_width = _model_meta_int(model_meta, "num_key_value_heads") * _model_meta_int(model_meta, "head_dim")
        d_unique = expand_group_scales(group_scales, width=unique_width, group_size=group_size)
        d = _expand_o_unique(d_unique, model_meta)
    else:
        d = expand_group_scales(group_scales, width=width, group_size=group_size)
    return d, [int(x) for x in best.tolist()]


def _resolve_weight_name(keys: dict[str, str], layer: int, projection: str) -> str:
    suffixes = (
        f"model.layers.{layer}.self_attn.{projection}.weight",
        f"model.layers.{layer}.mlp.{projection}.weight",
    )
    matches = [name for name in keys if any(name.endswith(s) for s in suffixes)]
    if len(matches) != 1:
        raise KeyError(
            f"expected one checkpoint tensor for layer={layer} projection={projection}, got {matches}"
        )
    return matches[0]


def _shared_weight_stat_from_checkpoint(
    checkpoint: Path,
    *,
    layer: int,
    domain: str,
) -> torch.Tensor:
    if domain not in {"attn_in", "mlp_in"}:
        raise ValueError("weight-aware comparator is only defined for attn_in/mlp_in")
    keys = list_safetensor_keys(checkpoint)
    weights: list[torch.Tensor] = []
    for projection in DOMAIN_PROJECTIONS[domain]:
        name = _resolve_weight_name(keys, layer, projection)
        weights.append(
            load_nvfp4_qat_dequant_weight(checkpoint, name, device="cpu").dequantized
        )
    return shared_input_weight_stat(tuple(weights))


def _register_recipe(recipes: dict[str, dict[str, Any]], recipe_id: str, **fields: Any) -> None:
    rec = dict(fields)
    if recipe_id in recipes and recipes[recipe_id] != rec:
        raise ValueError(f"recipe id collision for {recipe_id}")
    recipes[recipe_id] = rec


def build_candidate_scales(
    merged_stats_path: Path,
    *,
    config: dict[str, Any],
) -> Path:
    """Build the one fixed candidate scale artifact consumed by all eval shards."""
    merged_stats_path = Path(merged_stats_path)
    merged = _safe_torch_load(merged_stats_path)
    if str(merged.get("split")) != "discovery":
        raise ValueError("candidate scales may only be built from discovery stats")

    group_size = int(config.get("group_size", 64))
    layers = [int(x) for x in config.get("representative_layers", [])]
    if not layers:
        layers = sorted({_split_key(k)[0] for k in merged.get("stats", {})})
    pts_grid = candidate_pts_scales(
        log2_min=float(config.get("pts_log2_min", -1.0)),
        log2_max=float(config.get("pts_log2_max", 1.0)),
        points=int(config.get("pts_points", 33)),
    )
    model_meta = dict(merged.get("model_meta", {}))
    recipes: dict[str, dict[str, Any]] = {}
    scales: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    phase_best: dict[str, list[int]] = {}

    checkpoint_raw = config.get("checkpoint")
    checkpoint = Path(str(checkpoint_raw)) if checkpoint_raw else None
    run_aw = bool(config.get("run_weight_aware_balance", False))
    weight_stat_cache: dict[tuple[int, str], torch.Tensor] = {}

    for layer in layers:
        layer_key = str(layer)
        scales[layer_key] = {}
        available_domains = {
            _split_key(k)[1]
            for k in merged.get("stats", {})
            if _split_key(k)[0] == layer
        }
        for domain in DOMAINS:
            if domain not in available_domains:
                continue
            amplitude_full = _deploy_amplitude(merged, layer=layer, domain=domain)
            width_full = int(amplitude_full.numel())
            if width_full % group_size != 0:
                raise ValueError(
                    f"layer={layer} domain={domain} width={width_full} not divisible by {group_size}"
                )
            if domain == "o_in":
                amplitude_design = _o_unique_amplitude(amplitude_full, model_meta)
            else:
                amplitude_design = amplitude_full
            domain_scales: dict[str, torch.Tensor] = {}

            if bool(config.get("run_pts_layer", True)):
                for idx, c in enumerate(pts_grid):
                    rid = f"pts_layer_c{idx:02d}"
                    domain_scales[rid] = torch.full(
                        (width_full,), float(c.item()), dtype=torch.float32
                    )
                    _register_recipe(
                        recipes,
                        rid,
                        kind="pts_layer",
                        granularity=0,
                        alpha=0.0,
                        scalar=float(c.item()),
                        min_scale=float(config.get("min_scale", 0.5)),
                        max_scale=float(config.get("max_scale", 2.0)),
                        deployable=True,
                    )

            if bool(config.get("run_phase_g64", True)):
                d, best = _phase_g64_scale(
                    merged,
                    layer=layer,
                    domain=domain,
                    width=width_full,
                    pts_grid=pts_grid,
                    group_size=group_size,
                    model_meta=model_meta,
                )
                domain_scales["phase_g64"] = d
                phase_best[f"{layer}:{domain}"] = best
                _register_recipe(
                    recipes,
                    "phase_g64",
                    kind="phase_g64",
                    granularity=64,
                    alpha=0.0,
                    min_scale=float(config.get("min_scale", 0.5)),
                    max_scale=float(config.get("max_scale", 2.0)),
                    deployable=True,
                )

            for granularity in [int(x) for x in config.get("equalization_granularities", [16, 8, 4, 1])]:
                for alpha in [float(x) for x in config.get("alphas", [0.0, 0.25, 0.5, 0.75, 1.0])]:
                    rid = f"eq_g{granularity}_a{_alpha_slug(alpha)}"
                    d_design = build_equalization_scale(
                        amplitude_design,
                        granularity=granularity,
                        alpha=alpha,
                        group_size=group_size,
                        min_scale=float(config.get("min_scale", 0.5)),
                        max_scale=float(config.get("max_scale", 2.0)),
                    )
                    d = _expand_o_unique(d_design, model_meta) if domain == "o_in" else d_design
                    domain_scales[rid] = d
                    _register_recipe(
                        recipes,
                        rid,
                        kind="equalize",
                        granularity=granularity,
                        alpha=alpha,
                        min_scale=float(config.get("min_scale", 0.5)),
                        max_scale=float(config.get("max_scale", 2.0)),
                        deployable=True,
                    )

                    if domain == "o_in" and bool(config.get("run_o_free_oracle", True)):
                        free_rid = f"o_free_g{granularity}_a{_alpha_slug(alpha)}"
                        free_d = build_equalization_scale(
                            amplitude_full,
                            granularity=granularity,
                            alpha=alpha,
                            group_size=group_size,
                            min_scale=float(config.get("min_scale", 0.5)),
                            max_scale=float(config.get("max_scale", 2.0)),
                        )
                        domain_scales[free_rid] = free_d
                        _register_recipe(
                            recipes,
                            free_rid,
                            kind="equalize",
                            granularity=granularity,
                            alpha=alpha,
                            min_scale=float(config.get("min_scale", 0.5)),
                            max_scale=float(config.get("max_scale", 2.0)),
                            deployable=False,
                            diagnostic="o_free_oracle",
                        )

                    if bool(config.get("build_wide_bound_candidates", False)):
                        wrid = f"eqwide_g{granularity}_a{_alpha_slug(alpha)}"
                        wd_design = build_equalization_scale(
                            amplitude_design,
                            granularity=granularity,
                            alpha=alpha,
                            group_size=group_size,
                            min_scale=float(config.get("wide_bound_min_scale", 0.25)),
                            max_scale=float(config.get("wide_bound_max_scale", 4.0)),
                        )
                        wd = _expand_o_unique(wd_design, model_meta) if domain == "o_in" else wd_design
                        domain_scales[wrid] = wd
                        _register_recipe(
                            recipes,
                            wrid,
                            kind="equalize",
                            granularity=granularity,
                            alpha=alpha,
                            min_scale=float(config.get("wide_bound_min_scale", 0.25)),
                            max_scale=float(config.get("wide_bound_max_scale", 4.0)),
                            deployable=False,
                            diagnostic="wide_bound",
                        )

                if run_aw and domain in {"attn_in", "mlp_in"}:
                    if checkpoint is None:
                        raise ValueError("weight-aware balance requires config['checkpoint']")
                    cache_key = (layer, domain)
                    if cache_key not in weight_stat_cache:
                        weight_stat_cache[cache_key] = _shared_weight_stat_from_checkpoint(
                            checkpoint, layer=layer, domain=domain
                        )
                    rid = f"eqaw_g{granularity}"
                    d = build_weight_aware_equalization_scale(
                        amplitude_design,
                        weight_stat_cache[cache_key],
                        granularity=granularity,
                        beta=float(config.get("weight_aware_beta", 0.5)),
                        group_size=group_size,
                        min_scale=float(config.get("min_scale", 0.5)),
                        max_scale=float(config.get("max_scale", 2.0)),
                    )
                    domain_scales[rid] = d
                    _register_recipe(
                        recipes,
                        rid,
                        kind="equalize_aw",
                        granularity=granularity,
                        alpha=float(config.get("weight_aware_beta", 0.5)),
                        beta=float(config.get("weight_aware_beta", 0.5)),
                        min_scale=float(config.get("min_scale", 0.5)),
                        max_scale=float(config.get("max_scale", 2.0)),
                        deployable=True,
                    )

            scales[layer_key][domain] = domain_scales

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "split": "discovery",
        "config_sha256": str(merged.get("config_sha256")),
        "stats_merged_sha256": _sha256_file(merged_stats_path),
        "model_meta": model_meta,
        "pts_grid": pts_grid,
        "recipes": recipes,
        "scales": scales,
        "phase_g64_best_indices": phase_best,
    }
    out_path = merged_stats_path.parent / "candidate_scales.pt"
    torch.save(artifact, out_path)
    return out_path


def _sum_numeric_record(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, bool):
            dst[key] = bool(dst.get(key, False) or value)
        elif isinstance(value, (int, float)):
            dst[key] = dst.get(key, 0) + value
        elif key not in dst:
            dst[key] = value
        elif dst[key] != value:
            raise ValueError(f"non-numeric record field mismatch for {key}: {dst[key]!r} vs {value!r}")


def _derive_recovery_fields(rec: dict[str, Any], *, eps: float = 1e-12) -> None:
    pairs = (
        ("joint_conv_error_sum", "baseline_conv_error_sum", "joint_R_Y_conv"),
        ("joint_local_error_sum", "baseline_local_error_sum", "joint_R_Y_local"),
        ("activation_conv_error_sum", "baseline_activation_conv_error_sum", "activation_R_Y_conv"),
        ("activation_local_error_sum", "baseline_activation_local_error_sum", "activation_R_Y_local"),
        ("weight_local_error_sum", "baseline_weight_local_error_sum", "weight_R_Y_local"),
    )
    for num_key, den_key, out_key in pairs:
        if num_key not in rec or den_key not in rec:
            continue
        den = float(rec[den_key])
        if den < eps:
            rec[out_key] = None
            rec[f"{out_key}_valid"] = False
        else:
            rec[out_key] = 1.0 - float(rec[num_key]) / den
            rec[f"{out_key}_valid"] = True


def merge_scaling_eval(
    run_dir: Path,
    *,
    expected_num_shards: int,
) -> dict[str, Any]:
    """Merge raw candidate error sums and derive ratios only after global aggregation."""
    run_dir = Path(run_dir)
    paths = [run_dir / f"es_eval_shard{i}.pt" for i in range(expected_num_shards)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing eval shards: {missing}")
    shards = [_safe_torch_load(p) for p in paths]
    _validate_shard_common(
        shards,
        expected_num_shards=expected_num_shards,
        hash_fields=("config_sha256", "stats_merged_sha256", "candidate_scales_sha256"),
    )

    records: dict[str, dict[str, Any]] = {}
    for shard in shards:
        for key, src in shard.get("records", {}).items():
            dst = records.setdefault(key, {})
            _sum_numeric_record(dst, src)
    for rec in records.values():
        _derive_recovery_fields(rec)

    merged = {
        "schema_version": SCHEMA_VERSION,
        "split": shards[0]["split"],
        "config_sha256": shards[0]["config_sha256"],
        "stats_merged_sha256": shards[0]["stats_merged_sha256"],
        "candidate_scales_sha256": shards[0]["candidate_scales_sha256"],
        "sample_ids": sorted(
            str(x) for shard in shards for x in shard.get("sample_ids", [])
        ),
        "records": records,
    }
    out_path = run_dir / "es_eval_merged.pt"
    torch.save(merged, out_path)
    return merged


# ---------------------------------------------------------------------------
# Formal-model execution below. These functions are deliberately downstream
# of the stat/candidate barrier above.
# ---------------------------------------------------------------------------


@dataclass
class _ForwardState:
    sample_id: str = ""
    prompt_family: str = ""
    phase: str = "prefill"
    split: str = "discovery"


class _SourceTrajectoryHooks:
    """Capture raw X_in and then apply NVFP4 A4 in the same pre-hook.

    This ordering is important: collector sees the pre-current-Linear activation,
    while the actual source forward still consumes NVFP4 A4.
    """

    def __init__(
        self,
        model: nn.Module,
        scales: dict[str, torch.Tensor],
        *,
        collector: Callable[[str, nn.Linear, torch.Tensor, _ForwardState], None] | None,
        state: _ForwardState,
    ) -> None:
        self.model = model
        self.scales = scales
        self.collector = collector
        self.state = state
        self.handles: list[Any] = []

    def __enter__(self) -> "_SourceTrajectoryHooks":
        def make_hook(name: str):
            def hook(mod: nn.Linear, inputs: tuple[Any, ...]):
                if not inputs or not torch.is_tensor(inputs[0]):
                    return None
                x = inputs[0]
                if self.collector is not None:
                    self.collector(name, mod, x, self.state)
                scale = resolve_nvfp4_scale_for_module(self.scales, name).to(x.device)
                q = quantize_nvfp4_activation(
                    x.to(torch.bfloat16),
                    scale,
                    output_dtype=x.dtype,
                ).dequantized
                return (q,) + tuple(inputs[1:])

            return hook

        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Linear) and "lm_head" not in name and _projection(name) is not None:
                self.handles.append(mod.register_forward_pre_hook(make_hook(name)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _prompt_bank(split: str, samples_per_family: int):
    from Inference_Paradigm_Conversion.ipc_analysis.capture.prompts import (
        discovery_items,
        validation_items,
    )

    if split == "discovery":
        return discovery_items(samples_per_family)
    if split == "validation":
        return validation_items(samples_per_family)
    raise ValueError(f"split must be discovery|validation, got {split!r}")


def _load_model_meta(checkpoint: Path) -> dict[str, Any]:
    with (checkpoint / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {
        "num_hidden_layers": int(cfg["num_hidden_layers"]),
        "hidden_size": int(cfg["hidden_size"]),
        "intermediate_size": int(cfg["intermediate_size"]),
        "num_attention_heads": int(cfg["num_attention_heads"]),
        "num_key_value_heads": int(cfg["num_key_value_heads"]),
        "head_dim": int(cfg.get("head_dim", int(cfg["hidden_size"]) // int(cfg["num_attention_heads"]))),
    }


def _init_stat(width: int) -> dict[str, Any]:
    return {
        "sum_sq": torch.zeros(width, dtype=torch.float64),
        "max_abs": torch.zeros(width, dtype=torch.float64),
        "count": 0,
    }


def _sample_rows(x: torch.Tensor, *, max_rows: int) -> torch.Tensor:
    flat = x.detach().reshape(-1, x.shape[-1])
    return flat[: min(int(max_rows), int(flat.shape[0]))]


def _phase_error_accumulator(num_groups: int, points: int) -> dict[str, Any]:
    return {
        "error_sum": torch.zeros((num_groups, points), dtype=torch.float64),
        "count": 0,
    }


def _phase_g64_error_for_x(
    x: torch.Tensor,
    *,
    domain: str,
    module_names: tuple[str, ...],
    nv_scales: dict[str, torch.Tensor],
    pts_grid: torch.Tensor,
    model_meta: dict[str, Any],
    group_size: int,
) -> tuple[torch.Tensor, int]:
    """Activation-only restored error for every G64 phase candidate."""
    x = _sample_rows(x, max_rows=32).to(torch.bfloat16)
    k = int(x.shape[-1])
    if k % group_size != 0:
        raise ValueError(f"activation K={k} is not divisible by {group_size}")
    refs: list[torch.Tensor] = []
    for name in module_names:
        scale = resolve_nvfp4_scale_for_module(nv_scales, name).to(x.device)
        refs.append(
            quantize_nvfp4_activation(x, scale, output_dtype=torch.float32).dequantized
        )

    if domain == "o_in":
        hq = _model_meta_int(model_meta, "num_attention_heads")
        hkv = _model_meta_int(model_meta, "num_key_value_heads")
        hd = _model_meta_int(model_meta, "head_dim")
        repeat = hq // hkv
        groups_per_head = hd // group_size
        err = torch.zeros((hkv, groups_per_head, pts_grid.numel()), dtype=torch.float64)
    else:
        err = torch.zeros((k // group_size, pts_grid.numel()), dtype=torch.float64)

    for ci, c_cpu in enumerate(pts_grid):
        c = c_cpu.to(device=x.device, dtype=torch.float32)
        ah = quantize_hif4_tensor(
            x.float() / c,
            variant="full",
            output_dtype=torch.float32,
        ).dequantized * c
        for ref in refs:
            sq = (ah - ref).float().pow(2).reshape(x.shape[0], -1, group_size).sum(dim=(0, 2))
            if domain == "o_in":
                # [Hq * groups_per_head] -> [Hkv, repeat, groups_per_head].
                sq = sq.reshape(hkv, repeat, groups_per_head).sum(dim=1)
                err[..., ci] += sq.detach().cpu().to(torch.float64)
            else:
                err[:, ci] += sq.detach().cpu().to(torch.float64)
    if domain == "o_in":
        return err.reshape(-1, pts_grid.numel()), int(x.shape[0] * group_size * len(refs) * repeat)
    return err, int(x.shape[0] * group_size * len(refs))


def run_scaling_stats_shard(
    checkpoint: Path,
    out_dir: Path,
    *,
    split: str,
    shard_id: int,
    num_shards: int,
    device: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Stage A: source-trajectory raw statistics only; never chooses D."""
    from Inference_Paradigm_Conversion.ipc_analysis.capture.model_adapter import (
        load_source_model_for_capture,
    )

    checkpoint = Path(checkpoint)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_meta = _load_model_meta(checkpoint)
    rep_layers = {int(x) for x in config.get("representative_layers", [4, 18, 34])}
    samples_per_family = int(config.get("samples_per_family", 8))
    max_seq_len = int(config.get("max_seq_len", 256))
    decode_steps = int(config.get("decode_steps", 8))
    max_stat_rows = int(config.get("max_stat_rows_per_capture", 256))
    group_size = int(config.get("group_size", 64))
    pts_grid = candidate_pts_scales(
        log2_min=float(config.get("pts_log2_min", -1.0)),
        log2_max=float(config.get("pts_log2_max", 1.0)),
        points=int(config.get("pts_points", 33)),
    )
    config_hash = str(config.get("config_sha256") or _sha256_jsonable(config))

    device_t = torch.device(device)
    if device_t.type == "cuda":
        torch.cuda.set_device(device_t)
    model, tok = load_source_model_for_capture(checkpoint, device=device_t)
    nv_scales = load_nvfp4_activation_scales(resolve_activation_scale_path(checkpoint))
    nv_scales = {k: v.to(device_t) for k, v in nv_scales.items()}
    bank = _prompt_bank(split, samples_per_family)
    prompts = [item for i, item in enumerate(bank) if i % num_shards == shard_id]

    stats: dict[str, dict[str, Any]] = {}
    phase_errors: dict[str, dict[str, Any]] = {}
    state = _ForwardState(split=split)

    target_names_by_layer_domain: dict[tuple[int, str], tuple[str, ...]] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        li = _layer_idx(name)
        proj = _projection(name)
        if li not in rep_layers or proj is None:
            continue
        domain = _projection_to_domain(proj)
        target_names_by_layer_domain.setdefault((int(li), domain), tuple())
    # Resolve exact module names for each domain from the actual model rather than hard-coding prefix.
    for li, domain in list(target_names_by_layer_domain):
        names = []
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear) or _layer_idx(name) != li:
                continue
            if _projection(name) in DOMAIN_PROJECTIONS[domain]:
                names.append(name)
        if len(names) != len(DOMAIN_PROJECTIONS[domain]):
            raise RuntimeError(
                f"layer={li} domain={domain}: expected {DOMAIN_PROJECTIONS[domain]}, got {names}"
            )
        target_names_by_layer_domain[(li, domain)] = tuple(sorted(names))

    def collector(name: str, _mod: nn.Linear, x: torch.Tensor, fstate: _ForwardState) -> None:
        li = _layer_idx(name)
        proj = _projection(name)
        if li not in rep_layers or proj is None:
            return
        domain = _projection_to_domain(proj)
        if proj != CANONICAL_PROJECTION[domain]:
            return
        sample = _sample_rows(x, max_rows=max_stat_rows).detach()
        rec_key = _key(int(li), domain, fstate.phase)
        rec = stats.setdefault(rec_key, _init_stat(int(sample.shape[-1])))
        ss, ma, count = update_channel_stats(rec["sum_sq"], rec["max_abs"], int(rec["count"]), sample)
        rec["sum_sq"] = ss.cpu()
        rec["max_abs"] = ma.cpu()
        rec["count"] = count

        collect_phase = bool(config.get("collect_phase_g64_errors", True))
        phase_domains = set(config.get("phase_g64_domains", DOMAINS))
        if collect_phase and domain in phase_domains:
            names = target_names_by_layer_domain[(int(li), domain)]
            err, n = _phase_g64_error_for_x(
                sample,
                domain=domain,
                module_names=names,
                nv_scales=nv_scales,
                pts_grid=pts_grid,
                model_meta=model_meta,
                group_size=group_size,
            )
            er = phase_errors.setdefault(
                rec_key,
                _phase_error_accumulator(int(err.shape[0]), int(err.shape[1])),
            )
            er["error_sum"] += err
            er["count"] += int(n)

    with _SourceTrajectoryHooks(model, nv_scales, collector=collector, state=state):
        for item in prompts:
            state.sample_id = item.sample_id
            state.prompt_family = item.family
            enc = tok(item.text, return_tensors="pt", truncation=True, max_length=max_seq_len)
            batch = {k: v.to(device_t) for k, v in enc.items()}
            if "attention_mask" not in batch:
                batch["attention_mask"] = torch.ones_like(batch["input_ids"])

            state.phase = "prefill"
            warm = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=True,
            )
            past = warm.past_key_values
            next_id = warm.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            attn = torch.cat(
                [batch["attention_mask"], torch.ones_like(next_id, dtype=batch["attention_mask"].dtype)],
                dim=-1,
            )
            state.phase = "decode"
            current = next_id
            for _ in range(decode_steps):
                out = model(
                    input_ids=current,
                    attention_mask=attn,
                    past_key_values=past,
                    use_cache=True,
                )
                past = out.past_key_values
                current = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                attn = torch.cat(
                    [attn, torch.ones_like(current, dtype=attn.dtype)], dim=-1
                )

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "shard_id": int(shard_id),
        "num_shards": int(num_shards),
        "config_sha256": config_hash,
        "sample_ids": [item.sample_id for item in prompts],
        "model_meta": model_meta,
        "stats": stats,
        "phase_g64": phase_errors,
    }
    path = out_dir / f"es_stats_shard{shard_id}.pt"
    torch.save(artifact, path)
    del model
    if device_t.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "path": str(path),
        "num_prompts": len(prompts),
        "num_stat_keys": len(stats),
        "config_sha256": config_hash,
    }


def _module_name_map(model: nn.Module, layers: set[int]) -> dict[tuple[int, str], str]:
    out: dict[tuple[int, str], str] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        li = _layer_idx(name)
        proj = _projection(name)
        if li in layers and proj is not None:
            key = (int(li), proj)
            if key in out:
                raise RuntimeError(f"duplicate Linear mapping for {key}: {out[key]} and {name}")
            out[key] = name
    for li in layers:
        for proj in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ):
            if (li, proj) not in out:
                raise RuntimeError(f"missing model Linear for layer={li} projection={proj}")
    return out


@torch.no_grad()
def _capture_source_inputs(
    checkpoint: Path,
    *,
    split: str,
    shard_id: int,
    num_shards: int,
    device: str,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[int, str], str], dict[str, Any]]:
    """Capture small raw X_in samples while the real forward consumes NVFP4 A4."""
    from Inference_Paradigm_Conversion.ipc_analysis.capture.model_adapter import (
        load_source_model_for_capture,
    )

    checkpoint = Path(checkpoint)
    device_t = torch.device(device)
    if device_t.type == "cuda":
        torch.cuda.set_device(device_t)
    layers = {int(x) for x in config.get("representative_layers", [4, 18, 34])}
    model_meta = _load_model_meta(checkpoint)
    model, tok = load_source_model_for_capture(checkpoint, device=device_t)
    module_names = _module_name_map(model, layers)
    nv_scales = load_nvfp4_activation_scales(resolve_activation_scale_path(checkpoint))
    nv_scales = {k: v.to(device_t) for k, v in nv_scales.items()}
    state = _ForwardState(split=split)
    rows_prefill = int(config.get("max_eval_rows_prefill", 32))
    rows_decode = int(config.get("max_eval_rows_decode", 8))
    captures: list[dict[str, Any]] = []

    def collector(name: str, _mod: nn.Linear, x: torch.Tensor, fstate: _ForwardState) -> None:
        li = _layer_idx(name)
        proj = _projection(name)
        if li not in layers or proj is None:
            return
        domain = _projection_to_domain(proj)
        if proj != CANONICAL_PROJECTION[domain]:
            return
        limit = rows_prefill if fstate.phase == "prefill" else rows_decode
        sample = _sample_rows(x, max_rows=limit).to(torch.bfloat16).cpu()
        if sample.numel() == 0:
            return
        captures.append(
            {
                "sample_id": fstate.sample_id,
                "prompt_family": fstate.prompt_family,
                "split": fstate.split,
                "phase": fstate.phase,
                "layer": int(li),
                "domain": domain,
                "x": sample,
            }
        )

    bank = _prompt_bank(split, int(config.get("samples_per_family", 8)))
    prompts = [item for i, item in enumerate(bank) if i % num_shards == shard_id]
    max_seq_len = int(config.get("max_seq_len", 256))
    decode_steps = int(config.get("decode_steps", 8))
    with _SourceTrajectoryHooks(model, nv_scales, collector=collector, state=state):
        for item in prompts:
            state.sample_id = item.sample_id
            state.prompt_family = item.family
            enc = tok(item.text, return_tensors="pt", truncation=True, max_length=max_seq_len)
            batch = {k: v.to(device_t) for k, v in enc.items()}
            if "attention_mask" not in batch:
                batch["attention_mask"] = torch.ones_like(batch["input_ids"])
            state.phase = "prefill"
            warm = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=True,
            )
            past = warm.past_key_values
            current = warm.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            attn = torch.cat(
                [batch["attention_mask"], torch.ones_like(current, dtype=batch["attention_mask"].dtype)],
                dim=-1,
            )
            state.phase = "decode"
            for _ in range(decode_steps):
                out = model(
                    input_ids=current,
                    attention_mask=attn,
                    past_key_values=past,
                    use_cache=True,
                )
                past = out.past_key_values
                current = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                attn = torch.cat([attn, torch.ones_like(current, dtype=attn.dtype)], dim=-1)

    sample_ids = [item.sample_id for item in prompts]
    del model
    if device_t.type == "cuda":
        torch.cuda.empty_cache()
    return captures, module_names, {"model_meta": model_meta, "sample_ids": sample_ids}


def _tensor_error_sum(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        raise ValueError(f"tensor shape mismatch {tuple(a.shape)} != {tuple(b.shape)}")
    return float((a.float() - b.float()).pow(2).sum().item())


def _tensor_energy(a: torch.Tensor) -> float:
    return float(a.float().pow(2).sum().item())


def _candidate_activation_counts(
    a_nv: torch.Tensor,
    a_restored: torch.Tensor,
    payload: torch.Tensor,
) -> dict[str, int]:
    numel = int(a_restored.numel())
    out = {
        "activation_numel": numel,
        "hif4_zero_count": int((a_restored == 0).sum().item()),
        "nv_nonzero_to_hif4_zero_count": int(((a_nv != 0) & (a_restored == 0)).sum().item()),
        "hif4_boundary_count": int((payload >= 1.75).sum().item()),
    }
    p = payload.detach().float().reshape(-1)
    for idx, value in enumerate((0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75)):
        out[f"payload_bin_{idx}_count"] = int(torch.isclose(p, torch.tensor(value, device=p.device)).sum().item())
    return out


def _dispersion_sums(x: torch.Tensor, *, group_size: int = 64) -> dict[str, float | int]:
    if x.shape[-1] % group_size != 0:
        raise ValueError(f"dispersion width {x.shape[-1]} is not divisible by {group_size}")
    rows = x.detach().float().reshape(-1, x.shape[-1]).abs().reshape(-1, group_size)
    result: dict[str, float | int] = {"dispersion_group_count": int(rows.shape[0])}
    tiny = torch.finfo(torch.float32).tiny
    for sub in (16, 8, 4):
        amax = rows.reshape(rows.shape[0], group_size // sub, sub).amax(dim=-1)
        logs = torch.log2(amax.clamp_min(tiny))
        active = amax > 0
        ranges = torch.zeros(rows.shape[0], device=rows.device, dtype=torch.float32)
        for gi in range(rows.shape[0]):
            mask = active[gi]
            if bool(mask.any()):
                vals = logs[gi, mask]
                ranges[gi] = vals.max() - vals.min()
        result[f"sub{sub}_log2_amax_range_sum"] = float(ranges.sum().item())
    return result


def _activation_eval_record(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    d: torch.Tensor,
    nv_scale: torch.Tensor,
) -> dict[str, Any]:
    x32 = x.float()
    w32 = w.float()
    d32 = d.to(device=x.device, dtype=torch.float32)
    a_nv = quantize_nvfp4_activation(
        x.to(torch.bfloat16), nv_scale.to(x.device), output_dtype=torch.float32
    ).dequantized
    y_local = x32 @ w32.T
    y_nv = a_nv @ w32.T

    std_view = quantize_hif4_tensor(x32, variant="full", output_dtype=torch.float32)
    a_std = std_view.dequantized
    y_std = a_std @ w32.T

    cand_view = quantize_hif4_tensor(
        x32 / d32, variant="full", output_dtype=torch.float32
    )
    a_restored = cand_view.dequantized * d32
    y_cand = a_restored @ w32.T
    rec: dict[str, Any] = {
        "activation_conv_error_sum": _tensor_error_sum(y_cand, y_nv),
        "activation_local_error_sum": _tensor_error_sum(y_cand, y_local),
        "baseline_activation_conv_error_sum": _tensor_error_sum(y_std, y_nv),
        "baseline_activation_local_error_sum": _tensor_error_sum(y_std, y_local),
        "activation_output_numel": int(y_cand.numel()),
        "activation_value_conv_error_sum": _tensor_error_sum(a_restored, a_nv),
        "activation_value_local_error_sum": _tensor_error_sum(a_restored, x32),
        "activation_value_ref_energy_nv": _tensor_energy(a_nv),
        "activation_value_ref_energy_local": _tensor_energy(x32),
    }
    rec.update(
        _candidate_activation_counts(
            a_nv,
            a_restored,
            cand_view.metadata["payload_magnitude"].to(a_restored.device),
        )
    )
    before_dispersion = _dispersion_sums(x32)
    after_dispersion = _dispersion_sums(x32 / d32)
    rec["dispersion_group_count"] = int(before_dispersion["dispersion_group_count"])
    for sub in (16, 8, 4):
        key = f"sub{sub}_log2_amax_range_sum"
        rec[f"before_{key}"] = float(before_dispersion[key])
        rec[f"after_{key}"] = float(after_dispersion[key])
    return rec


def run_scaling_eval_shard(
    checkpoint: Path,
    out_dir: Path,
    *,
    candidate_scales_path: Path,
    split: str,
    shard_id: int,
    num_shards: int,
    device: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Stage B1: evaluate activation-only paths for every fixed candidate D."""
    checkpoint = Path(checkpoint)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_scales_path = Path(candidate_scales_path)
    candidates = _safe_torch_load(candidate_scales_path)
    config_hash = str(config.get("config_sha256") or _sha256_jsonable(config))
    if str(candidates.get("config_sha256")) != config_hash:
        raise ValueError("candidate scales config hash does not match eval config")
    stats_hash = str(candidates["stats_merged_sha256"])
    candidate_hash = _sha256_file(candidate_scales_path)

    captures, module_names, cap_meta = _capture_source_inputs(
        checkpoint,
        split=split,
        shard_id=shard_id,
        num_shards=num_shards,
        device=device,
        config=config,
    )
    raw_path = out_dir / f"es_raw_inputs_shard{shard_id}.pt"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "split": split,
            "shard_id": shard_id,
            "num_shards": num_shards,
            "config_sha256": config_hash,
            "stats_merged_sha256": stats_hash,
            "candidate_scales_sha256": candidate_hash,
            "sample_ids": cap_meta["sample_ids"],
            "module_names": {f"{k[0]}:{k[1]}": v for k, v in module_names.items()},
            "captures": captures,
        },
        raw_path,
    )

    device_t = torch.device(device)
    if device_t.type == "cuda":
        torch.cuda.set_device(device_t)
    nv_scales = load_nvfp4_activation_scales(resolve_activation_scale_path(checkpoint))
    records: dict[str, dict[str, Any]] = {}
    captures_by_layer_domain: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for cap in captures:
        captures_by_layer_domain[(int(cap["layer"]), str(cap["domain"]))].append(cap)

    for (layer, domain), domain_caps in sorted(captures_by_layer_domain.items()):
        layer_scales = candidates["scales"].get(str(layer), {}).get(domain, {})
        for projection in DOMAIN_PROJECTIONS[domain]:
            module_name = module_names[(layer, projection)]
            w = load_nvfp4_qat_dequant_weight(
                checkpoint, module_name + ".weight", device=device_t
            ).dequantized
            nv_scale = resolve_nvfp4_scale_for_module(nv_scales, module_name).to(device_t)
            for cap in domain_caps:
                phase = str(cap["phase"])
                x = cap["x"].to(device_t)
                for rid, d_cpu in layer_scales.items():
                    d = d_cpu.to(device_t)
                    rec = _activation_eval_record(x, w, d=d, nv_scale=nv_scale)
                    rec.update(
                        {
                            "sample_id": str(cap["sample_id"]),
                            "prompt_family": str(cap["prompt_family"]),
                            "phase": phase,
                            "layer": layer,
                            "projection": projection,
                            "domain": domain,
                            "recipe_id": rid,
                        }
                    )
                    key = f"{cap['sample_id']}|{phase}|{layer}|{projection}|{rid}"
                    if key in records:
                        meta_keys = (
                            "sample_id",
                            "prompt_family",
                            "phase",
                            "layer",
                            "projection",
                            "domain",
                            "recipe_id",
                        )
                        previous_meta = {k: records[key][k] for k in meta_keys}
                        numeric = {k: v for k, v in rec.items() if k not in meta_keys}
                        _sum_numeric_record(records[key], numeric)
                        records[key].update(previous_meta)
                    else:
                        records[key] = rec
            del w
            if device_t.type == "cuda":
                torch.cuda.empty_cache()

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "eval_kind": "activation",
        "split": split,
        "shard_id": int(shard_id),
        "num_shards": int(num_shards),
        "config_sha256": config_hash,
        "stats_merged_sha256": stats_hash,
        "candidate_scales_sha256": candidate_hash,
        "sample_ids": cap_meta["sample_ids"],
        "records": records,
    }
    out_path = out_dir / f"es_eval_shard{shard_id}.pt"
    torch.save(artifact, out_path)
    return {
        "path": str(out_path),
        "raw_inputs": str(raw_path),
        "num_records": len(records),
        "num_captures": len(captures),
    }


def _aggregate_by_domain_recipe(records: dict[str, dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    sums: dict[tuple[str, str], dict[str, float]] = {}
    for rec in records.values():
        if "domain" not in rec or "recipe_id" not in rec:
            continue
        key = (str(rec["domain"]), str(rec["recipe_id"]))
        dst = sums.setdefault(key, defaultdict(float))
        for field in (
            "activation_conv_error_sum",
            "activation_local_error_sum",
            "baseline_activation_conv_error_sum",
            "baseline_activation_local_error_sum",
            "joint_conv_error_sum",
            "joint_local_error_sum",
            "baseline_conv_error_sum",
            "baseline_local_error_sum",
            "weight_local_error_sum",
            "baseline_weight_local_error_sum",
            "joint_numel",
            "activation_output_numel",
        ):
            value = rec.get(field)
            if isinstance(value, (int, float)):
                dst[field] += float(value)
    return {k: dict(v) for k, v in sums.items()}


def _recovery(num: float, den: float, eps: float = 1e-12) -> float:
    if den < eps:
        return float("-inf")
    return 1.0 - num / den


def select_full_eval_candidates(
    run_dir: Path,
    *,
    candidate_scales_path: Path,
    config: dict[str, Any],
) -> Path:
    """ES2 gate: select a small deterministic subset for weight/joint W4A4."""
    run_dir = Path(run_dir)
    merged_path = run_dir / "es_eval_merged.pt"
    if not merged_path.is_file():
        raise FileNotFoundError(merged_path)
    merged = _safe_torch_load(merged_path)
    candidates = _safe_torch_load(Path(candidate_scales_path))
    agg = _aggregate_by_domain_recipe(merged.get("records", {}))
    recipes = candidates.get("recipes", {})
    selected: dict[str, list[str]] = {}
    details: dict[str, Any] = {}

    for domain in DOMAINS:
        scored: list[tuple[str, float]] = []
        for (d, rid), rec in agg.items():
            if d != domain:
                continue
            r = _recovery(
                float(rec.get("activation_conv_error_sum", math.inf)),
                float(rec.get("baseline_activation_conv_error_sum", 0.0)),
            )
            scored.append((rid, r))
        if not scored:
            continue
        score_map = dict(scored)
        keep: list[str] = []

        pts = sorted(
            rid for rid, _ in scored if recipes.get(rid, {}).get("kind") == "pts_layer"
        )
        # ES1 requires the full-W4A4 optimum over the same 33-point PTS grid; keeping
        # only the activation-best scalar would bias this comparison.
        keep.extend(pts)
        if any(rid == "phase_g64" for rid, _ in scored):
            keep.append("phase_g64")

        best_by_gran: list[tuple[int, str, float]] = []
        granularities = sorted({
            int(recipes[rid]["granularity"])
            for rid, _ in scored
            if recipes.get(rid, {}).get("kind") == "equalize"
            and recipes.get(rid, {}).get("deployable", True)
            and recipes.get(rid, {}).get("diagnostic") != "wide_bound"
        })
        for gran in granularities:
            options = [
                (rid, r)
                for rid, r in scored
                if recipes.get(rid, {}).get("kind") == "equalize"
                and int(recipes[rid].get("granularity", -1)) == gran
                and recipes.get(rid, {}).get("deployable", True)
            ]
            if options:
                rid, r = max(options, key=lambda x: (x[1], x[0]))
                best_by_gran.append((gran, rid, r))
        if best_by_gran:
            best_gran, best_rid, best_r = max(best_by_gran, key=lambda x: (x[2], x[0]))
            keep.append(best_rid)
            if best_r > 0:
                coarse = [
                    item
                    for item in best_by_gran
                    if item[0] >= 2 * best_gran and item[2] >= 0.9 * best_r
                ]
                if coarse:
                    keep.append(max(coarse, key=lambda x: (x[0], x[2], x[1]))[1])

        # Fixed-beta AW is a diagnostic of whether activation-only EQ has a false negative
        # due to weight penalty. Four granularities are cheap enough to retain explicitly.
        aw = sorted(
            rid
            for rid, _ in scored
            if recipes.get(rid, {}).get("kind") == "equalize_aw"
        )
        keep.extend(aw)
        keep = list(dict.fromkeys(keep))
        selected[domain] = keep
        details[domain] = {
            "selected": keep,
            "activation_R_Y_conv": {rid: score_map[rid] for rid in keep},
        }

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": merged.get("config_sha256"),
        "candidate_scales_sha256": _sha256_file(Path(candidate_scales_path)),
        "selected": selected,
        "details": details,
    }
    out = run_dir / "es3_candidate_subset.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2, sort_keys=True)
    return out


def _single_linear_full_record(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    d: torch.Tensor,
    nv_scale: torch.Tensor,
    w_std_q: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor]:
    x32 = x.float()
    w32 = w.float()
    d32 = d.to(device=x.device, dtype=torch.float32)
    a_nv = quantize_nvfp4_activation(
        x.to(torch.bfloat16), nv_scale.to(x.device), output_dtype=torch.float32
    ).dequantized
    y_local = x32 @ w32.T
    y_nv = a_nv @ w32.T

    a_std = quantize_hif4_tensor(x32, variant="full", output_dtype=torch.float32).dequantized
    y_joint_std = a_std @ w_std_q.T
    y_weight_std = x32 @ w_std_q.T

    w_scaled = w32 * d32.unsqueeze(0)
    w_view = quantize_hif4_tensor(w_scaled, variant="full", output_dtype=torch.float32)
    w_q = w_view.dequantized
    a_scaled = quantize_hif4_tensor(
        x32 / d32, variant="full", output_dtype=torch.float32
    ).dequantized
    y_joint = a_scaled @ w_q.T
    y_weight = (x32 / d32) @ w_q.T
    rec = {
        "joint_conv_error_sum": _tensor_error_sum(y_joint, y_nv),
        "joint_local_error_sum": _tensor_error_sum(y_joint, y_local),
        "baseline_conv_error_sum": _tensor_error_sum(y_joint_std, y_nv),
        "baseline_local_error_sum": _tensor_error_sum(y_joint_std, y_local),
        "weight_local_error_sum": _tensor_error_sum(y_weight, y_local),
        "baseline_weight_local_error_sum": _tensor_error_sum(y_weight_std, y_local),
        "joint_numel": int(y_joint.numel()),
        "weight_error_sum": _tensor_error_sum(w_q, w_scaled),
        "weight_ref_energy": _tensor_energy(w_scaled),
        "weight_numel": int(w_q.numel()),
        "weight_zero_count": int((w_q == 0).sum().item()),
        "weight_boundary_count": int((w_view.metadata["payload_magnitude"] >= 1.75).sum().item()),
    }
    return rec, w_q


def _evaluate_full_scale_subset(
    checkpoint: Path,
    *,
    raw: dict[str, Any],
    scale_artifact: dict[str, Any],
    selected_by_domain: dict[str, list[str]],
    device: str,
) -> dict[str, dict[str, Any]]:
    """Evaluate weight-only functional + joint W4A4 for one fixed scale subset."""
    checkpoint = Path(checkpoint)
    module_names = {
        (int(k.split(":", 1)[0]), k.split(":", 1)[1]): v
        for k, v in raw["module_names"].items()
    }
    captures_by_layer_domain: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for cap in raw["captures"]:
        captures_by_layer_domain[(int(cap["layer"]), str(cap["domain"]))].append(cap)

    device_t = torch.device(device)
    if device_t.type == "cuda":
        torch.cuda.set_device(device_t)
    nv_scales = load_nvfp4_activation_scales(resolve_activation_scale_path(checkpoint))
    records: dict[str, dict[str, Any]] = {}

    for (layer, domain), caps in sorted(captures_by_layer_domain.items()):
        selected = list(selected_by_domain.get(domain, []))
        if not selected:
            continue
        layer_scales = scale_artifact["scales"][str(layer)][domain]
        for projection in DOMAIN_PROJECTIONS[domain]:
            module_name = module_names[(layer, projection)]
            w = load_nvfp4_qat_dequant_weight(
                checkpoint, module_name + ".weight", device=device_t
            ).dequantized
            w_std_q = quantize_hif4_tensor(
                w.float(), variant="full", output_dtype=torch.float32
            ).dequantized
            nv_scale = resolve_nvfp4_scale_for_module(nv_scales, module_name).to(device_t)
            for rid in selected:
                if rid not in layer_scales:
                    raise KeyError(f"missing scale layer={layer} domain={domain} recipe={rid}")
                d = layer_scales[rid].to(device_t)
                w_scaled = w.float() * d.unsqueeze(0)
                w_view = quantize_hif4_tensor(
                    w_scaled, variant="full", output_dtype=torch.float32
                )
                w_q = w_view.dequantized
                weight_static = {
                    "weight_error_sum_once": _tensor_error_sum(w_q, w_scaled),
                    "weight_ref_energy_once": _tensor_energy(w_scaled),
                    "weight_numel_once": int(w_q.numel()),
                    "weight_zero_count_once": int((w_q == 0).sum().item()),
                    "weight_boundary_count_once": int((w_view.metadata["payload_magnitude"] >= 1.75).sum().item()),
                }
                recipe_record_keys: list[str] = []
                for cap in caps:
                    x = cap["x"].to(device_t)
                    x32 = x.float()
                    d32 = d.to(torch.float32)
                    a_nv = quantize_nvfp4_activation(
                        x.to(torch.bfloat16), nv_scale, output_dtype=torch.float32
                    ).dequantized
                    y_local = x32 @ w.float().T
                    y_nv = a_nv @ w.float().T
                    a_std = quantize_hif4_tensor(
                        x32, variant="full", output_dtype=torch.float32
                    ).dequantized
                    y_joint_std = a_std @ w_std_q.T
                    y_weight_std = x32 @ w_std_q.T
                    a_scaled = quantize_hif4_tensor(
                        x32 / d32, variant="full", output_dtype=torch.float32
                    ).dequantized
                    y_joint = a_scaled @ w_q.T
                    y_weight = (x32 / d32) @ w_q.T
                    rec = _activation_eval_record(x, w, d=d, nv_scale=nv_scale)
                    rec.update({
                        "sample_id": str(cap["sample_id"]),
                        "prompt_family": str(cap["prompt_family"]),
                        "phase": str(cap["phase"]),
                        "layer": layer,
                        "projection": projection,
                        "domain": domain,
                        "recipe_id": rid,
                        "joint_conv_error_sum": _tensor_error_sum(y_joint, y_nv),
                        "joint_local_error_sum": _tensor_error_sum(y_joint, y_local),
                        "baseline_conv_error_sum": _tensor_error_sum(y_joint_std, y_nv),
                        "baseline_local_error_sum": _tensor_error_sum(y_joint_std, y_local),
                        "weight_local_error_sum": _tensor_error_sum(y_weight, y_local),
                        "baseline_weight_local_error_sum": _tensor_error_sum(y_weight_std, y_local),
                        "joint_numel": int(y_joint.numel()),
                    })
                    key = f"{cap['sample_id']}|{cap['phase']}|{layer}|{projection}|{rid}"
                    recipe_record_keys.append(key)
                    if key in records:
                        meta_keys = (
                            "sample_id",
                            "prompt_family",
                            "phase",
                            "layer",
                            "projection",
                            "domain",
                            "recipe_id",
                        )
                        numeric = {k: v for k, v in rec.items() if k not in meta_keys}
                        _sum_numeric_record(records[key], numeric)
                    else:
                        records[key] = rec
                if not recipe_record_keys:
                    raise RuntimeError(
                        f"no raw captures for layer={layer} projection={projection} recipe={rid}"
                    )
                records[recipe_record_keys[0]].update(weight_static)
                del w_q, w_view, w_scaled
            del w, w_std_q
            if device_t.type == "cuda":
                torch.cuda.empty_cache()
    return records


def run_scaling_full_eval_shard(
    checkpoint: Path,
    out_dir: Path,
    *,
    candidate_scales_path: Path,
    subset_path: Path,
    shard_id: int,
    num_shards: int,
    device: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """ES3: weight-only functional and joint-W4A4 on the ES2 shortlist."""
    checkpoint = Path(checkpoint)
    out_dir = Path(out_dir)
    candidates = _safe_torch_load(Path(candidate_scales_path))
    with Path(subset_path).open("r", encoding="utf-8") as f:
        subset = json.load(f)
    raw = _safe_torch_load(out_dir / f"es_raw_inputs_shard{shard_id}.pt")
    candidate_hash = _sha256_file(Path(candidate_scales_path))
    if str(raw.get("candidate_scales_sha256")) != candidate_hash:
        raise ValueError("raw input artifact candidate scale hash mismatch")
    if str(subset.get("candidate_scales_sha256")) != candidate_hash:
        raise ValueError("ES3 subset candidate scale hash mismatch")
    config_hash = str(config.get("config_sha256") or _sha256_jsonable(config))
    if str(raw.get("config_sha256")) != config_hash:
        raise ValueError("raw input artifact config hash mismatch")

    module_names = {
        (int(k.split(":", 1)[0]), k.split(":", 1)[1]): v
        for k, v in raw["module_names"].items()
    }
    captures_by_layer_domain: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for cap in raw["captures"]:
        captures_by_layer_domain[(int(cap["layer"]), str(cap["domain"]))].append(cap)

    device_t = torch.device(device)
    if device_t.type == "cuda":
        torch.cuda.set_device(device_t)
    nv_scales = load_nvfp4_activation_scales(resolve_activation_scale_path(checkpoint))
    records: dict[str, dict[str, Any]] = {}

    for (layer, domain), caps in sorted(captures_by_layer_domain.items()):
        selected = list(subset.get("selected", {}).get(domain, []))
        if not selected:
            continue
        layer_scales = candidates["scales"][str(layer)][domain]
        for projection in DOMAIN_PROJECTIONS[domain]:
            module_name = module_names[(layer, projection)]
            w = load_nvfp4_qat_dequant_weight(
                checkpoint, module_name + ".weight", device=device_t
            ).dequantized
            w_std_q = quantize_hif4_tensor(
                w.float(), variant="full", output_dtype=torch.float32
            ).dequantized
            nv_scale = resolve_nvfp4_scale_for_module(nv_scales, module_name).to(device_t)
            for rid in selected:
                d = layer_scales[rid].to(device_t)
                # Quantize this transformed weight once; use the same exact tensor for all
                # prompt/phase samples. This avoids materializing many W candidates together.
                w_scaled = w.float() * d.unsqueeze(0)
                w_view = quantize_hif4_tensor(
                    w_scaled, variant="full", output_dtype=torch.float32
                )
                w_q = w_view.dequantized
                weight_static = {
                    "weight_error_sum_once": _tensor_error_sum(w_q, w_scaled),
                    "weight_ref_energy_once": _tensor_energy(w_scaled),
                    "weight_numel_once": int(w_q.numel()),
                    "weight_zero_count_once": int((w_q == 0).sum().item()),
                    "weight_boundary_count_once": int((w_view.metadata["payload_magnitude"] >= 1.75).sum().item()),
                }
                for cap in caps:
                    x = cap["x"].to(device_t)
                    x32 = x.float()
                    d32 = d.to(torch.float32)
                    a_nv = quantize_nvfp4_activation(
                        x.to(torch.bfloat16), nv_scale, output_dtype=torch.float32
                    ).dequantized
                    y_local = x32 @ w.float().T
                    y_nv = a_nv @ w.float().T
                    a_std = quantize_hif4_tensor(
                        x32, variant="full", output_dtype=torch.float32
                    ).dequantized
                    y_joint_std = a_std @ w_std_q.T
                    y_weight_std = x32 @ w_std_q.T
                    a_scaled = quantize_hif4_tensor(
                        x32 / d32, variant="full", output_dtype=torch.float32
                    ).dequantized
                    y_joint = a_scaled @ w_q.T
                    y_weight = (x32 / d32) @ w_q.T
                    rec = {
                        "sample_id": str(cap["sample_id"]),
                        "prompt_family": str(cap["prompt_family"]),
                        "phase": str(cap["phase"]),
                        "layer": layer,
                        "projection": projection,
                        "domain": domain,
                        "recipe_id": rid,
                        "joint_conv_error_sum": _tensor_error_sum(y_joint, y_nv),
                        "joint_local_error_sum": _tensor_error_sum(y_joint, y_local),
                        "baseline_conv_error_sum": _tensor_error_sum(y_joint_std, y_nv),
                        "baseline_local_error_sum": _tensor_error_sum(y_joint_std, y_local),
                        "weight_local_error_sum": _tensor_error_sum(y_weight, y_local),
                        "baseline_weight_local_error_sum": _tensor_error_sum(y_weight_std, y_local),
                        "joint_numel": int(y_joint.numel()),
                    }
                    key = f"{cap['sample_id']}|{cap['phase']}|{layer}|{projection}|{rid}"
                    if key in records:
                        meta_keys = (
                            "sample_id",
                            "prompt_family",
                            "phase",
                            "layer",
                            "projection",
                            "domain",
                            "recipe_id",
                        )
                        numeric = {k: v for k, v in rec.items() if k not in meta_keys}
                        _sum_numeric_record(records[key], numeric)
                    else:
                        records[key] = rec
                # Static weight fields are attached to one deterministic record only, so merge
                # and reports do not multiply parameter-space NMSE by prompt count.
                first_key = next(
                    k
                    for k in records
                    if k.endswith(f"|{layer}|{projection}|{rid}")
                )
                records[first_key].update(weight_static)
                del w_q, w_view, w_scaled
            del w, w_std_q
            if device_t.type == "cuda":
                torch.cuda.empty_cache()

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "eval_kind": "full",
        "split": str(raw["split"]),
        "shard_id": int(shard_id),
        "num_shards": int(num_shards),
        "config_sha256": config_hash,
        "stats_merged_sha256": str(raw["stats_merged_sha256"]),
        "candidate_scales_sha256": candidate_hash,
        "sample_ids": list(raw["sample_ids"]),
        "records": records,
    }
    out = out_dir / f"es_full_eval_shard{shard_id}.pt"
    torch.save(artifact, out)
    return {"path": str(out), "num_records": len(records)}


def merge_scaling_full_eval(
    run_dir: Path,
    *,
    expected_num_shards: int,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    paths = [run_dir / f"es_full_eval_shard{i}.pt" for i in range(expected_num_shards)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing full eval shards: {missing}")
    shards = [_safe_torch_load(p) for p in paths]
    _validate_shard_common(
        shards,
        expected_num_shards=expected_num_shards,
        hash_fields=("config_sha256", "stats_merged_sha256", "candidate_scales_sha256"),
    )
    records: dict[str, dict[str, Any]] = {}
    for shard in shards:
        for key, src in shard.get("records", {}).items():
            if key in records:
                _sum_numeric_record(records[key], src)
            else:
                records[key] = dict(src)
    for rec in records.values():
        _derive_recovery_fields(rec)
    merged = {
        "schema_version": SCHEMA_VERSION,
        "split": shards[0]["split"],
        "config_sha256": shards[0]["config_sha256"],
        "stats_merged_sha256": shards[0]["stats_merged_sha256"],
        "candidate_scales_sha256": shards[0]["candidate_scales_sha256"],
        "sample_ids": sorted(str(x) for s in shards for x in s["sample_ids"]),
        "records": records,
    }
    torch.save(merged, run_dir / "es_full_eval_merged.pt")
    return merged


def _refine_alpha_grid(center: float) -> list[float]:
    offsets = (-0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25)
    return sorted({min(1.0, max(0.0, float(center) + delta)) for delta in offsets})


def build_refined_candidate_scales(
    run_dir: Path,
    *,
    config: dict[str, Any],
) -> Path:
    """ES5: build the exact 7-point alpha neighborhoods around retained coarse EQ candidates."""
    run_dir = Path(run_dir)
    base_path = run_dir / "candidate_scales.pt"
    stats_path = run_dir / "es_stats_merged.pt"
    subset_path = run_dir / "es3_candidate_subset.json"
    full_path = run_dir / "es_full_eval_merged.pt"
    for path in (base_path, stats_path, subset_path, full_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    base = _safe_torch_load(base_path)
    merged_stats = _safe_torch_load(stats_path)
    with subset_path.open("r", encoding="utf-8") as f:
        subset = json.load(f)
    full_merged = _safe_torch_load(full_path)
    if str(subset.get("candidate_scales_sha256")) != _sha256_file(base_path):
        raise ValueError("ES3 subset does not match base candidate scales")
    if str(full_merged.get("candidate_scales_sha256")) != _sha256_file(base_path):
        raise ValueError("coarse full-eval does not match base candidate scales")

    selected: dict[str, list[str]] = {}
    recipes: dict[str, dict[str, Any]] = {}
    scales: dict[str, dict[str, dict[str, torch.Tensor]]] = {
        str(layer): {} for layer in config.get("representative_layers", [4, 18, 34])
    }
    sources: dict[str, dict[str, str]] = defaultdict(dict)
    model_meta = dict(merged_stats["model_meta"])
    group_size = int(config.get("group_size", 64))

    for domain in DOMAINS:
        coarse_rids = [
            rid
            for rid in subset.get("selected", {}).get(domain, [])
            if base.get("recipes", {}).get(rid, {}).get("kind") == "equalize"
            and base.get("recipes", {}).get(rid, {}).get("deployable", True)
            and not base.get("recipes", {}).get(rid, {}).get("diagnostic")
        ]
        if not coarse_rids:
            continue
        # ES2 retained at most two ordinary EQ granularities. If that invariant is broken,
        # stop rather than silently expanding ES5 cost.
        if len(coarse_rids) > 2:
            raise RuntimeError(
                f"domain={domain}: expected <=2 retained coarse EQ candidates, got {coarse_rids}"
            )
        domain_refined: list[str] = []
        for coarse_rid in coarse_rids:
            coarse = base["recipes"][coarse_rid]
            granularity = int(coarse["granularity"])
            center = float(coarse["alpha"])
            for alpha in _refine_alpha_grid(center):
                rid = f"eqref_g{granularity}_a{_alpha_slug(alpha)}"
                recipe = {
                    "kind": "equalize",
                    "granularity": granularity,
                    "alpha": alpha,
                    "min_scale": float(coarse.get("min_scale", config.get("min_scale", 0.5))),
                    "max_scale": float(coarse.get("max_scale", config.get("max_scale", 2.0))),
                    "deployable": True,
                    "refined": True,
                }
                if rid in recipes and recipes[rid] != recipe:
                    raise RuntimeError(f"refined recipe collision: {rid}")
                recipes[rid] = recipe
                sources[domain][rid] = coarse_rid
                domain_refined.append(rid)
        selected[domain] = sorted(set(domain_refined))

    for layer in [int(x) for x in config.get("representative_layers", [4, 18, 34])]:
        layer_key = str(layer)
        scales.setdefault(layer_key, {})
        for domain, rids in selected.items():
            amp_full = _deploy_amplitude(merged_stats, layer=layer, domain=domain)
            amp_design = _o_unique_amplitude(amp_full, model_meta) if domain == "o_in" else amp_full
            domain_scales: dict[str, torch.Tensor] = {}
            for rid in rids:
                recipe = recipes[rid]
                d_design = build_equalization_scale(
                    amp_design,
                    granularity=int(recipe["granularity"]),
                    alpha=float(recipe["alpha"]),
                    group_size=group_size,
                    min_scale=float(recipe["min_scale"]),
                    max_scale=float(recipe["max_scale"]),
                )
                domain_scales[rid] = (
                    _expand_o_unique(d_design, model_meta) if domain == "o_in" else d_design
                )
            scales[layer_key][domain] = domain_scales

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "split": "discovery",
        "config_sha256": base["config_sha256"],
        "stats_merged_sha256": base["stats_merged_sha256"],
        "base_candidate_scales_sha256": _sha256_file(base_path),
        "recipes": recipes,
        "scales": scales,
        "selected": selected,
        "refinement_sources": {d: dict(v) for d, v in sources.items()},
    }
    out = run_dir / "es5_refined_candidate_scales.pt"
    torch.save(artifact, out)
    return out


def run_scaling_refine_eval_shard(
    checkpoint: Path,
    out_dir: Path,
    *,
    refined_scales_path: Path,
    shard_id: int,
    num_shards: int,
    device: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate only ES5 refined alpha candidates, reusing Stage-B raw source inputs."""
    checkpoint = Path(checkpoint)
    out_dir = Path(out_dir)
    refined_scales_path = Path(refined_scales_path)
    refined = _safe_torch_load(refined_scales_path)
    raw = _safe_torch_load(out_dir / f"es_raw_inputs_shard{shard_id}.pt")
    config_hash = str(config.get("config_sha256") or _sha256_jsonable(config))
    if str(refined.get("config_sha256")) != config_hash:
        raise ValueError("refined scales config hash mismatch")
    if str(raw.get("config_sha256")) != config_hash:
        raise ValueError("raw input config hash mismatch")
    if str(refined.get("base_candidate_scales_sha256")) != str(raw.get("candidate_scales_sha256")):
        raise ValueError("refined scales were not derived from the raw-input base candidate artifact")
    if str(refined.get("stats_merged_sha256")) != str(raw.get("stats_merged_sha256")):
        raise ValueError("refined scales stats hash mismatch")

    records = _evaluate_full_scale_subset(
        checkpoint,
        raw=raw,
        scale_artifact=refined,
        selected_by_domain={d: list(v) for d, v in refined.get("selected", {}).items()},
        device=device,
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "eval_kind": "refine_full",
        "split": str(raw["split"]),
        "shard_id": int(shard_id),
        "num_shards": int(num_shards),
        "config_sha256": config_hash,
        "stats_merged_sha256": str(raw["stats_merged_sha256"]),
        "candidate_scales_sha256": str(raw["candidate_scales_sha256"]),
        "refined_scales_sha256": _sha256_file(refined_scales_path),
        "sample_ids": list(raw["sample_ids"]),
        "records": records,
    }
    out = out_dir / f"es5_refine_eval_shard{shard_id}.pt"
    torch.save(artifact, out)
    return {"path": str(out), "num_records": len(records)}


def merge_scaling_refine_eval(
    run_dir: Path,
    *,
    expected_num_shards: int,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    paths = [run_dir / f"es5_refine_eval_shard{i}.pt" for i in range(expected_num_shards)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing refine eval shards: {missing}")
    shards = [_safe_torch_load(p) for p in paths]
    _validate_shard_common(
        shards,
        expected_num_shards=expected_num_shards,
        hash_fields=(
            "config_sha256",
            "stats_merged_sha256",
            "candidate_scales_sha256",
            "refined_scales_sha256",
        ),
    )
    records: dict[str, dict[str, Any]] = {}
    for shard in shards:
        for key, src in shard.get("records", {}).items():
            if key in records:
                _sum_numeric_record(records[key], src)
            else:
                records[key] = dict(src)
    for rec in records.values():
        _derive_recovery_fields(rec)
    merged = {
        "schema_version": SCHEMA_VERSION,
        "split": shards[0]["split"],
        "config_sha256": shards[0]["config_sha256"],
        "stats_merged_sha256": shards[0]["stats_merged_sha256"],
        "candidate_scales_sha256": shards[0]["candidate_scales_sha256"],
        "refined_scales_sha256": shards[0]["refined_scales_sha256"],
        "sample_ids": sorted(str(x) for s in shards for x in s["sample_ids"]),
        "records": records,
    }
    torch.save(merged, run_dir / "es5_refine_eval_merged.pt")
    return merged


def _aggregate_full_by_domain_recipe(
    records: dict[str, dict[str, Any]],
) -> dict[tuple[str, str], dict[str, float]]:
    sums: dict[tuple[str, str], dict[str, float]] = {}
    for rec in records.values():
        if "domain" not in rec or "recipe_id" not in rec:
            continue
        key = (str(rec["domain"]), str(rec["recipe_id"]))
        dst = sums.setdefault(key, defaultdict(float))
        for field in (
            "joint_conv_error_sum",
            "joint_local_error_sum",
            "baseline_conv_error_sum",
            "baseline_local_error_sum",
            "weight_local_error_sum",
            "baseline_weight_local_error_sum",
            "joint_numel",
        ):
            value = rec.get(field)
            if isinstance(value, (int, float)):
                dst[field] += float(value)
    return {k: dict(v) for k, v in sums.items()}


def _recipe_objective(rec: dict[str, float], lambda_local: float) -> float:
    conv_den = float(rec.get("baseline_conv_error_sum", 0.0))
    local_den = float(rec.get("baseline_local_error_sum", 0.0))
    if conv_den < 1e-12 or local_den < 1e-12:
        return float("inf")
    return (
        float(rec.get("joint_conv_error_sum", math.inf)) / conv_den
        + float(lambda_local) * float(rec.get("joint_local_error_sum", math.inf)) / local_den
    )


def _recipe_passes_guards(rec: dict[str, float], config: dict[str, Any]) -> bool:
    wf_den = float(rec.get("baseline_weight_local_error_sum", 0.0))
    joint_den = float(rec.get("baseline_local_error_sum", 0.0))
    if wf_den < 1e-12 or joint_den < 1e-12:
        return False
    wf_ratio = float(rec.get("weight_local_error_sum", math.inf)) / wf_den
    joint_ratio = float(rec.get("joint_local_error_sum", math.inf)) / joint_den
    return (
        wf_ratio <= float(config.get("weight_functional_guard_ratio", 1.25))
        and joint_ratio <= float(config.get("local_output_guard_ratio", 1.05))
    )


def _select_global_domain_recipes(
    full_merged: dict[str, Any],
    candidates: dict[str, Any],
    *,
    config: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    agg = _aggregate_full_by_domain_recipe(full_merged.get("records", {}))
    recipes = candidates.get("recipes", {})
    lambda_local = float(config.get("output_objective_lambda_local", 0.25))
    selected: dict[str, str] = {}
    diagnostics: dict[str, Any] = {}
    for domain in DOMAINS:
        options: list[tuple[float, int, str, dict[str, float]]] = []
        rejected: list[dict[str, Any]] = []
        for (d, rid), rec in agg.items():
            if d != domain:
                continue
            recipe = recipes.get(rid, {})
            if not recipe.get("deployable", True):
                continue
            objective = _recipe_objective(rec, lambda_local)
            guard = _recipe_passes_guards(rec, config)
            granularity = int(recipe.get("granularity", 0))
            if guard and math.isfinite(objective):
                # For equal objective, larger granularity means fewer scale degrees of freedom.
                options.append((objective, -granularity, rid, rec))
            else:
                rejected.append({"recipe_id": rid, "objective": objective, "guard_pass": guard})
        if not options:
            diagnostics[domain] = {"enabled": False, "reason": "no_guard_passing_candidate", "rejected": rejected}
            continue
        options.sort(key=lambda x: (x[0], x[1], x[2]))
        objective, _, rid, rec = options[0]
        # D=I is always a legal fallback. A recipe whose conversion error is not better than
        # standard HiF4 is not enabled merely because it wins among bad candidates.
        r_conv = _recovery(
            float(rec["joint_conv_error_sum"]), float(rec["baseline_conv_error_sum"])
        )
        if r_conv <= 0:
            diagnostics[domain] = {
                "enabled": False,
                "reason": "best_candidate_not_better_than_standard",
                "best_recipe_id": rid,
                "joint_R_Y_conv": r_conv,
                "objective": objective,
                "rejected": rejected,
            }
            continue
        selected[domain] = rid
        diagnostics[domain] = {
            "enabled": True,
            "recipe_id": rid,
            "joint_R_Y_conv": r_conv,
            "joint_R_Y_local": _recovery(
                float(rec["joint_local_error_sum"]), float(rec["baseline_local_error_sum"])
            ),
            "objective": objective,
            "rejected": rejected,
        }
    return selected, diagnostics


def _layer_module(model: nn.Module, layer_idx: int) -> nn.Module:
    matches = [
        mod
        for name, mod in model.named_modules()
        if name.endswith(f"model.layers.{layer_idx}") or name == f"model.layers.{layer_idx}"
    ]
    if len(matches) != 1:
        # Some remote-code wrappers prepend another model prefix. Fall back to the unique
        # module whose final path components are layers.<idx>.
        matches = []
        for name, mod in model.named_modules():
            parts = name.split(".")
            if len(parts) >= 2 and parts[-2:] == ["layers", str(layer_idx)]:
                matches.append(mod)
    if len(matches) != 1:
        raise RuntimeError(f"cannot resolve unique decoder layer {layer_idx}; matches={len(matches)}")
    return matches[0]


def _all_target_linear_names(model: nn.Module) -> set[str]:
    return {
        name
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and "lm_head" not in name and _projection(name) is not None
    }


def _clone_layer_state_cpu(layer: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in layer.state_dict().items()}


def _restore_layer_state(layer: nn.Module, state: dict[str, torch.Tensor]) -> None:
    current = layer.state_dict()
    with torch.no_grad():
        for name, dst in current.items():
            src = state[name].to(device=dst.device, dtype=dst.dtype)
            dst.copy_(src)
    layer.load_state_dict(current, strict=True)


def _quantize_linear_from_fp32(linear: nn.Linear, weight_fp32: torch.Tensor) -> None:
    q = quantize_hif4_tensor(
        weight_fp32.float(), variant="full", output_dtype=linear.weight.dtype
    ).dequantized
    with torch.no_grad():
        linear.weight.copy_(q.to(device=linear.weight.device, dtype=linear.weight.dtype))


def _o_unique_from_expanded(d_o: torch.Tensor, model_meta: dict[str, Any]) -> torch.Tensor:
    hq = _model_meta_int(model_meta, "num_attention_heads")
    hkv = _model_meta_int(model_meta, "num_key_value_heads")
    hd = _model_meta_int(model_meta, "head_dim")
    repeat = hq // hkv
    heads = d_o.reshape(hkv, repeat, hd)
    ref = heads[:, :1, :]
    if not torch.equal(heads, ref.expand_as(heads)):
        raise ValueError("expanded o_in D is not GQA-tied")
    return ref[:, 0, :].reshape(-1)


def _apply_standard_hif4_layer(layer: nn.Module) -> None:
    for projection in (
        layer.self_attn.q_proj,
        layer.self_attn.k_proj,
        layer.self_attn.v_proj,
        layer.self_attn.o_proj,
        layer.mlp.gate_proj,
        layer.mlp.up_proj,
        layer.mlp.down_proj,
    ):
        _quantize_linear_from_fp32(projection, projection.weight.detach().float())


def _identity(width: int, device: torch.device) -> torch.Tensor:
    return torch.ones(width, dtype=torch.float32, device=device)


def _apply_folded_hif4_layer(
    layer: nn.Module,
    *,
    scales: dict[str, torch.Tensor],
    model_meta: dict[str, Any],
) -> None:
    device = layer.self_attn.q_proj.weight.device
    hidden = int(layer.self_attn.q_proj.weight.shape[1])
    intermediate = int(layer.mlp.down_proj.weight.shape[1])
    d_attn = scales.get("attn_in", _identity(hidden, device)).to(device=device, dtype=torch.float32)
    d_mlp = scales.get("mlp_in", _identity(hidden, device)).to(device=device, dtype=torch.float32)
    d_down = scales.get("down_in", _identity(intermediate, device)).to(device=device, dtype=torch.float32)
    if "o_in" in scales:
        d_o_expanded = scales["o_in"].to(device=device, dtype=torch.float32)
        d_o_unique = _o_unique_from_expanded(d_o_expanded, model_meta)
    else:
        d_o_expanded = _identity(hidden, device)
        d_o_unique = _identity(int(layer.self_attn.v_proj.weight.shape[0]), device)

    with torch.no_grad():
        layer.input_layernorm.weight.copy_(
            (layer.input_layernorm.weight.detach().float() / d_attn).to(layer.input_layernorm.weight.dtype)
        )
        layer.post_attention_layernorm.weight.copy_(
            (layer.post_attention_layernorm.weight.detach().float() / d_mlp).to(layer.post_attention_layernorm.weight.dtype)
        )

    q_src = layer.self_attn.q_proj.weight.detach().float()
    k_src = layer.self_attn.k_proj.weight.detach().float()
    v_src = layer.self_attn.v_proj.weight.detach().float()
    o_src = layer.self_attn.o_proj.weight.detach().float()
    g_src = layer.mlp.gate_proj.weight.detach().float()
    u_src = layer.mlp.up_proj.weight.detach().float()
    d_src = layer.mlp.down_proj.weight.detach().float()

    _quantize_linear_from_fp32(layer.self_attn.q_proj, q_src * d_attn.unsqueeze(0))
    _quantize_linear_from_fp32(layer.self_attn.k_proj, k_src * d_attn.unsqueeze(0))
    _quantize_linear_from_fp32(
        layer.self_attn.v_proj,
        (v_src * d_attn.unsqueeze(0)) / d_o_unique.unsqueeze(1),
    )
    _quantize_linear_from_fp32(layer.self_attn.o_proj, o_src * d_o_expanded.unsqueeze(0))
    _quantize_linear_from_fp32(layer.mlp.gate_proj, g_src * d_mlp.unsqueeze(0))
    _quantize_linear_from_fp32(
        layer.mlp.up_proj,
        (u_src * d_mlp.unsqueeze(0)) / d_down.unsqueeze(1),
    )
    _quantize_linear_from_fp32(layer.mlp.down_proj, d_src * d_down.unsqueeze(0))

    # Qwen3 target Linears are normally bias-free. If a remote-code variant has bias,
    # output-row inverse scaling must also be applied to keep the floating transform exact.
    with torch.no_grad():
        if layer.self_attn.v_proj.bias is not None:
            layer.self_attn.v_proj.bias.div_(d_o_unique.to(layer.self_attn.v_proj.bias.dtype))
        if layer.mlp.up_proj.bias is not None:
            layer.mlp.up_proj.bias.div_(d_down.to(layer.mlp.up_proj.bias.dtype))


@contextlib.contextmanager
def _activation_path_hooks(
    model: nn.Module,
    *,
    nv_scales: dict[str, torch.Tensor],
    hi_names: set[str],
) -> Iterator[None]:
    from Inference_Paradigm_Conversion.ipc_analysis.analysis.network_injection import (
        install_p2_activation_hooks,
    )

    matched = _all_target_linear_names(model)
    handles = install_p2_activation_hooks(
        model,
        scales=nv_scales,
        converted_names=hi_names,
        matched_names=matched,
    )
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _source_prefixes_for_block_replay(
    model: nn.Module,
    tok: Any,
    *,
    nv_scales: dict[str, torch.Tensor],
    prompts: list[Any],
    device: torch.device,
    max_seq_len: int,
    decode_steps: int,
) -> list[dict[str, Any]]:
    prefixes: list[dict[str, Any]] = []
    with _activation_path_hooks(model, nv_scales=nv_scales, hi_names=set()):
        for item in prompts:
            enc = tok(item.text, return_tensors="pt", truncation=True, max_length=max_seq_len)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
            prefixes.append(
                {
                    "sample_id": item.sample_id,
                    "prompt_family": item.family,
                    "phase": "prefill",
                    "input_ids": input_ids.detach().cpu(),
                }
            )
            warm = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            past = warm.past_key_values
            current = warm.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            full_ids = input_ids
            attn = torch.cat([attention_mask, torch.ones_like(current, dtype=attention_mask.dtype)], dim=-1)
            for _ in range(decode_steps):
                full_ids = torch.cat([full_ids, current], dim=-1)
                out = model(
                    input_ids=current,
                    attention_mask=attn,
                    past_key_values=past,
                    use_cache=True,
                )
                past = out.past_key_values
                current = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                attn = torch.cat([attn, torch.ones_like(current, dtype=attn.dtype)], dim=-1)
            prefixes.append(
                {
                    "sample_id": item.sample_id,
                    "prompt_family": item.family,
                    "phase": "decode",
                    "input_ids": full_ids.detach().cpu(),
                }
            )
    return prefixes


def _tree_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_tree_to_cpu(x) for x in value)
    if isinstance(value, list):
        return [_tree_to_cpu(x) for x in value]
    if isinstance(value, dict):
        return {k: _tree_to_cpu(v) for k, v in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported decoder-layer context object: {type(value)!r}")


def _tree_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_tree_to_device(x, device) for x in value)
    if isinstance(value, list):
        return [_tree_to_device(x, device) for x in value]
    if isinstance(value, dict):
        return {k: _tree_to_device(v, device) for k, v in value.items()}
    return value


def _layer_output_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    if hasattr(output, "last_hidden_state") and torch.is_tensor(output.last_hidden_state):
        return output.last_hidden_state
    raise TypeError(f"cannot extract decoder-layer hidden output from {type(output)!r}")


@torch.no_grad()
def _capture_direct_block_contexts(
    model: nn.Module,
    *,
    prefixes: list[dict[str, Any]],
    layer_indices: list[int],
    nv_scales: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[int, list[dict[str, Any]]]:
    """Run each source prefix once and capture exact per-layer args/kwargs/source output."""
    contexts: dict[int, list[dict[str, Any]]] = {int(li): [] for li in layer_indices}
    current_meta: dict[str, Any] = {}
    pending: dict[int, dict[str, Any]] = {}
    handles: list[Any] = []

    def make_pre(li: int):
        def pre_hook(_module, args, kwargs):
            if li in pending:
                raise RuntimeError(f"decoder layer {li} entered twice before returning")
            pending[li] = {
                "sample_id": current_meta["sample_id"],
                "prompt_family": current_meta["prompt_family"],
                "phase": current_meta["phase"],
                "args": _tree_to_cpu(args),
                "kwargs": _tree_to_cpu(kwargs),
            }
            return None

        return pre_hook

    def make_post(li: int):
        def post_hook(_module, _args, output):
            if li not in pending:
                raise RuntimeError(f"decoder layer {li} returned without captured input")
            rec = pending.pop(li)
            rec["source_out"] = _layer_output_tensor(output).detach().cpu()
            contexts[li].append(rec)
            return None

        return post_hook

    for li in layer_indices:
        layer = _layer_module(model, int(li))
        handles.append(layer.register_forward_pre_hook(make_pre(int(li)), with_kwargs=True))
        handles.append(layer.register_forward_hook(make_post(int(li))))
    try:
        with _activation_path_hooks(model, nv_scales=nv_scales, hi_names=set()):
            for item in prefixes:
                current_meta.clear()
                current_meta.update(
                    sample_id=str(item["sample_id"]),
                    prompt_family=str(item["prompt_family"]),
                    phase=str(item["phase"]),
                )
                ids = item["input_ids"].to(device)
                model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    use_cache=False,
                )
                if pending:
                    raise RuntimeError(f"unfinished decoder-layer contexts after forward: {sorted(pending)}")
    finally:
        for handle in handles:
            handle.remove()
    expected = len(prefixes)
    for li, rows in contexts.items():
        if len(rows) != expected:
            raise RuntimeError(f"layer {li}: expected {expected} contexts, captured {len(rows)}")
    return contexts


@contextlib.contextmanager
def _target_layer_hif4_hooks(model: nn.Module, layer_idx: int) -> Iterator[None]:
    handles: list[Any] = []

    def hook(_module: nn.Linear, inputs: tuple[Any, ...]):
        if not inputs or not torch.is_tensor(inputs[0]):
            return None
        x = inputs[0]
        q = quantize_hif4_tensor(
            x.float(), variant="full", output_dtype=x.dtype
        ).dequantized
        return (q,) + tuple(inputs[1:])

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _layer_idx(name) == layer_idx and _projection(name) is not None:
            handles.append(module.register_forward_pre_hook(hook))
    if len(handles) != 7:
        for handle in handles:
            handle.remove()
        raise RuntimeError(f"layer {layer_idx}: expected 7 HiF4 activation hooks, got {len(handles)}")
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def _direct_layer_forward(
    model: nn.Module,
    *,
    layer_idx: int,
    context: dict[str, Any],
    device: torch.device,
    hif4_activation: bool,
) -> torch.Tensor:
    layer = _layer_module(model, layer_idx)
    args = _tree_to_device(context["args"], device)
    kwargs = _tree_to_device(context["kwargs"], device)
    manager = _target_layer_hif4_hooks(model, layer_idx) if hif4_activation else contextlib.nullcontext()
    with manager:
        output = layer(*args, **kwargs)
    return _layer_output_tensor(output)


def _block_record(
    source: torch.Tensor,
    local: torch.Tensor,
    standard: torch.Tensor,
    optimized: torch.Tensor,
    *,
    phase: str,
) -> dict[str, Any]:
    if phase == "decode":
        source = source[:, -1:, :]
        local = local[:, -1:, :]
        standard = standard[:, -1:, :]
        optimized = optimized[:, -1:, :]
    return {
        "optimized_conv_error_sum": _tensor_error_sum(optimized, source),
        "optimized_local_error_sum": _tensor_error_sum(optimized, local),
        "standard_conv_error_sum": _tensor_error_sum(standard, source),
        "standard_local_error_sum": _tensor_error_sum(standard, local),
        "output_numel": int(optimized.numel()),
    }


def _aggregate_block_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    phase_rows: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for rec in records:
        dst = phase_rows[str(rec["phase"])]
        for key in (
            "optimized_conv_error_sum",
            "optimized_local_error_sum",
            "standard_conv_error_sum",
            "standard_local_error_sum",
            "output_numel",
        ):
            dst[key] += float(rec[key])
    phase_metrics: dict[str, Any] = {}
    for phase, rec in phase_rows.items():
        phase_metrics[phase] = {
            **dict(rec),
            "R_Y_conv": _recovery(rec["optimized_conv_error_sum"], rec["standard_conv_error_sum"]),
            "R_Y_local": _recovery(rec["optimized_local_error_sum"], rec["standard_local_error_sum"]),
        }
    valid = [v for v in phase_metrics.values() if math.isfinite(float(v["R_Y_conv"]))]
    return {
        "phases": phase_metrics,
        "mean_R_Y_conv": sum(float(v["R_Y_conv"]) for v in valid) / len(valid) if valid else float("-inf"),
        "mean_R_Y_local": sum(float(v["R_Y_local"]) for v in valid) / len(valid) if valid else float("-inf"),
    }


def _block_objective(metrics: dict[str, Any], lambda_local: float) -> float:
    vals = []
    for phase in ("prefill", "decode"):
        row = metrics.get("phases", {}).get(phase)
        if not row:
            continue
        r_conv = float(row["R_Y_conv"])
        r_local = float(row["R_Y_local"])
        if math.isfinite(r_conv) and math.isfinite(r_local):
            vals.append((1.0 - r_conv) + float(lambda_local) * (1.0 - r_local))
    return sum(vals) / len(vals) if vals else float("inf")


def _prepare_block_references_for_layer(
    model: nn.Module,
    *,
    layer_idx: int,
    contexts: list[dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    """Build local-FP and standard-HiF4 refs using direct decoder-layer replay."""
    layer = _layer_module(model, layer_idx)
    source_state = _clone_layer_state_cpu(layer)
    refs: list[dict[str, Any]] = []
    for context in contexts:
        _restore_layer_state(layer, source_state)
        local_out = _direct_layer_forward(
            model,
            layer_idx=layer_idx,
            context=context,
            device=device,
            hif4_activation=False,
        )

        _restore_layer_state(layer, source_state)
        _apply_standard_hif4_layer(layer)
        std_out = _direct_layer_forward(
            model,
            layer_idx=layer_idx,
            context=context,
            device=device,
            hif4_activation=True,
        )
        refs.append(
            {
                "sample_id": context["sample_id"],
                "prompt_family": context["prompt_family"],
                "phase": context["phase"],
                "context": context,
                "source_out": context["source_out"],
                "local_out": local_out.detach().cpu(),
                "standard_out": std_out.detach().cpu(),
            }
        )
    _restore_layer_state(layer, source_state)
    return source_state, refs


def _evaluate_block_mask(
    model: nn.Module,
    *,
    layer_idx: int,
    source_state: dict[str, torch.Tensor],
    refs: list[dict[str, Any]],
    model_meta: dict[str, Any],
    enabled_scales: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    layer = _layer_module(model, layer_idx)
    rows: list[dict[str, Any]] = []
    for ref in refs:
        _restore_layer_state(layer, source_state)
        _apply_folded_hif4_layer(layer, scales=enabled_scales, model_meta=model_meta)
        opt_out = _direct_layer_forward(
            model,
            layer_idx=layer_idx,
            context=ref["context"],
            device=device,
            hif4_activation=True,
        )
        rec = _block_record(
            ref["source_out"].to(device),
            ref["local_out"].to(device),
            ref["standard_out"].to(device),
            opt_out,
            phase=str(ref["phase"]),
        )
        rec.update(
            {
                "sample_id": str(ref["sample_id"]),
                "prompt_family": str(ref["prompt_family"]),
                "phase": str(ref["phase"]),
                "layer": int(layer_idx),
            }
        )
        rows.append(rec)
    _restore_layer_state(layer, source_state)
    return _aggregate_block_records(rows), rows


def _recipe_conflict_diagnostic(
    full_records: dict[str, dict[str, Any]],
    candidates: dict[str, Any],
    *,
    domain: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    recipes = candidates.get("recipes", {})
    by_layer: dict[int, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for rec in full_records.values():
        if rec.get("domain") != domain:
            continue
        li = int(rec["layer"])
        rid = str(rec["recipe_id"])
        dst = by_layer[li][rid]
        for field in (
            "joint_conv_error_sum",
            "joint_local_error_sum",
            "baseline_conv_error_sum",
            "baseline_local_error_sum",
        ):
            if isinstance(rec.get(field), (int, float)):
                dst[field] += float(rec[field])
    layer_best: dict[int, dict[str, Any]] = {}
    lambda_local = float(config.get("output_objective_lambda_local", 0.25))
    for li, recs in by_layer.items():
        opts = []
        for rid, rec in recs.items():
            recipe = recipes.get(rid, {})
            if not recipe.get("deployable", True):
                continue
            obj = _recipe_objective(dict(rec), lambda_local)
            if math.isfinite(obj):
                opts.append((obj, rid))
        if opts:
            _, rid = min(opts, key=lambda x: (x[0], x[1]))
            recipe = recipes[rid]
            layer_best[li] = {
                "recipe_id": rid,
                "granularity": int(recipe.get("granularity", 0)),
                "alpha": float(recipe.get("alpha", 0.0)),
            }
    alphas = [v["alpha"] for v in layer_best.values()]
    grans = [v["granularity"] for v in layer_best.values() if v["granularity"] > 0]
    alpha_conflict = bool(alphas) and max(alphas) - min(alphas) > 0.5
    gran_conflict = False
    if len(grans) >= 2:
        ordered = sorted({1, 4, 8, 16}.intersection(grans))
        index = {g: i for i, g in enumerate([1, 4, 8, 16])}
        if ordered:
            gran_conflict = max(index[g] for g in ordered) - min(index[g] for g in ordered) >= 2
    return {
        "layer_best": {str(k): v for k, v in sorted(layer_best.items())},
        "recipe_conflict": bool(alpha_conflict or gran_conflict),
        "alpha_conflict": alpha_conflict,
        "granularity_conflict": gran_conflict,
    }


@torch.no_grad()
def select_representative_recipe_and_policy(
    run_dir: Path,
    *,
    config: dict[str, Any],
) -> tuple[Path, Path]:
    """ES5/ES5-COMB: choose one recipe per domain and close it at decoder-layer level."""
    from Inference_Paradigm_Conversion.ipc_analysis.capture.model_adapter import (
        load_source_model_for_capture,
    )

    run_dir = Path(run_dir)
    candidates_path = run_dir / "candidate_scales.pt"
    full_path = run_dir / "es_full_eval_merged.pt"
    if not candidates_path.is_file() or not full_path.is_file():
        raise FileNotFoundError("candidate_scales.pt and es_full_eval_merged.pt are required")
    candidates = _safe_torch_load(candidates_path)
    full_merged = _safe_torch_load(full_path)
    refined_hash: str | None = None
    if bool(config.get("run_alpha_refine", True)):
        refined_path = run_dir / "es5_refined_candidate_scales.pt"
        refined_eval_path = run_dir / "es5_refine_eval_merged.pt"
        if not refined_path.is_file() or not refined_eval_path.is_file():
            raise FileNotFoundError(
                "ES5 alpha refinement is enabled but refined scales/eval artifacts are missing"
            )
        refined = _safe_torch_load(refined_path)
        refined_eval = _safe_torch_load(refined_eval_path)
        base_hash = _sha256_file(candidates_path)
        if str(refined.get("base_candidate_scales_sha256")) != base_hash:
            raise ValueError("refined candidate artifact does not match base candidates")
        if str(refined_eval.get("candidate_scales_sha256")) != base_hash:
            raise ValueError("refined eval does not match base candidates")
        refined_hash = _sha256_file(refined_path)
        if str(refined_eval.get("refined_scales_sha256")) != refined_hash:
            raise ValueError("refined eval does not match refined scale artifact")
        candidates["recipes"].update(refined.get("recipes", {}))
        for layer, domains in refined.get("scales", {}).items():
            candidates["scales"].setdefault(layer, {})
            for domain, recipes in domains.items():
                candidates["scales"][layer].setdefault(domain, {}).update(recipes)
        full_merged = dict(full_merged)
        full_records = dict(full_merged.get("records", {}))
        overlap = set(full_records).intersection(refined_eval.get("records", {}))
        if overlap:
            raise ValueError(f"coarse/refined full-eval record key collision: {sorted(overlap)[:3]}")
        full_records.update(refined_eval.get("records", {}))
        full_merged["records"] = full_records
    selected, diagnostics = _select_global_domain_recipes(
        full_merged, candidates, config=config
    )
    for domain in DOMAINS:
        diagnostics.setdefault(domain, {})["cross_layer"] = _recipe_conflict_diagnostic(
            full_merged.get("records", {}), candidates, domain=domain, config=config
        )

    checkpoint = Path(str(config["checkpoint"]))
    device = torch.device(str(config.get("selection_device", config.get("device", "cuda:0"))))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, tok = load_source_model_for_capture(checkpoint, device=device)
    nv_scales = load_nvfp4_activation_scales(resolve_activation_scale_path(checkpoint))
    nv_scales = {k: v.to(device) for k, v in nv_scales.items()}
    model_meta = _load_model_meta(checkpoint)
    prompts = _prompt_bank("discovery", int(config.get("samples_per_family", 8)))
    prefixes = _source_prefixes_for_block_replay(
        model,
        tok,
        nv_scales=nv_scales,
        prompts=prompts,
        device=device,
        max_seq_len=int(config.get("max_seq_len", 256)),
        decode_steps=int(config.get("decode_steps", 8)),
    )

    representative_layers = [int(x) for x in config.get("representative_layers", [4, 18, 34])]
    contexts_by_layer = _capture_direct_block_contexts(
        model,
        prefixes=prefixes,
        layer_indices=representative_layers,
        nv_scales=nv_scales,
        device=device,
    )
    layer_enabled: dict[str, list[str]] = {}
    block_summary: dict[str, Any] = {}
    block_rows: list[dict[str, Any]] = []
    lambda_local = float(config.get("output_objective_lambda_local", 0.25))
    for layer_idx in representative_layers:
        source_state, refs = _prepare_block_references_for_layer(
            model,
            layer_idx=layer_idx,
            contexts=contexts_by_layer[layer_idx],
            device=device,
        )
        enabled = [d for d in DOMAINS if d in selected]

        def scales_for(mask: list[str]) -> dict[str, torch.Tensor]:
            return {
                domain: candidates["scales"][str(layer_idx)][domain][selected[domain]].to(device)
                for domain in mask
            }

        metrics, rows = _evaluate_block_mask(
            model,
            layer_idx=layer_idx,
            source_state=source_state,
            refs=refs,
            model_meta=model_meta,
            enabled_scales=scales_for(enabled),
            device=device,
        )
        initial_metrics = metrics
        rollback_log: list[dict[str, Any]] = []
        while enabled and (
            float(metrics["mean_R_Y_conv"]) < 0.0
            or float(metrics["mean_R_Y_local"]) < 0.0
        ):
            trials: list[tuple[float, str, dict[str, Any], list[dict[str, Any]]]] = []
            for domain in list(enabled):
                trial_mask = [d for d in enabled if d != domain]
                trial_metrics, trial_rows = _evaluate_block_mask(
                    model,
                    layer_idx=layer_idx,
                    source_state=source_state,
                    refs=refs,
                    model_meta=model_meta,
                    enabled_scales=scales_for(trial_mask),
                    device=device,
                )
                trials.append(
                    (
                        _block_objective(trial_metrics, lambda_local),
                        domain,
                        trial_metrics,
                        trial_rows,
                    )
                )
            trials.sort(key=lambda x: (x[0], x[1]))
            _, removed, metrics, rows = trials[0]
            enabled.remove(removed)
            rollback_log.append(
                {
                    "removed_domain": removed,
                    "remaining": list(enabled),
                    "metrics": metrics,
                }
            )
        layer_enabled[str(layer_idx)] = list(enabled)
        block_summary[str(layer_idx)] = {
            "initial_enabled": [d for d in DOMAINS if d in selected],
            "final_enabled": list(enabled),
            "initial_metrics": initial_metrics,
            "final_metrics": metrics,
            "rollback": rollback_log,
        }
        for row in rows:
            row["enabled_domains"] = list(enabled)
            block_rows.append(row)
        _restore_layer_state(_layer_module(model, layer_idx), source_state)

    scale_payload: dict[str, dict[str, torch.Tensor]] = {}
    for layer_idx in [int(x) for x in config.get("representative_layers", [4, 18, 34])]:
        scale_payload[str(layer_idx)] = {}
        # Preserve every globally selected domain D, even if discovery block closure rolls
        # it back on one representative layer. ES6 needs the exact frozen D to test that
        # domain independently; representative_enabled_by_layer separately records rollback.
        for domain in selected:
            scale_payload[str(layer_idx)][domain] = (
                candidates["scales"][str(layer_idx)][domain][selected[domain]].cpu().float()
            )

    policy = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": str(checkpoint),
        "candidate_scales_sha256": _sha256_file(candidates_path),
        "refined_scales_sha256": refined_hash,
        "domain_recipes": selected,
        "recipe_definitions": {d: candidates["recipes"][rid] for d, rid in selected.items()},
        "domain_diagnostics": diagnostics,
        "representative_enabled_by_layer": layer_enabled,
        "block_summary": block_summary,
        "runtime_extra_ops": 0,
        "hif4_format_changed": False,
    }
    policy_path = run_dir / "best_scaling_policy.json"
    with policy_path.open("w", encoding="utf-8") as f:
        json.dump(policy, f, ensure_ascii=False, indent=2, sort_keys=True)
    scales_path = run_dir / "best_scaling_scales.pt"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "candidate_scales_sha256": _sha256_file(candidates_path),
            "refined_scales_sha256": refined_hash,
            "scales": scale_payload,
        },
        scales_path,
    )
    with (run_dir / "es5_block_replay.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": block_summary, "rows": block_rows}, f, ensure_ascii=False, indent=2)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return policy_path, scales_path


def _combine_layer_metrics(metrics_by_layer: dict[str, dict[str, Any]]) -> dict[str, Any]:
    conv = [float(v["mean_R_Y_conv"]) for v in metrics_by_layer.values() if math.isfinite(float(v["mean_R_Y_conv"]))]
    local = [float(v["mean_R_Y_local"]) for v in metrics_by_layer.values() if math.isfinite(float(v["mean_R_Y_local"]))]
    phase_values: dict[str, list[float]] = defaultdict(list)
    for metrics in metrics_by_layer.values():
        for phase, row in metrics.get("phases", {}).items():
            if math.isfinite(float(row["R_Y_conv"])):
                phase_values[phase].append(float(row["R_Y_conv"]))
    return {
        "mean_R_Y_conv": sum(conv) / len(conv) if conv else float("-inf"),
        "median_R_Y_conv": float(torch.tensor(conv).median().item()) if conv else float("-inf"),
        "mean_R_Y_local": sum(local) / len(local) if local else float("-inf"),
        "phase_mean_R_Y_conv": {
            phase: sum(vals) / len(vals) for phase, vals in sorted(phase_values.items()) if vals
        },
    }


@torch.no_grad()
def run_representative_validation(
    checkpoint: Path,
    run_dir: Path,
    *,
    device: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """ES6: freeze discovery scales/recipes; validation may only disable a domain."""
    from Inference_Paradigm_Conversion.ipc_analysis.capture.model_adapter import (
        load_source_model_for_capture,
    )

    checkpoint = Path(checkpoint)
    run_dir = Path(run_dir)
    policy_path = run_dir / "best_scaling_policy.json"
    scales_path = run_dir / "best_scaling_scales.pt"
    with policy_path.open("r", encoding="utf-8") as f:
        policy = json.load(f)
    representative_scales = _safe_torch_load(scales_path)["scales"]
    selected = dict(policy.get("domain_recipes", {}))
    rep_layers = [int(x) for x in config.get("representative_layers", [4, 18, 34])]

    device_t = torch.device(device)
    if device_t.type == "cuda":
        torch.cuda.set_device(device_t)
    model, tok = load_source_model_for_capture(checkpoint, device=device_t)
    nv_scales = load_nvfp4_activation_scales(resolve_activation_scale_path(checkpoint))
    nv_scales = {k: v.to(device_t) for k, v in nv_scales.items()}
    model_meta = _load_model_meta(checkpoint)
    prompts = _prompt_bank("validation", int(config.get("samples_per_family", 8)))
    prefixes = _source_prefixes_for_block_replay(
        model,
        tok,
        nv_scales=nv_scales,
        prompts=prompts,
        device=device_t,
        max_seq_len=int(config.get("max_seq_len", 256)),
        decode_steps=int(config.get("decode_steps", 8)),
    )

    contexts_by_layer = _capture_direct_block_contexts(
        model,
        prefixes=prefixes,
        layer_indices=rep_layers,
        nv_scales=nv_scales,
        device=device_t,
    )
    refs_by_layer: dict[int, tuple[dict[str, torch.Tensor], list[dict[str, Any]]]] = {}
    for li in rep_layers:
        refs_by_layer[li] = _prepare_block_references_for_layer(
            model,
            layer_idx=li,
            contexts=contexts_by_layer[li],
            device=device_t,
        )

    # Each globally selected domain is evaluated alone with exactly its frozen discovery D.
    domain_metrics: dict[str, dict[str, Any]] = {}
    validated_domains: list[str] = []
    for domain in DOMAINS:
        if domain not in selected:
            continue
        by_layer: dict[str, dict[str, Any]] = {}
        for li in rep_layers:
            source_state, refs = refs_by_layer[li]
            # best_scaling_scales.pt stores every globally selected frozen D, even when
            # discovery block closure rolled the domain back on this layer.
            d = representative_scales[str(li)][domain].to(device_t)
            metrics, _ = _evaluate_block_mask(
                model,
                layer_idx=li,
                source_state=source_state,
                refs=refs,
                model_meta=model_meta,
                enabled_scales={domain: d},
                device=device_t,
            )
            by_layer[str(li)] = metrics
        combined = _combine_layer_metrics(by_layer)
        domain_metrics[domain] = {"by_layer": by_layer, "combined": combined}
        if float(combined["mean_R_Y_conv"]) >= 0.0:
            validated_domains.append(domain)

    combined_by_layer: dict[str, dict[str, Any]] = {}
    combined_rows: list[dict[str, Any]] = []
    for li in rep_layers:
        source_state, refs = refs_by_layer[li]
        discovery_enabled = set(policy.get("representative_enabled_by_layer", {}).get(str(li), []))
        active = [d for d in validated_domains if d in discovery_enabled]
        layer_scales = {
            d: representative_scales[str(li)][d].to(device_t)
            for d in active
        }
        metrics, rows = _evaluate_block_mask(
            model,
            layer_idx=li,
            source_state=source_state,
            refs=refs,
            model_meta=model_meta,
            enabled_scales=layer_scales,
            device=device_t,
        )
        combined_by_layer[str(li)] = metrics
        for row in rows:
            row["enabled_domains"] = active
            combined_rows.append(row)
        _restore_layer_state(_layer_module(model, li), source_state)

    combined = _combine_layer_metrics(combined_by_layer)
    wins = sum(
        1
        for row in combined_rows
        if float(row["optimized_conv_error_sum"]) <= float(row["standard_conv_error_sum"])
    )
    win_rate = wins / len(combined_rows) if combined_rows else 0.0
    phase = combined.get("phase_mean_R_Y_conv", {})
    validation_pass = (
        float(combined["mean_R_Y_conv"]) >= float(config.get("validation_min_recovery", 0.05))
        and float(combined["median_R_Y_conv"]) >= 0.0
        and float(combined["mean_R_Y_local"]) >= 0.0
        and win_rate >= float(config.get("validation_min_win_rate", 0.60))
        and float(phase.get("prefill", float("-inf"))) >= 0.0
        and float(phase.get("decode", float("-inf"))) >= 0.0
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "split": "validation",
        "validated_domains": validated_domains,
        "domain_metrics": domain_metrics,
        "combined_by_layer": combined_by_layer,
        "combined": combined,
        "paired_win_rate": win_rate,
        "validation_pass": bool(validation_pass),
        "rows": combined_rows,
    }
    with (run_dir / "es6_validation.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    policy["validated_domains"] = validated_domains
    policy["validation"] = {
        "validation_pass": bool(validation_pass),
        "combined": combined,
        "paired_win_rate": win_rate,
    }
    with policy_path.open("w", encoding="utf-8") as f:
        json.dump(policy, f, ensure_ascii=False, indent=2, sort_keys=True)
    del model
    if device_t.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _scale_from_frozen_recipe(
    merged_stats: dict[str, Any],
    *,
    checkpoint: Path,
    layer: int,
    domain: str,
    recipe: dict[str, Any],
    config: dict[str, Any],
) -> torch.Tensor:
    group_size = int(config.get("group_size", 64))
    model_meta = dict(merged_stats["model_meta"])
    amp_full = _deploy_amplitude(merged_stats, layer=layer, domain=domain)
    width_full = int(amp_full.numel())
    amp_design = _o_unique_amplitude(amp_full, model_meta) if domain == "o_in" else amp_full
    kind = str(recipe["kind"])
    if kind == "pts_layer":
        return torch.full((width_full,), float(recipe["scalar"]), dtype=torch.float32)
    if kind == "phase_g64":
        grid = candidate_pts_scales(
            log2_min=float(config.get("pts_log2_min", -1.0)),
            log2_max=float(config.get("pts_log2_max", 1.0)),
            points=int(config.get("pts_points", 33)),
        )
        d, _ = _phase_g64_scale(
            merged_stats,
            layer=layer,
            domain=domain,
            width=width_full,
            pts_grid=grid,
            group_size=group_size,
            model_meta=model_meta,
        )
        return d
    if kind == "equalize":
        d_design = build_equalization_scale(
            amp_design,
            granularity=int(recipe["granularity"]),
            alpha=float(recipe["alpha"]),
            group_size=group_size,
            min_scale=float(recipe.get("min_scale", config.get("min_scale", 0.5))),
            max_scale=float(recipe.get("max_scale", config.get("max_scale", 2.0))),
        )
        return _expand_o_unique(d_design, model_meta) if domain == "o_in" else d_design
    if kind == "equalize_aw":
        if domain not in {"attn_in", "mlp_in"}:
            raise ValueError("equalize_aw is only legal for attn_in/mlp_in")
        wstat = _shared_weight_stat_from_checkpoint(checkpoint, layer=layer, domain=domain)
        return build_weight_aware_equalization_scale(
            amp_design,
            wstat,
            granularity=int(recipe["granularity"]),
            beta=float(recipe.get("beta", recipe.get("alpha", 0.5))),
            group_size=group_size,
            min_scale=float(recipe.get("min_scale", config.get("min_scale", 0.5))),
            max_scale=float(recipe.get("max_scale", config.get("max_scale", 2.0))),
        )
    raise ValueError(f"unsupported frozen recipe kind {kind!r}")


@torch.no_grad()
def instantiate_all_layer_policy(
    checkpoint: Path,
    representative_policy_path: Path,
    *,
    calibration_split: str,
    device: str,
    out_dir: Path,
    config: dict[str, Any],
) -> tuple[Path, Path]:
    """ES6.5: frozen recipe -> layer-specific D for every decoder layer, then block rollback."""
    checkpoint = Path(checkpoint)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with Path(representative_policy_path).open("r", encoding="utf-8") as f:
        policy = json.load(f)
    if not bool(policy.get("validation", {}).get("validation_pass", False)):
        raise RuntimeError("representative validation did not pass; all-layer instantiation is gated")
    validated_domains = list(policy.get("validated_domains", []))
    recipes = dict(policy.get("recipe_definitions", {}))
    model_meta = _load_model_meta(checkpoint)
    all_layers = list(range(int(model_meta["num_hidden_layers"])))

    stats_dir = out_dir / "all_layer_stats"
    stats_cfg = dict(config)
    stats_cfg.pop("config_sha256", None)
    stats_cfg["representative_layers"] = all_layers
    phase_domains = [d for d in validated_domains if recipes[d]["kind"] == "phase_g64"]
    stats_cfg["collect_phase_g64_errors"] = bool(phase_domains)
    stats_cfg["phase_g64_domains"] = phase_domains
    # This public function is a one-device closed-loop entry. The CLI may instead run the
    # same stats stage sharded and place es_stats_merged.pt in stats_dir before calling us.
    merged_path = stats_dir / "es_stats_merged.pt"
    if not merged_path.is_file():
        run_scaling_stats_shard(
            checkpoint,
            stats_dir,
            split=calibration_split,
            shard_id=0,
            num_shards=1,
            device=device,
            config=stats_cfg,
        )
        merged_path = merge_scaling_stats(stats_dir, expected_num_shards=1)
    merged = _safe_torch_load(merged_path)

    all_scales: dict[str, dict[str, torch.Tensor]] = {}
    for li in all_layers:
        all_scales[str(li)] = {}
        for domain in validated_domains:
            all_scales[str(li)][domain] = _scale_from_frozen_recipe(
                merged,
                checkpoint=checkpoint,
                layer=li,
                domain=domain,
                recipe=recipes[domain],
                config=config,
            ).cpu().float()

    # Real layer checks and deterministic per-layer rollback.
    from Inference_Paradigm_Conversion.ipc_analysis.capture.model_adapter import (
        load_source_model_for_capture,
    )

    device_t = torch.device(device)
    if device_t.type == "cuda":
        torch.cuda.set_device(device_t)
    model, tok = load_source_model_for_capture(checkpoint, device=device_t)
    nv_scales = load_nvfp4_activation_scales(resolve_activation_scale_path(checkpoint))
    nv_scales = {k: v.to(device_t) for k, v in nv_scales.items()}
    prompts = _prompt_bank(calibration_split, int(config.get("samples_per_family", 8)))
    prefixes = _source_prefixes_for_block_replay(
        model,
        tok,
        nv_scales=nv_scales,
        prompts=prompts,
        device=device_t,
        max_seq_len=int(config.get("max_seq_len", 256)),
        decode_steps=int(config.get("decode_steps", 8)),
    )
    contexts_by_layer = _capture_direct_block_contexts(
        model,
        prefixes=prefixes,
        layer_indices=all_layers,
        nv_scales=nv_scales,
        device=device_t,
    )
    threshold = float(config.get("all_layer_bad_recovery_threshold", -0.05))
    lambda_local = float(config.get("output_objective_lambda_local", 0.25))
    layer_enabled: dict[str, list[str]] = {}
    layer_checks: dict[str, Any] = {}
    for li in all_layers:
        source_state, refs = _prepare_block_references_for_layer(
            model,
            layer_idx=li,
            contexts=contexts_by_layer[li],
            device=device_t,
        )
        enabled = list(validated_domains)

        def scales_for(mask: list[str]) -> dict[str, torch.Tensor]:
            return {d: all_scales[str(li)][d].to(device_t) for d in mask}

        metrics, rows = _evaluate_block_mask(
            model,
            layer_idx=li,
            source_state=source_state,
            refs=refs,
            model_meta=model_meta,
            enabled_scales=scales_for(enabled),
            device=device_t,
        )
        initial = metrics
        rollback: list[dict[str, Any]] = []
        while enabled and float(metrics["mean_R_Y_conv"]) < threshold:
            trials = []
            for domain in list(enabled):
                mask = [d for d in enabled if d != domain]
                m, r = _evaluate_block_mask(
                    model,
                    layer_idx=li,
                    source_state=source_state,
                    refs=refs,
                    model_meta=model_meta,
                    enabled_scales=scales_for(mask),
                    device=device_t,
                )
                trials.append((_block_objective(m, lambda_local), domain, m, r))
            trials.sort(key=lambda x: (x[0], x[1]))
            _, removed, metrics, rows = trials[0]
            enabled.remove(removed)
            rollback.append({"removed_domain": removed, "remaining": list(enabled), "metrics": metrics})
        layer_enabled[str(li)] = list(enabled)
        layer_checks[str(li)] = {
            "initial_metrics": initial,
            "final_metrics": metrics,
            "rollback": rollback,
        }
        _restore_layer_state(_layer_module(model, li), source_state)

    final_scales = {
        li: {d: all_scales[li][d] for d in layer_enabled[li]}
        for li in all_scales
    }
    out_policy = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": str(checkpoint),
        "source_representative_policy": str(representative_policy_path),
        "domain_recipes": {d: policy["domain_recipes"][d] for d in validated_domains},
        "recipe_definitions": {d: recipes[d] for d in validated_domains},
        "validated_domains": validated_domains,
        "enabled_by_layer": layer_enabled,
        "layer_checks": layer_checks,
        "runtime_extra_ops": 0,
        "hif4_format_changed": False,
    }
    policy_out = out_dir / "best_scaling_policy_all_layers.json"
    scales_out = out_dir / "best_scaling_scales_all_layers.pt"
    with policy_out.open("w", encoding="utf-8") as f:
        json.dump(out_policy, f, ensure_ascii=False, indent=2, sort_keys=True)
    torch.save({"schema_version": SCHEMA_VERSION, "scales": final_scales}, scales_out)
    with (out_dir / "es65_all_layer_local_checks.json").open("w", encoding="utf-8") as f:
        json.dump(layer_checks, f, ensure_ascii=False, indent=2)
    del model
    if device_t.type == "cuda":
        torch.cuda.empty_cache()
    return policy_out, scales_out


def _model_meta_from_model(model: nn.Module) -> dict[str, Any]:
    cfg = model.config
    return {
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "hidden_size": int(cfg.hidden_size),
        "intermediate_size": int(cfg.intermediate_size),
        "num_attention_heads": int(cfg.num_attention_heads),
        "num_key_value_heads": int(cfg.num_key_value_heads),
        "head_dim": int(getattr(cfg, "head_dim", int(cfg.hidden_size) // int(cfg.num_attention_heads))),
    }


@torch.no_grad()
def apply_all_layer_policy_inplace(
    model: nn.Module,
    all_layer_policy_path: Path,
    all_layer_scales_path: Path,
) -> dict[str, Any]:
    """Fold a frozen all-layer policy into a fresh source model and HiF4-QDQ all 7 Linears/layer."""
    with Path(all_layer_policy_path).open("r", encoding="utf-8") as f:
        policy = json.load(f)
    scale_art = _safe_torch_load(Path(all_layer_scales_path))
    scales = scale_art["scales"]
    model_meta = _model_meta_from_model(model)
    n_layers = int(model_meta["num_hidden_layers"])
    if set(scales) - {str(i) for i in range(n_layers)}:
        raise ValueError("all-layer scale artifact contains invalid layer ids")
    enabled_by_layer = policy.get("enabled_by_layer", {})
    applied: dict[str, list[str]] = {}
    for li in range(n_layers):
        layer = _layer_module(model, li)
        enabled = list(enabled_by_layer.get(str(li), []))
        layer_scales = {
            domain: scales.get(str(li), {})[domain].to(layer.self_attn.q_proj.weight.device)
            for domain in enabled
        }
        _apply_folded_hif4_layer(layer, scales=layer_scales, model_meta=model_meta)
        applied[str(li)] = enabled
    return {
        "num_layers": n_layers,
        "applied_domains_by_layer": applied,
        "runtime_extra_ops": 0,
    }


@torch.no_grad()
def _collect_variant_logits(
    checkpoint: Path,
    *,
    variant: Literal["source", "standard", "optimized"],
    prompts: list[Any],
    device: torch.device,
    max_seq_len: int,
    all_layer_policy_path: Path | None = None,
    all_layer_scales_path: Path | None = None,
) -> dict[str, torch.Tensor]:
    from Inference_Paradigm_Conversion.ipc_analysis.capture.model_adapter import (
        load_source_model_for_capture,
    )

    model, tok = load_source_model_for_capture(checkpoint, device=device)
    nv_scales = load_nvfp4_activation_scales(resolve_activation_scale_path(checkpoint))
    nv_scales = {k: v.to(device) for k, v in nv_scales.items()}
    if variant == "standard":
        for li in range(int(model.config.num_hidden_layers)):
            _apply_standard_hif4_layer(_layer_module(model, li))
    elif variant == "optimized":
        if all_layer_policy_path is None or all_layer_scales_path is None:
            raise ValueError("optimized trajectory requires all-layer policy and scales")
        apply_all_layer_policy_inplace(model, all_layer_policy_path, all_layer_scales_path)
    elif variant != "source":
        raise ValueError(variant)

    hi_names = _all_target_linear_names(model) if variant in {"standard", "optimized"} else set()
    outputs: dict[str, torch.Tensor] = {}
    with _activation_path_hooks(model, nv_scales=nv_scales, hi_names=hi_names):
        for item in prompts:
            enc = tok(item.text, return_tensors="pt", truncation=True, max_length=max_seq_len)
            batch = {k: v.to(device) for k, v in enc.items()}
            if "attention_mask" not in batch:
                batch["attention_mask"] = torch.ones_like(batch["input_ids"])
            # BF16 is sufficient for stored trajectory artifacts; logits_distance promotes
            # both inputs to FP32 internally. This cuts host memory in half.
            logits = model(**batch, use_cache=False).logits.detach().to(torch.bfloat16).cpu()
            outputs[item.sample_id] = logits
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return outputs


@torch.no_grad()
def run_target_trajectory_check(
    checkpoint: Path,
    all_layer_policy_path: Path,
    all_layer_scales_path: Path,
    *,
    split: str,
    device: str,
    out_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compare full NVFP4 source / standard HiF4 / optimized HiF4 target trajectories."""
    from Inference_Paradigm_Conversion.ipc_analysis.analysis.network_injection import (
        logits_distance,
    )

    checkpoint = Path(checkpoint)
    out_dir = Path(out_dir)
    device_t = torch.device(device)
    if device_t.type == "cuda":
        torch.cuda.set_device(device_t)
    prompts = _prompt_bank(split, int(config.get("samples_per_family", 8)))
    max_seq_len = int(config.get("max_seq_len", 256))
    source = _collect_variant_logits(
        checkpoint,
        variant="source",
        prompts=prompts,
        device=device_t,
        max_seq_len=max_seq_len,
    )
    standard = _collect_variant_logits(
        checkpoint,
        variant="standard",
        prompts=prompts,
        device=device_t,
        max_seq_len=max_seq_len,
    )
    standard_metrics = {
        sample_id: logits_distance(source[sample_id], standard[sample_id])
        for sample_id in sorted(source)
    }
    del standard

    optimized = _collect_variant_logits(
        checkpoint,
        variant="optimized",
        prompts=prompts,
        device=device_t,
        max_seq_len=max_seq_len,
        all_layer_policy_path=Path(all_layer_policy_path),
        all_layer_scales_path=Path(all_layer_scales_path),
    )

    rows: list[dict[str, Any]] = []
    family_by_id = {item.sample_id: item.family for item in prompts}
    for sample_id in sorted(source):
        std = standard_metrics[sample_id]
        opt = logits_distance(source[sample_id], optimized[sample_id])
        rows.append(
            {
                "sample_id": sample_id,
                "prompt_family": family_by_id[sample_id],
                "standard_logits_nmse": float(std["logits_nmse"]),
                "standard_kl_last": float(std["kl_last"]),
                "optimized_logits_nmse": float(opt["logits_nmse"]),
                "optimized_kl_last": float(opt["kl_last"]),
            }
        )
    del optimized, source
    mean_std_kl = sum(r["standard_kl_last"] for r in rows) / len(rows)
    mean_opt_kl = sum(r["optimized_kl_last"] for r in rows) / len(rows)
    mean_std_nmse = sum(r["standard_logits_nmse"] for r in rows) / len(rows)
    mean_opt_nmse = sum(r["optimized_logits_nmse"] for r in rows) / len(rows)
    max_ratio = float(config.get("target_trajectory_max_kl_regression_ratio", 1.10))
    if mean_std_kl <= 1e-12:
        regression = mean_opt_kl > mean_std_kl + 1e-12
        kl_ratio = None
    else:
        kl_ratio = mean_opt_kl / mean_std_kl
        regression = kl_ratio > max_ratio
    family_summary: dict[str, Any] = {}
    for family in sorted({r["prompt_family"] for r in rows}):
        subset = [r for r in rows if r["prompt_family"] == family]
        family_summary[family] = {
            "standard_kl_last": sum(r["standard_kl_last"] for r in subset) / len(subset),
            "optimized_kl_last": sum(r["optimized_kl_last"] for r in subset) / len(subset),
            "standard_logits_nmse": sum(r["standard_logits_nmse"] for r in subset) / len(subset),
            "optimized_logits_nmse": sum(r["optimized_logits_nmse"] for r in subset) / len(subset),
        }
    result = {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "mean_standard_kl_last": mean_std_kl,
        "mean_optimized_kl_last": mean_opt_kl,
        "kl_ratio_optimized_over_standard": kl_ratio,
        "mean_standard_logits_nmse": mean_std_nmse,
        "mean_optimized_logits_nmse": mean_opt_nmse,
        "target_trajectory_regression": bool(regression),
        "threshold_ratio": max_ratio,
        "family_summary": family_summary,
        "rows": rows,
    }
    out_path = out_dir / "es65_target_trajectory.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with Path(all_layer_policy_path).open("r", encoding="utf-8") as f:
        policy = json.load(f)
    policy["target_trajectory"] = {
        "target_trajectory_regression": bool(regression),
        "mean_standard_kl_last": mean_std_kl,
        "mean_optimized_kl_last": mean_opt_kl,
        "kl_ratio_optimized_over_standard": kl_ratio,
    }
    with Path(all_layer_policy_path).open("w", encoding="utf-8") as f:
        json.dump(policy, f, ensure_ascii=False, indent=2, sort_keys=True)
    return result
