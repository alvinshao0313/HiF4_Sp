"""真实 packed NVFP4 权重的分块实验。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch

import nvfp4_hif4_torch as core


PAYLOAD_VARIANTS: dict[str, core.HiF4Config] = {
    "s1p2_native": core.HiF4Config(payload_format="s1p2", hierarchy_format="s1p2"),
    "e2m1_native": core.HiF4Config(payload_format="e2m1", hierarchy_format="e2m1"),
    "e2m1_fixed": core.HiF4Config(payload_format="e2m1", hierarchy_format="s1p2"),
    "bf16_range_matched": core.HiF4Config(payload_format="bf16_range_matched", hierarchy_format="s1p2"),
    "bf16_unclipped": core.HiF4Config(payload_format="bf16_unclipped", hierarchy_format="s1p2"),
}

MICRO_EXP_VARIANTS: dict[str, core.HiF4Config] = {
    "H00_no_exp": core.HiF4Config(enable_exp8=False, enable_exp4=False),
    "H10_exp8_only": core.HiF4Config(enable_exp8=True, enable_exp4=False),
    "H01_exp4_only": core.HiF4Config(enable_exp8=False, enable_exp4=True),
    "H11_full": core.HiF4Config(enable_exp8=True, enable_exp4=True),
}

TOP_SCALE_VARIANTS: dict[str, core.HiF4Config] = {
    mode: core.HiF4Config(scale_mode=mode)
    for mode in ("continuous", "bf16_math", "e6m2_only", "hardware")
}

GROUP_SIZE_VARIANTS: dict[str, core.HiF4Config] = {
    f"g{group_size}": core.HiF4Config(
        group_size=group_size,
        scale_mode="hardware",
        payload_format="s1p2",
        hierarchy_format="s1p2",
        enable_exp8=True,
        enable_exp4=True,
    )
    for group_size in (16, 32, 64)
}


def _config_summary(config: core.HiF4Config) -> dict[str, Any]:
    return {
        "group_size": config.group_size,
        "group_dim": config.group_dim,
        "scale_mode": config.scale_mode,
        "payload_format": config.payload_format,
        "hierarchy_format": config.hierarchy_format,
        "enable_exp8": config.enable_exp8,
        "enable_exp4": config.enable_exp4,
    }


class _MetricAccumulator:
    def __init__(self) -> None:
        self.sums = core.ErrorSums()
        self.chunk_count = 0

    def add(self, reference: torch.Tensor, approximation: torch.Tensor) -> None:
        self.sums = core.merge_error_sums(
            self.sums,
            core.compute_error_sums(reference, approximation),
        )
        self.chunk_count += 1

    def finalize(self) -> dict[str, Any]:
        return {
            "metrics": core.finalize_error_metrics(self.sums),
            "chunk_count": self.chunk_count,
        }


class _DecompositionAccumulator:
    def __init__(self) -> None:
        self.reference_energy = 0.0
        self.first_energy = 0.0
        self.second_energy = 0.0
        self.cross_energy = 0.0
        self.total_energy = 0.0

    def add(
        self,
        reference: torch.Tensor,
        first_error: torch.Tensor,
        second_error: torch.Tensor,
    ) -> None:
        reference64 = reference.detach().to(torch.float64)
        first64 = first_error.detach().to(torch.float64)
        second64 = second_error.detach().to(torch.float64)
        total64 = first64 + second64
        self.reference_energy += float((reference64 * reference64).sum().item())
        self.first_energy += float((first64 * first64).sum().item())
        self.second_energy += float((second64 * second64).sum().item())
        self.cross_energy += float((2.0 * first64 * second64).sum().item())
        self.total_energy += float((total64 * total64).sum().item())

    def finalize(self, first_name: str, second_name: str) -> dict[str, Any]:
        reconstructed = self.first_energy + self.second_energy + self.cross_energy
        residual = self.total_energy - reconstructed
        denominator = self.reference_energy if self.reference_energy > 0 else 1.0
        return {
            "reference_energy": self.reference_energy,
            first_name: self.first_energy,
            second_name: self.second_energy,
            "cross_energy": self.cross_energy,
            "total_energy": self.total_energy,
            "identity_residual": residual,
            "identity_residual_relative_to_reference": residual / denominator,
            f"{first_name}_normalized": self.first_energy / denominator,
            f"{second_name}_normalized": self.second_energy / denominator,
            "cross_energy_normalized": self.cross_energy / denominator,
            "total_energy_normalized": self.total_energy / denominator,
        }


def _nmse(bundle: dict[str, Any]) -> float:
    value = bundle["metrics"]["nmse"]
    if not isinstance(value, (int, float)):
        raise ValueError("real packed study requires non-zero references")
    return float(value)


def _micro_effects(variants: dict[str, Any]) -> dict[str, float]:
    l00 = _nmse(variants["H00_no_exp"])
    l10 = _nmse(variants["H10_exp8_only"])
    l01 = _nmse(variants["H01_exp4_only"])
    l11 = _nmse(variants["H11_full"])
    return {
        "exp8_mean_gain": ((l00 - l10) + (l01 - l11)) / 2.0,
        "exp4_mean_gain": ((l00 - l01) + (l10 - l11)) / 2.0,
        "interaction": l00 - l10 - l01 + l11,
    }


def _top_scale_effects(variants: dict[str, Any]) -> dict[str, float]:
    continuous = _nmse(variants["continuous"])
    bf16_math = _nmse(variants["bf16_math"])
    e6m2_only = _nmse(variants["e6m2_only"])
    hardware = _nmse(variants["hardware"])
    return {
        "bf16_math_mean_penalty": ((bf16_math - continuous) + (hardware - e6m2_only)) / 2.0,
        "e6m2_mean_penalty": ((e6m2_only - continuous) + (hardware - bf16_math)) / 2.0,
        "interaction": hardware - bf16_math - e6m2_only + continuous,
    }


def _group_size_result(variants: dict[str, Any]) -> dict[str, Any]:
    enriched: dict[str, Any] = {}
    for name, bundle in variants.items():
        config = GROUP_SIZE_VARIANTS[name]
        enriched[name] = {
            **bundle,
            "group_size": config.group_size,
            "full_three_level_hierarchy": bool(config.enable_exp8 and config.enable_exp4),
            "config": _config_summary(config),
        }
    nmse16 = _nmse(enriched["g16"])
    nmse32 = _nmse(enriched["g32"])
    nmse64 = _nmse(enriched["g64"])
    drop64_32 = nmse64 - nmse32
    drop32_16 = nmse32 - nmse16
    return {
        "reference": "只改变顶层S0共享范围；S1P2、每8元素指数和每4元素指数全部保留",
        "variants": enriched,
        "comparisons": {
            "nmse_drop_64_to_32": drop64_32,
            "nmse_drop_32_to_16": drop32_16,
            "relative_drop_64_to_32": None if nmse64 == 0 else drop64_32 / nmse64,
            "relative_drop_32_to_16": None if nmse32 == 0 else drop32_16 / nmse32,
            "recoverable_fraction_64_to_16": None if nmse64 == 0 else (nmse64 - nmse16) / nmse64,
            "diminishing_returns": drop32_16 < drop64_32,
        },
    }


class _RealBucket:
    def __init__(self) -> None:
        self.payload = {name: _MetricAccumulator() for name in PAYLOAD_VARIANTS}
        self.micro = {name: _MetricAccumulator() for name in MICRO_EXP_VARIANTS}
        self.top_scale = {name: _MetricAccumulator() for name in TOP_SCALE_VARIANTS}
        self.group_size = {name: _MetricAccumulator() for name in GROUP_SIZE_VARIANTS}
        self.native = {
            "fp32_carrier": _MetricAccumulator(),
            "bf16_carrier": _MetricAccumulator(),
            "bf16_projection": _MetricAccumulator(),
            "pts_fp32": _MetricAccumulator(),
            "pts_bf16": _MetricAccumulator(),
            "pts_bf16_projection": _MetricAccumulator(),
        }
        self.carrier_decomposition = _DecompositionAccumulator()
        self.pts_carrier_decomposition = _DecompositionAccumulator()
        self.payload_decomposition = _DecompositionAccumulator()

    def add(
        self,
        source: torch.Tensor,
        payload_reconstructions: dict[str, torch.Tensor],
        micro_reconstructions: dict[str, torch.Tensor],
        top_scale_reconstructions: dict[str, torch.Tensor],
        group_size_reconstructions: dict[str, torch.Tensor],
        bf16_carrier: torch.Tensor,
        bf16_hif4: torch.Tensor,
        pts_fp32: torch.Tensor,
        pts_bf16_projection: torch.Tensor,
        pts_bf16: torch.Tensor,
    ) -> None:
        for name, reconstruction in payload_reconstructions.items():
            self.payload[name].add(source, reconstruction)
        for name, reconstruction in micro_reconstructions.items():
            self.micro[name].add(source, reconstruction)
        for name, reconstruction in top_scale_reconstructions.items():
            self.top_scale[name].add(source, reconstruction)
        for name, reconstruction in group_size_reconstructions.items():
            self.group_size[name].add(source, reconstruction)

        s1p2 = payload_reconstructions["s1p2_native"]
        matched = payload_reconstructions["bf16_range_matched"]
        self.native["fp32_carrier"].add(source, s1p2)
        self.native["bf16_carrier"].add(source, bf16_hif4)
        self.native["bf16_projection"].add(source, bf16_carrier)
        self.native["pts_fp32"].add(source, pts_fp32)
        self.native["pts_bf16"].add(source, pts_bf16)
        self.native["pts_bf16_projection"].add(source, pts_bf16_projection)
        self.carrier_decomposition.add(
            source,
            bf16_carrier - source,
            bf16_hif4 - bf16_carrier,
        )
        self.pts_carrier_decomposition.add(
            source,
            pts_bf16_projection - source,
            pts_bf16 - pts_bf16_projection,
        )
        self.payload_decomposition.add(
            source,
            matched - source,
            s1p2 - matched,
        )

    def finalize(self) -> dict[str, Any]:
        payload_variants = {
            name: accumulator.finalize()
            for name, accumulator in self.payload.items()
        }
        micro_variants = {
            name: accumulator.finalize()
            for name, accumulator in self.micro.items()
        }
        top_scale_variants = {
            name: accumulator.finalize()
            for name, accumulator in self.top_scale.items()
        }
        group_size_variants = {
            name: accumulator.finalize()
            for name, accumulator in self.group_size.items()
        }
        native_variants = {
            name: accumulator.finalize()
            for name, accumulator in self.native.items()
        }
        s1p2_nmse = _nmse(payload_variants["s1p2_native"])
        e2m1_nmse = _nmse(payload_variants["e2m1_native"])
        fixed_nmse = _nmse(payload_variants["e2m1_fixed"])
        matched_nmse = _nmse(payload_variants["bf16_range_matched"])
        unclipped_nmse = _nmse(payload_variants["bf16_unclipped"])
        fp32_nmse = _nmse(native_variants["fp32_carrier"])
        bf16_nmse = _nmse(native_variants["bf16_carrier"])
        pts_fp32_nmse = _nmse(native_variants["pts_fp32"])
        pts_bf16_nmse = _nmse(native_variants["pts_bf16"])
        return {
            "native_conversion": {
                "reference": "packed NVFP4的FP32数学解码值",
                "variants": native_variants,
                "bf16_minus_fp32_carrier_nmse": bf16_nmse - fp32_nmse,
                "bf16_to_fp32_carrier_nmse_ratio": None
                if fp32_nmse == 0
                else bf16_nmse / fp32_nmse,
                "pts_fp32_minus_direct_nmse": pts_fp32_nmse - fp32_nmse,
                "pts_bf16_minus_direct_nmse": pts_bf16_nmse - fp32_nmse,
                "pts_bf16_minus_pts_fp32_nmse": pts_bf16_nmse - pts_fp32_nmse,
                "bf16_carrier_decomposition": self.carrier_decomposition.finalize(
                    "bf16_projection_energy",
                    "hif4_after_bf16_energy",
                ),
                "pts_bf16_carrier_decomposition": self.pts_carrier_decomposition.finalize(
                    "pts_bf16_projection_energy",
                    "hif4_after_pts_bf16_energy",
                ),
            },
            "payload_ablation": {
                "variants": payload_variants,
                "comparisons": {
                    "e2m1_minus_s1p2_nmse": e2m1_nmse - s1p2_nmse,
                    "e2m1_relative_change": None
                    if s1p2_nmse == 0
                    else (e2m1_nmse - s1p2_nmse) / s1p2_nmse,
                    "e2m1_fixed_minus_native_nmse": fixed_nmse - e2m1_nmse,
                    "s1p2_recoverable_to_bf16_range_matched": None
                    if s1p2_nmse == 0
                    else (s1p2_nmse - matched_nmse) / s1p2_nmse,
                    "s1p2_recoverable_to_bf16_unclipped": None
                    if s1p2_nmse == 0
                    else (s1p2_nmse - unclipped_nmse) / s1p2_nmse,
                },
                "s1p2_vs_bf16_decomposition": self.payload_decomposition.finalize(
                    "bf16_range_matched_floor_energy",
                    "s1p2_increment_energy",
                ),
            },
            "micro_exponent_ablation": {
                "variants": micro_variants,
                "effects": _micro_effects(micro_variants),
            },
            "top_scale_ablation": {
                "variants": top_scale_variants,
                "effects": _top_scale_effects(top_scale_variants),
            },
            "group_size_ablation": _group_size_result(group_size_variants),
        }


def _layer_from_name(name: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def _quantize_chunk(
    source: torch.Tensor,
    pts_scale: torch.Tensor,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    payload_reconstructions = {
        name: core.quantize_hif4(source, config=config).values
        for name, config in PAYLOAD_VARIANTS.items()
    }
    standard = payload_reconstructions["s1p2_native"]
    micro_reconstructions = {
        name: standard if name == "H11_full" else core.quantize_hif4(source, config=config).values
        for name, config in MICRO_EXP_VARIANTS.items()
    }
    top_scale_reconstructions = {
        name: standard if name == "hardware" else core.quantize_hif4(source, config=config).values
        for name, config in TOP_SCALE_VARIANTS.items()
    }
    group_size_reconstructions = {
        name: standard if name == "g64" else core.quantize_hif4(source, config=config).values
        for name, config in GROUP_SIZE_VARIANTS.items()
    }
    bf16_carrier = core.round_bfloat16(source)
    bf16_hif4 = core.quantize_hif4(bf16_carrier).values

    scale = pts_scale.to(device=source.device, dtype=torch.float32).reshape(())
    normalized = source / scale
    pts_fp32 = core.quantize_hif4(normalized).values * scale
    normalized_bf16 = core.round_bfloat16(normalized)
    pts_bf16_projection = normalized_bf16 * scale
    pts_bf16 = core.quantize_hif4(normalized_bf16).values * scale
    return (
        payload_reconstructions,
        micro_reconstructions,
        top_scale_reconstructions,
        group_size_reconstructions,
        bf16_carrier,
        bf16_hif4,
        pts_fp32,
        pts_bf16_projection,
        pts_bf16,
    )


def run_real_packed_study(
    checkpoint_path: Path,
    *,
    layers: tuple[int, ...],
    device: torch.device,
    chunk_groups: int = 4096,
    max_tensors: int | None = None,
) -> dict[str, Any]:
    """评测真实 packed NVFP4 权重的代表层。"""
    checkpoint_path = Path(checkpoint_path)
    if not layers:
        raise ValueError("layers must not be empty")
    if chunk_groups <= 0:
        raise ValueError("chunk_groups must be positive")

    selected_layers = tuple(sorted(set(int(layer) for layer in layers)))
    layer_pattern = "|".join(str(layer) for layer in selected_layers)
    include_regex = rf"\.layers\.(?:{layer_pattern})\."

    global_bucket = _RealBucket()
    category_buckets: dict[str, _RealBucket] = {}
    layer_buckets: dict[int, _RealBucket] = {}
    tensor_results: dict[str, Any] = {}

    for name, decoded, pts_scale in core.iter_nvfp4_packed_weights(
        checkpoint_path,
        include_regex=include_regex,
        exclude_regex=r"linear_attn|lm_head",
        max_tensors=max_tensors,
    ):
        if decoded.ndim != 2 or decoded.shape[-1] % 64 != 0:
            raise ValueError(f"tensor is not compatible with HiF4 group=64: {name} {tuple(decoded.shape)}")

        category = core.classify_tensor_category(name)
        layer = _layer_from_name(name)
        if layer is None:
            raise ValueError(f"cannot infer layer index from tensor name: {name}")
        category_bucket = category_buckets.setdefault(category, _RealBucket())
        layer_bucket = layer_buckets.setdefault(layer, _RealBucket())
        tensor_bucket = _RealBucket()

        for chunk_cpu in core.iter_group_aligned_chunks(
            decoded,
            group_size=64,
            group_dim=-1,
            chunk_groups=chunk_groups,
        ):
            source = chunk_cpu.to(device=device, dtype=torch.float32)
            (
                payload_reconstructions,
                micro_reconstructions,
                top_scale_reconstructions,
                group_size_reconstructions,
                bf16_carrier,
                bf16_hif4,
                pts_fp32,
                pts_bf16_projection,
                pts_bf16,
            ) = _quantize_chunk(source, pts_scale)
            for bucket in (global_bucket, category_bucket, layer_bucket, tensor_bucket):
                bucket.add(
                    source,
                    payload_reconstructions,
                    micro_reconstructions,
                    top_scale_reconstructions,
                    group_size_reconstructions,
                    bf16_carrier,
                    bf16_hif4,
                    pts_fp32,
                    pts_bf16_projection,
                    pts_bf16,
                )

        tensor_results[name] = {
            "shape": list(decoded.shape),
            "numel": int(decoded.numel()),
            "layer": layer,
            "category": category,
            **tensor_bucket.finalize(),
        }

    if not tensor_results:
        raise ValueError("no packed NVFP4 tensors matched the requested layers")

    return {
        "checkpoint": str(checkpoint_path),
        "layers": list(selected_layers),
        "chunk_groups": chunk_groups,
        "device": str(device),
        "tensor_count": len(tensor_results),
        "global": global_bucket.finalize(),
        "categories": {
            category: bucket.finalize()
            for category, bucket in sorted(category_buckets.items())
        },
        "layer_results": {
            str(layer): bucket.finalize()
            for layer, bucket in sorted(layer_buckets.items())
        },
        "tensors": tensor_results,
    }
