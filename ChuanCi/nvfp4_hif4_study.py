#!/usr/bin/env python3
"""NVFP4 / HiF4 综合实验编排与离线 HTML 报告生成。"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import nvfp4_hif4_torch as core
from nvfp4_hif4_real_study import run_real_packed_study


@dataclass(frozen=True)
class StudyConfig:
    """综合合成实验设置。"""

    seed: int = 20260723
    samples_per_repeat: int = 320_000
    repeats: int = 10
    distributions: tuple[str, ...] = core.DISTRIBUTION_NAMES


class MetricAccumulator:
    """以 FP64 原始量合并多个 repeat，避免直接平均 NMSE。"""

    def __init__(self) -> None:
        self.sums = core.ErrorSums()
        self.repeat_nmse: list[float] = []

    def add(self, reference: torch.Tensor, approximation: torch.Tensor) -> None:
        sums = core.compute_error_sums(reference, approximation)
        self.sums = core.merge_error_sums(self.sums, sums)
        metrics = core.finalize_error_metrics(sums)
        nmse = metrics["nmse"]
        if not isinstance(nmse, (int, float)):
            raise ValueError("study requires finite non-zero references")
        self.repeat_nmse.append(float(nmse))

    def finalize(self) -> dict[str, Any]:
        metrics = core.finalize_error_metrics(self.sums)
        mean_nmse = sum(self.repeat_nmse) / len(self.repeat_nmse) if self.repeat_nmse else 0.0
        if len(self.repeat_nmse) > 1:
            variance = sum((value - mean_nmse) ** 2 for value in self.repeat_nmse) / (len(self.repeat_nmse) - 1)
            std_nmse = math.sqrt(variance)
        else:
            std_nmse = 0.0
        return {
            "metrics": metrics,
            "repeat_nmse": self.repeat_nmse,
            "repeat_count": len(self.repeat_nmse),
            "mean_nmse": mean_nmse,
            "std_nmse": std_nmse,
        }


class EnergyDecompositionAccumulator:
    """累计 e_a + e_b 的平方误差分解。"""

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
        ref64 = reference.detach().to(torch.float64)
        first64 = first_error.detach().to(torch.float64)
        second64 = second_error.detach().to(torch.float64)
        total64 = first64 + second64
        self.reference_energy += float((ref64 * ref64).sum().item())
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


def _hif4_config_summary(config: core.HiF4Config) -> dict[str, Any]:
    return {
        "group_size": config.group_size,
        "group_dim": config.group_dim,
        "scale_mode": config.scale_mode,
        "payload_format": config.payload_format,
        "hierarchy_format": config.hierarchy_format,
        "enable_exp8": config.enable_exp8,
        "enable_exp4": config.enable_exp4,
    }


def _group_size_result(variants: dict[str, Any]) -> dict[str, Any]:
    enriched: dict[str, Any] = {}
    for name, bundle in variants.items():
        config = GROUP_SIZE_VARIANTS[name]
        enriched[name] = {
            **bundle,
            "group_size": config.group_size,
            "full_three_level_hierarchy": bool(config.enable_exp8 and config.enable_exp4),
            "config": _hif4_config_summary(config),
        }

    nmse16 = _numeric_nmse(enriched["g16"])
    nmse32 = _numeric_nmse(enriched["g32"])
    nmse64 = _numeric_nmse(enriched["g64"])
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


def _new_accumulators(names: list[str]) -> dict[str, MetricAccumulator]:
    return {name: MetricAccumulator() for name in names}


def _finalize_accumulators(accumulators: dict[str, MetricAccumulator]) -> dict[str, Any]:
    return {name: accumulator.finalize() for name, accumulator in accumulators.items()}


def _numeric_nmse(bundle: dict[str, Any]) -> float:
    value = bundle["metrics"]["nmse"]
    if not isinstance(value, (int, float)):
        raise ValueError("expected numeric NMSE")
    return float(value)


def _payload_source_result(
    variants: dict[str, Any],
    decomposition: EnergyDecompositionAccumulator,
) -> dict[str, Any]:
    s1p2_nmse = _numeric_nmse(variants["s1p2_native"])
    e2m1_nmse = _numeric_nmse(variants["e2m1_native"])
    fixed_nmse = _numeric_nmse(variants["e2m1_fixed"])
    matched_nmse = _numeric_nmse(variants["bf16_range_matched"])
    unclipped_nmse = _numeric_nmse(variants["bf16_unclipped"])
    return {
        "variants": variants,
        "comparisons": {
            "e2m1_minus_s1p2_nmse": e2m1_nmse - s1p2_nmse,
            "e2m1_relative_change": None if s1p2_nmse == 0 else (e2m1_nmse - s1p2_nmse) / s1p2_nmse,
            "e2m1_fixed_minus_native_nmse": fixed_nmse - e2m1_nmse,
            "s1p2_recoverable_to_bf16_range_matched": None
            if s1p2_nmse == 0
            else (s1p2_nmse - matched_nmse) / s1p2_nmse,
            "s1p2_recoverable_to_bf16_unclipped": None
            if s1p2_nmse == 0
            else (s1p2_nmse - unclipped_nmse) / s1p2_nmse,
        },
        "s1p2_vs_bf16_decomposition": decomposition.finalize(
            "bf16_range_matched_floor_energy",
            "s1p2_increment_energy",
        ),
    }


def _run_one_distribution(
    config: StudyConfig,
    distribution: str,
    device: torch.device,
) -> dict[str, Any]:
    same_source = _new_accumulators(["nvfp4", "hif4_s1p2"])
    native_conversion = _new_accumulators(
        [
            "fp32_carrier",
            "bf16_carrier",
            "bf16_projection",
            "pts_fp32",
            "pts_bf16",
            "pts_bf16_projection",
        ]
    )
    payload_acc = {
        "bf16_source": _new_accumulators(list(PAYLOAD_VARIANTS)),
        "nvfp4_source": _new_accumulators(list(PAYLOAD_VARIANTS)),
    }
    micro_acc = {
        "bf16_source": _new_accumulators(list(MICRO_EXP_VARIANTS)),
        "nvfp4_source": _new_accumulators(list(MICRO_EXP_VARIANTS)),
    }
    scale_acc = {
        "bf16_source": _new_accumulators(list(TOP_SCALE_VARIANTS)),
        "nvfp4_source": _new_accumulators(list(TOP_SCALE_VARIANTS)),
    }
    group_acc = {
        "bf16_source": _new_accumulators(list(GROUP_SIZE_VARIANTS)),
        "nvfp4_source": _new_accumulators(list(GROUP_SIZE_VARIANTS)),
    }
    carrier_decomposition = EnergyDecompositionAccumulator()
    pts_carrier_decomposition = EnergyDecompositionAccumulator()
    payload_decomposition = {
        "bf16_source": EnergyDecompositionAccumulator(),
        "nvfp4_source": EnergyDecompositionAccumulator(),
    }

    for repeat_index in range(config.repeats):
        seed = config.seed + repeat_index
        base = core.make_distribution(distribution, config.samples_per_repeat, seed).to(device)
        bf16_source = core.round_bfloat16(base)
        nv_simulation = core.simulate_nvfp4(base)
        nv_source = nv_simulation.values
        pts_scale = nv_simulation.global_scale.to(device=device, dtype=torch.float32).reshape(())

        # 同一 BF16 输入分别量化到 NVFP4 与标准 HiF4，比较格式本身。
        nv_from_bf16 = core.simulate_nvfp4(bf16_source).values
        hif4_from_bf16 = core.quantize_hif4(bf16_source).values
        same_source["nvfp4"].add(bf16_source, nv_from_bf16)
        same_source["hif4_s1p2"].add(bf16_source, hif4_from_bf16)

        # 原生 NVFP4 通过直接载体或保留全局 scale 的 PTS 路径转为 HiF4。
        fp32_hif4 = core.quantize_hif4(nv_source).values
        bf16_carrier = core.round_bfloat16(nv_source)
        bf16_hif4 = core.quantize_hif4(bf16_carrier).values
        normalized = nv_source / pts_scale
        pts_fp32 = core.quantize_hif4(normalized).values * pts_scale
        normalized_bf16 = core.round_bfloat16(normalized)
        pts_bf16_projection = normalized_bf16 * pts_scale
        pts_bf16 = core.quantize_hif4(normalized_bf16).values * pts_scale
        native_conversion["fp32_carrier"].add(nv_source, fp32_hif4)
        native_conversion["bf16_carrier"].add(nv_source, bf16_hif4)
        native_conversion["bf16_projection"].add(nv_source, bf16_carrier)
        native_conversion["pts_fp32"].add(nv_source, pts_fp32)
        native_conversion["pts_bf16"].add(nv_source, pts_bf16)
        native_conversion["pts_bf16_projection"].add(nv_source, pts_bf16_projection)
        carrier_decomposition.add(
            nv_source,
            bf16_carrier - nv_source,
            bf16_hif4 - bf16_carrier,
        )
        pts_carrier_decomposition.add(
            nv_source,
            pts_bf16_projection - nv_source,
            pts_bf16 - pts_bf16_projection,
        )

        for source_name, source in (("bf16_source", bf16_source), ("nvfp4_source", nv_source)):
            reconstructions: dict[str, torch.Tensor] = {}
            for variant_name, variant_config in PAYLOAD_VARIANTS.items():
                reconstruction = core.quantize_hif4(source, config=variant_config).values
                reconstructions[variant_name] = reconstruction
                payload_acc[source_name][variant_name].add(source, reconstruction)

            payload_decomposition[source_name].add(
                source,
                reconstructions["bf16_range_matched"] - source,
                reconstructions["s1p2_native"] - reconstructions["bf16_range_matched"],
            )

            for variant_name, variant_config in MICRO_EXP_VARIANTS.items():
                reconstruction = core.quantize_hif4(source, config=variant_config).values
                micro_acc[source_name][variant_name].add(source, reconstruction)

            for variant_name, variant_config in TOP_SCALE_VARIANTS.items():
                reconstruction = core.quantize_hif4(source, config=variant_config).values
                scale_acc[source_name][variant_name].add(source, reconstruction)

            for variant_name, variant_config in GROUP_SIZE_VARIANTS.items():
                reconstruction = core.quantize_hif4(source, config=variant_config).values
                group_acc[source_name][variant_name].add(source, reconstruction)

    same_source_final = _finalize_accumulators(same_source)
    same_nv = _numeric_nmse(same_source_final["nvfp4"])
    same_h = _numeric_nmse(same_source_final["hif4_s1p2"])

    native_final = _finalize_accumulators(native_conversion)
    fp32_nmse = _numeric_nmse(native_final["fp32_carrier"])
    bf16_nmse = _numeric_nmse(native_final["bf16_carrier"])
    pts_fp32_nmse = _numeric_nmse(native_final["pts_fp32"])
    pts_bf16_nmse = _numeric_nmse(native_final["pts_bf16"])

    payload_final: dict[str, Any] = {}
    for source_name in payload_acc:
        variants = _finalize_accumulators(payload_acc[source_name])
        payload_final[source_name] = _payload_source_result(
            variants,
            payload_decomposition[source_name],
        )

    return {
        "same_source_format": {
            "reference": "同一份BF16输入",
            "variants": same_source_final,
            "hif4_minus_nvfp4_nmse": same_h - same_nv,
            "hif4_to_nvfp4_nmse_ratio": None if same_nv == 0 else same_h / same_nv,
        },
        "native_conversion": {
            "reference": "NVFP4的FP32数学解码值",
            "variants": native_final,
            "bf16_minus_fp32_carrier_nmse": bf16_nmse - fp32_nmse,
            "bf16_to_fp32_carrier_nmse_ratio": None if fp32_nmse == 0 else bf16_nmse / fp32_nmse,
            "pts_fp32_minus_direct_nmse": pts_fp32_nmse - fp32_nmse,
            "pts_bf16_minus_direct_nmse": pts_bf16_nmse - fp32_nmse,
            "pts_bf16_minus_pts_fp32_nmse": pts_bf16_nmse - pts_fp32_nmse,
            "bf16_carrier_decomposition": carrier_decomposition.finalize(
                "bf16_projection_energy",
                "hif4_after_bf16_energy",
            ),
            "pts_bf16_carrier_decomposition": pts_carrier_decomposition.finalize(
                "pts_bf16_projection_energy",
                "hif4_after_pts_bf16_energy",
            ),
        },
        "payload_ablation": payload_final,
        "micro_exponent_ablation": {
            source_name: {"variants": _finalize_accumulators(values)}
            for source_name, values in micro_acc.items()
        },
        "top_scale_ablation": {
            source_name: {"variants": _finalize_accumulators(values)}
            for source_name, values in scale_acc.items()
        },
        "group_size_ablation": {
            source_name: _group_size_result(_finalize_accumulators(values))
            for source_name, values in group_acc.items()
        },
    }


def run_synthetic_study(config: StudyConfig, *, device: torch.device) -> dict[str, Any]:
    """运行完整合成实验。"""
    if config.samples_per_repeat <= 0 or config.samples_per_repeat % 64 != 0:
        raise ValueError("samples_per_repeat must be positive and divisible by 64")
    if config.repeats <= 0:
        raise ValueError("repeats must be positive")
    unknown = sorted(set(config.distributions) - set(core.DISTRIBUTION_NAMES))
    if unknown:
        raise ValueError(f"unknown distributions: {unknown}")

    distributions = {
        name: _run_one_distribution(config, name, device)
        for name in config.distributions
    }
    return {
        "schema_version": 2,
        "study": "nvfp4_hif4_comprehensive",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "requested_conda_env": "hif4",
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "device": str(device),
        },
        "config": asdict(config),
        "synthetic": {"distributions": distributions},
        "real_packed": None,
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return html.escape(value)
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return str(value)
        return f"{float(value):.{digits}g}"
    return html.escape(str(value))


def _nmse(bundle: dict[str, Any]) -> float:
    return float(bundle["metrics"]["nmse"])


def _metric_table(title: str, rows: list[tuple[str, float, str]]) -> str:
    max_value = max((value for _, value, _ in rows), default=1.0)
    max_value = max(max_value, 1e-30)
    body = []
    for label, value, explanation in rows:
        width = max(1.0, 100.0 * value / max_value)
        body.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td class='num'>{_fmt(value)}</td>"
            f"<td><div class='bar'><span style='width:{width:.2f}%'></span></div></td>"
            f"<td>{html.escape(explanation)}</td>"
            "</tr>"
        )
    return (
        f"<h4>{html.escape(title)}</h4>"
        "<table><thead><tr><th>方案</th><th>NMSE</th><th>相对大小</th><th>怎么理解</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _pct(value: float | None, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{100.0 * float(value):.{digits}f}%"


def _group_rows(group: dict[str, Any], explanation: str) -> list[tuple[str, float, str]]:
    return [
        (
            f"group={size}",
            _nmse(group["variants"][f"g{size}"]),
            explanation,
        )
        for size in (16, 32, 64)
    ]


def _render_mechanism_and_questions() -> str:
    return """
<section>
<h2>HiF4的误差从哪里产生</h2>
<p>HiF4是完整三级量化，而不是普通的每组一个scale。对第 <em>i</em> 个元素，可写成
<code>x̂ᵢ = sign(xᵢ) · S0(G) · 2^(e8ᵢ) · 2^(e4ᵢ) · qᵢ</code>。其中顶层S0由G个元素共享，e8每8个元素共享，e4每4个元素共享，qᵢ是S1P2 payload。</p>
<div class="grid">
<div class="card"><strong>顶层group共享误差</strong><br>64个元素共同决定S0；局部子组尺度差异过大时，两级1-bit指数未必完全补偿。</div>
<div class="card"><strong>局部指数离散误差</strong><br>e8和e4只能选择有限倍率，不能连续拟合每个子组的最优scale。</div>
<div class="card"><strong>payload码点误差</strong><br>归一化值最终必须落到有限S1P2码点，这是底层舍入误差。</div>
<div class="card"><strong>顶层scale表示误差</strong><br>理想连续S0还要经过BF16计算和E6M2存储。</div>
<div class="card"><strong>层级耦合误差</strong><br>S0变化会改变e8/e4判定和payload舍入区间，因此各项误差不能简单相加。</div>
</div>
<div class="note"><strong>分析原则：</strong>关闭组件后的误差增量用于证明该组件是否必要；提高某个组件精度后可恢复的误差，才用于判断后续算法应优先优化哪里。</div>
</section>
<section>
<h2>研究问题与实验假设</h2>
<ol>
<li>HiF4相对NVFP4是否在所有分布上都更准确，还是只在局部动态范围结构适合时占优？</li>
<li>标准group=64是否是主要瓶颈？若缩小到32和16，边际收益是否递减？</li>
<li>每8与每4元素指数是可替代关系，还是必须共同工作？</li>
<li>S1P2损失主要来自1.75范围限制，还是来自范围内部码点稀疏？</li>
<li>顶层BF16计算、E6M2存储和BF16中转载体是否值得成为主要优化方向？</li>
<li>保留NVFP4全局scale的PTS路径能否稳定降低HiF4重构误差？</li>
</ol>
</section>
"""


def _render_experiment_setup(result: dict[str, Any]) -> str:
    cfg = result["config"]
    env = result["environment"]
    real = result.get("real_packed")
    real_text = "未运行真实packed权重实验。"
    if real:
        total_numel = sum(int(item.get("numel", 0)) for item in real.get("tensors", {}).values())
        real_text = (
            f"真实实验读取packed NVFP4 checkpoint，选择层{html.escape(str(real['layers']))}，"
            f"共{real['tensor_count']}个线性权重张量、{total_numel:,}个元素；按完整64元素边界切chunk。"
        )
    return (
        "<section><h2>实验设置与评价指标</h2>"
        f"<p><strong>合成数据：</strong>{len(cfg['distributions'])}种分布；每种每次{int(cfg['samples_per_repeat']):,}个元素，"
        f"重复{int(cfg['repeats'])}次，seed={int(cfg['seed'])}。所有同表方案使用同一批输入。</p>"
        f"<p><strong>真实权重：</strong>{real_text}</p>"
        "<p><strong>group实验控制变量：</strong>group∈{16,32,64}时，payload始终为S1P2，层级始终为S1P2，"
        "每8元素指数和每4元素指数全部开启，只改变顶层S0共享范围。因此这不是单scale分组量化对比。</p>"
        "<p><strong>指标：</strong>NMSE = ||W−Ŵ||²/||W||²，越小越好；SQNR = 10log10(||W||²/||W−Ŵ||²)，越大越好。"
        "聚合时先累加FP64误差能量和参考能量，再计算总体NMSE，不直接平均各张量NMSE。</p>"
        f"<details><summary>运行环境</summary><pre>{html.escape(json.dumps(env, ensure_ascii=False, indent=2))}</pre></details>"
        "</section>"
    )


def _error_opportunities(source: dict[str, Any]) -> list[tuple[str, float, str]]:
    payload = source["payload_ablation"]
    group = source["group_size_ablation"]
    top = source["top_scale_ablation"]
    native = source["native_conversion"]
    s1p2 = _nmse(payload["variants"]["s1p2_native"])
    matched = _nmse(payload["variants"]["bf16_range_matched"])
    g64 = _nmse(group["variants"]["g64"])
    g16 = _nmse(group["variants"]["g16"])
    hardware = _nmse(top["variants"]["hardware"])
    continuous = _nmse(top["variants"]["continuous"])
    carrier = _nmse(native["variants"]["bf16_projection"])
    rows = [
        ("S1P2范围内码点离散", max(0.0, s1p2 - matched), "用同范围BF16替代S1P2后可恢复的误差"),
        ("顶层group共享范围", max(0.0, g64 - g16), "把S0共享范围从64缩到16后可恢复的误差"),
        ("顶层S0有限精度", max(0.0, hardware - continuous), "hardware S0相对连续S0的观测差值"),
        ("BF16中转载体", max(0.0, carrier), "只进行NVFP4解码值到BF16投影的误差"),
    ]
    return sorted(rows, key=lambda item: item[1], reverse=True)


def _render_integrated_analysis(result: dict[str, Any]) -> str:
    real = result.get("real_packed")
    if real:
        source = real["global"]
        scope = "真实packed权重"
    else:
        first = next(iter(result["synthetic"]["distributions"].values()))
        source = {
            "payload_ablation": first["payload_ablation"]["nvfp4_source"],
            "group_size_ablation": first["group_size_ablation"]["nvfp4_source"],
            "micro_exponent_ablation": first["micro_exponent_ablation"]["nvfp4_source"],
            "top_scale_ablation": first["top_scale_ablation"]["nvfp4_source"],
            "native_conversion": first["native_conversion"],
        }
        scope = "首个合成NVFP4分布"
    ranking = _error_opportunities(source)
    ranking_rows = "".join(
        "<tr>"
        f"<td>{index}</td><td>{html.escape(name)}</td><td class='num'>{_fmt(value)}</td>"
        f"<td>{html.escape(explanation)}</td></tr>"
        for index, (name, value, explanation) in enumerate(ranking, start=1)
    )
    micro = source["micro_exponent_ablation"]["variants"]
    h00 = _nmse(micro["H00_no_exp"])
    h11 = _nmse(micro["H11_full"])
    return (
        "<section><h2>误差来源综合排序</h2>"
        f"<p>下表基于{html.escape(scope)}，排序的是在当前对照实验中观察到的“可恢复误差机会”，不是严格正交、可加和的方差分解。"
        "由于S0、e8、e4和payload存在离散耦合，各数值之间可能重叠。</p>"
        "<table><thead><tr><th>排序</th><th>候选误差来源</th><th>可恢复NMSE</th><th>定义</th></tr></thead>"
        f"<tbody>{ranking_rows}</tbody></table>"
        f"<p><strong>结构必要性：</strong>关闭两级局部指数后，NMSE由{_fmt(h11)}升至{_fmt(h00)}。"
        "这说明三级结构是HiF4获得动态范围适应性的核心；但该差值不代表当前完整HiF4中仍残留同等大小的可优化误差。</p>"
        "</section>"
    )


def _render_algorithm_guidance(result: dict[str, Any]) -> str:
    real = result.get("real_packed")
    if real:
        source = real["global"]
    else:
        first = next(iter(result["synthetic"]["distributions"].values()))
        source = {
            "payload_ablation": first["payload_ablation"]["nvfp4_source"],
            "group_size_ablation": first["group_size_ablation"]["nvfp4_source"],
            "top_scale_ablation": first["top_scale_ablation"]["nvfp4_source"],
            "native_conversion": first["native_conversion"],
        }
    payload_recovery = source["payload_ablation"]["comparisons"]["s1p2_recoverable_to_bf16_range_matched"]
    group_recovery = source["group_size_ablation"]["comparisons"]["recoverable_fraction_64_to_16"]
    direct = _nmse(source["native_conversion"]["variants"]["fp32_carrier"])
    pts = _nmse(source["native_conversion"]["variants"]["pts_fp32"])
    pts_change = None if direct == 0 else (pts - direct) / direct
    return (
        "<section><h2>对后续HiF4量化算法的指导</h2>"
        "<div class='priority'><strong>优先级1：直接优化S1P2码点匹配。</strong>"
        f"同范围BF16对照可恢复{_pct(payload_recovery)}的标准S1P2误差。算法目标不应只压低amax，"
        "还应使归一化权重靠近S1P2可表示码点；可结合码点感知scale搜索、GPTQ式补偿和局部误差加权。</div>"
        "<div class='priority'><strong>优先级2：设计HiF4-aware局部平滑，而不是直接套SmoothQuant。</strong>"
        f"将顶层group从64缩到16的实验可恢复{_pct(group_recovery)}的g64误差。优化应同时约束每4元素内部、"
        "相邻4元素子组以及不同8元素子组的尺度关系，使其落入e4/e8可覆盖的二进制倍率范围。</div>"
        "<div class='priority'><strong>优先级3：以最终三级量化误差作为搜索目标。</strong>"
        "S0变化会触发e8、e4和payload舍入边界跳变，因此仅优化最大值、方差或连续scale代理目标可能与最终误差不一致。</div>"
        "<div class='priority'><strong>优先级4：谨慎使用保留原NVFP4 scale的PTS路径。</strong>"
        f"本实验中PTS-FP32相对直接路径变化{_pct(pts_change)}。保留外层scale并不保留HiF4内部指数分配；"
        "任何归一化或平滑方案都应重新评估最终层级判定。</div>"
        "<div class='priority'><strong>较低优先级：单独提高BF16倒数或顶层scale计算精度。</strong>"
        "只有当顶层S0消融显示稳定、显著收益时才值得投入；否则其收益通常小于payload和局部结构整形。</div>"
        "</section>"
    )


def _render_validity_and_appendix(result: dict[str, Any]) -> str:
    return (
        "<section><h2>HiF4的优势、局限与适用条件</h2>"
        "<div class='grid'>"
        "<div class='card'><strong>优势</strong><br>两级局部指数以很小元数据成本适配局部动态范围，避免单个离群值直接决定整个64元素组的量化步长。</div>"
        "<div class='card'><strong>局限</strong><br>每4元素内部没有更细scale，S1P2码点有限，且三个层级的离散判定相互耦合。</div>"
        "<div class='card'><strong>更适合</strong><br>局部尺度差异能被2倍指数层级覆盖、每4元素内部相对紧凑的权重分布。</div>"
        "<div class='card'><strong>不擅长</strong><br>4元素内部跨度极大，或多个子组尺度差异超过e8/e4覆盖范围的分布。</div>"
        "</div></section>"
        "<section><h2>有效性威胁与结论边界</h2>"
        "<ul>"
        "<li>报告分析的是权重重构误差；NMSE下降不自动等价于PPL、任务准确率、推理能力或端到端速度提升。</li>"
        "<li>group=16和32是诊断性非标准格式，用于估计顶层共享范围的损失，不代表现有HiF4硬件可直接部署。</li>"
        "<li>消融组件之间存在耦合，综合排序是可恢复机会排序，不是严格独立误差分解。</li>"
        "<li>真实实验只覆盖所选层和线性权重；全模型结论仍需后续PPL与下游任务验证。</li>"
        "<li>合成分布用于控制变量和解释机理，不能替代真实模型权重。</li>"
        "</ul>"
        "<details><summary>复现配置JSON</summary><pre>"
        f"{html.escape(json.dumps(result.get('config', {}), ensure_ascii=False, indent=2))}"
        "</pre></details></section>"
    )


def _render_real_packed_section(real: dict[str, Any] | None) -> str:
    if not real:
        return ""

    global_result = real["global"]
    native = global_result["native_conversion"]
    payload = global_result["payload_ablation"]
    micro = global_result.get("micro_exponent_ablation")
    top_scale = global_result.get("top_scale_ablation")
    group_size = global_result.get("group_size_ablation")
    native_rows = [
        ("FP32载体转HiF4", _nmse(native["variants"]["fp32_carrier"]), "packed NVFP4直接按FP32数学值转换"),
        ("BF16载体转HiF4", _nmse(native["variants"]["bf16_carrier"]), "先投影到BF16，再转HiF4"),
        ("只投影到BF16", _nmse(native["variants"]["bf16_projection"]), "不做HiF4，只测中转载体损失"),
    ]
    if "pts_fp32" in native["variants"]:
        native_rows.extend(
            [
                ("PTS-FP32（保留全局FP32 scale）", _nmse(native["variants"]["pts_fp32"]), "先除去NVFP4全局scale，量化后再乘回"),
                ("PTS-BF16（保留全局FP32 scale）", _nmse(native["variants"]["pts_bf16"]), "归一化值先投影到BF16，再量化并乘回scale"),
                ("PTS路径只做BF16投影", _nmse(native["variants"]["pts_bf16_projection"]), "只测归一化中间值的BF16载体损失"),
            ]
        )
    payload_rows = [
        ("S1P2原生", _nmse(payload["variants"]["s1p2_native"]), "标准HiF4，S0按amax/7计算"),
        ("E2M1原生", _nmse(payload["variants"]["e2m1_native"]), "完整替换格式，S0按amax/24计算"),
        ("E2M1沿用S1P2尺子", _nmse(payload["variants"]["e2m1_fixed"]), "错误设置的诊断对照"),
        ("BF16同范围上限", _nmse(payload["variants"]["bf16_range_matched"]), "只去掉S1P2码点舍入"),
        ("BF16不裁剪上限", _nmse(payload["variants"]["bf16_unclipped"]), "payload不再限制为4-bit"),
    ]
    micro_rows = [] if not micro else [
        (name, _nmse(bundle), "H11为完整HiF4；其余方案关闭一个或两个局部指数层级")
        for name, bundle in micro["variants"].items()
    ]
    top_scale_rows = [] if not top_scale else [
        (name, _nmse(bundle), "从连续S0逐步加入BF16计算与E6M2存储约束")
        for name, bundle in top_scale["variants"].items()
    ]
    group_rows = [] if not group_size else _group_rows(
        group_size,
        "完整三级量化：只缩小顶层S0共享范围，e8、e4和S1P2全部保留",
    )

    category_rows = []
    for category, category_result in real.get("categories", {}).items():
        category_payload = category_result["payload_ablation"]["variants"]
        category_group = category_result.get("group_size_ablation", {}).get("variants", {})
        s1p2 = _nmse(category_payload["s1p2_native"])
        e2m1 = _nmse(category_payload["e2m1_native"])
        g64 = _nmse(category_group["g64"]) if category_group else s1p2
        g32 = _nmse(category_group["g32"]) if category_group else float("nan")
        g16 = _nmse(category_group["g16"]) if category_group else float("nan")
        recovery = None if not math.isfinite(g16) or g64 == 0 else (g64 - g16) / g64
        category_rows.append(
            "<tr>"
            f"<td>{html.escape(category)}</td>"
            f"<td class='num'>{_fmt(s1p2)}</td>"
            f"<td class='num'>{_fmt(e2m1)}</td>"
            f"<td class='num'>{_fmt(g64)}</td>"
            f"<td class='num'>{_fmt(g32)}</td>"
            f"<td class='num'>{_fmt(g16)}</td>"
            f"<td class='num'>{_pct(recovery)}</td>"
            "</tr>"
        )

    micro_table = _metric_table("真实权重：两级指数位消融", micro_rows) if micro_rows else ""
    top_scale_table = _metric_table("真实权重：顶层S0消融", top_scale_rows) if top_scale_rows else ""
    group_table = _metric_table("真实权重：顶层group共享范围", group_rows) if group_rows else ""

    layer_rows = []
    for layer, layer_result in real.get("layer_results", {}).items():
        variants = layer_result["group_size_ablation"]["variants"]
        g64 = _nmse(variants["g64"])
        g32 = _nmse(variants["g32"])
        g16 = _nmse(variants["g16"])
        layer_rows.append(
            "<tr>"
            f"<td>{html.escape(str(layer))}</td>"
            f"<td class='num'>{_fmt(g64)}</td>"
            f"<td class='num'>{_fmt(g32)}</td>"
            f"<td class='num'>{_fmt(g16)}</td>"
            f"<td class='num'>{_pct(None if g64 == 0 else (g64 - g16) / g64)}</td>"
            "</tr>"
        )

    tensor_rows = []
    for name, tensor_result in real.get("tensors", {}).items():
        payload_variants = tensor_result["payload_ablation"]["variants"]
        group_variants = tensor_result.get("group_size_ablation", {}).get("variants", {})
        s1p2 = _nmse(payload_variants["s1p2_native"])
        e2m1 = _nmse(payload_variants["e2m1_native"])
        g64 = _nmse(group_variants["g64"]) if group_variants else s1p2
        g32 = _nmse(group_variants["g32"]) if group_variants else float("nan")
        g16 = _nmse(group_variants["g16"]) if group_variants else float("nan")
        tensor_rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{html.escape(str(tensor_result.get('category')))}</td>"
            f"<td>{html.escape(str(tensor_result.get('layer')))}</td>"
            f"<td class='num'>{_fmt(s1p2)}</td>"
            f"<td class='num'>{_fmt(e2m1)}</td>"
            f"<td class='num'>{_fmt(g32)}</td>"
            f"<td class='num'>{_fmt(g16)}</td>"
            f"<td class='num'>{_pct(None if g64 == 0 or not math.isfinite(g16) else (g64 - g16) / g64)}</td>"
            "</tr>"
        )

    group_analysis = ""
    if group_size:
        comparisons = group_size["comparisons"]
        group_analysis = (
            "<div class='analysis'><h4>结果分析：标准group=64是否是主要瓶颈？</h4>"
            f"<p>从64缩到32的NMSE绝对降低为{_fmt(comparisons['nmse_drop_64_to_32'])}，"
            f"相对降低{_pct(comparisons['relative_drop_64_to_32'])}；从32缩到16继续降低"
            f"{_fmt(comparisons['nmse_drop_32_to_16'])}，相对降低{_pct(comparisons['relative_drop_32_to_16'])}。"
            f"最终group=16相对group=64可恢复{_pct(comparisons['recoverable_fraction_64_to_16'])}的误差。"
            f"边际收益{'递减' if comparisons['diminishing_returns'] else '未递减'}。</p>"
            "<p><strong>机理解释：</strong>缩小group只改变S0的统计范围。如果改善明显，说明64元素内部不同8元素子组的尺度差异"
            "超过了e8/e4的有限倍率覆盖；如果改善很小，则主要瓶颈更可能位于每4元素内部或S1P2码点。</p></div>"
        )

    return (
        "<section><h2>真实packed NVFP4权重</h2>"
        f"<p>本节直接读取真实packed NVFP4 checkpoint，共评测 {real['tensor_count']} 个张量，层号为 {html.escape(str(real['layers']))}。"
        "输入从一开始就是NVFP4，不构造不存在的BF16→NVFP4→HiF4链路。</p>"
        "<h3>转换路径：中转载体和保留全局scale</h3>"
        "<p><strong>研究问题：</strong>误差是否主要来自FP32/BF16中转载体，保留NVFP4全局scale能否减少转换损失？</p>"
        + _metric_table("真实权重：FP32、BF16与保留全局scale路径", native_rows)
        + "<p class='analysis'><strong>算法含义：</strong>应比较最终HiF4重构误差，而不能只比较中间载体误差。外层scale被保留后，"
        "HiF4内部S0、e8、e4仍会重新判定，因此PTS不是数学上必然更优的等价变换。</p>"
        "<h3>Payload：范围还是码点密度</h3>"
        "<p><strong>研究问题：</strong>S1P2的损失来自1.75上限，还是来自上限内部有限码点？</p>"
        + _metric_table("真实权重：S1P2、E2M1与BF16上限", payload_rows)
        + "<p class='analysis'><strong>机理解释：</strong>同范围BF16保持S0、两级指数和1.75上限，只移除S1P2舍入。"
        "它与S1P2的差距可近似视为范围内部码点离散的可恢复机会；不裁剪BF16则进一步移除范围约束，但不是可部署4-bit格式。</p>"
        "<h3>局部指数：三级结构是否必要</h3>"
        + micro_table
        + "<p class='analysis'><strong>结论解释：</strong>H00/H10/H01/H11用于判断结构必要性。两级指数的增益具有交互作用，"
        "不能把单独关闭一级产生的差值直接相加为完整HiF4的剩余误差。</p>"
        "<h3>顶层S0：计算和存储精度</h3>"
        + top_scale_table
        + "<p class='analysis'><strong>算法含义：</strong>continuous、BF16数学、E6M2存储和hardware逐步加入约束。"
        "若hardware与continuous差距远小于payload对照差距，单独提高顶层scale精度不应成为首要方向。</p>"
        "<h3>顶层group共享范围</h3>"
        + group_table
        + group_analysis
        + "<h4>按线性层类别汇总</h4><table><thead><tr><th>类别</th><th>S1P2</th><th>E2M1</th><th>g64</th><th>g32</th><th>g16</th><th>64→16恢复比例</th></tr></thead>"
        f"<tbody>{''.join(category_rows)}</tbody></table>"
        + "<h4>按层号汇总group实验</h4><table><thead><tr><th>层</th><th>g64</th><th>g32</th><th>g16</th><th>64→16恢复比例</th></tr></thead>"
        f"<tbody>{''.join(layer_rows)}</tbody></table>"
        + "<details><summary>展开逐张量结果</summary><table><thead><tr><th>张量</th><th>类别</th><th>层</th><th>S1P2</th><th>E2M1</th><th>g32</th><th>g16</th><th>64→16恢复比例</th></tr></thead>"
        f"<tbody>{''.join(tensor_rows)}</tbody></table></details></section>"
    )


def _render_executive_summary(result: dict[str, Any]) -> str:
    distributions = result["synthetic"]["distributions"]
    distribution_count = len(distributions)
    hif4_wins = sum(
        data["same_source_format"]["hif4_minus_nvfp4_nmse"] < 0
        for data in distributions.values()
    )
    e2m1_bf16_worse = sum(
        data["payload_ablation"]["bf16_source"]["comparisons"]["e2m1_minus_s1p2_nmse"] > 0
        for data in distributions.values()
    )
    e2m1_nv_worse = sum(
        data["payload_ablation"]["nvfp4_source"]["comparisons"]["e2m1_minus_s1p2_nmse"] > 0
        for data in distributions.values()
    )

    real = result.get("real_packed")
    if real:
        real_payload = real["global"]["payload_ablation"]
        real_native = real["global"]["native_conversion"]
        e2m1_relative = real_payload["comparisons"]["e2m1_relative_change"]
        matched_recovery = real_payload["comparisons"]["s1p2_recoverable_to_bf16_range_matched"]
        carrier_ratio = real_native["bf16_to_fp32_carrier_nmse_ratio"]
        direct_nmse = _nmse(real_native["variants"]["fp32_carrier"])
        pts_fp32_nmse = _nmse(real_native["variants"]["pts_fp32"])
        pts_bf16_nmse = _nmse(real_native["variants"]["pts_bf16"])
        tensor_count = real["tensor_count"]
        real_text = (
            f"真实packed NVFP4的{tensor_count}个张量中，E2M1原生格式的总体误差比S1P2高"
            f"{100.0 * float(e2m1_relative):.1f}%。这说明重新计算S0虽然必要，但没有让E2M1胜过S1P2。"
        )
        recovery_text = (
            f"把S1P2 payload换成同范围BF16后，可恢复约{100.0 * float(matched_recovery):.1f}%的S1P2误差。"
            "主要损失来自4-bit码点稀疏，而不是1.75范围本身。"
        )
        carrier_text = (
            f"BF16中转载体的最终误差是FP32载体的{float(carrier_ratio):.4f}倍，"
            f"只增加约{100.0 * (float(carrier_ratio) - 1.0):.2f}%。"
        )
        pts_fp32_change = 0.0 if direct_nmse == 0 else (pts_fp32_nmse - direct_nmse) / direct_nmse
        pts_bf16_change = 0.0 if direct_nmse == 0 else (pts_bf16_nmse - direct_nmse) / direct_nmse
        pts_text = (
            f"保留NVFP4全局FP32 scale后，PTS-FP32相对直接路径变化{100.0 * pts_fp32_change:+.2f}%，"
            f"PTS-BF16变化{100.0 * pts_bf16_change:+.2f}%。负数代表误差降低。"
        )
        group_comparisons = real["global"]["group_size_ablation"]["comparisons"]
        group_text = (
            f"完整三级量化下，group从64缩到32可降低{_pct(group_comparisons['relative_drop_64_to_32'])}，"
            f"继续缩到16可再降低{_pct(group_comparisons['relative_drop_32_to_16'])}；"
            f"64→16共恢复{_pct(group_comparisons['recoverable_fraction_64_to_16'])}的g64误差。"
        )
    else:
        real_text = "真实packed权重尚未加入，本页结论暂时只来自合成数据。"
        recovery_values = [
            data["payload_ablation"]["nvfp4_source"]["comparisons"]["s1p2_recoverable_to_bf16_range_matched"]
            for data in distributions.values()
        ]
        matched_recovery = sum(float(value) for value in recovery_values) / len(recovery_values)
        recovery_text = f"在合成NVFP4输入上，同范围BF16平均可恢复约{100.0 * matched_recovery:.1f}%的S1P2误差。"
        carrier_text = "FP32与BF16中转载体的差距见各分布明细表。"
        pts_win_count = sum(
            _nmse(data["native_conversion"]["variants"]["pts_fp32"])
            < _nmse(data["native_conversion"]["variants"]["fp32_carrier"])
            for data in distributions.values()
        )
        pts_text = f"在{distribution_count}种合成分布中，PTS-FP32有{pts_win_count}种优于直接FP32路径。"
        group_recoveries = [
            data["group_size_ablation"]["nvfp4_source"]["comparisons"]["recoverable_fraction_64_to_16"]
            for data in distributions.values()
        ]
        finite_group_recoveries = [float(value) for value in group_recoveries if value is not None]
        mean_group_recovery = (
            sum(finite_group_recoveries) / len(finite_group_recoveries)
            if finite_group_recoveries
            else None
        )
        group_text = f"合成NVFP4输入中，group 64→16平均恢复{_pct(mean_group_recovery)}的g64误差。"

    return (
        "<section><h2>结论先行</h2>"
        "<p>先给出最容易使用的结论，后面的表格负责说明这些结论是怎样得到的。</p>"
        "<div class='grid'>"
        f"<div class='card'><strong>格式公平比较</strong><br>在{distribution_count}种合成分布中，HiF4在{hif4_wins}种分布上比NVFP4误差更小。它并非对所有分布都占优。</div>"
        f"<div class='card'><strong>S1P2还是E2M1</strong><br>BF16输入的{e2m1_bf16_worse}/{distribution_count}种分布、NVFP4输入的{e2m1_nv_worse}/{distribution_count}种分布中，E2M1误差更大。</div>"
        f"<div class='card'><strong>真实权重</strong><br>{html.escape(real_text)}</div>"
        f"<div class='card'><strong>BF16精度上限</strong><br>{html.escape(recovery_text)}</div>"
        f"<div class='card'><strong>顶层group共享</strong><br>{html.escape(group_text)}</div>"
        f"<div class='card'><strong>中转载体</strong><br>{html.escape(carrier_text)}</div>"
        f"<div class='card'><strong>PTS路径结论</strong><br>{html.escape(pts_text)}</div>"
        "</div></section>"
    )


def render_html_report(result: dict[str, Any]) -> str:
    """将 schema v2 结果渲染为单文件离线 HTML。"""
    distributions = result["synthetic"]["distributions"]
    sections: list[str] = []

    for distribution, data in distributions.items():
        same = data["same_source_format"]
        payload_bf = data["payload_ablation"]["bf16_source"]
        payload_nv = data["payload_ablation"]["nvfp4_source"]
        group_bf = data["group_size_ablation"]["bf16_source"]
        group_nv = data["group_size_ablation"]["nvfp4_source"]
        native = data["native_conversion"]

        same_rows = [
            ("NVFP4", _nmse(same["variants"]["nvfp4"]), "从同一份BF16权重直接量化"),
            ("HiF4（S1P2）", _nmse(same["variants"]["hif4_s1p2"]), "从同一份BF16权重直接量化"),
        ]
        payload_rows_bf = [
            ("S1P2原生", _nmse(payload_bf["variants"]["s1p2_native"]), "S0按最大值除以7计算"),
            ("E2M1原生", _nmse(payload_bf["variants"]["e2m1_native"]), "S0按最大值除以24重新计算"),
            ("E2M1但沿用S1P2尺子", _nmse(payload_bf["variants"]["e2m1_fixed"]), "只用于说明错误设置的影响"),
            ("BF16同范围上限", _nmse(payload_bf["variants"]["bf16_range_matched"]), "仍限制在0到1.75，只去掉4-bit舍入"),
            ("BF16不裁剪上限", _nmse(payload_bf["variants"]["bf16_unclipped"]), "payload完全保留BF16，不再是4-bit方案"),
        ]
        payload_rows_nv = [
            ("S1P2原生", _nmse(payload_nv["variants"]["s1p2_native"]), "输入是原生NVFP4解码值"),
            ("E2M1原生", _nmse(payload_nv["variants"]["e2m1_native"]), "输入相同，只改变HiF4底层格式"),
            ("BF16同范围上限", _nmse(payload_nv["variants"]["bf16_range_matched"]), "观察S1P2码点舍入还能恢复多少"),
            ("BF16不裁剪上限", _nmse(payload_nv["variants"]["bf16_unclipped"]), "观察payload不压缩时的绝对上限"),
        ]
        native_rows = [
            ("NVFP4→FP32载体→HiF4", _nmse(native["variants"]["fp32_carrier"]), "不先压到BF16"),
            ("NVFP4→BF16载体→HiF4", _nmse(native["variants"]["bf16_carrier"]), "先把解码值存成BF16"),
            ("只做NVFP4→BF16", _nmse(native["variants"]["bf16_projection"]), "只看中转载体本身的损失"),
            ("PTS-FP32（保留全局scale）", _nmse(native["variants"]["pts_fp32"]), "除去NVFP4全局scale后量化，再乘回原scale"),
            ("PTS-BF16（保留全局scale）", _nmse(native["variants"]["pts_bf16"]), "归一化值先存为BF16，再量化并乘回原scale"),
            ("PTS路径只做BF16投影", _nmse(native["variants"]["pts_bf16_projection"]), "只测归一化中间值的BF16损失"),
        ]

        micro_rows = [
            (name, _nmse(bundle), "误差越小越好")
            for name, bundle in data["micro_exponent_ablation"]["bf16_source"]["variants"].items()
        ]
        scale_rows = [
            (name, _nmse(bundle), "误差越小越好")
            for name, bundle in data["top_scale_ablation"]["bf16_source"]["variants"].items()
        ]
        group_rows_bf = _group_rows(
            group_bf,
            "完整三级量化，仅改变顶层S0共享范围",
        )
        group_rows_nv = _group_rows(
            group_nv,
            "输入为同一批NVFP4解码值；e8、e4和S1P2均保留",
        )
        group_cmp = group_nv["comparisons"]
        group_analysis = (
            "<div class='analysis'><strong>group结果分析：</strong>"
            f"NVFP4输入下，64→32相对降低{_pct(group_cmp['relative_drop_64_to_32'])}，"
            f"32→16相对降低{_pct(group_cmp['relative_drop_32_to_16'])}，"
            f"64→16共恢复{_pct(group_cmp['recoverable_fraction_64_to_16'])}的g64误差；"
            f"边际收益{'递减' if group_cmp['diminishing_returns'] else '未递减'}。"
            "该结果用于诊断顶层共享范围，不代表group=16/32是标准可部署HiF4格式。</div>"
        )

        sections.append(
            f"<section><h2>合成分布分析：{html.escape(distribution)}</h2>"
            "<p>这一节把同一批数值反复用于所有方案，因此差距来自格式或计算路径，而不是不同样本。"
            "合成分布的作用是控制变量、验证机理；它不能代替真实权重结论。</p>"
            "<h3>问题1：同一BF16源下，HiF4是否优于NVFP4？</h3>"
            + _metric_table("同一BF16输入：NVFP4与HiF4公平比较", same_rows)
            + "<p class='analysis'><strong>怎么解释：</strong>这里没有格式转换链路，两种格式都从同一BF16样本直接量化。"
            "差异反映两种格式的码点和scale组织方式对该分布的匹配程度，因此不能从单一分布推出普适优劣。</p>"
            "<h3>问题2：payload损失来自范围还是码点？</h3>"
            + _metric_table("S1P2、E2M1与BF16：BF16输入", payload_rows_bf)
            + _metric_table("S1P2、E2M1与BF16：NVFP4输入", payload_rows_nv)
            + "<p class='analysis'><strong>对照逻辑：</strong>E2M1原生会同步重算S0和两级指数，避免只换码点的不公平比较；"
            "同范围BF16只移除S1P2离散舍入，不移除1.75上限；不裁剪BF16用于估计取消4-bit payload后的精度上限。</p>"
            "<h3>问题3：顶层group共享范围是否限制精度？</h3>"
            + _metric_table("完整三级量化group实验：BF16输入", group_rows_bf)
            + _metric_table("完整三级量化group实验：NVFP4输入", group_rows_nv)
            + group_analysis
            + "<h3>问题4：NVFP4转HiF4时，载体和PTS是否重要？</h3>"
            + _metric_table("NVFP4转成HiF4：两种中转载体", native_rows)
            + "<p class='analysis'><strong>怎么解释：</strong>BF16 projection单独衡量载体损失；FP32/BF16 carrier衡量载体加HiF4；"
            "PTS则改变输入到HiF4层级判定的相位。最终误差由三层离散决策共同决定。</p>"
            "<h3>问题5：两级指数是否缺一不可？</h3>"
            + _metric_table("两级指数位消融", micro_rows)
            + "<p class='analysis'><strong>怎么解释：</strong>H00、H10、H01和H11构成2×2因子实验。H11显著优于单层方案时，"
            "说明e8与e4不是简单替代，而是通过先粗后细的倍率选择协同覆盖局部动态范围。</p>"
            "<h3>问题6：顶层S0精度是不是主要瓶颈？</h3>"
            + _metric_table("顶层S0计算方式消融", scale_rows)
            + "<p class='analysis'><strong>算法含义：</strong>若连续S0与hardware接近，则更高精度的S0数学并不能解决payload和局部结构误差，"
            "后续工作应把算力投入到码点感知和HiF4-aware分布整形。</p>"
            + "</section>"
        )

    env = result["environment"]
    cfg = result["config"]
    real = result.get("real_packed")
    executive_summary = _render_executive_summary(result)
    mechanism_and_questions = _render_mechanism_and_questions()
    experiment_setup = _render_experiment_setup(result)
    real_section = _render_real_packed_section(real)
    integrated_analysis = _render_integrated_analysis(result)
    algorithm_guidance = _render_algorithm_guidance(result)
    validity_and_appendix = _render_validity_and_appendix(result)
    real_status = "真实packed权重已完成" if real else "真实权重结果尚未运行"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HiF4量化误差归因与算法指导报告</title>
<style>
:root{{--blue:#174ea6;--blue2:#e8f0fe;--orange:#d97706;--ink:#172033;--muted:#5f6b7a;--line:#dbe3ef;--green:#137333;}}
*{{box-sizing:border-box}} body{{margin:0;background:#f5f7fb;color:var(--ink);font:15px/1.7 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:34px 24px 80px}} header,section{{background:white;border:1px solid var(--line);border-radius:16px;padding:28px;margin-bottom:22px;box-shadow:0 8px 25px rgba(25,55,100,.06)}}
h1{{margin:0 0 10px;font-size:34px;color:var(--blue)}} h2{{margin-top:0;color:var(--blue)}} h3{{margin-bottom:8px}} h4{{margin:28px 0 10px}}
.lead{{font-size:17px;color:var(--muted)}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-top:18px}} .card{{background:var(--blue2);border-radius:12px;padding:16px}}
table{{width:100%;border-collapse:collapse;margin:10px 0 24px}} th,td{{border-bottom:1px solid var(--line);padding:10px 9px;text-align:left;vertical-align:middle}} th{{background:#f7f9fd}} .num{{font-variant-numeric:tabular-nums;font-family:ui-monospace,monospace}}
.bar{{height:10px;background:#edf1f7;border-radius:9px;overflow:hidden;min-width:100px}} .bar span{{display:block;height:100%;background:var(--blue)}} code{{background:#eef2f7;padding:2px 5px;border-radius:5px}}
.note{{border-left:4px solid var(--orange);padding:10px 14px;background:#fff8ed}} .analysis{{border-left:4px solid var(--blue);padding:12px 15px;background:#f7f9fd;margin:12px 0 24px}} .priority{{border:1px solid #c9daf8;border-radius:10px;padding:15px;margin:12px 0;background:#f8fbff}} .priority strong{{color:var(--blue)}} ol,ul{{padding-left:24px}} details{{margin-top:14px}} @media(max-width:720px){{table{{font-size:13px}} main{{padding:16px 10px}}}}
</style>
</head>
<body><main>
<header>
<h1>HiF4量化误差归因与算法指导报告</h1>
<p class="lead">从格式机制、控制变量实验和真实packed权重三个层面，分析HiF4的优势、局限、主要损失来源，并给出后续量化算法的研发优先级。</p>
<div class="grid">
<div class="card"><strong>环境</strong><br>要求使用 Conda <code>hif4</code><br>实际记录：{html.escape(str(env.get('conda_default_env')))}</div>
<div class="card"><strong>样本</strong><br>每次 {_fmt(cfg['samples_per_repeat'],0)} 个数<br>重复 {_fmt(cfg['repeats'],0)} 次</div>
<div class="card"><strong>比较原则</strong><br>同一输入、同一分组、同一误差公式</div>
<div class="card"><strong>报告状态</strong><br>合成实验已生成<br>{html.escape(real_status)}</div>
</div>
</header>
{executive_summary}
<section>
<h2>为什么要做这些实验</h2>
<p>HiF4的最终误差同时受顶层S0共享范围、每8元素指数、每4元素指数、S1P2 payload以及顶层scale表示精度影响。只报告一个最终NMSE无法回答“算法应该优化哪一层”。</p>
<p>本报告采用控制变量和精度上限对照，把问题拆成格式公平比较、顶层group共享、局部指数、payload码点、顶层scale、转换路径六条证据链，并在真实packed权重上检查结论是否成立。</p>
<div class="note"><strong>关键设置：</strong>E2M1实验会同步按amax/24重算S0和两级指数；group=16/32/64实验始终保留e8、e4和S1P2，只有顶层S0共享范围变化。</div>
</section>
{mechanism_and_questions}
{experiment_setup}
{''.join(sections)}
{real_section}
{integrated_analysis}
{algorithm_guidance}
{validity_and_appendix}
<section>
<h2>怎样读这些结果</h2>
<p>NMSE越小越好。对照实验的关键不是某个数字单独多大，而是它只移除了哪一种约束：同范围BF16移除payload舍入，group实验只缩小S0共享范围，continuous S0移除顶层scale有限精度。</p>
<p>这些对照不是严格正交分解，因为S0变化可能改变e8/e4和payload的舍入结果。报告因此使用“可恢复误差机会”和“结构必要性”两类措辞，而不把所有差值相加成100%的误差构成。</p>
</section>
</main></body></html>"""


def write_study_outputs(result: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "NVFP4_HiF4_comprehensive_results.json"
    html_path = output_dir / "NVFP4_HiF4_comprehensive_report.html"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_html_report(result), encoding="utf-8")
    return json_path, html_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--samples-per-repeat", type=int, default=320_000)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--distributions", default=",".join(core.DISTRIBUTION_NAMES))
    parser.add_argument("--packed-checkpoint", type=Path)
    parser.add_argument("--layers", default="3,31,63")
    parser.add_argument("--chunk-groups", type=int, default=4096)
    parser.add_argument("--max-real-tensors", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    samples = 6_400 if args.quick and args.samples_per_repeat == 320_000 else args.samples_per_repeat
    repeats = 1 if args.quick and args.repeats == 10 else args.repeats
    config = StudyConfig(
        seed=args.seed,
        samples_per_repeat=samples,
        repeats=repeats,
        distributions=tuple(item.strip() for item in args.distributions.split(",") if item.strip()),
    )
    device = torch.device(args.device)
    result = run_synthetic_study(config, device=device)
    if args.packed_checkpoint is not None:
        layers = tuple(int(item.strip()) for item in args.layers.split(",") if item.strip())
        result["real_packed"] = run_real_packed_study(
            args.packed_checkpoint,
            layers=layers,
            device=device,
            chunk_groups=args.chunk_groups,
            max_tensors=args.max_real_tensors,
        )
    json_path, html_path = write_study_outputs(result, args.output_dir)
    print(json_path)
    print(html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
