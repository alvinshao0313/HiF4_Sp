"""Router contribution analysis and causal intervention scaffolding."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.production_puncture import (
    ProductionPunctureOp,
    frozen_router,
)

from .math_utils import exact_kl, l2
from .moe_decomposition import moe_three_way_decompose, router_id_weight_split
from .run_state import atomic_write_json, write_jsonl


def router_basic_metrics(r0: torch.Tensor, r1: torch.Tensor, *, top_k: int = 8) -> dict:
    a = r0.detach().to(dtype=torch.float64).reshape(-1)
    b = r1.detach().to(dtype=torch.float64).reshape(-1)
    p = torch.softmax(a, dim=-1)
    q = torch.softmax(b, dim=-1)
    router_kl = float((p * (torch.log(p.clamp_min(1e-300)) - torch.log(q.clamp_min(1e-300)))).sum())
    top0 = set(torch.topk(a, k=top_k).indices.tolist())
    top1 = set(torch.topk(b, k=top_k).indices.tolist())
    overlap = len(top0 & top1) / float(top_k)
    w0 = torch.zeros_like(a)
    w1 = torch.zeros_like(b)
    # approximate routing weights via softmax over full logits for L1/L2; production uses top-k renorm.
    w0[list(top0)] = torch.softmax(a[list(top0)], dim=-1)
    w1[list(top1)] = torch.softmax(b[list(top1)], dim=-1)
    return {
        "router_kl": router_kl,
        "top_k_exact": top0 == top1,
        "top_k_overlap": overlap,
        "routing_weight_l1": float((w0 - w1).abs().sum()),
        "routing_weight_l2": float(torch.linalg.vector_norm(w0 - w1)),
    }


def write_router_report(output_dir: Path, rows: list[dict], intervention_rows: list[dict]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "router_rows.jsonl", rows)
    write_jsonl(output_dir / "router_intervention_rows.jsonl", intervention_rows)
    (output_dir / "ROUTER_CONTRIBUTION_REPORT.md").write_text(
        "# Router Contribution Report\n\n"
        f"- decomposition_rows: {len(rows)}\n"
        f"- intervention_rows: {len(intervention_rows)}\n"
        "- Distinguishes top-k ID, routing weight, local MoE output, and final-loss causal contribution.\n",
        encoding="utf-8",
    )


def conditional_enable_o3(router_causal_rows: list[dict]) -> bool:
    """Legacy global O3 switch — forbidden on formal full-48 path; kept for rejection tests."""
    if not router_causal_rows:
        return False
    vals = [float(r["C_router"]) for r in router_causal_rows]
    return (sum(vals) / len(vals)) > 0.0 and sum(1 for v in vals if v > 0) >= max(1, len(vals) // 2)


def decide_o3_layers(
    holdout_by_layer: dict[str, Any] | dict[int, Any],
    *,
    causal_source_by_layer: dict[int, str] | None = None,
) -> list[int]:
    """Per-layer O3 gate (plan §16.2). No global enable_o3.

    For each layer:
      1. sample-level mean C_router over 4 states → n sample values (formal n=8)
      2. enter iff causal source is moe|both AND mean>0 AND median>0 AND positive_count>=5/8
    """
    o3: list[int] = []
    for raw_layer, payload in holdout_by_layer.items():
        layer = int(raw_layer)
        src = None
        if causal_source_by_layer is not None:
            src = causal_source_by_layer.get(layer)
        if src is None:
            src = payload.get("causal_source")
        if src not in {"moe", "both"}:
            continue
        sample_vals = payload.get("sample_level_C_router")
        if sample_vals is None:
            # allow pre-aggregated fields
            mean_c = payload.get("C_router")
            med_c = payload.get("C_router_median")
            pos = int(payload.get("C_router_positive_count") or 0)
            n = int(payload.get("C_router_n") or 0)
            if mean_c is None or med_c is None or n <= 0:
                continue
            need = 5 if n >= 8 else max(1, n // 2 + 1)
            if float(mean_c) > 0.0 and float(med_c) > 0.0 and pos >= need:
                o3.append(layer)
            continue
        vals = [float(v) for v in sample_vals]
        if not vals:
            continue
        n = len(vals)
        mean_c = sum(vals) / n
        med_c = float(sorted(vals)[n // 2])
        pos = sum(1 for v in vals if v > 0.0)
        need = 5 if n >= 8 else max(1, n // 2 + 1)
        if mean_c > 0.0 and med_c > 0.0 and pos >= need:
            o3.append(layer)
    return sorted(o3)
