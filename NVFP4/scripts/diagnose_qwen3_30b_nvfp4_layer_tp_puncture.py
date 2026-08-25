#!/usr/bin/env python3
"""Layer-level TP1/TP2 puncture for Qwen3-30B-A3B NVFP4 emulation semantics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
VLLM_ROOT = REPO_ROOT / "3rdparty" / "vllm"
if str(VLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VLLM_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import (  # noqa: E402
    load_qwen3_moe_layer_state,
    release_qwen3_moe_layer_state,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_model_spec import (  # noqa: E402
    QWEN3_30B_A3B_NVFP4,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import (  # noqa: E402
    native_nvfp4_linear,
    qdq_native_nvfp4,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot  # noqa: E402


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    diff = af - bf
    den = torch.linalg.vector_norm(af).clamp_min(1e-30)
    mse = diff.square().mean()
    power = af.square().mean().clamp_min(1e-30)
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rel_l2": float((torch.linalg.vector_norm(diff) / den).item()),
        "nmse": float((mse / power).item()),
    }


def _linear_full(x: torch.Tensor, weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    return native_nvfp4_linear(x, weight, scale_inv)


def _linear_col_tp2(x: torch.Tensor, weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    xq = qdq_native_nvfp4(x, scale_inv)
    parts = []
    for w in weight.chunk(2, dim=0):
        parts.append(F.linear(xq, w.to(device=x.device, dtype=xq.dtype)))
    return torch.cat(parts, dim=-1)


def _linear_row_tp2(x: torch.Tensor, weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    xq = qdq_native_nvfp4(x, scale_inv)
    xs = xq.chunk(2, dim=-1)
    ws = weight.chunk(2, dim=1)
    parts = [
        F.linear(xi, wi.to(device=x.device, dtype=xq.dtype))
        for xi, wi in zip(xs, ws, strict=True)
    ]
    return parts[0] + parts[1]


def _router(hidden: torch.Tensor, router_weight: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = F.linear(hidden.to(router_weight.dtype), router_weight)
    probs = torch.softmax(logits, dtype=torch.float32, dim=-1)
    weights, ids = torch.topk(probs, top_k, dim=-1)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).to(logits.dtype)
    return logits, weights, ids


def _w13_full(x: torch.Tensor, expert, gate_scale: torch.Tensor, up_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate = _linear_full(x, expert.gate_proj, gate_scale)
    up = _linear_full(x, expert.up_proj, up_scale)
    return gate, up, F.silu(gate) * up


def _w13_col_tp2(x: torch.Tensor, expert, gate_scale: torch.Tensor, up_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate = _linear_col_tp2(x, expert.gate_proj, gate_scale)
    up = _linear_col_tp2(x, expert.up_proj, up_scale)
    return gate, up, F.silu(gate) * up


def _moe_full(x: torch.Tensor, state) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits, routing_weights, selected = _router(x, state.router_weight, state.spec.top_k)
    output = torch.zeros_like(x)
    first: dict[str, torch.Tensor] = {}
    for expert_idx in selected.unique(sorted=True).tolist():
        expert = state.experts[int(expert_idx)]
        token_idx, topk_pos = torch.where(selected == expert_idx)
        current = x[token_idx]
        gate, up, swiglu = _w13_full(
            current,
            expert,
            expert.gate_metadata.input_global_scale_inv,
            expert.up_metadata.input_global_scale_inv,
        )
        down = _linear_full(swiglu, expert.down_proj, expert.down_metadata.input_global_scale_inv)
        output.index_add_(0, token_idx, down.to(output.dtype) * routing_weights[token_idx, topk_pos, None])
        if not first:
            first = {
                "expert_idx": torch.tensor(int(expert_idx)),
                "gate": gate,
                "up": up,
                "swiglu": swiglu,
                "down": down,
            }
    first["router_logits"] = logits
    first["selected_experts"] = selected
    return output, first


def _moe_tp2(x: torch.Tensor, state) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits, routing_weights, selected = _router(x, state.router_weight, state.spec.top_k)
    output = torch.zeros_like(x)
    first: dict[str, torch.Tensor] = {}
    for expert_idx in selected.unique(sorted=True).tolist():
        expert = state.experts[int(expert_idx)]
        token_idx, topk_pos = torch.where(selected == expert_idx)
        current = x[token_idx]
        gate, up, swiglu = _w13_col_tp2(
            current,
            expert,
            expert.gate_metadata.input_global_scale_inv,
            expert.up_metadata.input_global_scale_inv,
        )
        down = _linear_row_tp2(swiglu, expert.down_proj, expert.down_metadata.input_global_scale_inv)
        output.index_add_(0, token_idx, down.to(output.dtype) * routing_weights[token_idx, topk_pos, None])
        if not first:
            first = {
                "expert_idx": torch.tensor(int(expert_idx)),
                "gate": gate,
                "up": up,
                "swiglu": swiglu,
                "down": down,
            }
    first["router_logits"] = logits
    first["selected_experts"] = selected
    return output, first


def _scale_invariants(state) -> dict[str, Any]:
    qkv_inputs = [
        float(state.attention_metadata[p].input_global_scale_inv.item())
        for p in ("q_proj", "k_proj", "v_proj")
    ]
    qkv_weights = [
        float(state.attention_metadata[p].weight_global_scale.item())
        for p in ("q_proj", "k_proj", "v_proj")
    ]
    gate_inputs = [float(e.gate_metadata.input_global_scale_inv.item()) for e in state.experts]
    up_inputs = [float(e.up_metadata.input_global_scale_inv.item()) for e in state.experts]
    down_inputs = [float(e.down_metadata.input_global_scale_inv.item()) for e in state.experts]
    return {
        "qkv_input_global_scale_inv_all_equal": len(set(qkv_inputs)) == 1,
        "qkv_weight_global_scale_all_equal": len(set(qkv_weights)) == 1,
        "moe_a13_input_global_scale_inv_all_equal": len(set(gate_inputs + up_inputs)) == 1,
        "moe_a2_input_global_scale_inv_all_equal": len(set(down_inputs)) == 1,
        "group16_boundaries_tp2": {
            "hidden_half_mod16": (state.spec.hidden_size // 2) % 16,
            "q_out_half_mod16": (state.attention["q_proj"].shape[0] // 2) % 16,
            "kv_out_half_mod16": (state.attention["k_proj"].shape[0] // 2) % 16,
            "moe_intermediate_half_mod16": (state.spec.moe_intermediate_size // 2) % 16,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=QWEN3_30B_A3B_NVFP4)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-json", type=Path, default=Path("/tmp/hif4_task13_tp_parity/layer_puncture.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("layer TP puncture requires CUDA")
    device = torch.device("cuda")
    snapshot = Path(resolve_local_snapshot(args.checkpoint))
    state = load_qwen3_moe_layer_state(snapshot, args.layer, device)
    try:
        torch.manual_seed(args.seed)
        hidden = torch.randn(args.tokens, state.spec.hidden_size, device=device, dtype=torch.bfloat16)
        o_input = torch.randn(
            args.tokens,
            state.attention["o_proj"].shape[1],
            device=device,
            dtype=torch.bfloat16,
        )

        report: dict[str, Any] = {
            "checkpoint": str(snapshot),
            "layer": int(args.layer),
            "tokens": int(args.tokens),
            "seed": int(args.seed),
            "invariants": _scale_invariants(state),
            "metrics": {},
        }
        for proj in ("q_proj", "k_proj", "v_proj"):
            full = _linear_full(hidden, state.attention[proj], state.attention_metadata[proj].input_global_scale_inv)
            tp2 = _linear_col_tp2(hidden, state.attention[proj], state.attention_metadata[proj].input_global_scale_inv)
            report["metrics"][proj] = _metrics(full, tp2)

        full_o = _linear_full(o_input, state.attention["o_proj"], state.attention_metadata["o_proj"].input_global_scale_inv)
        tp2_o = _linear_row_tp2(o_input, state.attention["o_proj"], state.attention_metadata["o_proj"].input_global_scale_inv)
        report["metrics"]["o_proj_row_reduce"] = _metrics(full_o, tp2_o)

        router_full, _, selected_full = _router(hidden, state.router_weight, state.spec.top_k)
        router_tp2, _, selected_tp2 = _router(hidden, state.router_weight, state.spec.top_k)
        report["metrics"]["router_logits"] = _metrics(router_full, router_tp2)
        report["router_selected_experts_equal"] = bool(torch.equal(selected_full, selected_tp2))

        moe_full, first_full = _moe_full(hidden, state)
        moe_tp2, first_tp2 = _moe_tp2(hidden, state)
        for name in ("gate", "up", "swiglu", "down"):
            report["metrics"][f"moe_first_expert_{name}"] = _metrics(first_full[name], first_tp2[name])
        report["metrics"]["complete_moe_output"] = _metrics(moe_full, moe_tp2)
        report["moe_selected_experts_equal"] = bool(
            torch.equal(first_full["selected_experts"], first_tp2["selected_experts"])
        )

        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    finally:
        release_qwen3_moe_layer_state(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
