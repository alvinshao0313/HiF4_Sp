#!/usr/bin/env python3
"""P5: fused MoE parity under frozen routing (no re-route)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_model_spec import (
    load_qwen3_moe_model_spec,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import (
    native_nvfp4_linear,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import (
    load_qwen3_moe_layer_state,
    release_qwen3_moe_layer_state,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.nvfp4_operator_parity.common import (
    RESULT_ROOT,
    compare_tensors,
    ensure_cuda_lut,
    write_jsonl,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot


def iter_packs(result_root: Path):
    manifest = json.loads((result_root / "frozen_inputs_manifest.json").read_text(encoding="utf-8"))
    for sample in manifest["samples"]:
        for entry in sample["layers"]:
            path = result_root / entry["path"]
            pack = torch.load(path, map_location="cpu", weights_only=False)
            yield pack, path


def _load_keys(snapshot: Path, keys: list[str]) -> dict[str, torch.Tensor]:
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    by_shard: dict[str, list[str]] = {}
    for key in keys:
        by_shard.setdefault(index["weight_map"][key], []).append(key)
    out: dict[str, torch.Tensor] = {}
    for shard, shard_keys in by_shard.items():
        with safe_open(str(snapshot / shard), framework="pt", device="cpu") as handle:
            for key in shard_keys:
                out[key] = handle.get_tensor(key)
    return out


def _build_fused_experts(snapshot: Path, layer: int, device: torch.device, spec):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
        nvfp4_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
        Nvfp4QuantizationEmulationTritonExperts,
    )

    keys: list[str] = []
    for expert in range(spec.num_experts):
        base = f"model.layers.{layer}.mlp.experts.{expert}"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
                keys.append(f"{base}.{proj}.{suffix}")
    tensors = _load_keys(snapshot, keys)

    w1, w1_scale, w1_gscale = [], [], []
    w2, w2_scale, w2_gscale = [], [], []
    a13_scales, a2_scales = [], []
    for expert in range(spec.num_experts):
        base = f"model.layers.{layer}.mlp.experts.{expert}"
        gate = f"{base}.gate_proj"
        up = f"{base}.up_proj"
        down = f"{base}.down_proj"
        w1.append(torch.cat([tensors[f"{gate}.weight"], tensors[f"{up}.weight"]], dim=0))
        w1_scale.append(torch.cat([tensors[f"{gate}.weight_scale"], tensors[f"{up}.weight_scale"]], dim=0))
        w1_gscale.append(tensors[f"{gate}.weight_scale_2"].reshape(()).float())
        w2.append(tensors[f"{down}.weight"])
        w2_scale.append(tensors[f"{down}.weight_scale"])
        w2_gscale.append(tensors[f"{down}.weight_scale_2"].reshape(()).float())
        a13_scales.extend(
            [
                tensors[f"{gate}.input_scale"].reshape(()).float(),
                tensors[f"{up}.input_scale"].reshape(()).float(),
            ]
        )
        a2_scales.append(tensors[f"{down}.input_scale"].reshape(()).float())

    w1_t = torch.stack(w1).to(device)
    w1_scale_t = torch.stack(w1_scale).to(device)
    w1_gscale_t = torch.stack(w1_gscale).to(device)
    w2_t = torch.stack(w2).to(device)
    w2_scale_t = torch.stack(w2_scale).to(device)
    w2_gscale_t = torch.stack(w2_gscale).to(device)
    a1_gscale = (1.0 / torch.stack(a13_scales).max()).to(device)
    a2_gscale = (1.0 / torch.stack(a2_scales).max()).to(device)

    moe_config = FusedMoEConfig(
        num_experts=spec.num_experts,
        experts_per_token=spec.top_k,
        hidden_dim=spec.hidden_size,
        intermediate_size_per_partition=spec.moe_intermediate_size,
        num_local_experts=spec.num_experts,
        num_logical_experts=spec.num_experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device=str(device),
        routing_method=RoutingMethodType.TopK,
        moe_backend="emulation",
    )
    experts = Nvfp4QuantizationEmulationTritonExperts(
        moe_config=moe_config,
        quant_config=nvfp4_moe_quant_config(
            g1_alphas=w1_gscale_t,
            g2_alphas=w2_gscale_t,
            a1_gscale=a1_gscale,
            a2_gscale=a2_gscale,
            w1_scale=w1_scale_t,
            w2_scale=w2_scale_t,
        ),
    )
    return experts, w1_t, w2_t


def _semantic_forced(
    state,
    moe_in: torch.Tensor,
    selected: torch.Tensor,
    routing_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Routed MoE with frozen selected_experts / routing_weights (no re-route)."""
    flat = moe_in.reshape(-1, moe_in.shape[-1])
    selected = selected.reshape(flat.shape[0], -1)
    routing_w = routing_w.reshape(flat.shape[0], -1)
    output = torch.zeros_like(flat)
    weighted_sum = torch.zeros_like(flat)
    for expert_idx in selected.unique(sorted=True).tolist():
        expert = state.experts[int(expert_idx)]
        token_idx, topk_pos = torch.where(selected == expert_idx)
        current = flat[token_idx]
        gate = native_nvfp4_linear(current, expert.gate_proj, expert.gate_metadata.input_global_scale_inv)
        up = native_nvfp4_linear(current, expert.up_proj, expert.up_metadata.input_global_scale_inv)
        hidden = F.silu(gate) * up
        down = native_nvfp4_linear(hidden, expert.down_proj, expert.down_metadata.input_global_scale_inv)
        weighted = down * routing_w[token_idx, topk_pos, None]
        output.index_add_(0, token_idx, weighted.to(output.dtype))
        weighted_sum.index_add_(0, token_idx, weighted.to(output.dtype))
    return output, weighted_sum


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default=str(RESULT_ROOT))
    p.add_argument("--model_path", default="nvidia/Qwen3-30B-A3B-NVFP4")
    args = p.parse_args()

    device = torch.device(args.device)
    ensure_cuda_lut(device)
    out = Path(args.output_dir)
    snapshot = Path(resolve_local_snapshot(args.model_path))
    spec = load_qwen3_moe_model_spec(str(snapshot))
    rows: list[dict] = []

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    packs = list(iter_packs(out))
    by_layer: dict[int, list] = defaultdict(list)
    for pack, path in packs:
        by_layer[int(pack["layer"])].append((pack, path))

    for layer in sorted(by_layer):
        experts, w1_t, w2_t = _build_fused_experts(snapshot, layer, device, spec)
        state = load_qwen3_moe_layer_state(snapshot, layer, device)
        try:
            for pack, path in by_layer[layer]:
                moe_in = pack["moe_input"].to(device=device, dtype=torch.bfloat16).unsqueeze(0)
                selected = pack["selected_experts"].to(device=device).to(torch.int32).unsqueeze(0)
                routing_w = pack["routing_weights"].to(device=device, dtype=torch.bfloat16).unsqueeze(0)

                with torch.no_grad():
                    sem_out, _ = _semantic_forced(state, moe_in, selected, routing_w)

                    flat = moe_in.reshape(-1, moe_in.shape[-1])
                    fused_out = torch.zeros_like(flat)
                    ws13 = torch.zeros(
                        flat.shape[0] * spec.top_k * max(spec.moe_intermediate_size, spec.hidden_size),
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    ws2 = torch.zeros_like(ws13)
                    experts.apply(
                        output=fused_out,
                        hidden_states=flat,
                        w1=w1_t,
                        w2=w2_t,
                        topk_weights=routing_w.reshape(flat.shape[0], spec.top_k),
                        topk_ids=selected.reshape(flat.shape[0], spec.top_k).to(torch.int32),
                        activation=MoEActivation.SILU,
                        global_num_experts=spec.num_experts,
                        expert_map=None,
                        a1q_scale=None,
                        a2_scale=None,
                        workspace13=ws13,
                        workspace2=ws2,
                        expert_tokens_meta=None,
                        apply_router_weight_on_input=False,
                    )

                    p4_sum = torch.zeros_like(flat)
                    for expert in pack["experts"]:
                        p4_sum = p4_sum + expert["weighted_down"].to(
                            device=device, dtype=torch.bfloat16
                        ).unsqueeze(0)

                m_sem_fused = compare_tensors(sem_out, fused_out)
                m_p4_fused = compare_tensors(p4_sum, fused_out)
                m_p4_sem = compare_tensors(p4_sum, sem_out)
                row = {
                    "prompt_key": pack["prompt_key"],
                    "decode_index": int(pack["decode_index"]),
                    "layer": layer,
                    "bucket": pack["bucket"],
                    "pack_path": str(path.relative_to(out)),
                    "forced_routing": True,
                }
                for tag, metrics in (
                    ("semantic_vs_fused", m_sem_fused),
                    ("p4_weighted_sum_vs_fused", m_p4_fused),
                    ("p4_weighted_sum_vs_semantic", m_p4_sem),
                ):
                    for k, v in metrics.items():
                        row[f"{tag}_{k}"] = v
                rows.append(row)
                print(
                    f"[P5] {pack['prompt_key']} decode={pack['decode_index']} layer={layer} "
                    f"sem_vs_fused_max_abs={m_sem_fused['max_abs']:.6g}",
                    flush=True,
                )
        finally:
            release_qwen3_moe_layer_state(state)
            del experts, w1_t, w2_t, state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_jsonl(out / "P5_fused_moe_rows.jsonl", rows)
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_bucket[row["bucket"]].append(row)

    md = [
        "# P5 fused MoE parity (frozen routing)",
        "",
        f"- rows: {len(rows)}",
        "- note: selected_experts / routing_weights forced from frozen packs; router not re-run",
        "",
        "## Pair summary",
        "",
        "| pair | max_abs_max | rel_l2_max | exact_fraction_min |",
        "|---|---:|---:|---:|",
    ]
    for tag in ("semantic_vs_fused", "p4_weighted_sum_vs_fused", "p4_weighted_sum_vs_semantic"):
        if not rows:
            md.append(f"| {tag} |  |  |  |")
            continue
        md.append(
            f"| {tag} | {max(r[f'{tag}_max_abs'] for r in rows):.6g} | "
            f"{max(r[f'{tag}_rel_l2'] for r in rows):.6g} | "
            f"{min(r[f'{tag}_exact_fraction'] for r in rows):.6g} |"
        )
    md.extend(
        [
            "",
            "## By bucket",
            "",
            "| bucket | n | semantic_vs_fused_max_abs | p4_sum_vs_fused_max_abs |",
            "|---|---:|---:|---:|",
        ]
    )
    for bucket in ("focus_low_margin", "uniform_control", "post_tp_divergence_control_only"):
        group = by_bucket.get(bucket, [])
        if not group:
            continue
        md.append(
            f"| {bucket} | {len(group)} | "
            f"{max(r['semantic_vs_fused_max_abs'] for r in group):.6g} | "
            f"{max(r['p4_weighted_sum_vs_fused_max_abs'] for r in group):.6g} |"
        )
    md.append("")
    (out / "P5_fused_moe_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
