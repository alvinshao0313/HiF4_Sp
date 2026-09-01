#!/usr/bin/env python3
"""Capture frozen identical intermediate tensors from E0 semantic teacher-forcing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import (
    load_qwen3_moe_layer_state,
    release_qwen3_moe_layer_state,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core import moe_semantic_hif4 as sem
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import (
    NativeQwen3MoELayerRuntime,
    qwen3_moe_config_from_snapshot,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.causal_replay import (
    install_causal_sdpa,
    load_index,
    load_tensor,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.nvfp4_operator_parity.common import (
    DEFAULT_LAYERS,
    RESULT_ROOT,
    SMOKE_ROOT,
    all_probes,
    probe_bucket,
    tensor_checksum,
    write_json,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import read_jsonl
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot


def replay_ids(sample: dict, max_decode: int) -> list[int]:
    return [int(x) for x in sample["input_ids"]] + [int(x) for x in sample["output_ids"][:max_decode]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default=str(RESULT_ROOT))
    p.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    p.add_argument("--model_path", default="nvidia/Qwen3-30B-A3B-NVFP4")
    args = p.parse_args()

    install_causal_sdpa()
    device = torch.device(args.device)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    snapshot = Path(resolve_local_snapshot(args.model_path))
    config = qwen3_moe_config_from_snapshot(str(snapshot))
    e0_rows = {r["prompt_key"]: r for r in read_jsonl(SMOKE_ROOT / "normalized/E0.jsonl")}
    probes = all_probes()
    needed: dict[str, set[int]] = {}
    meta_by_key: dict[str, list[tuple[int, bool]]] = {}
    for key, dec, post in probes:
        needed.setdefault(key, set()).add(dec)
        meta_by_key.setdefault(key, []).append((dec, post))

    weight_map = load_index(snapshot)
    embed = load_tensor(snapshot, weight_map, "model.embed_tokens.weight").to(device=device, dtype=torch.bfloat16)
    rope = Qwen3MoeRotaryEmbedding(config).to(device)

    out_root = Path(args.output_dir) / "frozen_inputs"
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_samples = []

    for prompt_key, decode_set in needed.items():
        sample = e0_rows[prompt_key]
        max_decode = max(decode_set)
        ids = torch.tensor(replay_ids(sample, max_decode), dtype=torch.long, device=device)
        # Predict token at decode_index j using prefix ending at prompt_len+j-1
        prompt_len = len(sample["input_ids"])
        abs_positions = {dec: prompt_len + dec - 1 for dec in decode_set}
        if max(abs_positions.values()) >= ids.numel():
            raise RuntimeError(f"probe exceeds replay length for {prompt_key}")

        hidden = embed[ids].unsqueeze(0)  # [1,T,H]
        del ids
        torch.cuda.empty_cache()

        sample_manifest = {
            "prompt_key": prompt_key,
            "prompt_len": prompt_len,
            "max_decode": max_decode,
            "probes": [
                {
                    "decode_index": dec,
                    "post_tp_divergence_control_only": post,
                    "bucket": probe_bucket(prompt_key, dec, post),
                    "abs_position": abs_positions[dec],
                }
                for dec, post in meta_by_key[prompt_key]
            ],
            "layers": [],
        }

        for layer_index in layers:
            print(f"[frozen] {prompt_key} layer {layer_index}", flush=True)
            state = load_qwen3_moe_layer_state(snapshot, layer_index, device)
            runtime = NativeQwen3MoELayerRuntime(state).to(device).eval()
            position_ids = torch.arange(hidden.shape[1], device=device, dtype=torch.long).unsqueeze(0)
            position_embeddings = rope(hidden, position_ids)
            with torch.no_grad():
                layer_in = hidden
                attn_in = sem._rms_norm(layer_in, state.input_layernorm_weight, runtime.rms_norm_eps)
                attn = runtime.attention_projections(attn_in, None, position_embeddings)
                post_attn = layer_in + attn.o
                moe_in = sem._rms_norm(post_attn, state.post_attention_layernorm_weight, runtime.rms_norm_eps)
                moe = runtime.routed_moe(moe_in)
                layer_out = post_attn + moe.output

            # router reshape [T, E]
            router = moe.router_logits.reshape(hidden.shape[1], -1)
            selected = moe.selected_experts.reshape(hidden.shape[1], -1)
            routing_w = moe.routing_weights.reshape(hidden.shape[1], -1)

            for decode_index, post in meta_by_key[prompt_key]:
                pabs = abs_positions[decode_index]
                # expert contributions for this token
                token_experts = []
                token_idx = pabs
                flat_moe_in = moe_in[0, token_idx]
                for slot in range(selected.shape[1]):
                    expert_id = int(selected[token_idx, slot].item())
                    weight = float(routing_w[token_idx, slot].item())
                    expert = state.experts[expert_id]
                    with torch.no_grad():
                        gate = sem.native_nvfp4_linear(
                            flat_moe_in.unsqueeze(0),
                            expert.gate_proj,
                            expert.gate_metadata.input_global_scale_inv,
                        )[0]
                        up = sem.native_nvfp4_linear(
                            flat_moe_in.unsqueeze(0),
                            expert.up_proj,
                            expert.up_metadata.input_global_scale_inv,
                        )[0]
                        silu_up = F.silu(gate) * up
                        down = sem.native_nvfp4_linear(
                            silu_up.unsqueeze(0),
                            expert.down_proj,
                            expert.down_metadata.input_global_scale_inv,
                        )[0]
                        weighted = down * weight
                    token_experts.append(
                        {
                            "expert_id": expert_id,
                            "slot": slot,
                            "routing_weight": weight,
                            "gate": gate.detach().cpu().to(torch.bfloat16),
                            "up": up.detach().cpu().to(torch.bfloat16),
                            "w2_input": silu_up.detach().cpu().to(torch.bfloat16),
                            "down": down.detach().cpu().to(torch.bfloat16),
                            "weighted_down": weighted.detach().cpu().to(torch.bfloat16),
                            "a13_inv": float(expert.gate_metadata.input_global_scale_inv.float().item()),
                            "a2_inv": float(expert.down_metadata.input_global_scale_inv.float().item()),
                            "w13_global": float(expert.gate_metadata.weight_global_scale.float().item()),
                            "w2_global": float(expert.down_metadata.weight_global_scale.float().item()),
                        }
                    )

                pack = {
                    "schema_version": 1,
                    "prompt_key": prompt_key,
                    "decode_index": decode_index,
                    "post_tp_divergence_control_only": post,
                    "bucket": probe_bucket(prompt_key, decode_index, post),
                    "layer": layer_index,
                    "abs_position": pabs,
                    "layer_input": layer_in[0, pabs].detach().cpu().to(torch.bfloat16),
                    "qkv_input": attn_in[0, pabs].detach().cpu().to(torch.bfloat16),
                    "q": attn.q[0, pabs].detach().cpu().to(torch.bfloat16),
                    "k": attn.k[0, pabs].detach().cpu().to(torch.bfloat16),
                    "v": attn.v[0, pabs].detach().cpu().to(torch.bfloat16),
                    "o_input": attn.o_input[0, pabs].detach().cpu().to(torch.bfloat16),
                    "o": attn.o[0, pabs].detach().cpu().to(torch.bfloat16),
                    "post_attn": post_attn[0, pabs].detach().cpu().to(torch.bfloat16),
                    "moe_input": moe_in[0, pabs].detach().cpu().to(torch.bfloat16),
                    "router_logits": router[pabs].detach().cpu().to(torch.bfloat16),
                    "selected_experts": selected[pabs].detach().cpu().to(torch.int32),
                    "routing_weights": routing_w[pabs].detach().cpu().to(torch.bfloat16),
                    "moe_output": moe.output[0, pabs].detach().cpu().to(torch.bfloat16),
                    "layer_output": layer_out[0, pabs].detach().cpu().to(torch.bfloat16),
                    "experts": token_experts,
                    "scales": {
                        "qkv_input_inv": float(state.attention_metadata["q_proj"].input_global_scale_inv.float().item()),
                        "qkv_weight_global": float(state.attention_metadata["q_proj"].weight_global_scale.float().item()),
                        "o_input_inv": float(state.attention_metadata["o_proj"].input_global_scale_inv.float().item()),
                        "o_weight_global": float(state.attention_metadata["o_proj"].weight_global_scale.float().item()),
                    },
                }
                checksums = {
                    name: tensor_checksum(pack[name])
                    for name in (
                        "layer_input",
                        "qkv_input",
                        "q",
                        "k",
                        "v",
                        "o_input",
                        "o",
                        "moe_input",
                        "moe_output",
                    )
                }
                pack["checksums"] = checksums
                dest = out_root / prompt_key / f"decode_{decode_index:05d}"
                dest.mkdir(parents=True, exist_ok=True)
                path = dest / f"layer_{layer_index:02d}.pt"
                torch.save(pack, path)
                sample_manifest["layers"].append(
                    {
                        "decode_index": decode_index,
                        "layer": layer_index,
                        "path": str(path.relative_to(Path(args.output_dir))),
                        "checksums": checksums,
                    }
                )

            hidden = layer_out.detach()
            del runtime, attn, moe, post_attn, moe_in, layer_in, position_embeddings, layer_out
            release_qwen3_moe_layer_state(state)
            del state
            torch.cuda.empty_cache()

        manifest_samples.append(sample_manifest)

    write_json(
        Path(args.output_dir) / "frozen_inputs_manifest.json",
        {
            "schema_version": 1,
            "layers": layers,
            "probes": [
                {"prompt_key": k, "decode_index": d, "post_tp_divergence_control_only": p, "bucket": probe_bucket(k, d, p)}
                for k, d, p in probes
            ],
            "samples": manifest_samples,
        },
    )
    print(f"[frozen] done -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
