#!/usr/bin/env python3
"""Layerwise causal teacher-forcing replay for Qwen3-30B-A3B.

The existing training runtime is lazy/layerwise and intentionally uses no causal
mask for reconstruction calibration.  This diagnostic reuses its exact NVFP4 /
HiF4 linear semantics but patches SDPA inside this process to causal attention.
No production or E0-E7 runtime file is modified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.artifact import (
    load_conversion_state,
    select_layer_diag,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import (
    load_qwen3_moe_layer_state,
    release_qwen3_moe_layer_state,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core import moe_semantic_hif4 as sem
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import (
    NativeQwen3MoELayerRuntime,
    StudentQwen3MoELayerRuntime,
    StudentStepCache,
    build_moe_diag_state,
    qwen3_moe_config_from_snapshot,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PHASEA_ROOT,
    decode_bin,
    resolve_variant,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.metrics import (
    hidden_metrics,
    logit_metrics,
    router_metrics,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import write_jsonl
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot


def install_causal_sdpa() -> None:
    def causal_sdpa(module, query, key, value, attention_mask, *, scaling: float, dropout: float = 0.0):
        key_states = sem.repeat_kv(key, module.num_key_value_groups)
        value_states = sem.repeat_kv(value, module.num_key_value_groups)
        output = F.scaled_dot_product_attention(
            query,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=dropout if module.training else 0.0,
            is_causal=attention_mask is None,
            scale=scaling,
        )
        return output.transpose(1, 2).contiguous()

    sem._sdpa_attention_forward = causal_sdpa


def load_index(snapshot: Path) -> dict[str, str]:
    return json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]


def load_tensor(snapshot: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    shard = snapshot / weight_map[key]
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def final_head_tensors(snapshot: Path, weight_map: dict[str, str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    norm = load_tensor(snapshot, weight_map, "model.norm.weight").to(device=device, dtype=torch.bfloat16)
    if "lm_head.weight" in weight_map:
        lm_head = load_tensor(snapshot, weight_map, "lm_head.weight")
    else:
        lm_head = load_tensor(snapshot, weight_map, "model.embed_tokens.weight")
    return norm, lm_head.to(device=device, dtype=torch.bfloat16)


def predictor_positions(sample: dict) -> tuple[list[int], list[int], list[dict]]:
    prompt_len = len(sample["input_ids"])
    metadata = list(sample["positions"])
    decode_indices = [int(x["decode_index"]) for x in metadata]
    positions = [prompt_len + index - 1 for index in decode_indices]
    if min(positions) < 0:
        raise ValueError("prompt must contain at least one token")
    targets = [int(sample["output_ids"][index]) for index in decode_indices]
    return positions, targets, metadata


def replay_ids(sample: dict) -> list[int]:
    max_decode = int(sample["max_required_decode_index"])
    # To predict generated token j, the model consumes output tokens [0, j).
    return [int(x) for x in sample["input_ids"]] + [int(x) for x in sample["output_ids"][:max_decode]]


def build_runtime(layer_state, variant_name: str, artifact_state: dict | None, layer_index: int, device: torch.device):
    if variant_name == "E0":
        return NativeQwen3MoELayerRuntime(layer_state).to(device).eval(), None
    if artifact_state is None:
        diag = build_moe_diag_state(layer_state.spec, "fusable").to(device)
        use_r64 = variant_name == "E2"
        rot_order = "diag_then_rot"
    else:
        diag = build_moe_diag_state(layer_state.spec, str(artifact_state["diag_mode"])).to(device)
        record = artifact_state["layers"][str(layer_index)]
        diag.load_snapshot(select_layer_diag(record, "adopted"))
        use_r64 = bool(artifact_state["use_r64"])
        rot_order = str(artifact_state["rot_order"])
    runtime = StudentQwen3MoELayerRuntime(
        layer_state,
        diag,
        use_r64=use_r64,
        rot_order=rot_order,
    ).to(device).eval()
    return runtime, diag


def load_reference_pack(reference_dir: Path, prompt_key: str) -> dict:
    path = reference_dir / f"{prompt_key}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=["E0", "E1", "E2", "E3", "E4"])
    p.add_argument("--probe_plan", required=True)
    p.add_argument("--reference_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--phasea_root", default=str(DEFAULT_PHASEA_ROOT))
    p.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--min_e0_top1_parity", type=float, default=0.99)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    install_causal_sdpa()
    device = torch.device(args.device)
    variant = resolve_variant(args.variant)
    snapshot = Path(resolve_local_snapshot(args.model_path))
    config = qwen3_moe_config_from_snapshot(str(snapshot))
    if int(config.num_hidden_layers) != 48:
        raise RuntimeError(f"expected 48 layers, got {config.num_hidden_layers}")
    top_k = int(config.num_experts_per_tok)
    plan = json.loads(Path(args.probe_plan).read_text(encoding="utf-8"))
    samples = list(plan["samples"])
    if not samples:
        raise RuntimeError("empty probe plan")
    phasea_root = Path(args.phasea_root)
    artifact_path = variant.artifact_path(phasea_root)
    artifact_state = load_conversion_state(artifact_path) if artifact_path is not None else None
    reference_dir = Path(args.reference_dir)
    output_dir = Path(args.output_dir)
    reference_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    weight_map = load_index(snapshot)
    embed = load_tensor(snapshot, weight_map, "model.embed_tokens.weight").to(device=device, dtype=torch.bfloat16)
    hidden_cache: dict[str, torch.Tensor] = {}
    probe_abs: dict[str, list[int]] = {}
    targets: dict[str, list[int]] = {}
    metadata: dict[str, list[dict]] = {}
    reference_buffers: dict[str, dict[str, list[torch.Tensor]]] = {}
    references: dict[str, dict] = {}

    for sample in samples:
        key = str(sample["prompt_key"])
        ids = torch.tensor(replay_ids(sample), dtype=torch.long, device=device)
        hidden_cache[key] = embed[ids].detach().cpu().to(torch.bfloat16)
        positions, target_ids, meta = predictor_positions(sample)
        if max(positions) >= ids.numel():
            raise RuntimeError(f"probe position exceeds replay prefix for {key}")
        probe_abs[key] = positions
        targets[key] = target_ids
        metadata[key] = meta
        if args.variant == "E0":
            reference_buffers[key] = {"hidden": [], "router": []}
        else:
            references[key] = load_reference_pack(reference_dir, key)
    del embed
    torch.cuda.empty_cache()

    rope = Qwen3MoeRotaryEmbedding(config).to(device)
    metric_rows: list[dict] = []
    for layer_index in range(48):
        layer_state = load_qwen3_moe_layer_state(snapshot, layer_index, device)
        runtime, diag = build_runtime(layer_state, args.variant, artifact_state, layer_index, device)
        shared_step_cache = StudentStepCache.new() if args.variant != "E0" else None
        for sample in samples:
            key = str(sample["prompt_key"])
            hidden = hidden_cache[key].to(device=device, dtype=torch.bfloat16).unsqueeze(0)
            position_ids = torch.arange(hidden.shape[1], device=device, dtype=torch.long).unsqueeze(0)
            position_embeddings = rope(hidden, position_ids)
            with torch.no_grad():
                if args.variant == "E0":
                    result = runtime(
                        hidden,
                        attention_mask=None,
                        position_embeddings=position_embeddings,
                    )
                else:
                    result = runtime(
                        hidden,
                        attention_mask=None,
                        position_embeddings=position_embeddings,
                        step_cache=shared_step_cache,
                        use_ste=False,
                    )
            out = result.output
            pos = torch.tensor(probe_abs[key], device=device, dtype=torch.long)
            layer_hidden = out[0].index_select(0, pos).detach().cpu().to(torch.bfloat16)
            router = result.router_logits.reshape(1, hidden.shape[1], -1)[0].index_select(0, pos).detach().cpu().to(torch.bfloat16)
            if args.variant == "E0":
                reference_buffers[key]["hidden"].append(layer_hidden)
                reference_buffers[key]["router"].append(router)
            else:
                ref = references[key]
                for pidx, meta in enumerate(metadata[key]):
                    row = {
                        "scope": "layer",
                        "variant": args.variant,
                        "prompt_key": key,
                        "doc_id": sample.get("doc_id"),
                        "layer_id": layer_index,
                        "decode_index": int(meta["decode_index"]),
                        "bin": meta["bin"],
                        "reasons": meta["reasons"],
                    }
                    row.update(hidden_metrics(ref["hidden"][layer_index, pidx], layer_hidden[pidx]))
                    row.update(router_metrics(ref["router"][layer_index, pidx], router[pidx], top_k))
                    metric_rows.append(row)
            hidden_cache[key] = out[0].detach().cpu().to(torch.bfloat16)
            del hidden, out, result, position_embeddings
        del runtime
        if diag is not None:
            del diag
        release_qwen3_moe_layer_state(layer_state)
        torch.cuda.empty_cache()
        print(f"[{args.variant}] causal replay layer {layer_index + 1}/48", flush=True)

    norm_weight, lm_head = final_head_tensors(snapshot, weight_map, device)
    parity_total = 0
    parity_match = 0
    parity_mismatches: list[dict] = []
    for sample in samples:
        key = str(sample["prompt_key"])
        pos = torch.tensor(probe_abs[key], device=device, dtype=torch.long)
        selected = hidden_cache[key].to(device=device, dtype=torch.bfloat16).index_select(0, pos)
        normed = sem._rms_norm(selected, norm_weight, float(config.rms_norm_eps))
        with torch.no_grad():
            logits = F.linear(normed.to(lm_head.dtype), lm_head).detach().cpu().to(torch.float16)
        if args.variant == "E0":
            hidden_stack = torch.stack(reference_buffers[key]["hidden"], dim=0)
            router_stack = torch.stack(reference_buffers[key]["router"], dim=0)
            pack = {
                "schema_version": 1,
                "prompt_key": key,
                "doc_id": sample.get("doc_id"),
                "prompt_len": len(sample["input_ids"]),
                "decode_indices": [int(x["decode_index"]) for x in metadata[key]],
                "predictor_positions": probe_abs[key],
                "target_ids": targets[key],
                "top_k": top_k,
                "hidden": hidden_stack,
                "router": router_stack,
                "logits": logits,
            }
            torch.save(pack, reference_dir / f"{key}.pt")
            for pidx, target in enumerate(targets[key]):
                top1 = int(logits[pidx].argmax().item())
                parity_total += 1
                parity_match += int(top1 == int(target))
                if top1 != int(target):
                    parity_mismatches.append(
                        {
                            "prompt_key": key,
                            "decode_index": int(metadata[key][pidx]["decode_index"]),
                            "expected_vllm_greedy_token": int(target),
                            "semantic_top1_token": top1,
                        }
                    )
        else:
            ref = references[key]
            for pidx, meta in enumerate(metadata[key]):
                row = {
                    "scope": "logit",
                    "variant": args.variant,
                    "prompt_key": key,
                    "doc_id": sample.get("doc_id"),
                    "decode_index": int(meta["decode_index"]),
                    "bin": meta["bin"],
                    "reasons": meta["reasons"],
                }
                row.update(logit_metrics(ref["logits"][pidx], logits[pidx], targets[key][pidx]))
                metric_rows.append(row)

    if args.variant == "E0":
        parity = parity_match / max(parity_total, 1)
        parity_payload = {
            "num_probe_points": parity_total,
            "top1_matches_vllm_greedy": parity_match,
            "top1_parity": parity,
            "required_min_parity": args.min_e0_top1_parity,
            "mismatches": parity_mismatches,
        }
        (output_dir / "e0_semantic_parity.json").write_text(
            json.dumps(parity_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if parity < args.min_e0_top1_parity:
            raise RuntimeError(
                f"E0 semantic causal replay parity={parity:.4f} < {args.min_e0_top1_parity:.4f}; "
                "do not interpret hidden/router diagnostics"
            )
    else:
        write_jsonl(output_dir / "semantic_metrics.jsonl", metric_rows)
        meta = {
            "variant": args.variant,
            "num_samples": len(samples),
            "num_metric_rows": len(metric_rows),
            "reference_dir": str(reference_dir.resolve()),
            "artifact_path": str(artifact_path.resolve()) if artifact_path is not None else None,
            "attention_semantics": "causal SDPA process-local patch",
        }
        (output_dir / "semantic_metrics.meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
