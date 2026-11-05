"""Fold matrix_norm and export BF16 QDQ weights; C hardlinks frozen A/B shards."""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file, save_file

from ..e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from ..e2e_diag_reconstruction.core.moe_model_spec import Qwen3MoeModelSpec
from ..e2e_diag_reconstruction.evaluation.moe_materialize import (
    HIF4_RUNTIME_ABI_VERSION, _copy_json_without_quantization, _copy_tokenizer_assets,
    _copy_non_layer_tensors, _state_to_tensors, _write_runtime_abi_marker,
)
from ..module_design_ablation.parameters import LayerParameters, transformed_state
from .artifacts import atomic_save, load, read_json, sha256, state_digest, validate_root, verify_sources, write_json
from .config import MODELS, NATIVE
from .training import training_dir


def finish(output, weight_map, runtime, metadata):
    write_json({"metadata": {"total_size": sum((output / name).stat().st_size for name in set(weight_map.values()))},
                "weight_map": weight_map}, output / "model.safetensors.index.json")
    atomic_save(runtime, output / "hif4_runtime_spec.pt")
    _write_runtime_abi_marker(output, runtime)
    metadata["shards_sha256"] = {name: sha256(output / name) for name in sorted(set(weight_map.values()))}
    write_json({**metadata, "complete": True, "storage": "BF16 HiF4 QDQ values; logical W4A4 runtime; no packed compression claim"},
               output / "export.json")


def materialize(root, candidate, *, qualification=False):
    root = validate_root(root)
    verify_sources(root)
    if candidate not in MODELS:
        raise ValueError(candidate)
    output = root / ("qualification/models" if qualification else "models") / candidate
    output.mkdir(parents=True, exist_ok=False)
    if candidate == MODELS[2]:
        directory = root / ("qualification/router" if qualification else "router")
        aligned = read_json(directory / "manifest.json")
        expected = list(range(2 if qualification else 48))
        if not aligned["complete"] or aligned["completed_layers"] != expected:
            raise ValueError("router training is incomplete")
        origin = output.parent / aligned["winner"]
        manifest = read_json(origin / "export.json")
        if manifest["qualification"] != qualification or not manifest["complete"]:
            raise ValueError("router export initialization is invalid")
        index = read_json(origin / "model.safetensors.index.json")
        weight_map = index["weight_map"]
        router_shard = "model-routers.safetensors"
        if sha256(origin / router_shard) != manifest["shards_sha256"][router_shard]:
            raise ValueError("initial router export changed")
        for name in sorted(set(weight_map.values()) - {router_shard}):
            if sha256(origin / name) != manifest["shards_sha256"][name]:
                raise ValueError("initialization export changed")
            os.link(origin / name, output / name)
        routers = load_file(str(origin / router_shard))
        for i in expected:
            value = load(directory / f"layers/{i:03d}/selected.pt")["router_master"]
            routers[f"model.layers.{i}.mlp.gate.weight"] = value.to(torch.bfloat16).contiguous()
        if any(t.dtype != torch.bfloat16 for t in routers.values()):
            raise ValueError("router export requires BF16 weights")
        save_file(routers, str(output / router_shard))
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                     "generation_config.json", "vocab.json", "merges.txt", "chat_template.jinja"):
            if (origin / name).is_file():
                shutil.copyfile(origin / name, output / name)
        finish(output, weight_map, load(origin / "hif4_runtime_spec.pt"),
               {"candidate": candidate, "qualification": qualification, "initialization": str(origin),
                "non_router_shards_identical": True, "router_tensor_sha256": state_digest(routers)})
        return output
    directory = training_dir(root, candidate, qualification=qualification)
    spec = Qwen3MoeModelSpec.from_snapshot(NATIVE)
    _copy_json_without_quantization(NATIVE / "config.json", output / "config.json")
    _copy_tokenizer_assets(NATIVE, output)
    weight_map, routers, selected_hashes = {}, {}, {}
    _copy_non_layer_tensors(NATIVE, output, weight_map)
    for i in range(48):
        base = load_qwen3_moe_layer_state(NATIVE, i, "cuda:0")
        learned = LayerParameters(base, "matrix_norm")
        if not qualification or i < 2:
            selected = directory / f"layers/{i:03d}/selected.pt"
            learned.load_state_dict(load(selected)["parameters"], strict=True)
            selected_hashes[str(i)] = sha256(selected)
        with torch.no_grad():
            folded = transformed_state(base, learned)
            tensors = _state_to_tensors(folded)
        gate = f"model.layers.{i}.mlp.gate.weight"
        routers[gate] = tensors.pop(gate).cpu().contiguous()
        shard = f"model-layer-{i:05d}-of-00048.safetensors"
        save_file(tensors, str(output / shard))
        weight_map.update({key: shard for key in tensors})
        weight_map[gate] = "model-routers.safetensors"
        del base, learned, folded, tensors
        gc.collect()
        torch.cuda.empty_cache()
        print(f"exported {candidate} {i + 1}/48", flush=True)
    save_file(routers, str(output / "model-routers.safetensors"))
    runtime = {"runtime_schema_version": 2, "runtime_abi_version": HIF4_RUNTIME_ABI_VERSION,
               "model_type": "qwen3_moe", "variant": "fusable_r64", "algorithm_variant": "fusable_r64",
               "use_r64": True, "num_layers": spec.num_layers, "rot_order": "diag_then_rot",
               "hidden_size": spec.hidden_size, "head_dim": spec.head_dim, "moe_intermediate_size": spec.moe_intermediate_size,
               "num_experts": spec.num_experts, "top_k": spec.top_k, "identity_filled_layers": [],
               "online_activation_diag": {}, "online_activation_scale": {}}
    finish(output, weight_map, runtime, {"candidate": candidate, "qualification": qualification,
                                       "training_dir": str(directory), "selected_sha256": selected_hashes})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidate", choices=MODELS, required=True)
    parser.add_argument("--qualification", action="store_true")
    args = parser.parse_args()
    materialize(args.run, args.candidate, qualification=args.qualification)
