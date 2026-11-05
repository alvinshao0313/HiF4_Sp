"""Materialize all native 235B layers with selected matrix_norm folds and HiF4 QDQ."""
from __future__ import annotations

import argparse
import gc

import torch
from safetensors.torch import save_file

from ..e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from ..e2e_diag_reconstruction.core.moe_model_spec import Qwen3MoeModelSpec
from ..e2e_diag_reconstruction.evaluation.moe_materialize import (
    HIF4_RUNTIME_ABI_VERSION, _copy_json_without_quantization, _copy_tokenizer_assets,
    _copy_non_layer_tensors, _state_to_tensors, _write_runtime_abi_marker,
)
from ..module_design_ablation.parameters import LayerParameters, transformed_state
from .artifacts import atomic_save, load, read_json, sha256, state_digest, verify_inputs, verify_sources, write_json
from .config import NATIVE


def require_training(root, spec):
    training = root / "training"
    trained = read_json(training / "manifest.json")
    if (not trained["complete"] or trained["qualification"]
            or trained["source_snapshot"] != str(NATIVE)
            or trained["completed_layers"] != list(range(spec.num_layers))):
        raise ValueError(f"export requires all {spec.num_layers} selected native-initialized layers")
    for layer in range(spec.num_layers):
        path = training / f"layers/{layer:03d}/selected.pt"
        metrics = read_json(training / f"layers/{layer:03d}/metrics.json")
        if sha256(path) != metrics["selected_sha256"]:
            raise ValueError(f"selected layer changed after training: {layer}")
    return training


def materialize(root):
    root = verify_inputs(root)
    verify_sources(root)
    spec = Qwen3MoeModelSpec.from_snapshot(NATIVE)
    training = require_training(root, spec)
    output = root / "models/matrix_norm"
    output.mkdir(parents=True, exist_ok=False)
    _copy_json_without_quantization(NATIVE / "config.json", output / "config.json")
    _copy_tokenizer_assets(NATIVE, output)
    weight_map, selected_hashes, router_hashes = {}, {}, {}
    _copy_non_layer_tensors(NATIVE, output, weight_map)
    for layer in range(spec.num_layers):
        base = load_qwen3_moe_layer_state(NATIVE, layer, "cuda:0")
        learned = LayerParameters(base, "matrix_norm")
        path = training / f"layers/{layer:03d}/selected.pt"
        learned.load_state_dict(load(path)["parameters"], strict=True)
        selected_hashes[str(layer)] = sha256(path)
        native_router = state_digest({"router": base.router_weight.to(torch.bfloat16)})
        with torch.no_grad():
            folded = transformed_state(base, learned)
            if state_digest({"router": folded.router_weight}) != native_router:
                raise RuntimeError("matrix_norm export altered the frozen native router")
            tensors = _state_to_tensors(folded)
        shard = f"model-layer-{layer:05d}-of-{spec.num_layers:05d}.safetensors"
        save_file(tensors, str(output / shard))
        weight_map.update({name: shard for name in tensors})
        router_hashes[str(layer)] = native_router
        del base, learned, folded, tensors
        gc.collect()
        torch.cuda.empty_cache()
        print(f"exported {layer + 1}/{spec.num_layers}", flush=True)
    original = read_json(NATIVE / "model.safetensors.index.json")["weight_map"]
    expected_keys = {key for key in original if not key.endswith((".weight_scale", ".weight_scale_2", ".input_scale"))}
    if set(weight_map) != expected_keys:
        raise ValueError("export tensor keys do not cover the full native geometry")
    byte_count = sum((output / name).stat().st_size for name in set(weight_map.values()))
    write_json({"metadata": {"total_size": byte_count}, "weight_map": weight_map}, output / "model.safetensors.index.json")
    runtime = {"runtime_schema_version": 2, "runtime_abi_version": HIF4_RUNTIME_ABI_VERSION,
               "model_type": "qwen3_moe", "variant": "fusable_r64", "algorithm_variant": "fusable_r64",
               "use_r64": True, "num_layers": spec.num_layers, "rot_order": "diag_then_rot",
               "hidden_size": spec.hidden_size, "head_dim": spec.head_dim,
               "moe_intermediate_size": spec.moe_intermediate_size,
               "num_experts": spec.num_experts, "top_k": spec.top_k, "identity_filled_layers": [],
               "online_activation_diag": {}, "online_activation_scale": {}}
    atomic_save(runtime, output / "hif4_runtime_spec.pt")
    _write_runtime_abi_marker(output, runtime)
    write_json({"complete": True, "qualification": False, "smoke_only": False,
                "source_snapshot": str(NATIVE), "training_dir": str(training),
                "layers": spec.num_layers, "tensor_count": len(weight_map), "actual_shard_bytes": byte_count,
                "selected_sha256": selected_hashes, "frozen_router_sha256": router_hashes,
                "diagnostic_untrained_layers": [],
                "shards_sha256": {name: sha256(output / name) for name in sorted(set(weight_map.values()))},
                "storage": "BF16 HiF4 QDQ values plus runtime sidecars; logical W4A4; no packed compression or speed claim"},
               output / "export.json")
    verify_sources(root)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True)
    materialize(**vars(parser.parse_args()))
