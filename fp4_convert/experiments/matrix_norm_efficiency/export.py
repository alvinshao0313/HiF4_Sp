"""Fold learned M and norm into the existing HiF4 W4A4 inference format."""
import gc
import time

import torch
from safetensors.torch import save_file

from ..e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from ..e2e_diag_reconstruction.core.moe_model_spec import Qwen3MoeModelSpec
from ..e2e_diag_reconstruction.evaluation.moe_materialize import (
    HIF4_RUNTIME_ABI_VERSION, _copy_json_without_quantization, _copy_tokenizer_assets,
    _copy_non_layer_tensors, _state_to_tensors, _write_runtime_abi_marker,
)
from ..module_design_ablation.parameters import LayerParameters, transformed_state
from .artifacts import atomic_save, load, read_json, sha256, validate_root, verify_sources, write_json
from .config import LAYERS, NATIVE
from .training import training_dir


def materialize(root, candidate, *, qualification=False):
    started = time.monotonic()
    root = validate_root(root)
    verify_sources(root)
    directory = training_dir(root, candidate, qualification=qualification)
    training = read_json(directory / "manifest.json")
    if not torch.cuda.is_available():
        raise RuntimeError("HiF4 materialization requires CUDA in hif4")
    output = root / ("qualification/models" if qualification else "models") / candidate
    output.mkdir(parents=True, exist_ok=False)
    spec = Qwen3MoeModelSpec.from_snapshot(NATIVE)
    if spec.num_layers != LAYERS:
        raise ValueError("unexpected source model layer count")
    _copy_json_without_quantization(NATIVE / "config.json", output / "config.json")
    _copy_tokenizer_assets(NATIVE, output)
    weight_map, selected_hashes = {}, {}
    _copy_non_layer_tensors(NATIVE, output, weight_map)
    for layer in range(LAYERS):
        base = load_qwen3_moe_layer_state(NATIVE, layer, "cuda:0")
        learned = LayerParameters(base, "matrix_norm")
        if layer in training["completed_layers"]:
            selected = directory / f"layers/{layer:03d}/selected.pt"
            metrics = read_json(directory / f"layers/{layer:03d}/metrics.json")
            if sha256(selected) != metrics["selected_sha256"]:
                raise ValueError(f"selected parameters changed at layer {layer}")
            learned.load_state_dict(load(selected)["parameters"], strict=True)
            selected_hashes[str(layer)] = metrics["selected_sha256"]
        with torch.no_grad():
            folded = transformed_state(base, learned)
            tensors = _state_to_tensors(folded)
        shard = f"model-layer-{layer:05d}-of-{LAYERS:05d}.safetensors"
        save_file(tensors, str(output / shard))
        weight_map.update({key: shard for key in tensors})
        del base, learned, folded, tensors
        gc.collect()
        torch.cuda.empty_cache()
        print(f"exported {candidate}: {layer + 1}/{LAYERS}", flush=True)
    runtime = {"runtime_schema_version": 2, "runtime_abi_version": HIF4_RUNTIME_ABI_VERSION,
        "model_type": "qwen3_moe", "variant": "fusable_r64", "algorithm_variant": "fusable_r64",
        "use_r64": True, "num_layers": LAYERS, "rot_order": "diag_then_rot",
        "hidden_size": spec.hidden_size, "head_dim": spec.head_dim,
        "moe_intermediate_size": spec.moe_intermediate_size, "num_experts": spec.num_experts,
        "top_k": spec.top_k, "identity_filled_layers": [],
        "online_activation_diag": {}, "online_activation_scale": {}}
    write_json({"metadata": {"total_size": sum((output / name).stat().st_size
                                               for name in set(weight_map.values()))},
                "weight_map": weight_map}, output / "model.safetensors.index.json")
    atomic_save(runtime, output / "hif4_runtime_spec.pt")
    _write_runtime_abi_marker(output, runtime)
    verify_sources(root)
    write_json({"complete": True, "candidate": candidate, "qualification": qualification,
        "native_snapshot": str(NATIVE), "training_dir": str(directory),
        "trained_layers": training["completed_layers"], "selected_sha256": selected_hashes,
        "training_manifest_sha256": sha256(directory / "manifest.json"),
        "shards_sha256": {name: sha256(output / name) for name in sorted(set(weight_map.values()))},
        "storage": "BF16 HiF4 QDQ values; logical W4A4 runtime; no packed compression claim",
        "seconds": time.monotonic() - started}, output / "export.json")
    return output
