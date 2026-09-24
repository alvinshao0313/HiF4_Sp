"""Materialize all 48 selected layers into a standalone HiF4 checkpoint."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import (
    load_qwen3_moe_layer_state,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.moe_materialize import (
    HIF4_RUNTIME_ABI_VERSION, _copy_json_without_quantization, _copy_non_layer_tensors,
    _copy_tokenizer_assets, _state_to_tensors, _write_runtime_abi_marker,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.materialize import model_identity
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.transforms import (
    LayerParameters, transformed_state, fold_initial_diag,
)
from .artifact import atomic_save, load, load as load_artifact, read_json, sha256, write_json


def materialize(run_dir, output_dir, *, device="cpu", initialization_only=False):
    run = Path(run_dir).resolve()
    manifest = read_json(run / "manifest.json")
    if manifest.get("kind") != "progressive_error_cancellation":
        raise RuntimeError("wrong progressive run artifact")
    layers = manifest["completed_layers"]
    if not initialization_only and layers != list(range(48)):
        raise RuntimeError("materialization requires all 48 completed layers")
    snapshot = Path(manifest["source_snapshot"])
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"materialization output must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    initialization = load_artifact(run / "initialization.pt")
    _copy_json_without_quantization(snapshot / "config.json", out / "config.json")
    _copy_tokenizer_assets(snapshot, out)
    weight_map = {}
    _copy_non_layer_tensors(snapshot, out, weight_map)
    for index in range(48):
        native = load_qwen3_moe_layer_state(snapshot, index, device)
        base = fold_initial_diag(native, initialization[str(index)])
        del native
        params = LayerParameters(base, "group")
        if not initialization_only:
            params.load_state_dict(load_artifact(run / f"layers/L{index:02d}/selected.pt")["parameters"], strict=True)
        with torch.no_grad():
            tensors = _state_to_tensors(transformed_state(base, params))
        shard = f"model-layer-{index:05d}-of-00048.safetensors"
        save_file(tensors, str(out / shard))
        weight_map.update({name: shard for name in tensors})
        del base, params, tensors
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"materialized layer {index + 1}/48", flush=True)
    write_json({"metadata": {"total_size": sum(p.stat().st_size for p in out.glob("*.safetensors"))},
                "weight_map": weight_map}, out / "model.safetensors.index.json")
    spec = {"runtime_schema_version": 2, "runtime_abi_version": HIF4_RUNTIME_ABI_VERSION,
            "model_type": "qwen3_moe", "variant": "fusable_r64", "algorithm_variant": "fusable",
            "use_r64": True, "rot_order": "diag_then_rot", "num_layers": 48,
            "hidden_size": 2048, "head_dim": 128, "moe_intermediate_size": 768,
            "num_experts": 128, "top_k": 8, "identity_filled_layers": [],
            "online_activation_diag": {}, "online_activation_scale": {},
            "r64_placement": "qkv/gate/up/down contiguous G64; o per head; router none"}
    atomic_save(spec, out / "hif4_runtime_spec.pt")
    _write_runtime_abi_marker(out, spec)
    files = model_identity(out)
    write_json({"status": "COMPLETE", "kind": "progressive_error_cancellation_export",
                "source_run": str(run), "initialization_only": initialization_only,
                "source_snapshot": str(snapshot), "files": files,
                "protocol_sha256": sha256(run / "protocol.json")}, out / "export.json")
    # The established downstream runners use this marker to reject incomplete
    # exports before starting an expensive vLLM job.
    write_json({"complete": True, "kind": "progressive_error_cancellation_export",
                "source_run": str(run), "protocol_sha256": sha256(run / "protocol.json")},
               out / "non_equivalent_export.json")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--initialization_only", action="store_true")
    materialize(**vars(parser.parse_args()))
