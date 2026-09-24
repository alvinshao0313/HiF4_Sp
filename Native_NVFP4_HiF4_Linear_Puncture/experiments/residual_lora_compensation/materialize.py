"""Export learned group weights and residual adapters for actual vLLM inference."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.moe_materialize import (
    HIF4_RUNTIME_ABI_VERSION, _copy_json_without_quantization, _copy_tokenizer_assets,
    _copy_non_layer_tensors, _state_to_tensors, _write_runtime_abi_marker,
)
from .artifact import load, load_manifest, write_json, atomic_save, tensor_digest, sha256
from .transforms import fold_initial_diag, LayerParameters, transformed_state


def lora_tensors(parameters):
    return {name: value.detach().cpu().float().contiguous()
            for name, value in parameters.items() if "lora_" in name}


def export_layer(snapshot, initialization, parameters, cfg, index, out, device):
    native = load_qwen3_moe_layer_state(snapshot, index, device)
    base = fold_initial_diag(native, initialization[str(index)])
    del native
    learned = LayerParameters(base, cfg["matrix_sharing"], lora_mode=cfg["lora_mode"],
                              lora_rank=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"])
    learned.load_state_dict(parameters, strict=True)
    with torch.no_grad():
        tensors = _state_to_tensors(transformed_state(base, learned))
    shard = f"model-layer-{index:05d}-of-00048.safetensors"
    save_file(tensors, str(out / shard))
    result = {name: shard for name in tensors}
    del tensors, learned, base
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def materialize(run_dir, output_dir, *, device="cpu", smoke_base_model=None):
    run, out = Path(run_dir).resolve(), Path(output_dir).resolve()
    manifest = load_manifest(run, require_complete=smoke_base_model is None)
    smoke = bool(manifest["smoke_layers"])
    if smoke != (smoke_base_model is not None):
        raise ValueError("smoke exports require an explicit E4 base; formal exports require all 48 trained layers")
    completed = manifest["completed_layers"]
    if not completed or (smoke and completed != list(range(manifest["smoke_layers"]))):
        raise ValueError("training has not completed")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"materialization output must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    snapshot = Path(manifest["source_snapshot"])
    cfg = manifest["config"]
    initialization = load(run / "initialization.pt")
    _copy_json_without_quantization(snapshot / "config.json", out / "config.json")
    _copy_tokenizer_assets(snapshot, out)
    weight_map = {}
    if smoke:
        # The untrained suffix remains an explicitly labelled E4 suffix. It is
        # never presented as a completed experimental checkpoint.
        base = Path(smoke_base_model).resolve()
        base_marker = json.loads((base / "non_equivalent_export.json").read_text())
        if base_marker["init_sha256"] != manifest["init_sha256"] or not base_marker["initialization_only"]:
            raise ValueError("smoke base must be the matching E4 initialization")
        base_weight_map = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
        _copy_non_layer_tensors(base, out, weight_map)
        # Keep the E4 suffix and non-layer entries in the output index; trained
        # layer shards below replace only completed layer entries.
        weight_map.update(base_weight_map)
        replaced = {f"model-layer-{i:05d}-of-00048.safetensors" for i in completed}
        for shard in set(base_weight_map.values()) - replaced:
            if shard == "model-non-layer.safetensors":
                continue
            (out / shard).symlink_to(base / shard)
    else:
        _copy_non_layer_tensors(snapshot, out, weight_map)
    loras, checkpoints = {}, {}
    for index in completed:
        checkpoint = run / f"layers/{index:03d}/selected.pt"
        parameters = load(checkpoint)["parameters"]
        weight_map.update(export_layer(snapshot, initialization, parameters, cfg, index, out, device))
        loras[str(index)] = lora_tensors(parameters)
        checkpoints[str(index)] = sha256(checkpoint)
        print(f"materialized trained layer {index + 1}/{len(completed)}", flush=True)
    for index in set(range(48)) - set(completed):
        loras[str(index)] = {f"{branch}_lora_{part}": torch.zeros(shape, dtype=torch.float32)
                            for branch in ("attention", "moe")
                            for part, shape in (("A", (cfg["lora_rank"], 2048)), ("B", (2048, cfg["lora_rank"])))}
    write_json({"metadata": {"total_size": sum((out / p).stat().st_size for p in set(weight_map.values()))},
                "weight_map": weight_map}, out / "model.safetensors.index.json")
    spec = {"runtime_schema_version": 2, "runtime_abi_version": HIF4_RUNTIME_ABI_VERSION,
            "model_type": "qwen3_moe", "variant": "fusable_r64", "algorithm_variant": "fusable",
            "use_r64": True, "rot_order": "diag_then_rot", "num_layers": 48,
            "hidden_size": 2048, "head_dim": 128, "moe_intermediate_size": 768,
            "num_experts": 128, "top_k": 8, "identity_filled_layers": [],
            "online_activation_diag": {}, "online_activation_scale": {},
            "r64_placement": "qkv/gate/up/down contiguous G64; o per head; router none"}
    if cfg["lora_mode"] != "none":
        spec["residual_lora"] = {"schema_version": 1, "mode": cfg["lora_mode"],
                                 "rank": cfg["lora_rank"], "alpha": cfg["lora_alpha"],
                                 "compute_dtype": "float32", "layers": loras}
    atomic_save(spec, out / "hif4_runtime_spec.pt")
    reloaded = load(out / "hif4_runtime_spec.pt")
    hashes = {i: tensor_digest(values) for i, values in loras.items()}
    if cfg["lora_mode"] != "none":
        assert hashes == {i: tensor_digest(v) for i, v in reloaded["residual_lora"]["layers"].items()}
    _write_runtime_abi_marker(out, spec)
    write_json({"source_run": str(run), "init_sha256": manifest["init_sha256"], "complete": True,
                "smoke_only": smoke, "trained_layers": completed, "checkpoint_sha256": checkpoints,
                "lora_tensor_sha256": hashes, "runtime_spec_sha256": sha256(out / "hif4_runtime_spec.pt")},
               out / "residual_lora_export.json")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke_base_model")
    materialize(**vars(parser.parse_args()))
