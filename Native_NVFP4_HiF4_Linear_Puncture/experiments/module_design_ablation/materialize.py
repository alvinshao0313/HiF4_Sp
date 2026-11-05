"""Export native-start variants, including their learned activation matrices."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from ..e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from ..e2e_diag_reconstruction.core.moe_model_spec import Qwen3MoeModelSpec
from ..e2e_diag_reconstruction.evaluation.moe_materialize import (
    HIF4_RUNTIME_ABI_VERSION, _copy_json_without_quantization, _copy_tokenizer_assets,
    _copy_non_layer_tensors, _state_to_tensors, _write_runtime_abi_marker,
)
from ..non_equivalent_reconstruction.artifact import atomic_save, load, sha256, write_json
from .config import NATIVE_SNAPSHOT
from .parameters import LayerParameters, transformed_state
from .runtime import RUNTIME_KIND
from .variants import get_variant


def materialize(output_dir, *, variant, run_dir=None, diagnostic=False, device="cuda:0"):
    definition = get_variant(variant)
    if definition.is_reference:
        raise ValueError("native and historical references use their existing checkpoints")
    run = None if run_dir is None else Path(run_dir).resolve()
    manifest = None
    if definition.needs_training:
        if run is None:
            raise ValueError("trained variants require their own completed run")
        manifest = json.loads((run / "manifest.json").read_text())
        if (manifest.get("kind") != "module_design_ablation" or not manifest.get("complete")
                or manifest["config"]["variant"] != variant
                or Path(manifest["source_snapshot"]).resolve() != NATIVE_SNAPSHOT.resolve()):
            raise ValueError("incomplete or mismatched ablation training run")
        if not diagnostic and (manifest["config"]["phase"] != "formal" or manifest["completed_layers"] != list(range(48))):
            raise ValueError("formal export requires all 48 formally trained layers")
    elif run is not None or diagnostic:
        raise ValueError("untrained baselines do not take a run or diagnostic override")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    spec = Qwen3MoeModelSpec.from_snapshot(NATIVE_SNAPSHOT)
    _copy_json_without_quantization(NATIVE_SNAPSHOT / "config.json", output / "config.json")
    _copy_tokenizer_assets(NATIVE_SNAPSHOT, output)
    weight_map, rotation_files, selected_hashes = {}, {}, {}
    _copy_non_layer_tensors(NATIVE_SNAPSHOT, output, weight_map)
    for index in range(spec.num_layers):
        base = load_qwen3_moe_layer_state(NATIVE_SNAPSHOT, index, device)
        learned = LayerParameters(base, variant)
        if manifest is not None and index in manifest["completed_layers"]:
            selected = run / f"layers/{index:03d}/selected.pt"
            learned.load_state_dict(load(selected)["parameters"], strict=True)
            selected_hashes[str(index)] = sha256(selected)
        with torch.no_grad():
            folded = transformed_state(base, learned)
            tensors = _state_to_tensors(folded)
            if definition.transform == "rotation":
                rotations = {name: learned.rotations[name]().cpu().contiguous() for name in base.attention}
                rotations.update({name: torch.stack([
                    learned.rotations[f"expert_{expert}_{name}"]().cpu()
                    for expert in range(spec.num_experts)]).contiguous()
                    for name in ("gate_proj", "up_proj", "down_proj")})
                filename = f"hif4_rotation_layer_{index:03d}.safetensors"
                save_file(rotations, str(output / filename))
                rotation_files[str(index)] = filename
                del rotations
        shard = f"model-layer-{index:05d}-of-{spec.num_layers:05d}.safetensors"
        save_file(tensors, str(output / shard))
        weight_map.update({name: shard for name in tensors})
        del base, learned, folded, tensors
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"materialized layer {index + 1}/{spec.num_layers}", flush=True)
    write_json({"metadata": {"total_size": sum(p.stat().st_size for p in output.glob("model*.safetensors"))},
                "weight_map": weight_map}, output / "model.safetensors.index.json")
    family = (RUNTIME_KIND if definition.transform == "rotation" else
              "direct" if definition.transform == "direct" else "fusable_r64")
    runtime = {"runtime_schema_version": 2, "runtime_abi_version": HIF4_RUNTIME_ABI_VERSION,
               "model_type": "qwen3_moe", "variant": family, "algorithm_variant": family,
               "use_r64": family == "fusable_r64", "num_layers": spec.num_layers,
               "rot_order": "diag_then_rot", "hidden_size": spec.hidden_size, "head_dim": spec.head_dim,
               "moe_intermediate_size": spec.moe_intermediate_size, "num_experts": spec.num_experts,
               "top_k": spec.top_k, "rotation_files": rotation_files, "identity_filled_layers": [],
               "online_activation_diag": {}, "online_activation_scale": {}}
    atomic_save(runtime, output / "hif4_runtime_spec.pt")
    _write_runtime_abi_marker(output, runtime)
    write_json({"complete": True, "smoke_only": diagnostic, "variant": variant,
                "source_snapshot": str(NATIVE_SNAPSHOT), "source_run": str(run) if run else None,
                "selected_sha256": selected_hashes,
                "diagnostic_untrained_layers": [] if not diagnostic else [i for i in range(48) if str(i) not in selected_hashes]},
               output / "ablation_export.json")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--run_dir")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    materialize(**vars(parser.parse_args()))
