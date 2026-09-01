"""Streaming materializer for Qwen3-MoE HiF4 evaluation checkpoints."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.artifact import (
    load_conversion_state,
    select_layer_diag,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import (
    load_qwen3_moe_layer_state,
    release_qwen3_moe_layer_state,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_fold import (
    fold_fusable_moe_layer_state,
    fold_online_moe_layer_state,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import (
    build_moe_diag_state,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.formats import qdq_hif4_direct
from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import ensure_dir


# Bump when materialized weight semantics or the vLLM sidecar contract changes
# enough that old checkpoints are unsafe to reuse.
HIF4_RUNTIME_ABI_VERSION = 3

NVFP4_AUX_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")
NON_LAYER_KEYS = ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")


def _copy_json_without_quantization(src: Path, dst: Path) -> None:
    cfg = json.loads(src.read_text(encoding="utf-8"))
    cfg.pop("quantization_config", None)
    cfg["torch_dtype"] = "bfloat16"
    dst.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_diag(spec, diag_mode: str, z: dict[str, torch.Tensor]):
    diag = build_moe_diag_state(spec, diag_mode)
    diag.load_snapshot(z)
    return diag


def _write_runtime_abi_marker(out: Path, runtime_spec: dict[str, Any]) -> None:
    marker = {
        "runtime_abi_version": int(runtime_spec["runtime_abi_version"]),
        "runtime_schema_version": runtime_spec.get("runtime_schema_version"),
        "variant": runtime_spec.get("variant"),
        "algorithm_variant": runtime_spec.get("algorithm_variant", runtime_spec.get("variant")),
        "use_r64": bool(runtime_spec.get("use_r64", False)),
        "rot_order": runtime_spec.get("rot_order"),
        "note": "E2/E3-E7 must use optimized HiF4 Triton runtime; old materializations are not reusable.",
    }
    (out / "hif4_runtime_abi.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _online_scales_from_snapshot(z: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    expected = {"z_q", "z_k", "z_v", "z_o", "z_gate", "z_up", "z_down"}
    if set(z) != expected:
        raise ValueError(f"Online snapshot keys {sorted(z)} != {sorted(expected)}")
    return {
        name.replace("z_", "d_", 1): torch.exp2(value.to(torch.float32)).cpu()
        for name, value in z.items()
    }


def _copy_non_layer_tensors(src: Path, out: Path, weight_map: dict[str, str]) -> None:
    index = json.loads((src / "model.safetensors.index.json").read_text(encoding="utf-8"))
    by_shard: dict[str, list[str]] = {}
    for key in NON_LAYER_KEYS:
        shard = index["weight_map"].get(key)
        if shard is not None:
            by_shard.setdefault(shard, []).append(key)
    tensors: dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(str(src / shard), framework="pt", device="cpu") as handle:
            for key in keys:
                tensors[key] = handle.get_tensor(key).to(torch.bfloat16).cpu()
    if not tensors:
        raise RuntimeError("no non-layer tensors were copied for materialized checkpoint")
    shard_name = "model-non-layer.safetensors"
    save_file(tensors, out / shard_name)
    for key in tensors:
        weight_map[key] = shard_name


def _state_to_tensors(state) -> dict[str, torch.Tensor]:
    layer = state.layer_idx
    base = f"model.layers.{layer}"
    tensors: dict[str, torch.Tensor] = {
        f"{base}.input_layernorm.weight": state.input_layernorm_weight.to(torch.bfloat16).cpu(),
        f"{base}.post_attention_layernorm.weight": state.post_attention_layernorm_weight.to(torch.bfloat16).cpu(),
        f"{base}.self_attn.q_norm.weight": state.q_norm_weight.to(torch.bfloat16).cpu(),
        f"{base}.self_attn.k_norm.weight": state.k_norm_weight.to(torch.bfloat16).cpu(),
        f"{base}.mlp.gate.weight": state.router_weight.to(torch.bfloat16).cpu(),
    }
    for proj, weight in state.attention.items():
        tensors[f"{base}.self_attn.{proj}.weight"] = qdq_hif4_direct(
            weight.to(torch.float32), output_dtype=torch.bfloat16
        ).cpu()
    for expert_idx, expert in enumerate(state.experts):
        ebase = f"{base}.mlp.experts.{expert_idx}"
        tensors[f"{ebase}.gate_proj.weight"] = qdq_hif4_direct(expert.gate_proj, output_dtype=torch.bfloat16).cpu()
        tensors[f"{ebase}.up_proj.weight"] = qdq_hif4_direct(expert.up_proj, output_dtype=torch.bfloat16).cpu()
        tensors[f"{ebase}.down_proj.weight"] = qdq_hif4_direct(expert.down_proj, output_dtype=torch.bfloat16).cpu()
    return tensors


def materialize_moe_checkpoint(
    *,
    source_snapshot: str | Path,
    artifact_path: str | Path,
    output_dir: str | Path,
    diag_variant: str = "adopted",
    keep_eval_ckpt: bool = True,
) -> Path:
    src = Path(source_snapshot)
    out = ensure_dir(output_dir)
    state = load_conversion_state(artifact_path)
    if state.get("model_type") != "qwen3_moe":
        raise ValueError("MoE materializer requires schema v2 model_type=qwen3_moe")
    _copy_json_without_quantization(src / "config.json", out / "config.json")
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "special_tokens_map.json"):
        p = src / name
        if p.exists():
            target = out / name
            if target.exists():
                target.unlink()
            try:
                target.symlink_to(p)
            except OSError:
                shutil.copy2(p, target)

    weight_map: dict[str, str] = {}
    _copy_non_layer_tensors(src, out, weight_map)
    online_diag_by_layer: dict[str, dict[str, torch.Tensor]] = {}
    online_scale_by_layer: dict[str, dict[str, torch.Tensor]] = {}
    layers: dict[str, Any] = state["layers"]
    num_layers = int(state["num_layers"])
    present = {int(k) for k in layers}
    missing = [i for i in range(num_layers) if i not in present]
    for layer_idx in range(num_layers):
        if layer_idx % 8 == 0 or layer_idx + 1 == num_layers:
            print(
                f"[materialize] layer {layer_idx}/{num_layers - 1} "
                f"variant={diag_variant} present={layer_idx in present}",
                flush=True,
            )
        layer_state = load_qwen3_moe_layer_state(src, layer_idx, "cpu")
        try:
            if layer_idx in present:
                diag = _load_diag(
                    layer_state.spec,
                    state["diag_mode"],
                    select_layer_diag(layers[str(layer_idx)], diag_variant),
                )
            else:
                # Incomplete artifacts (smoke / mid-train) must still emit a full
                # runnable BF16 checkpoint; untrained layers stay identity DIAG.
                diag = build_moe_diag_state(layer_state.spec, state["diag_mode"])
            if state["diag_mode"] == "online":
                snapshot = diag.snapshot()
                online_diag_by_layer[str(layer_idx)] = snapshot
                online_scale_by_layer[str(layer_idx)] = _online_scales_from_snapshot(snapshot)
            if state["diag_mode"] == "fusable":
                layer_state = fold_fusable_moe_layer_state(
                    layer_state, diag, use_r64=bool(state["use_r64"])
                )
            elif state["diag_mode"] == "online":
                layer_state = fold_online_moe_layer_state(
                    layer_state,
                    diag,
                    use_r64=bool(state["use_r64"]),
                    rot_order=str(state["rot_order"]),
                )
            tensors = _state_to_tensors(layer_state)
            shard = f"model-layer-{layer_idx:05d}-of-{num_layers:05d}.safetensors"
            save_file(tensors, out / shard)
            for key in tensors:
                if key.endswith(NVFP4_AUX_SUFFIXES):
                    raise RuntimeError(f"materialized checkpoint leaked NVFP4 aux tensor {key}")
                weight_map[key] = shard
        finally:
            release_qwen3_moe_layer_state(layer_state)

    index = {"metadata": {"total_size": 0, "identity_filled_layers": missing}, "weight_map": weight_map}
    (out / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runtime_spec = {
        "runtime_schema_version": 2,
        "runtime_abi_version": HIF4_RUNTIME_ABI_VERSION,
        "model_type": "qwen3_moe",
        "variant": (
            "fusable_r64" if state["diag_mode"] == "online" and bool(state["use_r64"])
            else "fusable" if state["diag_mode"] == "online"
            else state["diag_mode"]
        ),
        "algorithm_variant": state["diag_mode"],
        "artifact_diag_variant": diag_variant,
        "use_r64": bool(state["use_r64"]),
        "rot_order": state["rot_order"],
        "num_layers": num_layers,
        "hidden_size": 2048,
        "head_dim": 128,
        "moe_intermediate_size": 768,
        "num_experts": 128,
        "top_k": 8,
        "identity_filled_layers": missing,
        "online_activation_diag": (
            online_diag_by_layer if state["diag_mode"] == "online" else {}
        ),
        "online_activation_scale": (
            online_scale_by_layer if state["diag_mode"] == "online" else {}
        ),
        "r64_placement": "qkv/gate/up/down contiguous G64; o per head; router none",
    }
    torch.save(runtime_spec, out / "hif4_runtime_spec.pt")
    _write_runtime_abi_marker(out, runtime_spec)
    if not keep_eval_ckpt:
        return out
    return out


def materialize_moe_identity_checkpoint(
    *,
    source_snapshot: str | Path,
    output_dir: str | Path,
    layer_indices: list[int] | None = None,
    use_r64: bool = False,
    keep_eval_ckpt: bool = True,
) -> Path:
    """Materialize direct-HiF4 or R64-only MoE checkpoints without DIAG artifacts."""
    src = Path(source_snapshot)
    out = ensure_dir(output_dir)
    _copy_json_without_quantization(src / "config.json", out / "config.json")
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "special_tokens_map.json"):
        p = src / name
        if p.exists():
            target = out / name
            if target.exists():
                target.unlink()
            try:
                target.symlink_to(p)
            except OSError:
                shutil.copy2(p, target)

    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    num_layers = int(cfg["num_hidden_layers"])
    layers = list(range(num_layers)) if layer_indices is None else [int(i) for i in layer_indices]
    if not layers:
        raise ValueError("layer_indices must not be empty")

    weight_map: dict[str, str] = {}
    _copy_non_layer_tensors(src, out, weight_map)
    for layer_idx in layers:
        layer_state = load_qwen3_moe_layer_state(src, layer_idx, "cpu")
        try:
            if use_r64:
                diag = build_moe_diag_state(layer_state.spec, "fusable")
                layer_state = fold_fusable_moe_layer_state(layer_state, diag, use_r64=True)
            tensors = _state_to_tensors(layer_state)
            shard = f"model-layer-{layer_idx:05d}-of-{len(layers):05d}.safetensors"
            save_file(tensors, out / shard)
            for key in tensors:
                if key.endswith(NVFP4_AUX_SUFFIXES):
                    raise RuntimeError(f"materialized checkpoint leaked NVFP4 aux tensor {key}")
                weight_map[key] = shard
        finally:
            release_qwen3_moe_layer_state(layer_state)

    index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
    (out / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runtime_spec = {
        "runtime_abi_version": HIF4_RUNTIME_ABI_VERSION,
        "model_type": "qwen3_moe",
        "variant": "r64" if use_r64 else "direct",
        "use_r64": bool(use_r64),
        "rot_order": "diag_then_rot",
        "num_layers": num_layers,
        "online_activation_diag": {},
        "r64_placement": "qkv/gate/up/down contiguous G64; o per head; router none",
    }
    torch.save(runtime_spec, out / "hif4_runtime_spec.pt")
    _write_runtime_abi_marker(out, runtime_spec)
    if not keep_eval_ckpt:
        return out
    return out
