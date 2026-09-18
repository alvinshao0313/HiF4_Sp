"""Export selected DIAG layers; all other tensors remain exactly baseline E1."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from safetensors.torch import save_file

from .run_state import atomic_write_json, read_json


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def materialize_candidate(*, checkpoints: dict[int, Path], output_dir: Path,
                          model_path: str, phasea_root: Path) -> Path:
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.build_llm import resolve_real_vllm_spec
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state, release_qwen3_moe_layer_state
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_fold import fold_fusable_moe_layer_state
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import build_moe_diag_state
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.moe_materialize import _state_to_tensors
    from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot

    _, base, _ = resolve_real_vllm_spec("E1", model_path=model_path, phasea_root=phasea_root)
    base_dir = Path(base.model_path).resolve()
    index = read_json(base_dir / "model.safetensors.index.json")
    out = Path(output_dir).resolve()
    if out == base_dir:
        raise ValueError("candidate output must not replace E1")
    provenance = {str(k): {"path": str(Path(v).resolve()), "sha256": sha256(v)}
                  for k, v in sorted(checkpoints.items())}
    expected = {"checkpoints": provenance, "base_dir": str(base_dir),
                "base_index_sha256": sha256(base_dir / "model.safetensors.index.json")}
    manifest_path = out / "candidate_manifest.json"
    if manifest_path.exists():
        saved = read_json(manifest_path)
        if any(saved[k] != v for k, v in expected.items()):
            raise RuntimeError("candidate provenance changed")
        for name, digest in saved["exported_sha256"].items():
            if sha256(out / name) != digest:
                raise RuntimeError(f"candidate export changed: {name}")
        return out
    out.mkdir(parents=True, exist_ok=True)
    replaced = {key for key in index["weight_map"]
                if any(key.startswith(f"model.layers.{layer}.") for layer in checkpoints)}
    replaced_shards = {index["weight_map"][key] for key in replaced}
    if any(shard in replaced_shards and key not in replaced
           for key, shard in index["weight_map"].items()):
        raise RuntimeError("expected one layer per baseline shard")
    for source in base_dir.iterdir():
        if not source.is_file() or source.name in replaced_shards or source.name == "model.safetensors.index.json":
            continue
        dest = out / source.name
        if not dest.exists():
            dest.symlink_to(source)
        elif not dest.is_symlink() or dest.resolve() != source:
            raise RuntimeError(f"unexpected existing candidate file {dest}")
    exported = {}
    snapshot = Path(resolve_local_snapshot(model_path))
    for layer, checkpoint in sorted(checkpoints.items()):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(payload["layer"]) != layer:
            raise ValueError("checkpoint layer mismatch")
        state = load_qwen3_moe_layer_state(snapshot, layer, "cpu")
        try:
            diag = build_moe_diag_state(state.spec, "fusable")
            diag.load_snapshot(payload["diag"])
            if not all(torch.isfinite(p).all() for p in diag.parameters()):
                raise ValueError("nonfinite DIAG")
            state = fold_fusable_moe_layer_state(state, diag, use_r64=False)
            tensors = _state_to_tensors(state)
            expected_keys = {key for key in replaced if key.startswith(f"model.layers.{layer}.")}
            if set(tensors) != expected_keys:
                raise RuntimeError("export tensor coverage mismatch")
            name = f"model-layer-{layer:05d}-of-00048.safetensors"
            save_file(tensors, out / name)
            exported[name] = sha256(out / name)
            for key in tensors:
                index["weight_map"][key] = name
        finally:
            release_qwen3_moe_layer_state(state)
    atomic_write_json(out / "model.safetensors.index.json", index)
    exported["model.safetensors.index.json"] = sha256(out / "model.safetensors.index.json")
    atomic_write_json(manifest_path, {**expected, "exported_sha256": exported,
                                   "status": "COMPLETE", "unchanged_layers": "symlink to exact E1 shards"})
    return out
