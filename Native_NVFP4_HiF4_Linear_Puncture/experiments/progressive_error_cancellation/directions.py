"""Frozen direct/JVP directions and deterministic shuffled controls."""
from __future__ import annotations

import hashlib
import random
import shutil
from pathlib import Path

import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import (
    NativeQwen3MoELayerRuntime,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.moe_layer_runtime import (
    build_qwen3_moe_layer_call,
)
from .artifact import atomic_save, tensor_sha256, write_json, sha256
from .data import assemble
from .jvp import fixed_route_native_jvp


def _hash_ids(ids):
    return hashlib.sha256("\0".join(ids).encode()).hexdigest()


def direct_direction(previous_student, previous_native, samples):
    return {s.sample_id: (previous_student[s.sample_id].float() - previous_native[s.sample_id].float()).detach().cpu()
            for s in samples}


def build_direction(*, method, source_layer, runtime, native_hidden, student_hidden,
                    samples, snapshot, device, output_dir=None, shuffle_seed=4242,
                    batch_groups=None):
    """Build one immutable direction cache for the layer being optimized.

    `native_hidden` and `student_hidden` are the input boundary caches ``b_l``.
    Direct uses ``c_l`` itself; JVP transports ``c_l`` through the current
    native layer at ``b_l^native`` with native top-k IDs held fixed.
    """
    if source_layer < 0:
        raise ValueError("source_layer must be nonnegative")
    direct = direct_direction(student_hidden, native_hidden, samples)
    permutation = {s.sample_id: s.sample_id for s in samples}
    if method == "shuffled":
        direction = {}
        if batch_groups is None:
            groups = {"all": [samples]}
        else:
            groups = batch_groups
        for group_name, group_batches in groups.items():
            for batch_index, batch in enumerate(group_batches):
                ids = [s.sample_id for s in batch]
                shuffled = list(ids)
                token = f"{shuffle_seed}:{source_layer}:{group_name}:{batch_index}"
                seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little")
                random.Random(seed).shuffle(shuffled)
                for sid, donor in zip(ids, shuffled):
                    permutation[sid] = donor
                    direction[sid] = direct[donor].clone()
        if set(direction) != set(direct):
            raise RuntimeError("shuffled batch groups do not cover every sample")
    elif method == "direct":
        direction = direct
    elif method == "jvp":
        direction = {}
        runtime.eval()
        for sample in samples:
            sid = sample.sample_id
            x = native_hidden[sid].unsqueeze(0).to(device)
            v = direct[sid].unsqueeze(0).to(device=device, dtype=x.dtype)
            call = build_qwen3_moe_layer_call(str(snapshot), x)
            with torch.no_grad():
                reference = runtime(
                    x, attention_mask=call.attention_mask,
                    position_embeddings=call.position_embeddings,
                )
                selected = reference.selected_experts.detach()
            direction[sid] = fixed_route_native_jvp(
                runtime, x, v, call, selected.detach()
            )[0][0].detach().float().cpu()
    elif method == "baseline":
        direction = {s.sample_id: torch.zeros_like(direct[s.sample_id]) for s in samples}
    else:
        raise ValueError(method)
    for sid, value in direction.items():
        if value.ndim != 2 or not torch.isfinite(value).all():
            raise RuntimeError(f"invalid direction tensor for {sid}")
    manifest = {
        "status": "COMPLETE", "source_layer": source_layer,
        "target_layer": source_layer, "method": method,
        "boundary": "input_c_l",
        "shuffle_seed": shuffle_seed if method == "shuffled" else None,
        "shuffle_scope": "fixed_batch_permutation" if method == "shuffled" else None,
        "batch_groups": {
            name: [[s.sample_id for s in batch] for batch in groups]
            for name, groups in batch_groups.items()
        } if method == "shuffled" and batch_groups is not None else None,
        "sample_ids": [s.sample_id for s in samples],
        "sample_id_hash": _hash_ids([s.sample_id for s in samples]),
        "permutation": permutation,
        "samples": {},
        "jvp": {"derivative": "STE surrogate", "routing": "fixed native top-k"
                } if method == "jvp" else None,
    }
    for sid, value in direction.items():
        manifest["samples"][sid] = {"path": f"{sid}.pt", "shape": list(value.shape),
                                     "sha256": tensor_sha256(value),
                                     "norm": float(value.norm())}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        for sid, value in direction.items():
            atomic_save(value, output_dir / f"{sid}.pt")
        write_json(manifest, output_dir / "manifest.json")
    return direction, manifest


def load_direction(path, expected_manifest=None):
    path = Path(path)
    manifest = __import__("json").loads((path / "manifest.json").read_text())
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError("direction cache is incomplete")
    if expected_manifest is not None and manifest != expected_manifest:
        raise RuntimeError("direction manifest changed")
    result = {}
    for sid, row in manifest["samples"].items():
        value = torch.load(path / row["path"], map_location="cpu", weights_only=False)
        if list(value.shape) != row["shape"] or tensor_sha256(value) != row["sha256"]:
            raise RuntimeError(f"direction hash changed: {sid}")
        result[sid] = value
    return result, manifest


def compact_direction(path, direction, manifest):
    """Retain reproducible provenance after a layer leaves the active window."""
    path = Path(path)
    values = [value.float().reshape(-1) for _, value in sorted(direction.items())]
    flat = torch.cat(values) if values else torch.zeros(0)
    first_sid = manifest["sample_ids"][0] if manifest["sample_ids"] else None
    probe = direction[first_sid][: min(4, direction[first_sid].shape[0]), : min(16, direction[first_sid].shape[1])].float() if first_sid else torch.zeros(0)
    write_json({
        "status": "COMPACT",
        "manifest_sha256": sha256(path / "manifest.json"),
        "method": manifest["method"],
        "source_layer": manifest["source_layer"],
        "sample_id_hash": manifest["sample_id_hash"],
        "sample_count": len(direction),
        "global_fp32_sum": float(flat.sum()),
        "global_fp32_abs_sum": float(flat.abs().sum()),
        "global_l2": float(flat.norm()),
        "probe_sample_id": first_sid,
        "probe_shape": list(probe.shape),
        "probe_sha256": tensor_sha256(probe),
    }, path.parent / "direction_summary.json")
    if first_sid:
        atomic_save(probe, path.parent / "direction_probe.pt")
    shutil.rmtree(path)
