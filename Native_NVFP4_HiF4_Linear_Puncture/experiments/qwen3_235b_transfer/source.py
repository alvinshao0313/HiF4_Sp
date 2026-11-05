"""Validate the pinned source files actually needed by the current stage."""
from __future__ import annotations

import json
from pathlib import Path

from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.artifact import sha256
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot
from .config import source_snapshot
from .fetch_model import MODEL, REVISION

SMOKE_PREFIXES = ("model.embed_tokens.", "model.layers.0.")


def check_source(run_root: Path, *, smoke: bool) -> dict:
    selection = json.loads((run_root / "source_selection.json").read_text())
    if selection.get("revision") != REVISION or selection.get("source_snapshot") != str(source_snapshot()):
        raise ValueError("source selection differs from the pinned snapshot")
    lfs = json.loads((run_root / "source_lfs.json").read_text())
    if lfs["revision"] != REVISION or lfs["model"] != MODEL:
        raise ValueError("LFS checksums differ from the pinned source")
    prefixes = SMOKE_PREFIXES if smoke else None
    snapshot = resolve_local_snapshot(str(source_snapshot()), tensor_prefixes=prefixes)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted({shard for key, shard in index.items() if prefixes is None or key.startswith(prefixes)})
    checked = {}
    for shard in shards:
        path = snapshot / shard
        expected = lfs["files"][shard]
        if path.stat().st_size != expected["size"]:
            raise ValueError(f"incomplete source shard: {shard}")
        digest = sha256(path)
        if digest != expected["sha256"]:
            raise ValueError(f"source shard checksum mismatch: {shard}")
        checked[shard] = digest
    return {"source_snapshot": str(snapshot), "revision": REVISION,
            "tensor_prefixes": prefixes, "verified_shards_sha256": checked,
            "full_model_download_complete": not smoke}
