"""Release only this RUN's completed one-time diagnostic export shard links."""
from collections import defaultdict
from pathlib import Path

from .artifacts import now, read_json, sha256, verify_sources, write_json
from .config import MODELS, NATIVE, S1K


def diagnostic_shards(root):
    root = Path(root).resolve()
    verify_sources(root)
    models = (MODELS[0], MODELS[2])
    protected_paths = [NATIVE / "model.safetensors.index.json", root / "calibration/s1k_clean/manifest.json"]
    protected_paths += [d / f"{s}.pt" for d in (S1K / "calibration", root / "calibration/s1k_clean",
                       root / "calibration/native_traces") for s in ("train", "val")]
    protected = {str(p): sha256(p) for p in protected_paths}
    targets, inodes = [], defaultdict(list)
    for model in models:
        if not read_json(root / "qualification/inference" / model / "metrics.json")["passed"]:
            raise ValueError("diagnostic TP2 inference must complete before cleanup")
        directory = root / "qualification/models" / model
        export = read_json(directory / "export.json")
        if not export["complete"] or not export["qualification"]:
            raise ValueError("only completed diagnostic exports may be removed")
        index = read_json(directory / "model.safetensors.index.json")
        for name in sorted(set(index["weight_map"].values())):
            path = directory / name
            if (not path.is_file() or path.is_symlink()
                    or not path.resolve().is_relative_to(root / "qualification/models")):
                raise ValueError(f"unexpected diagnostic shard path: {path}")
            stat = path.stat()
            # The inference processes have exited and these immutable shard
            # hashes were checked at export. Later formal stages never read them.
            item = {"path": str(path), "logical_bytes": stat.st_size, "allocated_bytes": stat.st_blocks * 512,
                    "device": stat.st_dev, "inode": stat.st_ino, "link_count": stat.st_nlink,
                    "sha256_at_export": export["shards_sha256"][name],
                    "reason": "completed one-time diagnostic export; no later-stage weight reference"}
            targets.append(item)
            inodes[(stat.st_dev, stat.st_ino)].append(item)
    released = sum(values[0]["allocated_bytes"] for values in inodes.values() if len(values) == values[0]["link_count"])
    record = {"status": "planned", "created_at": now(), "targets": targets,
              "expected_released_unique_allocated_bytes": released, "protected": protected}
    write_json(record, root / "qualification/cleanup.json")
    for item in targets:
        Path(item["path"]).unlink()
    if any(Path(item["path"]).exists() for item in targets):
        raise RuntimeError("diagnostic shard cleanup is incomplete")
    if any(sha256(path) != checksum for path, checksum in protected.items()):
        raise RuntimeError("protected input changed during diagnostic cleanup")
    record.update(status="complete", finished_at=now(), released_unique_allocated_bytes=released)
    write_json(record, root / "qualification/cleanup.json")
