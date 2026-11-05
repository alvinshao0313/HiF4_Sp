"""Finite HTTP range transfer with exact interval and final LFS SHA256 checks.

Each request is at most 64 MiB, avoiding a single multi-gigabyte connection.
An error stops the affected transfer; there is no automatic retry loop.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests
from huggingface_hub.utils import WeakFileLock

from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.artifact import sha256, write_json
from .config import source_snapshot
from .fetch_model import ENDPOINT, MODEL, REVISION

CHUNK_BYTES = 64 * 1024 * 1024


def download(name: str, expected: dict) -> dict:
    snapshot = source_snapshot()
    model_cache = snapshot.parents[1]
    cache = snapshot.parents[2]
    digest, size = expected["sha256"], expected["size"]
    blob = model_cache / "blobs" / digest
    incomplete = blob.with_name(blob.name + ".incomplete")
    lock = cache / ".locks" / model_cache.name / (digest + ".lock")
    destination = snapshot / name
    url = f"{ENDPOINT}/{MODEL}/resolve/{REVISION}/{name}"
    started = time.monotonic()
    with WeakFileLock(lock):
        if not blob.is_file():
            offset = incomplete.stat().st_size if incomplete.exists() else 0
            if offset > size:
                raise ValueError(f"partial file exceeds expected size: {name}")
            with requests.Session() as session:
                while offset < size:
                    stop = min(offset + CHUNK_BYTES, size) - 1
                    interval = f"bytes {offset}-{stop}/{size}"
                    with session.get(url, headers={"Range": f"bytes={offset}-{stop}"},
                                     stream=True, timeout=(15, 60)) as response:
                        if response.status_code != 206 or response.headers.get("Content-Range") != interval:
                            raise ValueError(f"unexpected range response for {name}: "
                                             f"status={response.status_code}, range={response.headers.get('Content-Range')}, "
                                             f"expected={interval}")
                        written = 0
                        with incomplete.open("ab") as stream:
                            for chunk in response.iter_content(1024 * 1024):
                                if written + len(chunk) > stop - offset + 1:
                                    raise ValueError(f"range body exceeds requested interval: {name}")
                                stream.write(chunk)
                                written += len(chunk)
                        if written != stop - offset + 1:
                            raise ValueError(f"incomplete range body: {name}: {written}")
                    offset = stop + 1
                    print(json.dumps({"event": "range_complete", "file": name, "bytes": offset,
                                      "expected_bytes": size, "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
            if incomplete.stat().st_size != size or sha256(incomplete) != digest:
                raise ValueError(f"LFS size or SHA256 mismatch: {name}")
            incomplete.replace(blob)
        elif blob.stat().st_size != size or sha256(blob) != digest:
            raise ValueError(f"existing LFS blob size or SHA256 mismatch: {name}")
        if destination.exists() or destination.is_symlink():
            if not destination.is_file() or sha256(destination) != digest:
                raise ValueError(f"existing snapshot file differs: {name}")
        else:
            destination.symlink_to(os.path.relpath(blob, destination.parent))
    result = {"file": name, "bytes": size, "sha256": digest}
    print(json.dumps({"event": "file_complete", **result}), flush=True)
    return result


def fetch(run_root: Path, output_dir: Path, first_shard: int, last_shard: int) -> None:
    lfs = json.loads((run_root / "source_lfs.json").read_text())
    if lfs["model"] != MODEL or lfs["revision"] != REVISION:
        raise ValueError("source LFS metadata differs from the pinned model")
    names = sorted(lfs["files"])
    if not 1 <= first_shard <= last_shard <= len(names):
        raise ValueError("invalid shard range")
    names = names[first_shard - 1:last_shard]
    output_dir.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "model": MODEL, "revision": REVISION, "files": names,
              "source_snapshot": str(source_snapshot()), "chunk_bytes": CHUNK_BYTES,
              "started_at": datetime.now(timezone.utc).isoformat()}
    write_json(status, output_dir / "status.json")
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            completed = list(pool.map(lambda name: download(name, lfs["files"][name]), names))
        status.update(status="complete", completed=completed)
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(status, output_dir / "status.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--first_shard", type=int, required=True)
    parser.add_argument("--last_shard", type=int, required=True)
    args = parser.parse_args()
    fetch(args.run_root.resolve(), args.output_dir.resolve(), args.first_shard, args.last_shard)
