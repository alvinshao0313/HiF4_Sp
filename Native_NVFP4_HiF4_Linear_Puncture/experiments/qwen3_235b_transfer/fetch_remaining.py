"""Download remaining shards concurrently into the existing isolated HTTP cache."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download

from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.artifact import write_json
from .config import source_snapshot
from .fetch_model import ENDPOINT, MODEL, REVISION


def fetch(run_root: Path) -> None:
    root = run_root / "http_download"
    status_path = root / "parallel_status.json"
    if status_path.exists():
        raise FileExistsError(status_path)
    source = json.loads((root / "source_files.json").read_text())
    if source["revision"] != REVISION or source["model"] != MODEL:
        raise ValueError("download source differs from the pinned model")
    names = sorted(name for name in source["files"] if name.endswith(".safetensors"))
    # The original four-worker transfer owns the first four shards. Hub file
    # locks also prevent duplicate writes if it reaches a later shard first.
    names = names[4:]
    status = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "model": MODEL, "revision": REVISION, "files": names, "workers": 8}
    write_json(status, status_path)

    def download(name):
        path = Path(hf_hub_download(MODEL, name, revision=REVISION, endpoint=ENDPOINT,
                                   cache_dir=source_snapshot().parents[2]))
        if path.stat().st_size != source["files"][name]:
            raise ValueError(f"unexpected downloaded shard size: {name}")
        print(json.dumps({"event": "file_complete", "file": name, "bytes": path.stat().st_size}), flush=True)
        return name

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            completed = list(pool.map(download, names))
        status.update(status="complete", completed_files=completed)
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(status, status_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True, type=Path)
    fetch(parser.parse_args().run_root.resolve())
