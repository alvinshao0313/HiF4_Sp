"""Download the fixed public source checkpoint and record exact provenance."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.artifact import write_json

MODEL = "nvidia/Qwen3-235B-A22B-NVFP4"
REVISION = "21cfa2c9e152032eb60647ee7b46a2bbcd8d76d2"
ENDPOINT = "https://huggingface.co"


def fetch(run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    status_path = run_root / "download_status.json"
    if status_path.exists():
        raise FileExistsError(f"download status already exists: {status_path}")
    status = {"model": MODEL, "revision": REVISION, "status": "running",
              "started_at": datetime.now(timezone.utc).isoformat()}
    write_json(status, status_path)
    try:
        info = HfApi(endpoint=ENDPOINT).model_info(MODEL, revision=REVISION, files_metadata=True)
        if info.sha != REVISION:
            raise ValueError("source revision differs from the pinned revision")
        expected = {entry.rfilename: entry.size for entry in info.siblings}
        write_json({"model": MODEL, "revision": REVISION, "files": expected,
                    "total_bytes": sum(expected.values())}, run_root / "source_files.json")
        snapshot = Path(snapshot_download(MODEL, revision=REVISION, endpoint=ENDPOINT,
                                          max_workers=4))
        for name, size in expected.items():
            if (snapshot / name).stat().st_size != size:
                raise ValueError(f"downloaded file has incorrect size: {name}")
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        for shard in set(index["weight_map"].values()):
            if not (snapshot / shard).is_file():
                raise FileNotFoundError(shard)
        status.update(status="complete", snapshot=str(snapshot),
                      completed_at=datetime.now(timezone.utc).isoformat())
        print(json.dumps(status), flush=True)
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(status, status_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True, type=Path)
    fetch(parser.parse_args().run_root)
