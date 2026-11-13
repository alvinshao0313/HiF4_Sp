"""Verify cached weights against a pinned official ModelScope file manifest."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "download_manifest.json").read_text())
    destination = Path(manifest["snapshot"])
    destination.mkdir(parents=True, exist_ok=False)
    old = Path(manifest["existing_cache"])
    state = {"status": "running", "pid": os.getpid(),
             "started_at": datetime.now(timezone.utc).isoformat()}
    save(root / "download_status.json", state)

    def fetch(entry):
        name, expected = entry["Path"], entry["Sha256"]
        target, cached = destination / name, old / name
        reuse = (name.endswith(".safetensors") and cached.is_file()
                 and cached.stat().st_size == entry["Size"] and sha256(cached) == expected)
        if reuse:
            target.symlink_to(cached)
            source = "verified_existing_weight"
        else:
            partial = target.with_suffix(target.suffix + ".partial")
            url = (f"https://modelscope.cn/models/{manifest['repo_id']}/resolve/"
                   f"{manifest['revision']}/{name}")
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                            "--retry", "2", "--connect-timeout", "30",
                            "--output", str(partial), url], check=True)
            if partial.stat().st_size != entry["Size"] or sha256(partial) != expected:
                raise ValueError(f"Official file size/hash mismatch: {name}")
            partial.replace(target)
            source = "downloaded_official"
        result = {"name": name, "size": entry["Size"], "sha256": expected, "source": source}
        print(json.dumps(result), flush=True)
        return result

    try:
        small = [entry for entry in manifest["files"] if not entry["Path"].endswith(".safetensors")]
        weights = [entry for entry in manifest["files"] if entry["Path"].endswith(".safetensors")]
        records = [fetch(entry) for entry in small]
        save(root / "metadata_ready.json", {"snapshot": str(destination), "files": records})
        with ThreadPoolExecutor(max_workers=2) as pool:
            records.extend(pool.map(fetch, weights))
        from safetensors import safe_open

        dtype_counts = {}
        tensor_names = set()
        for entry in weights:
            with safe_open(destination / entry["Path"], framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if name in tensor_names:
                        raise ValueError(f"Duplicate tensor: {name}")
                    tensor_names.add(name)
                    dtype = handle.get_slice(name).get_dtype()
                    dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        index = json.loads((destination / "model.safetensors.index.json").read_text())
        config = json.loads((destination / "config.json").read_text())
        if tensor_names != set(index["weight_map"]) or set(dtype_counts) != {"BF16"}:
            raise ValueError("Incomplete tensor coverage or non-BF16 checkpoint")
        if config["torch_dtype"] != "bfloat16" or "quantization_config" in config:
            raise ValueError("Expected the unquantized official BF16 config")
        save(root / "model_verification.json", {"files": records, "tensor_dtypes": dtype_counts,
             "tensor_count": len(tensor_names), "index_coverage_verified": True,
             "revision": manifest["revision"], "snapshot": str(destination)})
        state.update(status="completed", exit_code=0, files=len(records), tensor_dtypes=dtype_counts)
    except Exception as exc:
        state.update(status="failed", exit_code=1, error=repr(exc))
        raise
    finally:
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        save(root / "download_status.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    download(parser.parse_args().run_dir)
