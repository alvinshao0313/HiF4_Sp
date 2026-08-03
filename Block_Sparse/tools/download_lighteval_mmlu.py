#!/usr/bin/env python3
"""Download lighteval/mmlu subsets from official Hugging Face Hub (no mirror).

By default skips configs already present in the local datasets cache.
Pass --force to re-download everything.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def cached_configs() -> set[str]:
    cache = Path.home() / ".cache" / "huggingface" / "datasets" / "lighteval___mmlu"
    if not cache.is_dir():
        return set()
    return {p.name for p in cache.iterdir() if p.is_dir()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download all subsets even if cached.",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        default=True,
        help="Only download missing subsets (default).",
    )
    args = parser.parse_args()

    os.environ.pop("HF_ENDPOINT", None)
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    print("HF_ENDPOINT=", repr(os.environ.get("HF_ENDPOINT")), flush=True)

    from datasets import get_dataset_config_names, load_dataset

    names = get_dataset_config_names("lighteval/mmlu")
    have = cached_configs()
    if args.force:
        todo = list(names)
    else:
        todo = [n for n in names if n not in have]

    print(f"total subsets: {len(names)}", flush=True)
    print(f"already cached: {len(have)}", flush=True)
    print(f"to download: {len(todo)}", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return 0

    ok = 0
    fail: list[tuple[str, str]] = []
    for i, name in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] loading {name}", flush=True)
        try:
            load_dataset("lighteval/mmlu", name)
            ok += 1
        except Exception as e:
            fail.append((name, repr(e)))
            print(f"FAIL {name}: {e}", flush=True)

    print(f"done ok={ok} fail={len(fail)}", flush=True)
    for name, err in fail:
        print(f"  - {name}: {err}", flush=True)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
