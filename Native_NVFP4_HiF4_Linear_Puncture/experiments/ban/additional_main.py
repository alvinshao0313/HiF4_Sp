"""Pin the three supplemental task datasets, then use the existing Ban CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .artifact import sha256
from .evaluation_main import install_observations


def install_datasets(manifest):
    from datasets import Dataset, DatasetDict
    from lighteval.tasks.lighteval_task import LightevalTask

    routes = json.loads(Path(manifest).read_text(encoding="utf-8"))
    for route in routes.values():
        for split in route["splits"].values():
            if sha256(split["path"]) != split["sha256"]:
                raise ValueError(f"pinned task Arrow changed: {split['path']}")

    def download(task):
        if task.dataset_path not in routes:
            raise ValueError(f"dataset was not pinned for this run: {task.dataset_path}")
        route = routes[task.dataset_path]
        if route.get("revision") is not None and task.dataset_revision != route["revision"]:
            raise ValueError(f"task revision differs from the pinned dataset: {task.dataset_path}")
        return DatasetDict({key: Dataset.from_file(value["path"])
                            for key, value in route["splits"].items()})

    LightevalTask.download_dataset_worker = staticmethod(download)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    args, _ = parser.parse_known_args()
    install_datasets(args.dataset_manifest)
    install_observations(args.output_dir)
    remaining = sys.argv[1:]
    index = remaining.index("--dataset_manifest")
    del remaining[index:index + 2]
    sys.argv = [sys.argv[0], *remaining]
    from .reasoning_main import main as reasoning_main
    reasoning_main()


if __name__ == "__main__":
    main()
