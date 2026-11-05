"""Load the RUN's hashed Arrow files in both audit and actual main.py."""
from pathlib import Path

from .artifacts import read_json, sha256


def install(root):
    from datasets import Dataset, DatasetDict
    from lighteval.tasks.lighteval_task import LightevalTask
    root = Path(root)
    evidence = read_json(root / "datasets/manifest.json")
    routes = {entry["repo"]: entry for entry in evidence.values()}
    original = LightevalTask.download_dataset_worker

    def download(task):
        if task.dataset_path not in routes:
            return original(task)
        entry = routes[task.dataset_path]
        if task.dataset_revision != entry["task_revision"]:
            raise ValueError(f"task revision differs from preparation: {task.dataset_path}")
        data = {}
        for split, item in entry["splits"].items():
            path = root / item["path"]
            if sha256(path) != item["sha256"]:
                raise ValueError(f"pinned dataset changed: {path}")
            data[split] = Dataset.from_file(str(path))
        return DatasetDict(data)

    LightevalTask.download_dataset_worker = staticmethod(download)
