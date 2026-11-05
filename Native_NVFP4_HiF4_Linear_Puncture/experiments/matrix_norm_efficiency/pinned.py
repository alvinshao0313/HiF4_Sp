"""Route evaluation exclusively to this RUN's audited Arrow files."""
from pathlib import Path

from .artifacts import read_json, sha256


def manifest(root):
    root = Path(root)
    value = read_json(root / "evaluation/data/manifest.json")
    if value.get("complete") is not True:
        raise ValueError("evaluation data preparation is incomplete")
    for task in value["tasks"].values():
        for split in task["splits"].values():
            path = (root / split["path"]).resolve()
            if not path.is_relative_to(root.resolve()) or sha256(path) != split["sha256"]:
                raise ValueError(f"evaluation Arrow changed or escaped RUN: {path}")
    return value


def install(root):
    from datasets import Dataset, DatasetDict
    from lighteval.tasks.lighteval_task import LightevalTask

    root = Path(root)
    evidence = manifest(root)
    routes = {entry["repo"]: entry for entry in evidence["tasks"].values()}

    def download(task):
        if task.dataset_path not in routes:
            raise ValueError(f"task is absent from this RUN's evaluation manifest: {task.dataset_path}")
        entry = routes[task.dataset_path]
        if (task.dataset_revision not in (entry["registry_revision"], entry["task_revision"])
                or task.dataset_config_name != entry["subset"]):
            raise ValueError("task revision or subset differs from audited preparation")
        # The shared registry leaves several revisions unset. Bind both the
        # loader and serialized task metadata to the actual audited revision.
        task.dataset_revision = entry["task_revision"]
        task.config.hf_revision = entry["task_revision"]
        data = {}
        for name, item in entry["splits"].items():
            path = root / item["path"]
            if sha256(path) != item["sha256"]:
                raise ValueError(f"evaluation Arrow changed: {path}")
            data[name] = Dataset.from_file(str(path))
        return DatasetDict(data)

    LightevalTask.download_dataset_worker = staticmethod(download)
