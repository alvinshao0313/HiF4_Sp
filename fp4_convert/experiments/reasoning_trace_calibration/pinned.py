"""Bind lighteval task loaders to the audited Arrow contents."""
from .artifacts import read_json, sha256
from .config import MMLU_DATA
from .sources import evaluation_questions


def install():
    from datasets import Dataset, DatasetDict
    from lighteval.tasks.lighteval_task import LightevalTask
    _, evidence = evaluation_questions()
    routes = {"TIGER-Lab/MMLU-Pro": ("mmlu_pro", "test"), "openai/gsm8k": ("gsm8k", "test"),
              "lighteval/code_generation_lite": ("lcb", "test"), "yentinglin/aime_2025": ("aime25", "train")}
    original = LightevalTask.download_dataset_worker

    def download(task):
        if task.dataset_path not in routes:
            return original(task)
        key, split = routes[task.dataset_path]
        entry = evidence[key]
        if sha256(entry["path"]) != entry["sha256"]:
            raise ValueError("audited task Arrow changed")
        data = {split: Dataset.from_file(entry["path"])}
        if key == "mmlu_pro":
            manifest = read_json(MMLU_DATA / "dataset_manifest.json")
            if task.dataset_revision != manifest["revision"]:
                raise ValueError("MMLU-Pro task revision differs from the confirmed protocol")
            validation = MMLU_DATA / "data/validation.arrow"
            if sha256(validation) != manifest["splits"]["validation"]["sha256"]:
                raise ValueError("pinned MMLU validation changed")
            data["validation"] = Dataset.from_file(str(validation))
        return DatasetDict(data)

    LightevalTask.download_dataset_worker = staticmethod(download)
