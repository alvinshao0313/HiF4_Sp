"""Experiment constants and read-only identities of existing artifacts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from ..qwen3_mxfp4.artifacts import REPO, file_hash, save

HERE = Path(__file__).resolve().parent
RESULTS = REPO / "Native_NVFP4_HiF4_Linear_Puncture/results"
RUNS = {
    "fht32": RESULTS / "qwen3_mxfp4/fht32_tp4_20261110T032038Z",
    "r16x2": RESULTS / "qwen3_mxfp4/r16x2_gpu03_20261110T040449Z",
}
MODELS = {name: root / "checkpoints" / name for name, root in RUNS.items()}
TASKS = {
    "mmlu_pro": {"tag": "rotation_mmlu_pro_avg10|0", "max_new_tokens": 32768},
    "lcb": {"tag": "rotation_lcb_v6_avg10|0", "max_new_tokens": 38912},
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def model_view(source, target):
    target.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.is_file() and path.name != "conversion.json":
            (target / path.name).symlink_to(path.resolve())


def install_datasets(manifest):
    """Load exactly the pinned Arrow bytes; never use offline revision fallback."""
    from datasets import Dataset, DatasetDict
    from lighteval.tasks.lighteval_task import LightevalTask

    routes = read(manifest)
    for route in routes.values():
        for split in route["splits"].values():
            if file_hash(split["path"]) != split["sha256"]:
                raise ValueError(f"Pinned Arrow changed: {split['path']}")

    def download(task):
        route = routes[task.dataset_path]
        if task.dataset_revision != route.get("revision") or task.dataset_config_name != route["subset"]:
            raise ValueError("Task revision/subset differs from the pinned inputs")
        return DatasetDict({key: Dataset.from_file(value["path"])
                            for key, value in route["splits"].items()})

    LightevalTask.download_dataset_worker = staticmethod(download)


def observe_generation(output):
    """Record real request tokens and finish reasons without changing engine kwargs."""
    from vllm import LLM

    original = LLM.generate
    root = Path(output) / "generation_observations"
    root.mkdir(parents=True, exist_ok=False)
    batch = 0

    def generate(self, *args, **kwargs):
        nonlocal batch
        result = original(self, *args, **kwargs)
        save(root / f"batch_{batch:05d}.json", [
            {"request_id": row.request_id, "prompt_token_ids": row.prompt_token_ids,
             "outputs": [{"index": value.index, "output_tokens": len(value.token_ids),
                          "finish_reason": value.finish_reason, "stop_reason": value.stop_reason}
                         for value in row.outputs]} for row in result])
        batch += 1
        return result

    LLM.generate = generate
