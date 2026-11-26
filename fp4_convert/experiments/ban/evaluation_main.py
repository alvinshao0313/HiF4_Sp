"""Run the existing reasoning CLI against a pinned local GSM8K test split."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .artifact import sha256, write_json


def install_dataset(path, expected_sha256):
    from datasets import Dataset, DatasetDict
    from lighteval.tasks.lighteval_task import LightevalTask

    path = Path(path).resolve()
    if sha256(path) != expected_sha256:
        raise ValueError("GSM8K test Arrow differs from the evaluation manifest")
    original = LightevalTask.download_dataset_worker

    def download(task):
        if task.dataset_path == "openai/gsm8k":
            return DatasetDict({"test": Dataset.from_file(str(path))})
        return original(task)

    LightevalTask.download_dataset_worker = staticmethod(download)


def install_observations(output):
    from vllm import LLM

    initialize, generate = LLM.__init__, LLM.generate
    root = Path(output).resolve() / "generation_observations"
    root.mkdir(parents=True, exist_ok=False)
    batch = 0

    def init(self, *args, **kwargs):
        # Ban counters must see every input position. Apply the same cache
        # setting to all three groups so prefix reuse cannot alter coverage.
        kwargs["enable_prefix_caching"] = False
        initialize(self, *args, **kwargs)

    def run(self, *args, **kwargs):
        nonlocal batch
        result = generate(self, *args, **kwargs)
        rows = [{"request_id": request.request_id,
                 "prompt_token_ids": request.prompt_token_ids,
                 "outputs": [{"index": value.index, "output_tokens": len(value.token_ids),
                              "finish_reason": value.finish_reason, "stop_reason": value.stop_reason}
                             for value in request.outputs]}
                for request in result]
        write_json(root / f"batch_{batch:04d}.json", rows)
        batch += 1
        return result

    LLM.__init__, LLM.generate = init, run


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gsm8k_arrow", required=True)
    parser.add_argument("--gsm8k_sha256", required=True)
    args, remaining = parser.parse_known_args()
    install_dataset(args.gsm8k_arrow, args.gsm8k_sha256)
    forwarded = argparse.ArgumentParser(add_help=False)
    forwarded.add_argument("--output_dir", required=True)
    base, _ = forwarded.parse_known_args(remaining)
    install_observations(base.output_dir)
    sys.argv = [sys.argv[0], *remaining]
    from .reasoning_main import main as reasoning_main
    reasoning_main()


if __name__ == "__main__":
    main()
