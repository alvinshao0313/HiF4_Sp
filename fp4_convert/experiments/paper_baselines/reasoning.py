"""Seeded lighteval runs using main.py's tracker, isolated from active callers."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from . import entry  # Register quantization in both the parent and spawned workers.


def observe(output):
    from vllm import LLM

    original = LLM.generate
    directory = output / "generation_observations"
    directory.mkdir()
    batch = 0

    def generate(model, *args, **kwargs):
        nonlocal batch
        sampler = kwargs["sampling_params"]
        if sampler.n != 1:
            raise ValueError("Each paper round requires exactly one generation per question")
        result = original(model, *args, **kwargs)
        records = []
        for response in result:
            if len(response.outputs) != 1:
                raise ValueError("Unexpected generation count")
            answer = response.outputs[0]
            if answer.cumulative_logprob is not None and not math.isfinite(answer.cumulative_logprob):
                raise ValueError("Nonfinite generation logprob")
            records.append({"request_id": response.request_id, "prompt_token_ids": response.prompt_token_ids,
                            "output_tokens": len(answer.token_ids), "finish_reason": answer.finish_reason})
        fields = ("seed", "n", "temperature", "top_p", "top_k", "min_p", "presence_penalty",
                  "frequency_penalty", "repetition_penalty", "max_tokens", "stop")
        report = {"sampling": {name: getattr(sampler, name) for name in fields}, "responses": records}
        (directory / f"batch_{batch:05d}.json").write_text(json.dumps(report) + "\n")
        batch += 1
        return result

    LLM.generate = generate


def install_data(manifest):
    from datasets import Dataset, DatasetDict
    from lighteval.tasks.lighteval_task import LightevalTask

    routes = json.loads(Path(manifest).read_text())
    for route in routes.values():
        for split in route["splits"].values():
            if hashlib.sha256(Path(split["path"]).read_bytes()).hexdigest() != split["sha256"]:
                raise ValueError(f"Pinned task data changed: {split['path']}")

    def download(task):
        route = routes[task.dataset_path]
        if task.dataset_config_name != route["subset"]:
            raise ValueError("Task subset differs from the pinned dataset")
        task.dataset_revision = task.config.hf_revision = route["revision"]
        return DatasetDict({name: (Dataset.from_file(item["path"]) if item["format"] == "arrow"
                                  else Dataset.from_json(item["path"]))
                            for name, item in route["splits"].items()})

    LightevalTask.download_dataset_worker = staticmethod(download)


def evaluate(plan, config, task, seed, output, qualification=False, tp=2, max_samples=None):
    # Follow the existing seeded BF16 runner: keep shared main.py and active
    # experiments untouched while using its tracker and the same Pipeline.
    from main import CustomEvaluationTracker
    from lighteval.models.model_input import GenerationParameters
    from lighteval.models.vllm.vllm_model import VLLMModelConfig, VLLMModel
    from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

    spec = json.loads(plan.read_text())
    candidate = next(row for row in spec["configs"] if row["id"] == config)
    settings = spec["reasoning"][candidate["model"]][task]
    if max_samples is not None and (qualification or not 1 <= max_samples <= settings["count"]):
        raise ValueError("Use a positive in-range sample count independently of legacy qualification")
    sample_limit = 128 if qualification else max_samples
    install_data(plan.parent / "data/manifest.json")
    output.mkdir(parents=True, exist_ok=False)
    view = output / "model"
    view.mkdir()
    for file in Path(candidate["checkpoint"]).iterdir():
        if file.is_file() or file.name == "paper_metadata":
            (view / file.name).symlink_to(file.resolve())
    generation = {**settings["sampling"], "seed": seed,
                  "max_new_tokens": 8 if qualification else 32768}
    engine = VLLMModelConfig(
        model_name=str(view), trust_remote_code=True, tensor_parallel_size=tp,
        gpu_memory_utilization=.90, max_model_length=40960, dtype="auto",
        enforce_eager=False, kv_cache_dtype="bfloat16", max_num_seqs=128,
        max_num_batched_tokens=8192, seed=seed, override_chat_template=True,
        generation_parameters=GenerationParameters(**generation),
    )
    (output / "runtime_config.json").write_text(json.dumps(engine.model_dump(mode="json"), indent=2) + "\n")
    observe(output)
    # lighteval deletes LLM before scoring. Close its worker processes explicitly
    # before deletion, as in the qualified QA runner.
    cleanup = VLLMModel.cleanup

    def close(model):
        model.model.llm_engine.engine_core.shutdown(timeout=30)
        cleanup(model)

    VLLMModel.cleanup = close
    tracker = CustomEvaluationTracker(output_dir=str(output), short_model_name="model", save_details=True)
    pipeline = None
    try:
        pipeline = Pipeline(
            tasks=settings["task_name"] + "|0",
            pipeline_parameters=PipelineParameters(launcher_type=ParallelismManager.VLLM,
                                                   max_samples=sample_limit),
            evaluation_tracker=tracker, model_config=engine,
        )
        documents = [doc for docs in pipeline.sampling_docs.values() for doc in docs]
        expected = sample_limit if sample_limit is not None else settings["count"]
        if len(documents) != expected or any(doc.num_samples != 1 for doc in documents):
            raise ValueError("Wrong task coverage or number of generations per question")
        prompts = []
        for doc in documents:
            prompt = pipeline.model.prompt_manager.prepare_prompt(doc)
            tokens = pipeline.model.tokenizer.encode(prompt, add_special_tokens=False)
            if len(tokens) + 32768 > 40960:
                raise ValueError(f"Prompt would exceed the agreed context: {doc.id}")
            prompts.append({"id": str(doc.id), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                            "token_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                            "prompt_tokens": len(tokens), "gold": doc.get_golds()})
        (output / "prompt_audit.json").write_text(json.dumps(prompts, indent=2) + "\n")
        pipeline.evaluate()
        pipeline.show_results()
        tracker.general_config_logger.model_name = "model"
        pipeline.save_and_push_results()
    finally:
        if pipeline is not None and hasattr(pipeline.model, "model"):
            close(pipeline.model)
        VLLMModel.cleanup = cleanup


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("config")
    parser.add_argument("task")
    parser.add_argument("seed", type=int, choices=[1234, 1235, 1236])
    parser.add_argument("output", type=Path)
    parser.add_argument("--qualification", action="store_true")
    parser.add_argument("--tp", type=int, choices=(1, 2), default=2)
    parser.add_argument("--max-samples", type=int)
    evaluate(**vars(parser.parse_args()))
