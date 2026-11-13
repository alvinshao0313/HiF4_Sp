"""One BF16 round using the root entry's tracker and the same lighteval path.

Keep round-specific engine seeds here: shared main.py has no seed CLI and is
still a dependency of other active experiments. Prompt/option RNG stays 1234.
"""
import argparse
import json
from pathlib import Path

from .download import save


def evaluate(root, phase):
    from main import CustomEvaluationTracker, get_short_model_name
    from lighteval.models.model_input import GenerationParameters
    from lighteval.models.vllm.vllm_model import VLLMModelConfig
    from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

    manifest = json.loads((root / "manifest.json").read_text())
    config = manifest["phases"][phase]
    directory = root / phase
    model = str(directory / "model")
    short_name = get_short_model_name(model)
    tracker = CustomEvaluationTracker(output_dir=str(directory / "evaluation"),
                                     short_model_name=short_name, save_details=True)
    model_config = VLLMModelConfig(
        model_name=model, trust_remote_code=True, tensor_parallel_size=8,
        gpu_memory_utilization=0.45, max_model_length=40960, dtype="auto",
        enforce_eager=True, kv_cache_dtype="bfloat16", kv_cache_memory_bytes=20 * 1024**3,
        max_num_seqs=128, max_num_batched_tokens=4096, seed=config["seed"],
        generation_parameters=GenerationParameters(
            temperature=0.6, top_p=0.95, top_k=20, min_p=0,
            max_new_tokens=config["max_new_tokens"],
        ),
    )
    save(directory / "runtime_config.json", model_config.model_dump(mode="json"))
    print(f"GPQA round={phase}, engine_seed={config['seed']}, n=1, questions={config['questions']}", flush=True)
    pipeline = Pipeline(
        tasks="gpqa_diamond_qwen_bf16_single|0",
        pipeline_parameters=PipelineParameters(
            launcher_type=ParallelismManager.VLLM,
            max_samples=config["questions"] if config["kind"] == "probe" else None,
            custom_tasks_directory=str(Path(__file__).with_name("task.py")),
        ), evaluation_tracker=tracker, model_config=model_config,
    )
    pipeline.evaluate()
    pipeline.show_results()
    tracker.general_config_logger.model_name = short_name
    pipeline.save_and_push_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    args = parser.parse_args()
    evaluate(args.run_dir.resolve(), args.phase)
