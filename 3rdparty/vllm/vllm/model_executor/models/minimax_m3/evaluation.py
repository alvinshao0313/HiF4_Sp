# SPDX-License-Identifier: Apache-2.0
"""MiniMax-specific LLM settings while retaining the existing lighteval runner."""

import json
from pathlib import Path


def require_complete_text_checkpoint(model_path):
    root = Path(model_path).resolve()
    config = json.loads((root / "config.json").read_text())
    if config.get("architectures") != ["MiniMaxM3SparseForConditionalGeneration"]:
        raise ValueError("Expected a local MiniMax-M3.1-Flash checkpoint")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    missing = sorted(name for name in shards if not (root / name).is_file())
    if missing:
        raise FileNotFoundError(f"Download incomplete: {len(missing)} weight shards missing; first: {missing[:3]}")
    return root


def engine_kwargs(model_path, tensor_parallel_size, max_model_len,
                  gpu_memory_utilization=0.9, max_num_seqs=1,
                  max_num_batched_tokens=2048, pipeline_parallel_size=1):
    return dict(model=str(model_path), tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        max_model_len=max_model_len, dtype="bfloat16", enforce_eager=True,
        block_size=128, kv_cache_dtype="auto", enable_prefix_caching=False,
        disable_hybrid_kv_cache_manager=True, linear_backend="emulation",
        moe_backend="emulation", gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens,
        trust_remote_code=False)


def generation_parameters(temperature, top_p, max_new_tokens):
    from lighteval.models.model_input import GenerationParameters
    # Both 0 and -1 disable top-k in vLLM; lighteval accepts only nonnegative values.
    return GenerationParameters(temperature=temperature, top_p=top_p,
        top_k=0, min_p=0.0, max_new_tokens=max_new_tokens)


def evaluate(args):
    # Imported here so the generation/check commands do not import lighteval.
    from main import CustomEvaluationTracker
    from lighteval.models.vllm.vllm_model import VLLMModel, VLLMModelConfig
    from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
    from vllm import LLM

    if args.dataset_manifest:
        from tools.minimax_m3_eval.data import install
        install(args.dataset_manifest)

    class MiniMaxVLLMModel(VLLMModel):
        def _create_auto_model(self, config):
            self.model_args = engine_kwargs(config.model_name, config.tensor_parallel_size,
                config.max_model_length, config.gpu_memory_utilization,
                config.max_num_seqs, config.max_num_batched_tokens,
                config.pipeline_parallel_size)
            self.model_args["seed"] = config.seed
            return LLM(**self.model_args)

    if args.max_samples is None and any(
            task.split("|")[0] == "mmlu_pro" for task in args.datasets.split(",")):
        args.max_samples = 300
    config = VLLMModelConfig(model_name=args.model_path, trust_remote_code=False,
        tensor_parallel_size=args.tensor_parallel_size, max_model_length=args.max_model_len,
        pipeline_parallel_size=args.pipeline_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=args.batch_size,
        max_num_batched_tokens=args.max_num_batched_tokens, dtype="bfloat16",
        override_chat_template=True, enforce_eager=True,
        generation_parameters=generation_parameters(args.temperature,
            args.top_p, args.max_new_tokens))
    model = MiniMaxVLLMModel(config)
    name = "MiniMax-M3.1-Flash"
    tracker = CustomEvaluationTracker(output_dir=args.output_dir, short_model_name=name,
        save_details=True)
    parameters = PipelineParameters(launcher_type=ParallelismManager.VLLM,
        max_samples=args.max_samples, custom_tasks_directory=str(args.repo_root / "tasks/custom_tasks.py"),
        reasoning_tags=[("<mm:think>", "</mm:think>")])
    pipeline = Pipeline(tasks=args.datasets, pipeline_parameters=parameters,
        evaluation_tracker=tracker, model=model)
    pipeline.evaluate()
    pipeline.show_results()
    tracker.general_config_logger.model_name = name
    pipeline.save_and_push_results()
