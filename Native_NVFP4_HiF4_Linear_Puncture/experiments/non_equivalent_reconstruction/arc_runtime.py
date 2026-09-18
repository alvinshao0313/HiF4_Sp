"""ARC evaluation with explicit ownership of vLLM processes and output thread."""
from __future__ import annotations

import os


def evaluate_arc(model, spec_path):
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.vllm_runner import (
        _patch_lm_eval_transformers, _pick_metric,
    )
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.lm_eval_vllm import (
        build_lm_eval_vllm_kwargs, patch_lm_eval_vllm_generate_compat,
    )
    _patch_lm_eval_transformers()
    patch_lm_eval_vllm_generate_compat()
    from lm_eval import simple_evaluate
    from lm_eval.models.vllm_causallms import VLLM

    os.environ["HIF4_RUNTIME_SPEC_PATH"] = str(spec_path.resolve())
    kwargs = build_lm_eval_vllm_kwargs(
        model_path=str(model), hif4_runtime_spec_path=str(spec_path),
        max_model_len=4096, max_num_batched_tokens=4096, seed=42,
    )
    lm = VLLM(batch_size="auto", **kwargs)
    core = lm.model.llm_engine.engine_core
    # This experiment uses vLLM's synchronous multiprocessing client for TP=2.
    output_thread = core.output_queue_thread
    try:
        result = simple_evaluate(
            model=lm, tasks=["arc_easy", "arc_challenge"],
            num_fewshot=0, batch_size="auto",
        )
    finally:
        core.shutdown(timeout=60)
        output_thread.join(timeout=60)
        if output_thread.is_alive():
            raise RuntimeError("vLLM output thread did not terminate after engine shutdown")
    print("ARC engine shutdown and output thread join completed", flush=True)
    scores = {}
    for task, task_result in result["results"].items():
        if not isinstance(task_result, dict):
            continue
        _, value = _pick_metric(task_result)
        if value is not None:
            scores[task] = value
    return {
        "backend": "lm_eval_vllm", "tensor_parallel_size": 2,
        "kv_cache_dtype": "bfloat16", "enforce_eager": True,
        "max_num_batched_tokens": 4096, "scores": scores,
        "raw_results": result["results"], "hif4_runtime_spec_path": str(spec_path),
    }
