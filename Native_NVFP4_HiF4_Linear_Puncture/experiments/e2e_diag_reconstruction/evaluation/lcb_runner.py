"""LiveCodeBench evaluation through the shared vLLM/lighteval path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import DEFAULT_MODEL_PATH
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.vllm_runner import (
    cleanup_materialized_eval_spec,
    resolve_vllm_eval_spec,
    run_main_py_lighteval,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import ensure_dir, write_json


def run_livecodebench_vllm(
    *,
    variant: str,
    output_dir: Path,
    model_path: str = DEFAULT_MODEL_PATH,
    artifact_path: str | Path | None = None,
    artifact_diag_variant: str = "adopted",
    device: str = "cuda",
) -> dict[str, Any]:
    spec = resolve_vllm_eval_spec(
        variant=variant,
        model_path=model_path,
        artifact_path=artifact_path,
        artifact_diag_variant=artifact_diag_variant,
        output_dir=output_dir,
        device=device,
    )
    try:
        eval_root = ensure_dir(output_dir / "eval" / "livecodebench")
        results = run_main_py_lighteval(
            model_path=spec.model_path,
            output_dir=eval_root / "vllm_run",
            datasets="lcb:codegeneration_v6|0",
            max_samples=None,
            max_new_tokens=38912,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            fake_act_quant=spec.fake_act_quant,
            disable_thinking=False,
            hif4_runtime_spec_path=spec.hif4_runtime_spec_path,
            native_nvfp4=spec.native_nvfp4,
        )
        write_json(eval_root / "metrics.json", results)
        return results
    finally:
        cleanup_materialized_eval_spec(spec)
