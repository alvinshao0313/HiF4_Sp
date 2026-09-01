"""Build in-memory semantic models and run ARC / MMLU-Pro / AIME."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.artifact import (
    load_and_apply_conversion_artifact,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import (
    DEFAULT_MODEL_PATH,
    E2ETrainConfig,
    TARGET_LINEAR_COUNT,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.semantic_hif4 import (
    enable_eval_weight_cache,
    iter_switchable_linears,
    set_layer_runtime_mode,
    upgrade_semantic_model_inplace,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.vllm_runner import (
    cleanup_materialized_eval_spec,
    resolve_vllm_eval_spec,
    run_arc_vllm,
    run_aime25_avg5_vllm,
    run_mmlu_pro_300_vllm,
    run_mmlu_vllm,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot
from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import ensure_dir, write_json
from Native_NVFP4_HiF4_Linear_Puncture.src.semantic_model import load_native_nvfp4_semantic_model

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.common import (
    REASONING_EVAL_NUM_GPUS,
    VLLM_TP2_EVAL_GROUPS,
    require_visible_cuda_count,
)

EVAL_VARIANTS = ("native_nvfp4", "direct_hif4", "r64_only", "artifact")


def _eval_group_metrics_path(output_dir: Path, group: str) -> Path:
    if group == "arc":
        return output_dir / "eval" / "arc" / "metrics.json"
    if group == "mmlu":
        return output_dir / "eval" / "mmlu" / "metrics.json"
    if group == "mmlu_pro_300":
        return output_dir / "eval" / "mmlu_pro" / "metrics.json"
    if group == "aime25_avg5":
        return output_dir / "eval" / "aime25" / "metrics.json"
    raise ValueError(f"unknown eval group {group!r}")


def _load_group_metrics(output_dir: Path, group: str) -> dict[str, Any] | None:
    path = _eval_group_metrics_path(output_dir, group)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _patch_lm_eval_transformers() -> None:
    import transformers

    orig = transformers.__class__.__getattr__

    def patched(self, name):  # noqa: ANN001
        if name == "AutoModelForVision2Seq":
            return transformers.AutoModelForImageTextToText
        return orig(self, name)

    transformers.__class__.__getattr__ = patched
    setattr(transformers, "AutoModelForVision2Seq", transformers.AutoModelForImageTextToText)


def build_eval_model(
    variant: str,
    device: torch.device | str,
    *,
    model_path: str = DEFAULT_MODEL_PATH,
    artifact_path: str | Path | None = None,
    artifact_diag_variant: str = "adopted",
):
    if variant not in EVAL_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    if variant == "artifact" and artifact_path is None:
        raise ValueError("--artifact_path is required for variant=artifact")
    snapshot = resolve_local_snapshot(model_path)
    model, _index = load_native_nvfp4_semantic_model(snapshot, device=device)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=True)
    if variant == "native_nvfp4":
        cfg = E2ETrainConfig.for_test(diag_mode="online", use_r64=False)
        n = upgrade_semantic_model_inplace(model, cfg)
        if n != TARGET_LINEAR_COUNT:
            raise RuntimeError(f"coverage {n} != {TARGET_LINEAR_COUNT}")
        for layer in model.model.layers:
            set_layer_runtime_mode(layer, "native_nvfp4")
        return model, tokenizer
    if variant == "direct_hif4":
        cfg = E2ETrainConfig.for_test(diag_mode="online", use_r64=False)
        upgrade_semantic_model_inplace(model, cfg)
        for layer in model.model.layers:
            set_layer_runtime_mode(layer, "hif4_eval")
            enable_eval_weight_cache(layer)
        return model, tokenizer
    if variant == "r64_only":
        cfg = E2ETrainConfig.for_test(diag_mode="online", use_r64=True)
        upgrade_semantic_model_inplace(model, cfg)
        for layer in model.model.layers:
            set_layer_runtime_mode(layer, "hif4_eval")
            enable_eval_weight_cache(layer)
        return model, tokenizer
    load_and_apply_conversion_artifact(model, artifact_path, diag_variant=artifact_diag_variant)
    if len(iter_switchable_linears(model)) != TARGET_LINEAR_COUNT:
        raise RuntimeError("artifact apply did not restore 252 wrappers")
    return model, tokenizer


def _pick_metric(task_result: dict) -> tuple[str, float | None]:
    for key in ("acc,none", "acc", "acc_norm,none", "acc_norm"):
        if key in task_result and isinstance(task_result[key], (int, float)):
            return key, float(task_result[key])
    return "", None


def run_arc(model, tokenizer, *, device: str, output_dir: Path) -> dict[str, Any]:
    _patch_lm_eval_transformers()
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=8, device=device)
    out = simple_evaluate(
        model=lm,
        tasks=["arc_easy", "arc_challenge"],
        num_fewshot=0,
        batch_size=8,
    )
    scores = {}
    for t, trez in out["results"].items():
        if isinstance(trez, dict):
            _k, v = _pick_metric(trez)
            if v is not None:
                scores[t] = v
    payload = {"scores": scores, "raw_results": out["results"]}
    ensure_dir(output_dir / "eval" / "arc")
    write_json(output_dir / "eval" / "arc" / "metrics.json", payload)
    return payload


def run_mmlu_pro_300(
    *,
    variant: str,
    output_dir: Path,
    model_path: str = DEFAULT_MODEL_PATH,
    artifact_path: str | Path | None = None,
    artifact_diag_variant: str = "adopted",
    device: str = "cuda",
) -> dict[str, Any]:
    require_visible_cuda_count(REASONING_EVAL_NUM_GPUS)
    return run_mmlu_pro_300_vllm(
        variant=variant,
        output_dir=output_dir,
        model_path=model_path,
        artifact_path=artifact_path,
        artifact_diag_variant=artifact_diag_variant,
        device=device,
    )


def run_aime25_avg5(
    *,
    variant: str,
    output_dir: Path,
    model_path: str = DEFAULT_MODEL_PATH,
    artifact_path: str | Path | None = None,
    artifact_diag_variant: str = "adopted",
    device: str = "cuda",
) -> dict[str, Any]:
    require_visible_cuda_count(REASONING_EVAL_NUM_GPUS)
    return run_aime25_avg5_vllm(
        variant=variant,
        output_dir=output_dir,
        model_path=model_path,
        artifact_path=artifact_path,
        artifact_diag_variant=artifact_diag_variant,
        device=device,
    )


def _set_eval_seed(eval_seed: int) -> None:
    # Seed only host RNGs here. Do not touch torch.cuda before vLLM starts:
    # in-process lm_eval → vLLM still forks TP workers, and a pre-initialized
    # CUDA context in the parent triggers "Cannot re-initialize CUDA in forked
    # subprocess". GPU seeding is owned by vLLM via its own seed kwarg.
    random.seed(eval_seed)
    torch.manual_seed(eval_seed)


def run_eval_groups(
    *,
    variant: str,
    groups: list[str],
    output_dir: Path,
    device: str,
    artifact_path: str | None,
    artifact_diag_variant: str = "adopted",
    model_path: str,
    eval_seed: int = 42,
) -> dict[str, Any]:
    if any(g in VLLM_TP2_EVAL_GROUPS for g in groups):
        require_visible_cuda_count(REASONING_EVAL_NUM_GPUS)
    _set_eval_seed(eval_seed)
    summary_path = output_dir / "eval" / "eval_summary.json"
    out: dict[str, Any] = {}
    if summary_path.is_file():
        existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(existing_summary, dict):
            out.update(existing_summary)
    out["variant"] = variant
    out["artifact_diag_variant"] = artifact_diag_variant
    out["eval_seed"] = int(eval_seed)
    spec = resolve_vllm_eval_spec(
        variant=variant,
        model_path=model_path,
        artifact_path=artifact_path,
        artifact_diag_variant=artifact_diag_variant,
        output_dir=output_dir,
        device=device,
    )
    try:
        if "arc" in groups:
            existing = _load_group_metrics(output_dir, "arc")
            if existing is not None:
                print(f"reuse existing ARC metrics: {output_dir / 'eval' / 'arc' / 'metrics.json'}", flush=True)
                out["arc"] = existing
            else:
                out["arc"] = run_arc_vllm(spec=spec, output_dir=output_dir, seed=int(eval_seed))
        if "mmlu" in groups:
            existing = _load_group_metrics(output_dir, "mmlu")
            if existing is not None:
                print(f"reuse existing MMLU metrics: {output_dir / 'eval' / 'mmlu' / 'metrics.json'}", flush=True)
                out["mmlu"] = existing
            else:
                out["mmlu"] = run_mmlu_vllm(spec=spec, output_dir=output_dir, seed=int(eval_seed))
        if "mmlu_pro_300" in groups:
            existing = _load_group_metrics(output_dir, "mmlu_pro_300")
            if existing is not None:
                print(
                    f"reuse existing MMLU-Pro metrics: {output_dir / 'eval' / 'mmlu_pro' / 'metrics.json'}",
                    flush=True,
                )
                out["mmlu_pro_300"] = existing
            else:
                out["mmlu_pro_300"] = run_mmlu_pro_300_vllm(
                    variant=variant,
                    output_dir=output_dir,
                    model_path=model_path,
                    artifact_path=artifact_path,
                    artifact_diag_variant=artifact_diag_variant,
                    device=device,
                )
        if "aime25_avg5" in groups:
            existing = _load_group_metrics(output_dir, "aime25_avg5")
            if existing is not None:
                print(
                    f"reuse existing AIME25 metrics: {output_dir / 'eval' / 'aime25' / 'metrics.json'}",
                    flush=True,
                )
                out["aime25_avg5"] = existing
            else:
                out["aime25_avg5"] = run_aime25_avg5_vllm(
                    variant=variant,
                    output_dir=output_dir,
                    model_path=model_path,
                    artifact_path=artifact_path,
                    artifact_diag_variant=artifact_diag_variant,
                    device=device,
                )
        ensure_dir(output_dir / "eval")
        write_json(summary_path, out)
        return out
    finally:
        cleanup_materialized_eval_spec(spec)
