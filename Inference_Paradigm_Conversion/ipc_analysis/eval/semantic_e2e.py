"""Semantic E2E: experiment-side QDQ + standard GEMM (not runtime kernel perf)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

REPO_ROOT = Path("/home/shaoyuantian/program/HiF4_Sp")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# transformers AutoModelForVision2Seq compatibility for older lm_eval
import transformers  # noqa: E402

_orig = transformers.__class__.__getattr__


def _patched(self, name):  # noqa: ANN001
    if name == "AutoModelForVision2Seq":
        return transformers.AutoModelForImageTextToText
    return _orig(self, name)


transformers.__class__.__getattr__ = _patched
setattr(transformers, "AutoModelForVision2Seq", transformers.AutoModelForImageTextToText)

from lm_eval import simple_evaluate  # noqa: E402
from lm_eval.models.huggingface import HFLM  # noqa: E402

from Inference_Paradigm_Conversion.ipc_analysis.analysis.network_injection import (  # noqa: E402
    convert_linear_weight_inplace,
    install_activation_qdq_hooks,
    install_p2_activation_hooks,
)
from Inference_Paradigm_Conversion.ipc_analysis.config import (  # noqa: E402
    load_experiment_config,
    resolve_activation_scale_file,
)
from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import (  # noqa: E402
    load_nvfp4_activation_scales,
)
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import (  # noqa: E402
    atomic_write_json,
    ensure_dir,
)


def convert_all_linears_to_hif4(model: nn.Module) -> list[str]:
    converted = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "lm_head" not in name:
            convert_linear_weight_inplace(mod)
            converted.append(name)
    return converted


def apply_semantic_path(
    model: nn.Module,
    *,
    role: str,
    path_id: str,
    scales: dict[str, torch.Tensor] | None,
) -> list[Any]:
    """role: source|target. Returns hook handles (keep alive on caller)."""
    if path_id == "P1_semantic":
        if role == "target":
            convert_all_linears_to_hif4(model)
        return install_activation_qdq_hooks(model, path_id="P1_semantic", scales=None)
    if path_id == "P2_matched_semantic":
        if scales is None:
            raise ValueError("P2_matched_semantic requires NVFP4 activation scales")
        converted: set[str] = set()
        if role == "target":
            converted = set(convert_all_linears_to_hif4(model))
        matched = {
            n
            for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and "lm_head" not in n
        }
        return install_p2_activation_hooks(
            model,
            scales=scales,
            converted_names=converted,
            matched_names=matched,
        )
    raise ValueError(path_id)


def _pick_metric(task_result: dict) -> tuple[str, float | None]:
    for key in ("acc,none", "acc", "acc_norm,none", "acc_norm"):
        if key in task_result and isinstance(task_result[key], (int, float)):
            return key, float(task_result[key])
    return "", None


def run_lm_eval_arc(
    model,
    tokenizer,
    *,
    tasks: list[str],
    batch_size: str,
    device: str,
) -> dict[str, Any]:
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=device)
    out = simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
        batch_size=batch_size,
    )
    results = out["results"]
    scores: dict[str, float] = {}
    keys: dict[str, str] = {}
    for t, trez in results.items():
        if not isinstance(trez, dict):
            continue
        k, v = _pick_metric(trez)
        if v is not None:
            scores[t] = v
            keys[t] = k
    return {"scores": scores, "metric_keys": keys, "raw_results": results}


def run_one(
    *,
    path_id: str,
    role: str,
    device: str,
    batch_size: str,
    tasks: list[str],
    out_dir: Path,
) -> dict[str, Any]:
    cfg = load_experiment_config()
    ckpt = cfg.source_checkpoint_path()
    device_t = torch.device(device)
    if device_t.type == "cuda":
        torch.cuda.set_device(device_t)
    torch.set_num_threads(4)

    from Inference_Paradigm_Conversion.ipc_analysis.capture.model_adapter import (
        load_source_model_for_capture,
    )

    model, tok = load_source_model_for_capture(ckpt, device=device_t)
    scales = None
    if path_id == "P2_matched_semantic":
        scales = load_nvfp4_activation_scales(resolve_activation_scale_file(ckpt))
        scales = {k: v.to(device_t) for k, v in scales.items()}

    handles = apply_semantic_path(model, role=role, path_id=path_id, scales=scales)
    model._ipc_e2e_handles = handles  # type: ignore[attr-defined]

    eval_out = run_lm_eval_arc(
        model, tok, tasks=tasks, batch_size=batch_size, device=str(device_t)
    )
    record = {
        "e2e_kind": "semantic_e2e",
        "path_id": path_id,
        "role": role,
        "checkpoint": str(ckpt),
        "tasks": tasks,
        "scores": eval_out["scores"],
        "metric_keys": eval_out["metric_keys"],
        "device": str(device_t),
        "note": "QDQ + BF16 GEMM; not runtime kernel",
    }
    out_dir = ensure_dir(out_dir)
    atomic_write_json(out_dir / f"semantic_{path_id}_{role}.json", record)
    del model
    if device_t.type == "cuda":
        torch.cuda.empty_cache()
    return record


@torch.no_grad()
def _run_checkpoint_variant_arc(
    checkpoint: Path,
    *,
    variant: str,
    device: str,
    batch_size: str,
    tasks: list[str],
    all_layer_policy_path: Path | None = None,
    all_layer_scales_path: Path | None = None,
) -> dict[str, Any]:
    from Inference_Paradigm_Conversion.ipc_analysis.capture.model_adapter import (
        load_source_model_for_capture,
    )
    from Inference_Paradigm_Conversion.ipc_analysis.analysis.hif4_scaling_pipeline import (
        apply_all_layer_policy_inplace,
    )

    device_t = torch.device(device)
    if device_t.type == "cuda":
        torch.cuda.set_device(device_t)
    model, tok = load_source_model_for_capture(checkpoint, device=device_t)
    scales = load_nvfp4_activation_scales(resolve_activation_scale_file(checkpoint))
    scales = {k: v.to(device_t) for k, v in scales.items()}
    matched = {
        n
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and "lm_head" not in n
    }
    if variant == "source_nvfp4":
        converted: set[str] = set()
    elif variant == "standard_hif4":
        converted = set(convert_all_linears_to_hif4(model))
    elif variant == "optimized_hif4":
        if all_layer_policy_path is None or all_layer_scales_path is None:
            raise ValueError("optimized_hif4 requires all-layer policy/scales")
        apply_all_layer_policy_inplace(model, all_layer_policy_path, all_layer_scales_path)
        converted = matched
    else:
        raise ValueError(variant)
    handles = install_p2_activation_hooks(
        model,
        scales=scales,
        converted_names=converted,
        matched_names=matched,
    )
    model._ipc_e2e_handles = handles  # type: ignore[attr-defined]
    eval_out = run_lm_eval_arc(
        model,
        tok,
        tasks=tasks,
        batch_size=batch_size,
        device=str(device_t),
    )
    record = {
        "e2e_kind": "semantic_e2e",
        "variant": variant,
        "checkpoint": str(checkpoint),
        "tasks": tasks,
        "scores": eval_out["scores"],
        "metric_keys": eval_out["metric_keys"],
        "device": str(device_t),
        "note": "QDQ + BF16 GEMM semantic evaluation; optimized path uses offline folded parameters only",
    }
    for handle in handles:
        handle.remove()
    del model
    if device_t.type == "cuda":
        torch.cuda.empty_cache()
    return record


@torch.no_grad()
def _optimized_cache_consistency_smoke(
    checkpoint: Path,
    *,
    device: str,
    all_layer_policy_path: Path,
    all_layer_scales_path: Path,
    steps: int = 4,
) -> dict[str, Any]:
    from Inference_Paradigm_Conversion.ipc_analysis.capture.model_adapter import (
        load_source_model_for_capture,
    )
    from Inference_Paradigm_Conversion.ipc_analysis.analysis.hif4_scaling_pipeline import (
        apply_all_layer_policy_inplace,
    )
    from Inference_Paradigm_Conversion.ipc_analysis.capture.prompts import discovery_items

    device_t = torch.device(device)
    if device_t.type == "cuda":
        torch.cuda.set_device(device_t)
    model, tok = load_source_model_for_capture(checkpoint, device=device_t)
    apply_all_layer_policy_inplace(model, all_layer_policy_path, all_layer_scales_path)
    scales = load_nvfp4_activation_scales(resolve_activation_scale_file(checkpoint))
    scales = {k: v.to(device_t) for k, v in scales.items()}
    matched = {
        n
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and "lm_head" not in n
    }
    handles = install_p2_activation_hooks(
        model,
        scales=scales,
        converted_names=matched,
        matched_names=matched,
    )
    prompt = discovery_items(8)[0].text
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=256)
    prefix = enc["input_ids"].to(device_t)
    attention = enc.get("attention_mask", torch.ones_like(prefix)).to(device_t)

    no_cache_logits = model(input_ids=prefix, attention_mask=attention, use_cache=False).logits[:, -1, :]
    warm = model(input_ids=prefix, attention_mask=attention, use_cache=True)
    cache_logits = warm.logits[:, -1, :]
    past = warm.past_key_values
    rows: list[dict[str, Any]] = []
    current_prefix = prefix
    current_attention = attention
    for step in range(steps):
        diff = (no_cache_logits.float() - cache_logits.float())
        denom = no_cache_logits.float().pow(2).mean().clamp_min(1e-12)
        nmse = float((diff.pow(2).mean() / denom).item())
        cosine = float(torch.nn.functional.cosine_similarity(
            no_cache_logits.float(), cache_logits.float(), dim=-1
        ).mean().item())
        top1_equal = bool(
            torch.equal(
                no_cache_logits.argmax(dim=-1), cache_logits.argmax(dim=-1)
            )
        )
        allclose = bool(torch.allclose(no_cache_logits, cache_logits, rtol=2e-2, atol=2e-2))
        rows.append(
            {
                "step": step,
                "nmse": nmse,
                "cosine": cosine,
                "max_abs": float(diff.abs().max().item()),
                "top1_equal": top1_equal,
                "allclose_rtol2e-2_atol2e-2": allclose,
            }
        )
        next_id = no_cache_logits.argmax(dim=-1, keepdim=True)
        current_prefix = torch.cat([current_prefix, next_id], dim=-1)
        current_attention = torch.cat(
            [current_attention, torch.ones_like(next_id, dtype=current_attention.dtype)], dim=-1
        )
        no_cache_logits = model(
            input_ids=current_prefix,
            attention_mask=current_attention,
            use_cache=False,
        ).logits[:, -1, :]
        cache_out = model(
            input_ids=next_id,
            attention_mask=current_attention,
            past_key_values=past,
            use_cache=True,
        )
        past = cache_out.past_key_values
        cache_logits = cache_out.logits[:, -1, :]
    passed = all(r["top1_equal"] and r["cosine"] >= 0.999 for r in rows)
    for handle in handles:
        handle.remove()
    del model
    if device_t.type == "cuda":
        torch.cuda.empty_cache()
    return {"steps": rows, "passed": bool(passed)}


@torch.no_grad()
def run_equalized_arc_e2e(
    checkpoint: Path,
    *,
    run_dir: Path,
    device: str,
    all_layer_policy_path: Path,
    all_layer_scales_path: Path,
    batch_size: str = "4",
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    run_dir = ensure_dir(Path(run_dir))
    tasks = ["arc_easy", "arc_challenge"]
    with Path(all_layer_policy_path).open("r", encoding="utf-8") as f:
        policy = json.load(f)
    if bool(policy.get("target_trajectory", {}).get("target_trajectory_regression", False)):
        trajectory_warning = True
    else:
        trajectory_warning = False
    o_enabled = any(
        "o_in" in domains for domains in policy.get("enabled_by_layer", {}).values()
    )
    cache_smoke = None
    if o_enabled:
        cache_smoke = _optimized_cache_consistency_smoke(
            checkpoint,
            device=device,
            all_layer_policy_path=all_layer_policy_path,
            all_layer_scales_path=all_layer_scales_path,
            steps=4,
        )
        atomic_write_json(run_dir / "es7_cache_consistency.json", cache_smoke)

    source = _run_checkpoint_variant_arc(
        checkpoint,
        variant="source_nvfp4",
        device=device,
        batch_size=batch_size,
        tasks=tasks,
    )
    atomic_write_json(run_dir / "semantic_P2_matched_semantic_source.json", source)
    standard = _run_checkpoint_variant_arc(
        checkpoint,
        variant="standard_hif4",
        device=device,
        batch_size=batch_size,
        tasks=tasks,
    )
    atomic_write_json(run_dir / "semantic_P2_matched_semantic_target.json", standard)
    optimized = _run_checkpoint_variant_arc(
        checkpoint,
        variant="optimized_hif4",
        device=device,
        batch_size=batch_size,
        tasks=tasks,
        all_layer_policy_path=all_layer_policy_path,
        all_layer_scales_path=all_layer_scales_path,
    )
    atomic_write_json(run_dir / "semantic_P2_hif4_equalized_semantic_target.json", optimized)

    deltas = {
        task: {
            "source_nvfp4": source["scores"].get(task),
            "standard_hif4": standard["scores"].get(task),
            "optimized_hif4": optimized["scores"].get(task),
            "optimized_minus_standard": (
                optimized["scores"].get(task) - standard["scores"].get(task)
                if optimized["scores"].get(task) is not None and standard["scores"].get(task) is not None
                else None
            ),
        }
        for task in tasks
    }
    comparable = [v for v in deltas.values() if v["optimized_minus_standard"] is not None]
    candidate_for_extended = bool(
        comparable
        and all(float(v["optimized_minus_standard"]) >= 0.0 for v in comparable)
        and any(float(v["optimized_minus_standard"]) > 0.0 for v in comparable)
        and (cache_smoke is None or bool(cache_smoke["passed"]))
        and not trajectory_warning
    )
    result = {
        "schema_version": 1,
        "tasks": tasks,
        "source": source,
        "standard": standard,
        "optimized": optimized,
        "deltas": deltas,
        "o_in_enabled": o_enabled,
        "cache_consistency": cache_smoke,
        "target_trajectory_regression": trajectory_warning,
        "candidate_for_extended_e2e": candidate_for_extended,
    }
    atomic_write_json(run_dir / "es7_e2e_summary.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path-id", choices=["P1_semantic", "P2_matched_semantic"], required=True)
    ap.add_argument("--role", choices=["source", "target"], required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", default="4")
    ap.add_argument("--tasks", default="arc_easy,arc_challenge")
    ap.add_argument("--out-dir", type=Path, required=True)
    ns = ap.parse_args(argv)
    rec = run_one(
        path_id=ns.path_id,
        role=ns.role,
        device=ns.device,
        batch_size=ns.batch_size,
        tasks=[t.strip() for t in ns.tasks.split(",") if t.strip()],
        out_dir=ns.out_dir,
    )
    print(json.dumps({k: rec[k] for k in ("path_id", "role", "scores")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
