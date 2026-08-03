"""High-level reorder pipeline (public API for future main.py integration)."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import multiprocessing as mp

import torch

from .activation_collector import collect_down_inputs
from .config import LayerSearchResult, SearchConfig
from .hierarchical_greedy import optimize_layer_permutation
from .model_permutation import apply_mlp_permutation_, discover_swiglu_mlps, get_mlp_modules

logger = logging.getLogger("permutation_optimization.pipeline")


def _hash_indices(indices: list[int]) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(indices).encode()).hexdigest()[:16]


def _metrics_row(
    result: LayerSearchResult,
    layer_index: int,
    config: SearchConfig,
    elapsed: float,
) -> dict[str, Any]:
    split = result.extra.get("split", {})
    split_audit = (
        {
            "search_rows": split["search_rows"],
            "validation_rows": split["validation_rows"],
            "overlap_rows": split["overlap_rows"],
            "search_indices_sha256": _hash_indices(split["search_indices"]),
            "validation_indices_sha256": _hash_indices(split["validation_indices"]),
        }
        if split
        else {}
    )
    return {
        "layer_name": result.layer_name,
        "layer_index": layer_index,
        "selected_candidate": result.extra.get("selected_candidate", ""),
        "accepted": bool(result.accepted),
        "rejection_reason": result.extra.get("rejection_reason", ""),
        "search_split_seed": int(config.seed),
        "validation_seeds": [int(s) for s in config.validation_seeds],
        "identity_hif4_loss": float(result.identity_hif4_loss),
        "optimized_hif4_loss": float(result.optimized_hif4_loss),
        "identity_output_nrmse": float(result.identity_output_nrmse),
        "optimized_output_nrmse": float(result.optimized_output_nrmse),
        "candidate_metrics": result.baseline_metrics,
        "split_audit": split_audit,
        "proxy_audit": result.extra.get("proxy_audit", {}),
        "refinement": result.extra.get("refinement", {}),
        "elapsed_sec": float(elapsed),
    }


def _optimize_one_layer_job(
    layer_name: str,
    mlp_input: torch.Tensor,
    up_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    config: SearchConfig,
) -> LayerSearchResult:
    """Top-level worker for ProcessPoolExecutor (must be picklable)."""
    return optimize_layer_permutation(
        layer_name, mlp_input, up_weight, gate_weight, down_weight, config
    )


@torch.no_grad()
def reorder_model_mlps(
    model: torch.nn.Module,
    calibration_batches,
    config: SearchConfig,
    device: torch.device,
    layer_indices: list[int] | None = None,
    metrics_path: Path | None = None,
    num_workers: int = 1,
    max_rows_per_batch: int | None = None,
    apply_candidate_name: str = "selected",
) -> dict[str, Any]:
    """Discover SwiGLU MLPs, collect activations, optimize, apply perms.

    ``num_workers > 1`` parallelizes per-layer search (CPU-heavy) after a single
    activation collection pass. Weights are copied to CPU before fan-out.

    ``apply_candidate_name="selected"`` applies the robust-selector permutation
    per layer (identity when rejected). Any other value must be a candidate
    name from the search pool (e.g. ``"q99_sort_desc"``); that candidate's
    permutation is then applied to EVERY selected layer.
    """
    specs = discover_swiglu_mlps(model)
    if layer_indices is not None:
        allow = set(layer_indices)
        specs = [s for s in specs if s.layer_index in allow]
    if not specs:
        raise RuntimeError("No SwiGLU MLP layers selected")

    logger.info("Collecting down_proj activations for %d MLPs ...", len(specs))
    cache = collect_down_inputs(
        model,
        specs,
        calibration_batches,
        input_device=device,
        max_rows=config.activation_rows,
        seed=config.seed,
        max_rows_per_batch=max_rows_per_batch,
    )

    # Snapshot CPU tensors for parallel workers.
    jobs: list[tuple] = []
    for spec in specs:
        gate, up, down = get_mlp_modules(model, spec)
        x = cache.mlp_input_by_layer[spec.name].float().cpu().contiguous()
        wu = up.weight.detach().float().cpu().contiguous()
        wg = gate.weight.detach().float().cpu().contiguous()
        wd = down.weight.detach().float().cpu().contiguous()
        jobs.append((spec, x, wu, wg, wd))
        del cache.by_layer[spec.name]
        del cache.mlp_input_by_layer[spec.name]

    results_by_name: dict[str, LayerSearchResult] = {}
    workers = max(1, int(num_workers))
    logger.info("Optimizing %d layers with num_workers=%d ...", len(jobs), workers)

    # Free model GPU memory during search so workers can use CUDA for G8/G64 matrices.
    model_device = next(model.parameters()).device
    if workers > 1 and model_device.type == "cuda":
        model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Moved model to CPU for parallel search (will restore after)")

    if workers == 1:
        for spec, x, wu, wg, wd in jobs:
            logger.info("Optimizing %s (d_ff=%d) ...", spec.name, spec.intermediate_size)
            t0 = time.time()
            result = optimize_layer_permutation(spec.name, x, wu, wg, wd, config)
            elapsed = time.time() - t0
            result.extra["elapsed_sec"] = elapsed
            results_by_name[spec.name] = result
            logger.info(
                "  %s accepted=%s hier_loss=%.6f id_loss=%.6f hier_nrmse=%.6f id_nrmse=%.6f (%.1fs)",
                spec.name,
                result.accepted,
                result.optimized_hif4_loss,
                result.identity_hif4_loss,
                result.optimized_output_nrmse,
                result.identity_output_nrmse,
                elapsed,
            )
            if metrics_path is not None:
                row = _metrics_row(result, spec.layer_index, config, elapsed)
                with metrics_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
    else:
        # Process pool: each layer independent. Stream metrics as workers finish.
        if metrics_path is not None:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text("")
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            fut_map = {
                ex.submit(_optimize_one_layer_job, spec.name, x, wu, wg, wd, config): (spec, time.time())
                for spec, x, wu, wg, wd in jobs
            }
            done_n = 0
            for fut in as_completed(fut_map):
                spec, t0 = fut_map[fut]
                result = fut.result()
                elapsed = time.time() - t0
                result.extra["elapsed_sec"] = elapsed
                results_by_name[spec.name] = result
                done_n += 1
                logger.info(
                    "  [%d/%d] %s accepted=%s hier_loss=%.6f id_loss=%.6f hier_nrmse=%.6f id_nrmse=%.6f (%.1fs)",
                    done_n,
                    len(jobs),
                    spec.name,
                    result.accepted,
                    result.optimized_hif4_loss,
                    result.identity_hif4_loss,
                    result.optimized_output_nrmse,
                    result.identity_output_nrmse,
                    elapsed,
                )
                if metrics_path is not None:
                    row = _metrics_row(result, spec.layer_index, config, elapsed)
                    with metrics_path.open("a") as f:
                        f.write(json.dumps(row) + "\n")

    if workers > 1 and model_device.type == "cuda":
        model.to(model_device)
        logger.info("Restored model to %s after parallel search", model_device)

    # Apply in layer order; rewrite metrics sorted by layer (progress already streamed above).
    permutations: dict[str, torch.Tensor] = {}
    candidate_permutations: dict[str, dict[str, torch.Tensor]] = {}
    best_candidate_permutations: dict[str, torch.Tensor] = {}
    results = []
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("")

    for spec in specs:
        result = results_by_name[spec.name]
        exported = result.extra.get("candidate_permutations", {})
        for cand_name, cand_perm in exported.items():
            candidate_permutations.setdefault(cand_name, {})[spec.name] = (
                cand_perm.detach().to(device="cpu", dtype=torch.long).contiguous()
            )
        best_candidate_permutations[spec.name] = (
            result.candidate_permutation.detach()
            .to(device="cpu", dtype=torch.long)
            .contiguous()
        )
        if apply_candidate_name == "selected":
            applied_perm = result.permutation
        else:
            if apply_candidate_name not in exported:
                raise KeyError(
                    f"apply_candidate_name {apply_candidate_name!r} not found in "
                    f"candidate pool for {spec.name}: {sorted(exported)}"
                )
            applied_perm = exported[apply_candidate_name]
        gate, up, down = get_mlp_modules(model, spec)
        apply_mlp_permutation_(gate, up, down, applied_perm)
        permutations[spec.name] = applied_perm.detach().to(
            device="cpu", dtype=torch.long
        ).contiguous()
        row = _metrics_row(
            result,
            spec.layer_index,
            config,
            float(result.extra.get("elapsed_sec", 0.0)),
        )
        results.append(row)
        if metrics_path is not None:
            with metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")

    return {
        "permutations": permutations,
        "candidate_permutations": candidate_permutations,
        "best_candidate_permutations": best_candidate_permutations,
        "applied_candidate": apply_candidate_name,
        "results": results,
        "specs": specs,
    }
