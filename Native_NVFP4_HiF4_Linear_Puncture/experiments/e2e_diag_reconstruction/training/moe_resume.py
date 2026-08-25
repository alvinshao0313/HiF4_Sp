"""Resume helpers for lazy Qwen3-MoE DIAG reconstruction.

This module keeps resume compatibility and prefix replay separate from the main
training loop. A resumed layer must receive the same progressive hidden states
as an uninterrupted run, while preserving old artifact semantics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.artifact import (
    layer_artifact_dir,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import (
    E2ETrainConfig,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import (
    load_qwen3_moe_layer_state,
    release_qwen3_moe_layer_state,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import (
    NativeQwen3MoELayerRuntime,
    StudentQwen3MoELayerRuntime,
    build_moe_diag_state,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.batching import (
    DynamicCalibrationCollator,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.calibration import (
    CalibrationSample,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.layer_runtime import (
    ProgressiveHiddenCache,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import load_pt, read_json


PropagateFn = Callable[..., ProgressiveHiddenCache]


def load_resume_prefix_records(
    cfg: E2ETrainConfig,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Load layer records and summaries for layers before ``start_layer``.

    Schema-v3 runs preserve both candidate and adopted DIAGs. Older runs may
    only contain ``best_diag.pt``; in that case the adopted DIAG is also used as
    the candidate because the original pre-rollback candidate is unrecoverable.
    """

    layer_records: dict[int, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    for layer_idx in range(cfg.start_layer):
        d = layer_artifact_dir(cfg.output_dir, layer_idx)
        metrics = read_json(d / "metrics.json")
        adopted_z = load_pt(d / "best_diag.pt", map_location="cpu")

        candidate_path = d / "candidate_best_diag.pt"
        candidate_z = (
            load_pt(candidate_path, map_location="cpu")
            if candidate_path.is_file()
            else {name: value.clone() for name, value in adopted_z.items()}
        )

        rollback = bool(metrics.get("rollback", False))
        best_epoch = int(metrics.get("best_epoch", -1))
        candidate_best_epoch = int(metrics.get("candidate_best_epoch", best_epoch))
        layer_records[layer_idx] = {
            "accepted": bool(metrics.get("accepted", not rollback)),
            "rollback": rollback,
            "best_epoch": best_epoch,
            "candidate_best_epoch": candidate_best_epoch,
            "candidate_z": candidate_z,
            "adopted_z": adopted_z,
            "z": adopted_z,
            "loss_rollback_applied": bool(metrics.get("loss_rollback_applied", False)),
            "router_rollback_applied": bool(metrics.get("router_rollback_applied", False)),
        }
        summaries.append(metrics)

    return layer_records, summaries


def resume_prefix_uses_native(cfg: E2ETrainConfig, metrics: dict[str, Any]) -> bool:
    """Return the propagation path that was used when an old layer finished.

    New split-rollback runs propagate native hidden states only for a loss
    rollback. Router-only rollback keeps the adopted student DIAG active.
    Legacy runs had one rollback bit and propagated native whenever it fired.
    """

    if cfg.calib_input_mode == "teacher":
        return True
    if "loss_rollback_applied" in metrics:
        return bool(metrics["loss_rollback_applied"])
    return bool(metrics.get("rollback", False))


def replay_moe_prefix(
    cfg: E2ETrainConfig,
    snapshot: Path,
    samples: list[CalibrationSample],
    collator: DynamicCalibrationCollator,
    x_cache: ProgressiveHiddenCache,
    device: torch.device,
    layer_records: dict[int, dict[str, Any]],
    summaries: list[dict[str, Any]],
    *,
    propagate_fn: PropagateFn,
) -> ProgressiveHiddenCache:
    """Replay layers ``[0, start_layer)`` using their original adopted path."""

    if len(summaries) != cfg.start_layer:
        raise ValueError(
            f"resume summary prefix length {len(summaries)} != start_layer {cfg.start_layer}"
        )

    for layer_idx in range(cfg.start_layer):
        state = load_qwen3_moe_layer_state(snapshot, layer_idx, device)
        try:
            metrics = summaries[layer_idx]
            if resume_prefix_uses_native(cfg, metrics):
                runtime = NativeQwen3MoELayerRuntime(state).to(device).eval()
                x_cache = propagate_fn(
                    runtime,
                    snapshot,
                    samples,
                    collator,
                    x_cache,
                    device,
                    cfg.diag_batch_size,
                    student=False,
                )
                continue

            diag_state = build_moe_diag_state(state.spec, cfg.diag_mode).to(device)
            diag_state.load_snapshot(layer_records[layer_idx]["adopted_z"])
            runtime = StudentQwen3MoELayerRuntime(
                state,
                diag_state,
                use_r64=cfg.use_r64,
                rot_order=cfg.rot_order,
            ).to(device).eval()
            x_cache = propagate_fn(
                runtime,
                snapshot,
                samples,
                collator,
                x_cache,
                device,
                cfg.diag_batch_size,
                student=True,
            )
        finally:
            release_qwen3_moe_layer_state(state)

    return x_cache
