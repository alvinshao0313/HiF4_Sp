"""Objective study scaffolding: O0/O1/O2/O3_full/O3_topk/O4 with 50:50 Wiki:S1K fairness."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable

import torch

from .config import (
    OBJECTIVE_TRAIN_SEED,
    S1K_SHARED_CALIB,
    WIKITEXT2_SHARED_CALIB,
)
from .run_state import atomic_write_json


OBJECTIVES = ("O0", "O1", "O2", "O3_full", "O3_topk", "O4")


def local_q_nmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    num = torch.linalg.vector_norm(pred.float() - target.float()) ** 2
    den = (torch.linalg.vector_norm(target.float()) ** 2).clamp_min(1e-12)
    return num / den


def cumulative_residual_nmse(
    upstream_residual: torch.Tensor,
    branch_pred: torch.Tensor,
    target_residual: torch.Tensor,
) -> torch.Tensor:
    """L_cum = ||R_up + branch_theta - R_target||^2 / ||R_target||^2"""
    pred = upstream_residual.float() + branch_pred.float()
    num = torch.linalg.vector_norm(pred - target_residual.float()) ** 2
    den = (torch.linalg.vector_norm(target_residual.float()) ** 2).clamp_min(1e-12)
    return num / den


def build_objective_split_manifest(
    *,
    output_path: Path,
    n_train: int = 64,
    n_val: int = 16,
    seed: int = OBJECTIVE_TRAIN_SEED,
    excluded_sample_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Freeze a 50:50 objective split disjoint from the causal cohort."""
    import torch as _torch

    wiki = _torch.load(WIKITEXT2_SHARED_CALIB / "calibration" / "train.pt", map_location="cpu", weights_only=False)
    s1k = _torch.load(S1K_SHARED_CALIB / "calibration" / "train.pt", map_location="cpu", weights_only=False)
    rng = random.Random(int(seed))
    excluded = {str(x) for x in excluded_sample_ids}
    wiki_ids = [s.sample_id for s in wiki if str(s.sample_id) not in excluded]
    s1k_ids = [s.sample_id for s in s1k if str(s.sample_id) not in excluded]
    n_each_train = n_train // 2
    n_each_val = n_val // 2
    required_each = n_each_train + n_each_val
    if len(wiki_ids) < required_each or len(s1k_ids) < required_each:
        raise RuntimeError(
            "not enough non-causal calibration samples for objective split: "
            f"wiki={len(wiki_ids)} s1k={len(s1k_ids)} required_each={required_each}"
        )
    wiki_sel = rng.sample(wiki_ids, required_each)
    s1k_sel = rng.sample(s1k_ids, required_each)
    train_ids = wiki_sel[:n_each_train] + s1k_sel[:n_each_train]
    val_ids = wiki_sel[n_each_train:] + s1k_sel[n_each_train:]
    if set(train_ids) & set(val_ids):
        raise RuntimeError("objective train/val split overlap")
    if (set(train_ids) | set(val_ids)) & excluded:
        raise RuntimeError("objective split leaked causal discovery/holdout samples")
    payload = {
        "status": "FROZEN",
        "seed": seed,
        "source_ratio": {"wikitext2": 0.5, "s1k_original": 0.5},
        "train_ids": train_ids,
        "val_ids": val_ids,
        "excluded_causal_sample_ids": sorted(excluded),
        "note": (
            "Formal O0 must be retrained on this mix; old pure-S1K O0 is historical only. "
            "Objective train/val are disjoint from all causal discovery/holdout samples."
        ),
        "objectives": list(OBJECTIVES),
        "o4_top_k_layers_only": 4,
    }
    atomic_write_json(output_path, payload)
    return payload


def decide_objective_scopes(
    causal_source_by_layer: dict[int, str],
    o3_layers: list[int] | None = None,
) -> dict[str, Any]:
    """Map causal attribution to DIAG parameter scopes. No MoE preset."""
    o3_set = {int(x) for x in (o3_layers or [])}
    scopes: dict[int, dict[str, Any]] = {}
    for layer, source in causal_source_by_layer.items():
        layer_i = int(layer)
        if source == "attention":
            scopes[layer_i] = {"params": ["D_QKV", "D_VO"], "losses": ["O0", "O1_A", "O2_A", "O4"]}
        elif source == "moe":
            losses = ["O0", "O1_M", "O2_M", "O4"]
            if layer_i in o3_set:
                losses = ["O0", "O1_M", "O2_M", "O3_full", "O3_topk", "O4"]
            scopes[layer_i] = {"params": ["D_GU", "D_UD"], "losses": losses}
        elif source == "both":
            losses = ["O0", "O1_A", "O1_M", "O2_A", "O2_M", "joint", "O4"]
            if layer_i in o3_set:
                losses = [
                    "O0",
                    "O1_A",
                    "O1_M",
                    "O2_A",
                    "O2_M",
                    "joint",
                    "O3_full",
                    "O3_topk",
                    "O4",
                ]
            scopes[layer_i] = {
                "params": ["D_QKV", "D_VO", "D_GU", "D_UD"],
                "losses": losses,
            }
        elif source == "unstable/compensatory":
            scopes[layer_i] = {"params": [], "losses": ["O0"], "skipped_reason": source}
        else:
            raise ValueError(f"unknown causal source {source}")
        if "O3_conditional" in scopes[layer_i]["losses"]:
            raise RuntimeError("O3_conditional must not appear in formal objective scopes")
    return scopes


def assert_formal_objective_scope(objective: dict) -> list[int]:
    """Reject legacy enable_o3 / O3_conditional; require explicit o3_layers."""
    if "enable_o3" in objective:
        raise RuntimeError(
            "formal objective_scope must not contain enable_o3; use o3_layers: list[int]"
        )
    if "O3_conditional" in objective.get("objectives", []) or objective.get("o3_conditional"):
        raise RuntimeError(
            "formal objective_scope must not use O3_conditional; use O3_full/O3_topk via o3_layers"
        )
    if "o3_layers" not in objective:
        raise RuntimeError("FROZEN objective_scope missing required field o3_layers")
    return [int(x) for x in objective["o3_layers"]]


def require_current_state_recapture_for_topk(manifest: dict) -> None:
    if not manifest.get("current_state_recapture", False):
        raise RuntimeError(
            "Top-K accumulation-aware candidate requires sequential current-state "
            "recapture after earlier-layer updates; frozen baseline states are forbidden"
        )
