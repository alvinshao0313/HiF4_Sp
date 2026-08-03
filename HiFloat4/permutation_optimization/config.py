"""Search configuration and result dataclasses for HiF4 MLP hierarchical permutation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")


def _require_non_negative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative int, got {value!r}")


def _require_weight(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


def _require_fraction(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a float in (0, 0.5), got {value!r}")
    if value <= 0.0 or value >= 0.5:
        raise ValueError(f"{name} must be in (0, 0.5), got {value!r}")


def _require_positive_float(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0.0:
        raise ValueError(f"{name} must be a positive float, got {value!r}")


@dataclass(frozen=True)
class SearchConfig:
    activation_rows: int = 512
    validation_fraction: float = 0.2
    weight_rows: int = 512
    candidate_window: int = 128
    neighbor_k: int = 32
    beam_width_g4: int = 4
    exact_rerank_g4: int = 8
    beam_width_g64: int = 4
    activation_loss_weight: float = 0.5
    weight_loss_weight: float = 0.5
    range_loss_weight: float = 0.2
    refine_passes: int = 2
    refine_bad_blocks: int = 32
    # Hard cap on accepted swaps per stage (G8 / G4 / channel) per pass.
    # Without this, best-improvement while-loops can run for many hours on large d_ff.
    refine_max_swaps_per_stage: int = 16
    improvement_tol: float = 1e-8
    seed: int = 42
    eps: float = 1e-8
    # Multi-split robust acceptance.
    validation_seeds: tuple[int, ...] = (42, 43, 44)
    min_relative_improvement: float = 0.001
    min_validation_wins: int = 2
    improvement_std_multiplier: float = 2.0
    max_bf16_reorder_drift: float = 0.002
    # Per-layer C4-proxy vs real-G64 ranking audit.
    proxy_audit_enabled: bool = True
    proxy_audit_candidates: int = 128
    # Candidate-pool local refinement (seeded local search).
    refine_enabled: bool = True
    refine_seed_candidates: tuple[str, ...] = ("identity", "q99_sort_desc", "hierarchical")
    refine_max_rounds: int = 2
    refine_candidates_per_round: int = 64
    refine_min_proxy_gain: float = 1e-5

    def __post_init__(self) -> None:
        _require_positive_int("activation_rows", self.activation_rows)
        _require_fraction("validation_fraction", self.validation_fraction)
        _require_positive_int("weight_rows", self.weight_rows)
        _require_positive_int("candidate_window", self.candidate_window)
        _require_positive_int("neighbor_k", self.neighbor_k)
        _require_positive_int("beam_width_g4", self.beam_width_g4)
        _require_positive_int("exact_rerank_g4", self.exact_rerank_g4)
        _require_positive_int("beam_width_g64", self.beam_width_g64)
        _require_weight("activation_loss_weight", self.activation_loss_weight)
        _require_weight("weight_loss_weight", self.weight_loss_weight)
        _require_weight("range_loss_weight", self.range_loss_weight)
        _require_non_negative_int("refine_passes", self.refine_passes)
        _require_positive_int("refine_bad_blocks", self.refine_bad_blocks)
        _require_positive_int("refine_max_swaps_per_stage", self.refine_max_swaps_per_stage)
        _require_positive_float("improvement_tol", self.improvement_tol)
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed must be an int, got {self.seed!r}")
        _require_positive_float("eps", self.eps)
        if self.neighbor_k > 2 * self.candidate_window:
            raise ValueError(
                f"neighbor_k ({self.neighbor_k}) cannot exceed "
                f"2 * candidate_window ({2 * self.candidate_window})"
            )
        if (
            not isinstance(self.validation_seeds, tuple)
            or len(self.validation_seeds) < 2
            or not all(isinstance(s, int) and not isinstance(s, bool) for s in self.validation_seeds)
        ):
            raise ValueError(
                f"validation_seeds must be a tuple of >= 2 ints, got {self.validation_seeds!r}"
            )
        _require_positive_float("min_relative_improvement", self.min_relative_improvement)
        _require_positive_int("min_validation_wins", self.min_validation_wins)
        if self.min_validation_wins > len(self.validation_seeds):
            raise ValueError(
                f"min_validation_wins ({self.min_validation_wins}) cannot exceed "
                f"len(validation_seeds) ({len(self.validation_seeds)})"
            )
        _require_positive_float("improvement_std_multiplier", self.improvement_std_multiplier)
        _require_positive_float("max_bf16_reorder_drift", self.max_bf16_reorder_drift)
        if not isinstance(self.proxy_audit_enabled, bool):
            raise ValueError(f"proxy_audit_enabled must be bool, got {self.proxy_audit_enabled!r}")
        _require_positive_int("proxy_audit_candidates", self.proxy_audit_candidates)
        if not isinstance(self.refine_enabled, bool):
            raise ValueError(f"refine_enabled must be bool, got {self.refine_enabled!r}")
        if (
            not isinstance(self.refine_seed_candidates, tuple)
            or not self.refine_seed_candidates
            or not all(isinstance(s, str) and s for s in self.refine_seed_candidates)
        ):
            raise ValueError(
                f"refine_seed_candidates must be a non-empty tuple of str, "
                f"got {self.refine_seed_candidates!r}"
            )
        _require_positive_int("refine_max_rounds", self.refine_max_rounds)
        _require_positive_int("refine_candidates_per_round", self.refine_candidates_per_round)
        _require_positive_float("refine_min_proxy_gain", self.refine_min_proxy_gain)


@dataclass(frozen=True)
class MLPLayerSpec:
    name: str
    layer_index: int
    gate_name: str
    up_name: str
    down_name: str
    intermediate_size: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("MLPLayerSpec.name must be non-empty")
        if self.intermediate_size <= 0:
            raise ValueError(
                f"intermediate_size must be positive, got {self.intermediate_size}"
            )
        if self.intermediate_size % 64 != 0:
            raise ValueError(
                f"intermediate_size must be divisible by 64, got {self.intermediate_size}"
            )


@dataclass
class LayerSearchResult:
    layer_name: str
    permutation: torch.Tensor
    candidate_permutation: torch.Tensor
    baseline_metrics: dict[str, dict[str, float]]
    identity_hif4_loss: float
    optimized_hif4_loss: float
    identity_output_nrmse: float
    optimized_output_nrmse: float
    accepted: bool
    g4_groups: list[list[int]] = field(default_factory=list)
    g8_groups: list[list[int]] = field(default_factory=list)
    g64_groups: list[list[int]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
