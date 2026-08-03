"""Candidate pool aggregation and robust multi-split acceptance.

The final accept decision is made on deployment-consistent ``total_nrmse``
measured on independent validation splits. Proxy ``hif4_loss`` is a
search/diagnostic signal only and never decides acceptance alone. Random
permutations are negative controls: they are evaluated and recorded but are
never eligible for deployment; if a random control beats every structured
candidate, the search is considered broken and the decision falls back to
identity with an explicit alarm reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import SearchConfig
from .objective import DeploymentMetrics

REASON_ACCEPTED = "accepted"
REASON_NO_STRUCTURED = "no_structured_candidate_beats_identity"
REASON_INSUFFICIENT_WINS = "insufficient_validation_wins"
REASON_BELOW_THRESHOLD = "relative_improvement_below_threshold"
REASON_NOT_ABOVE_VARIANCE = "improvement_not_above_split_variance"
REASON_DRIFT_ABOVE = "bf16_reorder_drift_above_threshold"
REASON_RANDOM_BEST = "random_negative_control_outperformed_structured_candidates"


@dataclass(frozen=True)
class CandidateMetrics:
    """Per-split deployment metrics for one candidate permutation."""

    name: str
    permutation: torch.Tensor
    split_metrics: tuple[DeploymentMetrics, ...]
    eligible_for_deployment: bool


@dataclass(frozen=True)
class CandidateDecision:
    selected_name: str
    selected_permutation: torch.Tensor
    accepted: bool
    rejection_reason: str
    aggregate_metrics: dict[str, dict[str, float]]


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(torch.tensor(values, dtype=torch.float64).std(unbiased=True).item())


def _aggregate_one(
    cand: CandidateMetrics, identity: CandidateMetrics
) -> dict[str, float]:
    totals = [m.total_nrmse for m in cand.split_metrics]
    drifts = [m.bf16_reorder_drift for m in cand.split_metrics]
    residuals = [m.quantization_residual_nrmse for m in cand.split_metrics]
    id_totals = [m.total_nrmse for m in identity.split_metrics]
    mean_total = sum(totals) / len(totals)
    mean_id = sum(id_totals) / len(id_totals)
    wins = sum(1 for t, i in zip(totals, id_totals) if t < i)
    rel_impr = (mean_id - mean_total) / mean_id if mean_id > 0 else 0.0
    return {
        "mean_total_nrmse": mean_total,
        "std_total_nrmse": _std(totals),
        "mean_quantization_residual_nrmse": sum(residuals) / len(residuals),
        "mean_bf16_reorder_drift": sum(drifts) / len(drifts),
        "wins_vs_identity": float(wins),
        "relative_improvement_pct": rel_impr * 100.0,
    }


def _check_acceptance(
    cand: CandidateMetrics,
    identity: CandidateMetrics,
    config: SearchConfig,
) -> str:
    """Return REASON_ACCEPTED if ``cand`` passes every robustness check."""
    totals = [m.total_nrmse for m in cand.split_metrics]
    id_totals = [m.total_nrmse for m in identity.split_metrics]
    wins = sum(1 for t, i in zip(totals, id_totals) if t < i)
    if wins < config.min_validation_wins:
        return REASON_INSUFFICIENT_WINS
    mean_total = sum(totals) / len(totals)
    mean_id = sum(id_totals) / len(id_totals)
    if not (mean_total <= mean_id * (1.0 - config.min_relative_improvement)):
        return REASON_BELOW_THRESHOLD
    deltas = [i - t for t, i in zip(totals, id_totals)]
    mean_delta = sum(deltas) / len(deltas)
    if not (mean_delta > config.improvement_std_multiplier * _std(deltas)):
        return REASON_NOT_ABOVE_VARIANCE
    mean_drift = sum(m.bf16_reorder_drift for m in cand.split_metrics) / len(
        cand.split_metrics
    )
    if mean_drift > config.max_bf16_reorder_drift:
        return REASON_DRIFT_ABOVE
    return REASON_ACCEPTED


def select_candidate(
    candidates: list[CandidateMetrics], config: SearchConfig
) -> CandidateDecision:
    """Pick the deployment candidate under the multi-split robust rule.

    Always returns a decision; when rejecting, ``selected_*`` is identity.
    """
    identity = next((c for c in candidates if c.name == "identity"), None)
    if identity is None:
        raise ValueError("candidates must include an 'identity' baseline")
    if not candidates:
        raise ValueError("candidates must be non-empty")

    aggregate = {c.name: _aggregate_one(c, identity) for c in candidates}

    def _mean_total(c: CandidateMetrics) -> float:
        return aggregate[c.name]["mean_total_nrmse"]

    best_overall = min(candidates, key=lambda c: (_mean_total(c), c.name))
    if not best_overall.eligible_for_deployment:
        return CandidateDecision(
            selected_name="identity",
            selected_permutation=identity.permutation,
            accepted=False,
            rejection_reason=REASON_RANDOM_BEST,
            aggregate_metrics=aggregate,
        )

    structured = [
        c for c in candidates if c.eligible_for_deployment and c.name != "identity"
    ]
    better = [c for c in structured if aggregate[c.name]["wins_vs_identity"] > 0]
    if not better:
        return CandidateDecision(
            selected_name="identity",
            selected_permutation=identity.permutation,
            accepted=False,
            rejection_reason=REASON_NO_STRUCTURED,
            aggregate_metrics=aggregate,
        )

    best = min(better, key=lambda c: (_mean_total(c), c.name))
    reason = _check_acceptance(best, identity, config)
    if reason == REASON_ACCEPTED:
        return CandidateDecision(
            selected_name=best.name,
            selected_permutation=best.permutation,
            accepted=True,
            rejection_reason=reason,
            aggregate_metrics=aggregate,
        )
    return CandidateDecision(
        selected_name="identity",
        selected_permutation=identity.permutation,
        accepted=False,
        rejection_reason=reason,
        aggregate_metrics=aggregate,
    )
