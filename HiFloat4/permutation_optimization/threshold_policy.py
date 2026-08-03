"""Pure threshold-gating policy for per-layer permutation selection.

For each layer, the searched candidate permutation is applied only when its
real W4A4 MLP output error improves on identity by at least ``threshold_pct``
percent. Negative-gain layers always fall back to identity. No model loading,
no quantization — this module is a pure function of search metrics.
"""

from __future__ import annotations

import torch


def relative_output_gain_pct(
    identity_output_nrmse: float,
    candidate_output_nrmse: float,
    eps: float = 1e-12,
) -> float:
    """Relative W4A4 output-error improvement of candidate vs identity, in %."""
    if identity_output_nrmse <= eps:
        return 0.0
    return 100.0 * (identity_output_nrmse - candidate_output_nrmse) / identity_output_nrmse


def _validate_candidate_perm(name: str, perm: torch.Tensor) -> torch.Tensor:
    if not isinstance(perm, torch.Tensor) or perm.ndim != 1:
        raise ValueError(f"candidate permutation for {name} must be a 1-D tensor")
    perm_long = perm.detach().to(dtype=torch.long, device="cpu").contiguous()
    n = perm_long.numel()
    if n == 0:
        raise ValueError(f"candidate permutation for {name} is empty")
    if int(perm_long.min().item()) < 0 or int(perm_long.max().item()) >= n:
        raise ValueError(f"candidate permutation for {name} has out-of-range values")
    if torch.unique(perm_long).numel() != n:
        raise ValueError(
            f"candidate permutation for {name} must cover each index exactly once"
        )
    return perm_long


def build_threshold_gated_permutations(
    summary: dict,
    candidate_permutations: dict[str, torch.Tensor],
    threshold_pct: float,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Pick candidate vs identity per layer under a gain threshold.

    ``summary`` must carry ``results`` rows with ``layer_name``,
    ``identity_output_nrmse`` and ``optimized_output_nrmse`` (the best
    structured candidate's mean total NRMSE). ``candidate_permutations`` maps
    layer name to that candidate's permutation. Returns (gated_map, report).
    """
    if not isinstance(threshold_pct, (int, float)) or isinstance(threshold_pct, bool):
        raise ValueError(f"threshold_pct must be a non-negative number, got {threshold_pct!r}")
    if threshold_pct < 0:
        raise ValueError(f"threshold_pct must be non-negative, got {threshold_pct!r}")

    metrics_by_layer: dict[str, tuple[float, float]] = {}
    for row in summary.get("results", []):
        metrics_by_layer[row["layer_name"]] = (
            float(row["identity_output_nrmse"]),
            float(row["optimized_output_nrmse"]),
        )

    gated: dict[str, torch.Tensor] = {}
    layers_report: list[dict] = []
    n_reordered = 0
    for layer_name in sorted(candidate_permutations):
        if layer_name not in metrics_by_layer:
            raise KeyError(
                f"summary has no metrics for layer {layer_name!r}; refusing to guess"
            )
        identity_nrmse, candidate_nrmse = metrics_by_layer[layer_name]
        cand = _validate_candidate_perm(layer_name, candidate_permutations[layer_name])
        gain_pct = relative_output_gain_pct(identity_nrmse, candidate_nrmse)
        use_reorder = gain_pct > 0.0 and gain_pct + 1e-12 >= float(threshold_pct)
        identity = torch.arange(cand.numel(), dtype=torch.long)
        gated[layer_name] = cand if use_reorder else identity
        n_reordered += int(use_reorder)
        layers_report.append(
            {
                "layer_name": layer_name,
                "identity_output_nrmse": identity_nrmse,
                "candidate_output_nrmse": candidate_nrmse,
                "relative_gain_pct": gain_pct,
                "use_reorder": bool(use_reorder),
            }
        )

    report = {
        "threshold_pct": float(threshold_pct),
        "n_layers": len(layers_report),
        "n_reordered": n_reordered,
        "n_identity": len(layers_report) - n_reordered,
        "layers": layers_report,
    }
    return gated, report
