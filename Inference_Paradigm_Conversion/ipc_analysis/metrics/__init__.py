"""Unified error metrics, streaming accumulators, and stratified statistics."""

from Inference_Paradigm_Conversion.ipc_analysis.metrics.statistics import (
    pearson_with_bootstrap,
    spearman_with_bootstrap,
)
from Inference_Paradigm_Conversion.ipc_analysis.metrics.streaming import ErrorAccumulator
from Inference_Paradigm_Conversion.ipc_analysis.metrics.tensor_metrics import (
    compute_group_metrics,
    compute_pair_metrics,
)

__all__ = [
    "compute_pair_metrics",
    "compute_group_metrics",
    "ErrorAccumulator",
    "pearson_with_bootstrap",
    "spearman_with_bootstrap",
]
