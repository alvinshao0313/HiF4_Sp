from __future__ import annotations

import math

import numpy as np
import torch

from Inference_Paradigm_Conversion.ipc_analysis.metrics.statistics import (
    pearson_with_bootstrap,
    spearman_with_bootstrap,
)
from Inference_Paradigm_Conversion.ipc_analysis.metrics.streaming import ErrorAccumulator
from Inference_Paradigm_Conversion.ipc_analysis.metrics.tensor_metrics import compute_pair_metrics


def test_chunked_accumulator_matches_oneshot():
    torch.manual_seed(0)
    ref = torch.randn(1000, 64)
    tgt = ref + 0.1 * torch.randn_like(ref)
    one = compute_pair_metrics(ref, tgt)
    acc = ErrorAccumulator(reservoir_size=100_000)
    for i in range(0, ref.shape[0], 64):
        acc.update(ref[i : i + 64], tgt[i : i + 64])
    fin = acc.finalize()
    assert math.isclose(fin["nmse"], one["nmse"], rel_tol=1e-12, abs_tol=0)
    assert math.isclose(fin["reference_energy"], one["reference_energy"], rel_tol=1e-12)
    assert math.isclose(fin["error_energy"], one["error_energy"], rel_tol=1e-12)
    assert math.isclose(fin["cosine"], one["cosine"], rel_tol=1e-12)


def test_bootstrap_seed_reproducible():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 0.5 * x + rng.normal(size=200) * 0.1
    clusters = np.repeat(np.arange(20), 10)
    a = pearson_with_bootstrap(x, y, seed=123, repeats=50, cluster_ids=clusters)
    b = pearson_with_bootstrap(x, y, seed=123, repeats=50, cluster_ids=clusters)
    assert a == b
    c = spearman_with_bootstrap(x, y, seed=7, repeats=50, cluster_ids=clusters)
    d = spearman_with_bootstrap(x, y, seed=7, repeats=50, cluster_ids=clusters)
    assert c == d
