from types import SimpleNamespace

import pytest
import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse import directions
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.artifacts import (
    ActiveClock, BudgetExpired, read_json, tensor_hash,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.tests.test_objectives import ToyRuntime


def setup_refresh():
    torch.manual_seed(18)
    runtime = ToyRuntime()
    samples = {str(n): {"input_ids": torch.arange(n)} for n in (2, 3, 5, 6)}
    for item in samples.values():
        item["token_sha256"] = tensor_hash(item["input_ids"])
    ds = SimpleNamespace(samples=samples, ids=lambda split: list(samples), protocol={"batches": [list(samples)]})
    inputs = {s: torch.randn(len(item["input_ids"]), 3) for s, item in samples.items()}
    logits = {s: torch.randn(len(item["input_ids"]), 7) for s, item in samples.items()}
    # Toy runtime carries a batch dimension; real full-vocabulary readers do not.
    native = SimpleNamespace(logits=lambda s: logits[s].unsqueeze(0))
    baseline = SimpleNamespace(layer=lambda s, layer: {"input": inputs[s]})
    return dict(runtime=runtime, ds=ds, native=native, baseline=baseline, epoch=0, step=0,
                provenance={"source": "unit"}, clock=ActiveClock(now=lambda: 0.), budget=10.)


def test_full_refresh_has_one_parameter_snapshot_and_common_denominator(tmp_path):
    kwargs = setup_refresh()
    cache = directions.build(tmp_path / "refresh_0", **kwargs)
    assert cache.manifest["status"] == "COMPLETE"
    assert set(cache.manifest["samples"]) == set(kwargs["ds"].samples)
    for sid in kwargs["ds"].samples:
        item = cache.sample(sid, 16)
        assert item["gradient"].dtype == torch.float32
    with torch.no_grad():
        kwargs["runtime"].student.learned.weight.add_(.001)
    kwargs.update(epoch=1, step=1, previous=cache)
    second = directions.build(tmp_path / "refresh_1", **kwargs)
    d = second.manifest["diagnostics"]
    assert .9 < d["gradient_cosine"] <= 1.
    assert abs(d["actual_kl_change"]-d["predicted_kl_change"]) < 1e-4


def test_expired_refresh_remains_incomplete(tmp_path, monkeypatch):
    kwargs = setup_refresh()
    now = [0.]
    kwargs["clock"] = ActiveClock(now=lambda: now[0])
    original = directions.collect_direction
    def collect(*args):
        result = original(*args)
        now[0] += 6.
        return result
    monkeypatch.setattr(directions, "collect_direction", collect)
    path = tmp_path / "refresh"
    with pytest.raises(BudgetExpired):
        directions.build(path, **kwargs)
    assert read_json(path / "manifest.json")["status"] == "RUNNING"
    with pytest.raises(RuntimeError, match="provenance"):
        directions.Directions(path, provenance=kwargs["provenance"], ds=kwargs["ds"], layer=8)
