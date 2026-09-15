"""Scope-freeze and objective-split isolation tests."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.objective_study import (
    build_objective_split_manifest,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.run_state import read_json


def test_objective_split_excludes_all_causal_samples(tmp_path: Path):
    wiki = [SimpleNamespace(sample_id=f"w{i}") for i in range(80)]
    s1k = [SimpleNamespace(sample_id=f"s{i}") for i in range(80)]
    excluded = {"w0", "w1", "w2", "s0", "s1", "s2"}

    def _fake_load(path, *args, **kwargs):
        text = str(path)
        if "wikitext2" in text:
            return wiki
        return s1k

    out = tmp_path / "objective_split_manifest.json"
    with mock.patch("torch.load", side_effect=_fake_load):
        payload = build_objective_split_manifest(
            output_path=out,
            n_train=16,
            n_val=8,
            seed=7,
            excluded_sample_ids=excluded,
        )

    assert payload["status"] == "FROZEN"
    assert set(payload["train_ids"]).isdisjoint(payload["val_ids"])
    assert (set(payload["train_ids"]) | set(payload["val_ids"])).isdisjoint(excluded)
    assert set(payload["excluded_causal_sample_ids"]) == excluded
    assert payload["objectives"] == ["O0", "O1", "O2", "O3_full", "O3_topk", "O4"]
    assert "o3_conditional" not in payload
    assert read_json(out) == payload
