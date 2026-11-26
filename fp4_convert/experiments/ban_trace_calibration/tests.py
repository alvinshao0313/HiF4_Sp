"""Regression checks for the experiment's data and comparison contracts."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

from .config import TASKS, budget_config, model_view
from .data import budget_inputs, sample_windows
from .report import clustered_bootstrap, protocol, require_coverage
from .run import check_pairs, evaluation_command


def training_rows():
    return [{"id": f"{domain}-{i}", "domain": domain, "tokens": list(range(4000 + i)),
             "prompt_tokens": 2300, "source": f"{domain}-{i}.json", "source_sha256": "fixed"}
            for domain in ("math", "code") for i in range(3)]


def test_windows_are_balanced_reproducible_original_tokens():
    rows = training_rows()
    a, data = sample_windows(rows, nsamples=12)
    b, again = sample_windows(rows, nsamples=12)
    assert torch.equal(a, b) and data == again
    assert a.shape == (12, 2048) and a.dtype == torch.long
    assert Counter(w["domain"] for w in data["windows"]) == {"math": 6, "code": 6}
    by_id = {r["id"]: r for r in rows}
    for ids, window in zip(a, data["windows"]):
        source = by_id[window["id"]]
        assert window["end_exclusive"] > source["prompt_tokens"]
        assert ids.tolist() == source["tokens"][window["start"]:window["end_exclusive"]]


def test_short_traces_are_recorded_and_not_sampled():
    rows = training_rows() + [{"id": "short", "domain": "math", "tokens": [1], "prompt_tokens": 0}]
    _, data = sample_windows(rows, nsamples=4)
    assert data["excluded"][0]["id"] == "short"
    assert all(row["id"] != "short" for row in data["windows"])
    with pytest.raises(ValueError, match="no code"):
        sample_windows([row for row in rows if row["domain"] == "math"], nsamples=4)


def test_budget_selects_first_eight_original_validation_prompts():
    rows = [{"id": f"{d}-{i}", "domain": d, "tokens": [i, 100, 200, 300],
             "prompt_tokens": 2, "source": "original"} for d in ("math", "code") for i in range(16)]
    selected = budget_inputs(rows)
    assert len(selected) == 16
    assert [row["id"] for row in selected[:8]] == [f"math-{i}" for i in range(8)]
    assert [row["input_ids"] for row in selected[8:]] == [[i, 100] for i in range(8)]


@pytest.mark.parametrize("pairs", [["0,1", "1,2"], [], ["0,1", "2,3", "4,5", "6,7"]])
def test_resource_pairs_reject_overlap_and_excess(pairs):
    with pytest.raises(ValueError):
        check_pairs(pairs)


def test_fresh_views_exclude_predictions_and_unrelated_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("config.json", "chat_template.jinja", "hif4_runtime_spec.pt", "shard.safetensors", "old_report.json"):
        (source / name).write_text("original")
    (source / "old_cache").mkdir()
    (source / "old_cache/GENERATIVE.parquet").write_text("prior predictions")
    for name in ("first", "second"):
        view = model_view(source, tmp_path / name)
        assert (view / "shard.safetensors").resolve() == source / "shard.safetensors"
        assert not (view / "old_cache").exists() and not (view / "old_report.json").exists()


@pytest.mark.parametrize("arm", ["top8", "c4", "trace"])
def test_formal_mmlu_is_full_and_short_is_explicit(tmp_path, arm):
    model = tmp_path / "source"
    model.mkdir()
    (model / "config.json").write_text("{}")
    manifest = {"model": str(model), "spec": "sidecar"}
    command = evaluation_command(tmp_path, manifest, arm, "mmlu_pro", tmp_path / "formal", "artifact", 0.7)
    assert TASKS["mmlu_pro"]["count"] == 12032
    assert "--max_samples" not in command
    assert command[command.index("--max_new_tokens") + 1] == "32768"
    assert command[command.index("--expert_pruning_method") + 1] == ("none" if arm == "top8" else "Ban")
    short = evaluation_command(tmp_path, manifest, arm, "mmlu_pro", tmp_path / "short", "artifact", 0.7, short=True)
    assert short[short.index("--max_samples") + 1] == "2"
    assert short[short.index("--model_path") + 1] != command[command.index("--model_path") + 1]


def test_full_coverage_rejects_300_and_duplicate_ids():
    reference = {str(i) for i in range(12032)}
    with pytest.raises(ValueError, match="coverage"):
        require_coverage([str(i) for i in range(300)], reference, "mmlu_pro")
    with pytest.raises(ValueError, match="coverage"):
        require_coverage(["0", "0"], {"0"}, "task")
    require_coverage(sorted(reference), reference, "mmlu_pro")


def test_bootstrap_keeps_repeats_with_question_and_seed():
    values = np.array([[1, -1], [-1, 1], [0, 0]])
    assert clustered_bootstrap(values) == [0, 0]
    assert clustered_bootstrap(np.ones((3, 175))) == [100, 100]


def test_protocol_only_ignores_relocated_model_and_sidecar():
    a = {"model_name": "a", "seed": 1234, "additional_config": {"hif4_runtime_spec_path": "one"}}
    b = {"model_name": "b", "seed": 1234, "additional_config": {"hif4_runtime_spec_path": "two"}}
    assert protocol(a) == protocol(b)
    b["seed"] = 1235
    assert protocol(a) != protocol(b)


@pytest.mark.parametrize("target", [float("nan"), float("inf"), 4.9, 8.1])
def test_budget_rejects_targets_outside_route_bounds(target):
    with pytest.raises(ValueError, match="target experts"):
        budget_config(target)
