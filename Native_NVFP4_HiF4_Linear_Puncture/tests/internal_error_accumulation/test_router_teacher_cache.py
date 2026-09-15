"""Unit tests for E0 router teacher cache schema / isolation (plan O3.1 G 8–9)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.router_objective import (
    build_production_topk_teacher,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.router_teacher_cache import (
    STATUS_EMPTY,
    build_router_teacher_cache,
    reshape_flat_router_logits,
    slice_router_logits_by_lengths,
)


def test_reshape_flat_router_logits_requires_exact_bt():
    flat = torch.randn(12, 8)
    out = reshape_flat_router_logits(flat, batch_size=3, seq_len=4)
    assert tuple(out.shape) == (3, 4, 8)
    with pytest.raises(RuntimeError, match="guess sample boundaries"):
        reshape_flat_router_logits(torch.randn(11, 8), batch_size=3, seq_len=4)


def test_slice_router_logits_drops_padding_rows():
    logits = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
    lengths = torch.tensor([2, 4])
    parts = slice_router_logits_by_lengths(logits, lengths)
    assert parts[0].shape == (2, 3)
    assert parts[1].shape == (4, 3)
    assert torch.equal(parts[0], logits[0, :2])
    assert torch.equal(parts[1], logits[1, :4])


def test_empty_layers_returns_empty_without_scanning(tmp_path: Path):
    manifest_in = {
        "train_ids": ["a", "b"],
        "val_ids": ["c"],
        "source_ratio": {"wikitext2": 0.5, "s1k_original": 0.5},
    }
    out = build_router_teacher_cache(
        run_root=tmp_path,
        model_path="unused-when-empty",
        objective_split_manifest=manifest_in,
        layers=[],
        batch_size=2,
    )
    assert out["status"] == STATUS_EMPTY
    assert out["layers"] == []
    written = json.loads((tmp_path / "60_objective" / "router_teacher_cache" / "manifest.json").read_text())
    assert written["status"] == STATUS_EMPTY


def test_synthetic_cache_payload_alignment_and_isolation(tmp_path: Path):
    """Simulate Lxx_train/val payloads: sample/layer/length align, no padding, no ID leak."""
    cache_dir = tmp_path / "60_objective" / "router_teacher_cache"
    cache_dir.mkdir(parents=True)
    layer = 7
    top_k = 3
    num_experts = 11

    def _payload(sid: str, length: int) -> dict:
        logits = torch.randn(length, num_experts, dtype=torch.bfloat16)
        ids, weights = build_production_topk_teacher(logits.float(), top_k=top_k, norm_topk_prob=True)
        return {
            "sample_id": sid,
            "layer": layer,
            "length": length,
            "router_logits": logits,
            "topk_ids": ids.to(torch.int64),
            "topk_weights": weights.to(torch.float32),
        }

    train = {
        "train_a": _payload("train_a", 5),
        "train_b": _payload("train_b", 9),
    }
    val = {
        "val_x": _payload("val_x", 4),
    }
    # Deliberately ensure no padding rows were stored.
    for split_name, payload in (("train", train), ("val", val)):
        for sid, row in payload.items():
            assert row["router_logits"].shape[0] == row["length"]
            assert row["topk_ids"].shape[0] == row["length"]
            assert row["topk_weights"].shape[0] == row["length"]
            assert row["layer"] == layer
            assert row["sample_id"] == sid
        torch.save(payload, cache_dir / f"L{layer:02d}_{split_name}.pt")

    train_loaded = torch.load(cache_dir / f"L{layer:02d}_train.pt", map_location="cpu", weights_only=False)
    val_loaded = torch.load(cache_dir / f"L{layer:02d}_val.pt", map_location="cpu", weights_only=False)
    assert set(train_loaded) == {"train_a", "train_b"}
    assert set(val_loaded) == {"val_x"}
    assert set(train_loaded).isdisjoint(set(val_loaded))
