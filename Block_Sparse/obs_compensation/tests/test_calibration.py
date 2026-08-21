from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from obs_compensation.calibration import (
    build_calibration_samples,
    make_calibration_sample,
)
from obs_compensation.config import OBSCompensationConfig


def test_make_calibration_sample_valid():
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    sample = make_calibration_sample(ids)
    ids[0, 0] = 99
    assert sample.input_ids.tolist() == [[1, 2, 3]]
    assert sample.attention_mask.tolist() == [[1, 1, 1]]
    assert sample.input_ids.device.type == "cpu"
    assert sample.input_ids.is_contiguous()


def test_make_calibration_sample_rejects_bad_inputs():
    with pytest.raises(ValueError, match="\\[1, T\\]"):
        make_calibration_sample(torch.tensor([1, 2]))
    with pytest.raises(ValueError, match="too short"):
        make_calibration_sample(torch.tensor([[1]], dtype=torch.long))
    with pytest.raises(TypeError, match="integer"):
        make_calibration_sample(torch.tensor([[1.0, 2.0]]))


class _FakeTokenizer:
    def __call__(self, text, add_special_tokens=False, return_tensors="pt", truncation=False):
        del add_special_tokens, truncation, return_tensors
        # deterministic: length from text length
        ids = [i + 1 for i in range(max(len(text), 2))]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def _cfg(tmp_path, **kwargs):
    base = dict(
        model_path="tiny",
        source_artifacts_dir=tmp_path / "src",
        output_dir=tmp_path / "out",
        calibration_dataset="s1k",
        calibration_samples=2,
        sequence_length=4,
        obs_percdamp=0.01,
        solver_block_size=8,
        obs_order_policy="auto",
        dtype="float32",
        device="cpu",
        seed=0,
        trust_remote_code=True,
    )
    base.update(kwargs)
    return OBSCompensationConfig(**base)


def test_build_s1k_samples(monkeypatch, tmp_path):
    import obs_compensation.calibration as cal

    rows = [{"text": "abcd"}, {"text": "efghij"}, {"text": "kl"}]
    captured = {}

    def fake_load_dataset(dataset_id, split="train"):
        captured["dataset_id"] = dataset_id
        captured["split"] = split
        return rows

    monkeypatch.setattr(cal, "load_dataset", fake_load_dataset)
    samples = build_calibration_samples(_FakeTokenizer(), _cfg(tmp_path, calibration_samples=2, sequence_length=3, seed=0))
    assert captured["dataset_id"] == "simplescaling/s1K-1.1_tokenized"
    assert captured["split"] == "train"
    assert len(samples) == 2
    assert all(s.input_ids.shape[1] <= 3 for s in samples)
    assert all(s.input_ids.shape[1] >= 2 for s in samples)


def test_s1k_empty_text_raises(monkeypatch, tmp_path):
    import obs_compensation.calibration as cal

    monkeypatch.setattr(
        cal,
        "load_dataset",
        lambda *a, **k: [{"text": "   "}],
    )
    with pytest.raises(ValueError, match="empty"):
        build_calibration_samples(
            _FakeTokenizer(),
            _cfg(tmp_path, calibration_samples=1, seed=0),
        )


def test_build_wikitext2_samples(monkeypatch, tmp_path):
    import obs_compensation.calibration as cal

    captured = {}

    class _DS(list):
        pass

    def fake_load_dataset(dataset_id, config=None, split="train"):
        captured["dataset_id"] = dataset_id
        captured["config"] = config
        captured["split"] = split
        return _DS(
            [
                {"text": "hello world"},
                {"text": ""},
                {"text": "more text here"},
            ]
        )

    monkeypatch.setattr(cal, "load_dataset", fake_load_dataset)

    class Tok:
        def __call__(self, text, add_special_tokens=False, return_tensors="pt", truncation=False):
            del add_special_tokens, truncation, return_tensors
            # long enough corpus
            ids = list(range(1, 40))
            assert "\n\n" in text
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    samples = build_calibration_samples(
        Tok(),
        _cfg(
            tmp_path,
            calibration_dataset="wikitext2",
            calibration_samples=3,
            sequence_length=8,
            seed=1,
        ),
    )
    assert captured["dataset_id"] == "Salesforce/wikitext"
    assert captured["config"] == "wikitext-2-raw-v1"
    assert len(samples) == 3
    assert all(s.input_ids.shape == (1, 8) for s in samples)


def test_wikitext_too_short_raises(monkeypatch, tmp_path):
    import obs_compensation.calibration as cal

    monkeypatch.setattr(
        cal,
        "load_dataset",
        lambda *a, **k: [{"text": "abc"}],
    )

    class Tok:
        def __call__(self, text, add_special_tokens=False, return_tensors="pt", truncation=False):
            return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}

    with pytest.raises(ValueError, match="corpus has"):
        build_calibration_samples(
            Tok(),
            _cfg(
                tmp_path,
                calibration_dataset="wikitext2",
                sequence_length=16,
                calibration_samples=1,
            ),
        )
