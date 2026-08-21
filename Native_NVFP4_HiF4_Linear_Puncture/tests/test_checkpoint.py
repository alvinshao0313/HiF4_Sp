"""Checkpoint / preflight gates (synthetic; no 8B download)."""

from __future__ import annotations

from pathlib import Path

import pytest

from Native_NVFP4_HiF4_Linear_Puncture.src import checkpoint as checkpoint_mod
from Native_NVFP4_HiF4_Linear_Puncture.src import config as config_mod

OLD_DEQUANT_ID = "Qmodel/Qwen3-8B-FPQuant-QAT-NVFP4-Dequant-BF16-NoHadamard"
NATIVE_MODEL_ID = "DASLab/Qwen3-8B-FPQuant-QAT-NVFP4"
SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def test_missing_local_snapshot_never_falls_back_to_old_dequant_ckpt(monkeypatch):
    tried_ids: list[str] = []

    def fake_snapshot_download(*, repo_id: str, local_files_only: bool = False, **kwargs):
        tried_ids.append(repo_id)
        assert local_files_only is True
        raise FileNotFoundError(
            "Native NVFP4 checkpoint is not fully available in local HF cache."
        )

    monkeypatch.setattr(checkpoint_mod, "snapshot_download", fake_snapshot_download)

    with pytest.raises((FileNotFoundError, RuntimeError, SystemExit, OSError)):
        checkpoint_mod.resolve_local_snapshot(NATIVE_MODEL_ID)

    assert tried_ids == [NATIVE_MODEL_ID]
    assert OLD_DEQUANT_ID not in tried_ids


def test_preflight_requires_packed_weight_triplet_for_every_target_linear():
    coverage = {
        "packed_weight_coverage": 251,
        "weight_scale_coverage": 252,
        "weight_global_scale_coverage": 252,
        "target_linear_count": 252,
    }
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        checkpoint_mod.assert_packed_triplet_coverage(coverage)


def test_preflight_requires_activation_global_scale_for_every_target_linear():
    coverage = {
        "activation_global_scale_coverage": 250,
        "target_linear_count": 252,
    }
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        checkpoint_mod.assert_activation_global_scale_coverage(coverage)


def test_rotation_source_must_be_explicit():
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        checkpoint_mod.validate_rotation_source(None)
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        checkpoint_mod.validate_rotation_source("")
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        checkpoint_mod.validate_rotation_source("guessed_h16")
    checkpoint_mod.validate_rotation_source("checkpoint_tensor")
    checkpoint_mod.validate_rotation_source("reconstructed_from_official_config")


def test_config_rejects_non_nvfp4_forward_dtype():
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        config_mod.validate_forward_dtype("bf16")
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        config_mod.validate_forward_dtype("mxfp8")
    config_mod.validate_forward_dtype("nvfp4")


def test_src_tree_forbids_inference_paradigm_conversion_import():
    if not SRC_DIR.is_dir():
        pytest.skip("src/ not present yet")
    offenders: list[str] = []
    for path in SRC_DIR.rglob("*.py"):
        if "Inference_Paradigm_Conversion" in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(SRC_DIR.parent)))
    assert offenders == []
