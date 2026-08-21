from __future__ import annotations

import json
from pathlib import Path

import pytest

from obs_compensation.config import OBSCompensationConfig, build_config, parse_args


def _valid_kwargs(tmp_path: Path) -> dict:
    return {
        "model_path": "Qwen/Qwen3.5-4B",
        "source_artifacts_dir": tmp_path / "source",
        "output_dir": tmp_path / "out",
        "calibration_dataset": "s1k",
        "calibration_samples": 8,
        "sequence_length": 128,
        "obs_percdamp": 0.01,
        "solver_block_size": 64,
        "obs_order_policy": "auto",
        "dtype": "bfloat16",
        "device": "cpu",
        "seed": 42,
        "trust_remote_code": True,
    }


def test_config_accepts_valid_values(tmp_path):
    cfg = OBSCompensationConfig(**_valid_kwargs(tmp_path))
    cfg.validate_paths(require_source_exists=False)


def test_config_rejects_invalid_numeric_values(tmp_path):
    base = _valid_kwargs(tmp_path)
    for key, value in {
        "calibration_samples": 0,
        "sequence_length": 1,
        "obs_percdamp": 0.0,
        "solver_block_size": 0,
    }.items():
        kwargs = dict(base)
        kwargs[key] = value
        with pytest.raises(ValueError):
            OBSCompensationConfig(**kwargs)


def test_config_rejects_output_equal_to_source(tmp_path):
    path = tmp_path / "same"
    kwargs = _valid_kwargs(tmp_path)
    kwargs["source_artifacts_dir"] = path
    kwargs["output_dir"] = path
    with pytest.raises(ValueError, match="must differ"):
        OBSCompensationConfig(**kwargs)


def test_config_rejects_invalid_dataset_and_dtype(tmp_path):
    base = _valid_kwargs(tmp_path)
    with pytest.raises(ValueError, match="calibration_dataset"):
        OBSCompensationConfig(**{**base, "calibration_dataset": "c4"})
    with pytest.raises(ValueError, match="dtype"):
        OBSCompensationConfig(**{**base, "dtype": "float64"})


def test_config_rejects_invalid_order_policy(tmp_path):
    base = _valid_kwargs(tmp_path)
    with pytest.raises(ValueError, match="obs_order_policy"):
        OBSCompensationConfig(**{**base, "obs_order_policy": "mask_count"})


def test_parse_args_defaults_order_policy_auto():
    args = parse_args(
        [
            "--source_artifacts_dir",
            "/tmp/source",
            "--output_dir",
            "/tmp/out",
        ]
    )
    assert args.obs_order_policy == "auto"
    assert args.model_path is None


def test_build_config_uses_source_model_path_when_missing(tmp_path):
    args = parse_args(
        [
            "--source_artifacts_dir",
            str(tmp_path / "source"),
            "--output_dir",
            str(tmp_path / "out"),
            "--device",
            "cpu",
        ]
    )
    cfg = build_config(args, "tiny-model")
    assert cfg.model_path == "tiny-model"


def test_build_config_rejects_model_path_mismatch(tmp_path):
    args = parse_args(
        [
            "--model_path",
            "other-model",
            "--source_artifacts_dir",
            str(tmp_path / "source"),
            "--output_dir",
            str(tmp_path / "out"),
        ]
    )
    with pytest.raises(ValueError, match="does not exactly match"):
        build_config(args, "tiny-model")


def test_validate_paths_rejects_nonempty_output(tmp_path):
    source = tmp_path / "source"
    out = tmp_path / "out"
    source.mkdir()
    out.mkdir()
    (out / "marker.txt").write_text("x", encoding="utf-8")
    cfg = OBSCompensationConfig(**{**_valid_kwargs(tmp_path), "source_artifacts_dir": source, "output_dir": out})
    with pytest.raises(ValueError, match="non-empty"):
        cfg.validate_paths(require_source_exists=True)


def test_obs_percdamp_upper_bound(tmp_path):
    base = _valid_kwargs(tmp_path)
    OBSCompensationConfig(**{**base, "obs_percdamp": 1.0})
    with pytest.raises(ValueError):
        OBSCompensationConfig(**{**base, "obs_percdamp": 1.01})
