"""Theoretical grid-scale activation validation tests."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from Native_NVFP4_HiF4_Linear_Puncture.src.config import TARGET_PROJECTIONS, load_config
from Native_NVFP4_HiF4_Linear_Puncture.src.formats import qdq_hif4_direct
from Native_NVFP4_HiF4_Linear_Puncture.src.grid_scale_validation import (
    REQUIRED_FORMAL_LAYERS,
    STANDARD_S0_DIVISOR,
    THEORY_GRID_SCALE,
    THEORY_S0_DIVISOR,
    aggregate_by_projection,
    aggregate_rows,
    assert_distinct_run_ids,
    build_arg_parser,
    build_hif4_config,
    configs_differ_only_in_s0_divisor,
    evaluate_capture,
    expected_capture_paths,
    nonzero_to_zero_rate,
    theory_s0_divisor,
    validate_capture_manifest,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.native_nvfp4 import qdq_nvfp4_post_rotation
import Native_NVFP4_HiF4_Linear_Puncture.src.grid_scale_validation as gsv


def _std_cfg(**kwargs):
    params = {
        "group_size": 64,
        "s0_divisor": STANDARD_S0_DIVISOR,
        "e8_threshold": 4.0,
        "e4_threshold": 2.0,
        "s0_mode": "hardware",
    }
    params.update(kwargs)
    return build_hif4_config(**params)


def test_theory_s0_divisor_scales_grid_up():
    d = theory_s0_divisor(7.0)
    assert abs(d - 5.387535138334382) < 1e-12
    assert d < 7.0
    assert abs(d - THEORY_S0_DIVISOR) < 1e-12
    assert abs(THEORY_GRID_SCALE - 1.299295470055) < 1e-12


def test_theory_s0_divisor_rejects_non_positive():
    with pytest.raises(ValueError):
        theory_s0_divisor(0.0)
    with pytest.raises(ValueError):
        theory_s0_divisor(-7.0)


def test_standard_and_theory_configs_differ_only_in_s0_divisor():
    standard = _std_cfg(s0_divisor=7.0)
    theory = _std_cfg(s0_divisor=theory_s0_divisor(7.0))
    assert configs_differ_only_in_s0_divisor(standard, theory)
    d_std = asdict(standard)
    d_theory = asdict(theory)
    for key in d_std:
        if key == "s0_divisor":
            assert d_std[key] != d_theory[key]
            assert d_theory[key] < d_std[key]
        else:
            assert d_std[key] == d_theory[key]
    assert standard.group_dim == -1
    assert theory.group_dim == -1
    assert standard.enable_exp8 is True
    assert standard.enable_exp4 is True


def test_nonzero_to_zero_rate_and_all_zero_nan():
    source = torch.tensor([1.0, 0.0, 2.0, 3.0])
    target = torch.tensor([0.0, 0.0, 2.0, 0.0])
    assert abs(nonzero_to_zero_rate(source, target) - (2.0 / 3.0)) < 1e-12
    assert math.isnan(nonzero_to_zero_rate(torch.zeros(4), target))


def _synthetic_capture() -> dict:
    torch.manual_seed(0)
    return {
        "module_name": "synthetic.layers.2.self_attn.q_proj",
        "layer_idx": 2,
        "projection": "q_proj",
        "split": "val",
        "x_rot_bf16": torch.randn(4, 64, dtype=torch.bfloat16),
        "input_global_scale_fp32": torch.tensor(0.25, dtype=torch.float32),
    }


def test_evaluate_capture_synthetic_64d_semantics():
    capture = _synthetic_capture()
    standard = _std_cfg()
    theory = _std_cfg(s0_divisor=theory_s0_divisor(7.0))
    out = evaluate_capture(
        capture,
        hif4_base_config=standard,
        hif4_theory_config=theory,
        nvfp4_group_size=16,
        device=torch.device("cpu"),
    )
    x = capture["x_rot_bf16"].to(dtype=torch.float32)
    scale = capture["input_global_scale_fp32"]
    a_n = qdq_nvfp4_post_rotation(x, scale, group_size=16).to(torch.float32)
    a_h_std = qdq_hif4_direct(x, config=standard, output_dtype=torch.float32)
    a_h_theory = qdq_hif4_direct(x, config=theory, output_dtype=torch.float32)
    assert a_n.shape == x.shape == a_h_std.shape == a_h_theory.shape
    assert a_n.dtype == a_h_std.dtype == a_h_theory.dtype == torch.float32
    assert out["num_elements"] == 4 * 64
    for key in (
        "error_energy_std_vs_an",
        "error_energy_theory_vs_an",
        "error_energy_std_vs_xrot",
        "error_energy_theory_vs_xrot",
        "reference_energy_an",
        "reference_energy_xrot",
    ):
        assert out[key] >= 0.0
        assert math.isfinite(out[key])


def test_evaluate_capture_quantizes_the_same_x_rot(monkeypatch):
    capture = _synthetic_capture()
    seen = []

    def fake_hif4(x, *, config=None, output_dtype=None):
        seen.append(x.detach().cpu().clone())
        return x.to(dtype=output_dtype or torch.float32)

    monkeypatch.setattr(gsv, "qdq_hif4_direct", fake_hif4)
    evaluate_capture(
        capture,
        hif4_base_config=_std_cfg(),
        hif4_theory_config=_std_cfg(s0_divisor=theory_s0_divisor(7.0)),
        nvfp4_group_size=16,
        device=torch.device("cpu"),
    )
    assert len(seen) == 2
    assert torch.equal(seen[0], seen[1])


def test_evaluate_uses_nvfp4_qdq_as_an_reference(monkeypatch):
    capture = _synthetic_capture()
    called = {"n": 0}
    real = gsv.qdq_nvfp4_post_rotation

    def wrapped(x, scale, group_size=16):
        called["n"] += 1
        return real(x, scale, group_size=group_size)

    monkeypatch.setattr(gsv, "qdq_nvfp4_post_rotation", wrapped)
    evaluate_capture(
        capture,
        hif4_base_config=_std_cfg(),
        hif4_theory_config=_std_cfg(s0_divisor=theory_s0_divisor(7.0)),
        nvfp4_group_size=16,
        device=torch.device("cpu"),
    )
    assert called["n"] == 1


def test_recovery_is_nan_when_std_error_energy_is_zero(monkeypatch):
    capture = _synthetic_capture()

    def identity_nvfp4(x, scale, group_size=16):
        return x

    def identity_hif4(x, *, config=None, output_dtype=None):
        return x.to(dtype=output_dtype or torch.float32)

    monkeypatch.setattr(gsv, "qdq_nvfp4_post_rotation", identity_nvfp4)
    monkeypatch.setattr(gsv, "qdq_hif4_direct", identity_hif4)
    out = evaluate_capture(
        capture,
        hif4_base_config=_std_cfg(),
        hif4_theory_config=_std_cfg(s0_divisor=theory_s0_divisor(7.0)),
        nvfp4_group_size=16,
        device=torch.device("cpu"),
    )
    assert out["error_energy_std_vs_an"] == 0.0
    assert math.isnan(out["recovery_mse_vs_an"])
    assert math.isnan(out["recovery_mse_vs_xrot"])


def _metric_row(
    *,
    split: str,
    projection: str,
    num_elements: int,
    e_std_an: float,
    e_theory_an: float,
    e_std_x: float,
    e_theory_x: float,
    ref_an: float,
    ref_x: float,
    module_name: str = "m",
    layer_idx: int = 2,
) -> dict:
    n = num_elements
    return {
        "split": split,
        "projection": projection,
        "module_name": module_name,
        "layer_idx": layer_idx,
        "num_elements": n,
        "error_energy_std_vs_an": e_std_an,
        "error_energy_theory_vs_an": e_theory_an,
        "mse_std_vs_an": e_std_an / n,
        "mse_theory_vs_an": e_theory_an / n,
        "recovery_mse_vs_an": (
            float("nan") if e_std_an == 0.0 else 1.0 - e_theory_an / e_std_an
        ),
        "error_energy_std_vs_xrot": e_std_x,
        "error_energy_theory_vs_xrot": e_theory_x,
        "reference_energy_an": ref_an,
        "reference_energy_xrot": ref_x,
        "zero_rate_an": 0.1,
        "zero_rate_hif4_std": 0.2,
        "zero_rate_hif4_theory": 0.15,
        "an_nonzero_to_hif4_std_zero_rate": 0.05,
        "an_nonzero_to_hif4_theory_zero_rate": 0.04,
    }


def test_aggregate_uses_total_energy_not_module_mean():
    rows = [
        _metric_row(
            split="val",
            projection="q_proj",
            num_elements=10,
            e_std_an=10.0,
            e_theory_an=1.0,
            e_std_x=8.0,
            e_theory_x=4.0,
            ref_an=20.0,
            ref_x=30.0,
            module_name="small",
        ),
        _metric_row(
            split="val",
            projection="q_proj",
            num_elements=1000,
            e_std_an=100.0,
            e_theory_an=90.0,
            e_std_x=50.0,
            e_theory_x=40.0,
            ref_an=200.0,
            ref_x=300.0,
            module_name="large",
        ),
    ]
    agg = aggregate_rows(rows)
    expected_mse_std = 110.0 / 1010.0
    mean_mse_std = (1.0 + 0.1) / 2.0
    assert abs(agg["global_mse_std_vs_an"] - expected_mse_std) < 1e-12
    assert abs(agg["global_mse_std_vs_an"] - mean_mse_std) > 0.1
    expected_recovery = 1.0 - 91.0 / 110.0
    mean_recovery = (0.9 + 0.1) / 2.0
    assert abs(agg["global_recovery_vs_an"] - expected_recovery) < 1e-12
    assert abs(agg["global_recovery_vs_an"] - mean_recovery) > 0.1
    assert abs(agg["global_nmse_std_vs_an"] - (110.0 / 220.0)) < 1e-12
    assert agg["num_modules_positive_recovery"] == 2
    assert abs(agg["global_mse_theory_vs_an"] - (91.0 / 1010.0)) < 1e-12


def test_aggregate_by_projection_has_14_rows_and_energy_weighting():
    rows = []
    for split in ("cal", "val"):
        for i, proj in enumerate(TARGET_PROJECTIONS):
            rows.append(
                _metric_row(
                    split=split,
                    projection=proj,
                    num_elements=10 + i,
                    e_std_an=2.0,
                    e_theory_an=1.0,
                    e_std_x=3.0,
                    e_theory_x=1.5,
                    ref_an=4.0,
                    ref_x=5.0,
                    module_name=f"{split}_{proj}",
                    layer_idx=2,
                )
            )
    out = aggregate_by_projection(rows)
    assert len(out) == 14
    splits = {(r["split"], r["projection"]) for r in out}
    assert len(splits) == 14
    q_val = next(r for r in out if r["split"] == "val" and r["projection"] == "q_proj")
    assert abs(q_val["recovery_vs_an"] - 0.5) < 1e-12
    assert q_val["num_modules"] == 1


def test_cli_has_no_theory_scale_flag():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--run-id",
                "a",
                "--capture-run-id",
                "b",
                "--theory-scale",
                "1.2",
            ]
        )


def test_run_ids_must_differ():
    with pytest.raises(ValueError):
        assert_distinct_run_ids("same", "same")
    assert_distinct_run_ids("src", "dst")


def test_expected_capture_paths_are_70():
    config = load_config()
    paths = expected_capture_paths(config, Path("/tmp/fake_run"))
    assert len(paths) == 70
    assert len(config.formal_module_names) == 35
    assert tuple(config.experiment.formal_layers) == REQUIRED_FORMAL_LAYERS


def test_manifest_gate_rejects_wrong_mode(tmp_path: Path):
    config = load_config()
    manifest = {
        "capture_mode": "smoke",
        "capture_coverage": "35/35",
        "module_count": 35,
        "capture_point": "post_rotation_pre_activation_quant",
        "source_semantic_version": "native_nvfp4_rot_a4_v1",
        "formal_layers": [2, 10, 18, 26, 34],
    }
    with pytest.raises(ValueError, match="capture_mode"):
        validate_capture_manifest(config, tmp_path, manifest)


def test_manifest_gate_rejects_missing_file(tmp_path: Path):
    config = load_config()
    capture_dir = tmp_path
    cap_root = capture_dir / "captures"
    cap_root.mkdir()
    paths = expected_capture_paths(config, capture_dir)
    for path in paths[:-1]:
        path.write_bytes(b"x")
    manifest = {
        "capture_mode": "formal",
        "capture_coverage": "35/35",
        "module_count": 35,
        "capture_point": "post_rotation_pre_activation_quant",
        "source_semantic_version": "native_nvfp4_rot_a4_v1",
        "formal_layers": [2, 10, 18, 26, 34],
    }
    with pytest.raises(FileNotFoundError):
        validate_capture_manifest(config, capture_dir, manifest)


def test_manifest_gate_accepts_complete_files(tmp_path: Path):
    config = load_config()
    cap_root = tmp_path / "captures"
    cap_root.mkdir()
    for path in expected_capture_paths(config, tmp_path):
        path.write_bytes(b"x")
    manifest = {
        "capture_mode": "formal",
        "capture_coverage": "35/35",
        "module_count": 35,
        "capture_point": "post_rotation_pre_activation_quant",
        "source_semantic_version": "native_nvfp4_rot_a4_v1",
        "formal_layers": [2, 10, 18, 26, 34],
    }
    validate_capture_manifest(config, tmp_path, manifest)


def test_module_source_has_locked_theory_constants():
    source = Path(gsv.__file__).read_text(encoding="utf-8")
    assert "THEORY_GRID_SCALE: float = 1.299295470055" in source
    assert "--theory-scale" not in source
    assert "checkpoint" not in source
    assert "linear_cases" not in source
    assert "resolve_local_snapshot" not in source
