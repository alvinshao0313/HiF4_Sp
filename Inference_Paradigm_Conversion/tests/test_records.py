from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from Inference_Paradigm_Conversion.ipc_analysis.config import (
    load_experiment_config,
    resolve_representative_layers,
    validate_experiment_config,
)
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import atomic_write_json, read_json
from Inference_Paradigm_Conversion.ipc_analysis.records import (
    LinearDecompositionRecord,
    RootCauseRecord,
    RunManifest,
    TensorMetricRecord,
)


def test_schema_roundtrip():
    m = RunManifest(
        run_id="t",
        git_commit="none",
        device="cpu",
        torch_version="x",
        status="ok",
        notes="",
        source_checkpoint="Qmodel/x",
        source_semantics="nvfp4_qat_fake_dequant_bf16",
        source_weight_dtype="bfloat16",
        source_weight_tensor_count=1,
        num_hidden_layers=36,
        resolved_representative_layers=[4, 18, 34],
        hadamard_runtime="disabled",
        nvfp4_activation_scale_file="scales.safetensors",
        nvfp4_activation_scale_count=252,
        matched_activation_module_count=252,
    )
    m2 = RunManifest.from_dict(m.to_dict())
    assert m2.resolved_representative_layers == [4, 18, 34]
    assert m2.source_semantics == "nvfp4_qat_fake_dequant_bf16"

    t = TensorMetricRecord(
        path_id="P1_semantic",
        layer_idx=0,
        module_name="model.layers.0.mlp.down_proj",
        projection="down_proj",
        phase="prefill",
        reference_name="W_N",
        target_name="W_H",
        nmse=0.1,
        sqnr_db=10.0,
        cosine=0.99,
        mae=0.01,
        mean_signed_error=0.0,
        max_abs_error=1.0,
        relative_norm_change=0.0,
        reference_energy=1.0,
        target_energy=1.0,
        error_energy=0.1,
        error_p50=0.0,
        error_p90=0.0,
        error_p99=0.0,
        error_p99_9=0.0,
        top1pct_error_energy_share=0.2,
        numel=10,
    )
    assert TensorMetricRecord.from_dict(t.to_dict()).nmse == 0.1

    l = LinearDecompositionRecord(
        path_id="P2_matched_semantic",
        layer_idx=1,
        module_name="m",
        projection="q_proj",
        phase="decode",
        sample_id="s0",
        energy_wn_an=1.0,
        energy_delta_w_an=0.1,
        energy_wn_delta_a=0.2,
        energy_delta_w_delta_a=0.01,
        cross_dw_an_wn_da=0.0,
        cross_dw_an_dw_da=0.0,
        cross_wn_da_dw_da=0.0,
        residual_rel=1e-12,
    )
    assert LinearDecompositionRecord.from_dict(l.to_dict()).energy_delta_w_delta_a == 0.01

    r = RootCauseRecord(
        cause_id="C1",
        hypothesis_id="H1a",
        mechanism="16_to_64_dispersion",
        path_id="P1_semantic",
        evidence_class="controlled_causal_evidence",
        recoverable_error_fraction=0.3,
        affected_scope="weight_groups",
        metric_name="output_nmse",
        metric_value=0.05,
    )
    assert RootCauseRecord.from_dict(r.to_dict()).cause_id == "C1"


def test_config_rejects_ordinary_bf16_semantics(tmp_path: Path):
    src = Path(
        "Inference_Paradigm_Conversion/configs/qwen3_8b_nvfp4_qat_formal.yaml"
    )
    text = src.read_text(encoding="utf-8").replace(
        "source_semantics: nvfp4_qat_fake_dequant_bf16",
        "source_semantics: bf16",
    )
    bad = tmp_path / "bad.yaml"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="nvfp4_qat_fake_dequant_bf16"):
        load_experiment_config(bad)


def test_formal_config_loads():
    cfg = load_experiment_config()
    assert cfg.model.source_semantics == "nvfp4_qat_fake_dequant_bf16"
    assert cfg.model.hadamard_runtime == "disabled"
    assert "Qwen3-8B-FPQuant-QAT-NVFP4-Dequant-BF16-NoHadamard" in cfg.model.source_checkpoint
    assert "Qwen3-8B/" not in cfg.model.source_checkpoint or "FPQuant" in cfg.model.source_checkpoint
    validate_experiment_config(cfg)


def test_representative_layers_qwen3_8b():
    assert resolve_representative_layers(36) == [4, 18, 34]


def test_atomic_json_roundtrip(tmp_path: Path):
    path = tmp_path / "a.json"
    atomic_write_json(path, {"x": 1, "y": [1, 2]})
    assert read_json(path)["x"] == 1
