from __future__ import annotations

import json

import torch
from safetensors.torch import save_file

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.artifact import (
    save_conversion_artifact,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import E2ETrainConfig
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation import moe_materialize
from Native_NVFP4_HiF4_Linear_Puncture.tests.e2e_diag_reconstruction.test_moe_fold import _state


def test_moe_materialize_writes_layer_shard_index_and_sidecar(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "qwen3_moe", "quantization_config": {"quant_method": "modelopt"}}),
        encoding="utf-8",
    )
    save_file(
        {
            "model.embed_tokens.weight": torch.ones(2, 4, dtype=torch.bfloat16),
            "model.norm.weight": torch.ones(4, dtype=torch.bfloat16),
            "lm_head.weight": torch.ones(2, 4, dtype=torch.bfloat16),
        },
        source / "non-layer.safetensors",
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "model.embed_tokens.weight": "non-layer.safetensors",
                    "model.norm.weight": "non-layer.safetensors",
                    "lm_head.weight": "non-layer.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = E2ETrainConfig.for_test(output_dir=str(tmp_path / "run"))
    adopted = {
        "z_qkv": torch.zeros(2048),
        "z_vo": torch.zeros(512),
        "z_gu": torch.zeros(2048),
        "z_ud": torch.zeros(2, 768),
    }
    candidate = {name: value.clone() for name, value in adopted.items()}
    candidate["z_gu"].fill_(0.25)
    artifact = save_conversion_artifact(
        cfg=cfg,
        layer_records={
            0: {
                "accepted": False,
                "rollback": True,
                "best_epoch": 0,
                "candidate_z": candidate,
                "adopted_z": adopted,
                "z": adopted,
                "router_rollback_applied": True,
            }
        },
        out_dir=tmp_path / "run",
    )
    monkeypatch.setattr(moe_materialize, "load_qwen3_moe_layer_state", lambda *_args, **_kwargs: _state())
    monkeypatch.setattr(moe_materialize, "release_qwen3_moe_layer_state", lambda _state: None)
    out = moe_materialize.materialize_moe_checkpoint(
        source_snapshot=source,
        artifact_path=artifact,
        output_dir=tmp_path / "materialized",
        diag_variant="candidate",
    )
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert "quantization_config" not in config
    assert config["torch_dtype"] == "bfloat16"
    index = json.loads((out / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index["weight_map"]
    assert index["weight_map"]["model.embed_tokens.weight"] == "model-non-layer.safetensors"
    assert index["weight_map"]["model.norm.weight"] == "model-non-layer.safetensors"
    assert index["weight_map"]["lm_head.weight"] == "model-non-layer.safetensors"
    assert all(not key.endswith((".weight_scale", ".weight_scale_2", ".input_scale")) for key in index["weight_map"])
    sidecar = torch.load(out / "hif4_runtime_spec.pt", map_location="cpu", weights_only=False)
    assert sidecar["model_type"] == "qwen3_moe"
    assert sidecar["variant"] == "fusable"
    assert sidecar["artifact_diag_variant"] == "candidate"
