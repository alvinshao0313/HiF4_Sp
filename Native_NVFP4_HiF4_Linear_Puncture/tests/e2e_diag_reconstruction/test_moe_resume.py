from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.artifact import (
    save_layer_artifacts,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import (
    E2ETrainConfig,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training import (
    moe_resume,
    moe_trainer_resume,
)


def _z(value: float) -> dict[str, torch.Tensor]:
    return {
        "z_qkv": torch.full((2,), value),
        "z_vo": torch.full((1,), value),
        "z_gu": torch.full((2,), value),
        "z_ud": torch.full((1, 2), value),
    }


def test_load_resume_prefix_records_supports_legacy_and_v3(tmp_path):
    save_layer_artifacts(
        tmp_path,
        0,
        z=_z(0.1),
        metrics={"best_epoch": 1, "rollback": True, "accepted": False},
        train_log=[],
    )
    save_layer_artifacts(
        tmp_path,
        1,
        z=_z(0.2),
        candidate_z=_z(0.3),
        candidate_metrics={"candidate_best_epoch": 2},
        metrics={
            "best_epoch": 2,
            "candidate_best_epoch": 2,
            "rollback": True,
            "accepted": False,
            "loss_rollback_applied": False,
            "router_rollback_applied": True,
        },
        train_log=[],
    )
    cfg = E2ETrainConfig.for_test(output_dir=str(tmp_path), start_layer=2, end_layer=2)

    records, summaries = moe_resume.load_resume_prefix_records(cfg)

    assert set(records) == {0, 1}
    assert len(summaries) == 2
    torch.testing.assert_close(records[0]["candidate_z"]["z_gu"], _z(0.1)["z_gu"])
    torch.testing.assert_close(records[0]["adopted_z"]["z_gu"], _z(0.1)["z_gu"])
    torch.testing.assert_close(records[1]["candidate_z"]["z_gu"], _z(0.3)["z_gu"])
    torch.testing.assert_close(records[1]["adopted_z"]["z_gu"], _z(0.2)["z_gu"])
    assert records[1]["router_rollback_applied"] is True


def test_resume_prefix_policy_preserves_legacy_and_split_rollback_semantics():
    progressive = E2ETrainConfig.for_test(calib_input_mode="progressive_student")
    teacher = E2ETrainConfig.for_test(calib_input_mode="teacher")

    assert moe_resume.resume_prefix_uses_native(progressive, {"rollback": False}) is False
    assert moe_resume.resume_prefix_uses_native(progressive, {"rollback": True}) is True
    assert (
        moe_resume.resume_prefix_uses_native(
            progressive,
            {"rollback": True, "loss_rollback_applied": False, "router_rollback_applied": True},
        )
        is False
    )
    assert (
        moe_resume.resume_prefix_uses_native(
            progressive,
            {"rollback": True, "loss_rollback_applied": True, "router_rollback_applied": False},
        )
        is True
    )
    assert moe_resume.resume_prefix_uses_native(teacher, {"rollback": False}) is True


def test_replay_moe_prefix_uses_adopted_student_or_native(monkeypatch):
    cfg = E2ETrainConfig.for_test(start_layer=3, end_layer=3)
    records = {
        0: {"adopted_z": _z(0.1)},
        1: {"adopted_z": _z(0.2)},
        2: {"adopted_z": _z(0.3)},
    }
    summaries = [
        {"rollback": False},
        {"rollback": True},
        {"rollback": True, "loss_rollback_applied": False, "router_rollback_applied": True},
    ]
    released: list[int] = []
    calls: list[tuple[int, bool]] = []

    class DummyRuntime:
        def __init__(self, state):
            self.state = state

        def to(self, _device):
            return self

        def eval(self):
            return self

    class DummyDiag:
        def to(self, _device):
            return self

        def load_snapshot(self, snapshot):
            self.snapshot = snapshot

    monkeypatch.setattr(
        moe_resume,
        "load_qwen3_moe_layer_state",
        lambda _snapshot, layer_idx, _device: SimpleNamespace(
            layer_idx=layer_idx,
            spec=SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(moe_resume, "release_qwen3_moe_layer_state", lambda state: released.append(state.layer_idx))
    monkeypatch.setattr(moe_resume, "NativeQwen3MoELayerRuntime", DummyRuntime)
    monkeypatch.setattr(
        moe_resume,
        "StudentQwen3MoELayerRuntime",
        lambda state, _diag, **_kwargs: DummyRuntime(state),
    )
    monkeypatch.setattr(moe_resume, "build_moe_diag_state", lambda _spec, _mode: DummyDiag())

    def propagate(runtime, _snapshot, _samples, _collator, x_cache, _device, _batch_size, *, student):
        calls.append((runtime.state.layer_idx, student))
        return [*x_cache, runtime.state.layer_idx]

    out = moe_resume.replay_moe_prefix(
        cfg,
        Path("/tmp/source"),
        [],
        object(),
        [],
        torch.device("cpu"),
        records,
        summaries,
        propagate_fn=propagate,
    )

    assert out == [0, 1, 2]
    assert calls == [(0, True), (1, False), (2, True)]
    assert released == [0, 1, 2]


def test_resume_runtime_config_rejects_trajectory_changes(tmp_path):
    cfg = E2ETrainConfig.for_test(output_dir=str(tmp_path), start_layer=1, end_layer=1)
    previous = cfg.to_dict()
    previous["start_layer"] = 0
    previous["diag_batch_size"] = cfg.diag_batch_size + 1
    (tmp_path / "config.json").write_text(json.dumps(previous), encoding="utf-8")

    with pytest.raises(ValueError, match="diag_batch_size"):
        moe_trainer_resume._assert_resume_runtime_config(cfg)

    previous["diag_batch_size"] = cfg.diag_batch_size
    (tmp_path / "config.json").write_text(json.dumps(previous), encoding="utf-8")
    moe_trainer_resume._assert_resume_runtime_config(cfg)


def test_resume_wrapper_merges_prefix_and_does_not_overwrite_config(monkeypatch, tmp_path):
    cfg = E2ETrainConfig.for_test(output_dir=str(tmp_path), start_layer=2, end_layer=2)
    prefix_records = {0: {"z": "old0"}, 1: {"z": "old1"}}
    prefix_summaries = [{"layer_id": 0}, {"layer_id": 1}]
    saved: dict[str, object] = {}
    writes: list[tuple[Path, object]] = []

    monkeypatch.setattr(moe_trainer_resume, "assert_resume_artifacts", lambda _cfg: None)
    monkeypatch.setattr(moe_trainer_resume, "_assert_resume_runtime_config", lambda _cfg: None)
    monkeypatch.setattr(moe_trainer_resume, "resolve_local_snapshot", lambda _model: str(tmp_path))
    monkeypatch.setattr(
        moe_trainer_resume,
        "_load_tokenizer",
        lambda _snapshot: SimpleNamespace(pad_token_id=0, eos_token_id=0, pad_token=None),
    )
    monkeypatch.setattr(moe_trainer_resume, "build_or_load_calibration", lambda *_args, **_kwargs: (["train"], ["val"]))
    monkeypatch.setattr(moe_trainer_resume, "DynamicCalibrationCollator", lambda _pad: "collator")
    monkeypatch.setattr(moe_trainer_resume, "load_resume_prefix_records", lambda _cfg: (prefix_records, prefix_summaries))
    monkeypatch.setattr(moe_trainer_resume, "replay_moe_prefix", lambda *_args, **_kwargs: "replayed-cache")

    base = moe_trainer_resume._base
    original_cache = lambda *_args, **_kwargs: "initial-cache"
    original_calibration = lambda *_args, **_kwargs: (["base-train"], ["base-val"])

    def original_save(*, cfg, layer_records, out_dir):
        saved["layers"] = dict(layer_records)
        saved["out_dir"] = Path(out_dir)
        return Path(out_dir) / "conversion_state.pt"

    def original_write(path, payload):
        writes.append((Path(path), payload))

    def fake_base_train(_cfg, _device):
        assert base.build_initial_moe_hidden_cache(None) == "replayed-cache"
        assert base.build_or_load_calibration(None) == (["train"], ["val"])
        base.save_conversion_artifact(
            cfg=_cfg,
            layer_records={2: {"z": "new2"}},
            out_dir=tmp_path,
        )
        base.write_json(tmp_path / "config.json", {"should_not": "overwrite"})
        base.write_json(tmp_path / "summary.json", {"layers": [{"layer_id": 2}]})

    monkeypatch.setattr(base, "build_initial_moe_hidden_cache", original_cache)
    monkeypatch.setattr(base, "build_or_load_calibration", original_calibration)
    monkeypatch.setattr(base, "save_conversion_artifact", original_save)
    monkeypatch.setattr(base, "write_json", original_write)
    monkeypatch.setattr(base, "train_qwen3_moe_lazy", fake_base_train)

    moe_trainer_resume.train_qwen3_moe_lazy(cfg, torch.device("cpu"))

    assert set(saved["layers"]) == {0, 1, 2}
    assert all(path.name != "config.json" for path, _payload in writes)
    summary_payload = next(payload for path, payload in writes if path.name == "summary.json")
    assert summary_payload["layers"] == [
        {"layer_id": 0},
        {"layer_id": 1},
        {"layer_id": 2},
    ]
    assert base.build_initial_moe_hidden_cache is original_cache
    assert base.build_or_load_calibration is original_calibration
    assert base.save_conversion_artifact is original_save
    assert base.write_json is original_write
