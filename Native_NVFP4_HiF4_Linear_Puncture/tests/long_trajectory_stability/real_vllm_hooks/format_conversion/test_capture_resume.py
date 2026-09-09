from pathlib import Path

import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.hook_state import HookCaptureState
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.worker_hooks import _capture
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion import VARIANTS
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.run_go import ArtifactTask


def test_feature_scan_records_only_requested_inputs(tmp_path: Path):
    state = HookCaptureState("E0", 0, 2, str(tmp_path), capture_level="feature_scan")
    state.begin_sample("s", 3, {2: 0})
    state.active_rows = [0]
    state.active_decode_indices = [0]
    x = torch.ones(1, 64)
    for boundary in ("input_norm", "post_attn_norm", "moe_out", "layer_out"):
        for role in ("normalized", "updated_residual", "branch"):
            _capture(state, boundary=boundary, layer=1, tensor=x, role=role)
    assert [(r["boundary"], r["role"]) for r in state.records] == [
        ("input_norm", "normalized"), ("post_attn_norm", "normalized")]


def test_artifact_resume_checks_inputs_outputs_and_parameters(tmp_path: Path):
    source, out = tmp_path / "source", tmp_path / "out"
    source.write_text("source")
    out.write_text("actual result")
    task = ArtifactTask(tmp_path, "test", [source], {"variant": "E0"})
    assert not task.reusable()
    task.finish([out])
    assert task.reusable()
    assert not ArtifactTask(tmp_path, "test", [source], {"variant": "E1"}).reusable()
    out.write_text("partial")
    assert not task.reusable()
    task.finish([out])
    source.write_text("changed source")
    assert not ArtifactTask(tmp_path, "test", [source], {"variant": "E0"}).reusable()


def test_new_experiments_exclude_e4():
    assert VARIANTS == ("E0", "E1", "E2", "E3")
