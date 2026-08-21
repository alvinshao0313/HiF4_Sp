"""Toy tests for activation_viz_pipeline hooks (no Qwen load)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from Inference_Paradigm_Conversion.ipc_analysis.analysis.activation_viz_pipeline import (
    ACTIVATION_VIZ_CONTEXT,
    CaptureBuffers,
    _cat_point_chunks,
    _deterministic_flat_indices,
    _expand_nvfp4_element_fields,
    _make_recording_hook,
    _sample_seed,
    attach_theoretical_grids,
    jensen_shannon,
    merge_activation_viz_shards,
)
from Inference_Paradigm_Conversion.ipc_analysis.formats.hif4 import quantize_hif4_tensor
from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import quantize_nvfp4_activation
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import write_csv


def test_recording_hook_matches_manual_nvfp4_linear():
    torch.manual_seed(0)
    linear = nn.Linear(64, 32, bias=False).to(torch.bfloat16)
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    scale = torch.tensor(2.5, dtype=torch.float32)

    a_n_ref = quantize_nvfp4_activation(
        x, scale, output_dtype=torch.bfloat16, collect_metadata=False
    ).dequantized
    y_ref = linear(a_n_ref)

    buffers = CaptureBuffers()
    ACTIVATION_VIZ_CONTEXT.sample_id = "toy_000"
    ACTIVATION_VIZ_CONTEXT.prompt_family = "natural"
    ACTIVATION_VIZ_CONTEXT.split = "discovery"
    ACTIVATION_VIZ_CONTEXT.phase = "prefill"
    ACTIVATION_VIZ_CONTEXT.decode_step = -1
    ACTIVATION_VIZ_CONTEXT.record_enabled = True

    handle = linear.register_forward_pre_hook(
        _make_recording_hook(
            "toy.layers.4.mlp.down_proj",
            layer_idx=4,
            projection="down_proj",
            scale=scale,
            is_representative=True,
            buffers=buffers,
            max_point_samples=128,
        )
    )
    try:
        y_hook = linear(x)
    finally:
        handle.remove()
        ACTIVATION_VIZ_CONTEXT.record_enabled = False

    torch.testing.assert_close(y_hook, y_ref, rtol=0, atol=0)
    assert len(buffers.summary_rows) == 1
    assert len(buffers.point_chunks) == 1
    recorded_an = buffers.point_chunks[0]["a_nvfp4"]
    seed = _sample_seed("toy_000", "prefill", -1, 4, "down_proj")
    idx = _deterministic_flat_indices(a_n_ref.numel(), 128, seed)
    torch.testing.assert_close(
        recorded_an,
        a_n_ref.reshape(-1).to(torch.float32)[idx].cpu(),
        rtol=0,
        atol=0,
    )


def test_ah_side_compute_does_not_change_output():
    torch.manual_seed(1)
    linear = nn.Linear(64, 16, bias=True).to(torch.bfloat16)
    x = torch.randn(3, 64, dtype=torch.bfloat16)
    scale = torch.tensor(1.0, dtype=torch.float32)

    def run(record: bool, representative: bool) -> torch.Tensor:
        buffers = CaptureBuffers()
        ACTIVATION_VIZ_CONTEXT.sample_id = "toy_001"
        ACTIVATION_VIZ_CONTEXT.prompt_family = "math"
        ACTIVATION_VIZ_CONTEXT.split = "discovery"
        ACTIVATION_VIZ_CONTEXT.phase = "decode"
        ACTIVATION_VIZ_CONTEXT.decode_step = 0
        ACTIVATION_VIZ_CONTEXT.record_enabled = record
        handle = linear.register_forward_pre_hook(
            _make_recording_hook(
                "toy.layers.18.self_attn.q_proj",
                layer_idx=18,
                projection="q_proj",
                scale=scale,
                is_representative=representative,
                buffers=buffers,
                max_point_samples=64,
            )
        )
        try:
            return linear(x).detach().clone(), buffers
        finally:
            handle.remove()
            ACTIVATION_VIZ_CONTEXT.record_enabled = False

    y_rec, buf_rec = run(True, True)
    y_off, buf_off = run(False, True)
    y_nonrep, _ = run(True, False)
    torch.testing.assert_close(y_rec, y_off, rtol=0, atol=0)
    torch.testing.assert_close(y_rec, y_nonrep, rtol=0, atol=0)
    assert len(buf_rec.summary_rows) == 1
    assert len(buf_off.summary_rows) == 0


def test_recorded_an_is_linear_input():
    """Hook return value is what Linear receives; equals recorded a_nvfp4 path."""
    torch.manual_seed(2)
    linear = nn.Linear(64, 8, bias=False).to(torch.bfloat16)
    x = torch.randn(1, 64, dtype=torch.bfloat16)
    scale = torch.tensor(0.75, dtype=torch.float32)
    buffers = CaptureBuffers()
    seen: dict[str, torch.Tensor] = {}

    def spy(_m, inputs):
        # Registered after recording hook → sees post-QDQ activation.
        seen["in"] = inputs[0].detach().clone()

    ACTIVATION_VIZ_CONTEXT.sample_id = "toy_002"
    ACTIVATION_VIZ_CONTEXT.prompt_family = "arc_mmlu"
    ACTIVATION_VIZ_CONTEXT.split = "validation"
    ACTIVATION_VIZ_CONTEXT.phase = "prefill"
    ACTIVATION_VIZ_CONTEXT.decode_step = -1
    ACTIVATION_VIZ_CONTEXT.record_enabled = True
    h_rec = linear.register_forward_pre_hook(
        _make_recording_hook(
            "toy.layers.34.mlp.up_proj",
            layer_idx=34,
            projection="up_proj",
            scale=scale,
            is_representative=True,
            buffers=buffers,
            max_point_samples=64,
        )
    )
    h_spy = linear.register_forward_pre_hook(spy)
    try:
        _ = linear(x)
    finally:
        h_rec.remove()
        h_spy.remove()
        ACTIVATION_VIZ_CONTEXT.record_enabled = False

    assert "in" in seen
    a_n_manual = quantize_nvfp4_activation(
        x, scale, output_dtype=torch.bfloat16, collect_metadata=False
    ).dequantized
    torch.testing.assert_close(seen["in"], a_n_manual, rtol=0, atol=0)
    # HiF4 side path exists and differs or equals, but must be recorded.
    assert buffers.point_chunks[0]["a_hif4"].numel() > 0
    ah = quantize_hif4_tensor(x, output_dtype=torch.bfloat16).dequantized
    seed = _sample_seed("toy_002", "prefill", -1, 34, "up_proj")
    idx = _deterministic_flat_indices(ah.numel(), 64, seed)
    torch.testing.assert_close(
        buffers.point_chunks[0]["a_hif4"],
        ah.reshape(-1).to(torch.float32)[idx].cpu(),
        rtol=0,
        atol=0,
    )


def test_nvfp4_local_scale_repeat_alignment():
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    scale = torch.tensor(1.0)
    view = quantize_nvfp4_activation(x, scale, collect_metadata=True)
    payload, local = _expand_nvfp4_element_fields(view.metadata, 2, 64)
    assert payload.shape == (2, 64)
    assert local.shape == (2, 64)
    # Each 16-block shares one scale.
    for t in range(2):
        for g in range(4):
            block = local[t, g * 16 : (g + 1) * 16]
            assert torch.equal(block, block[0].expand_as(block))


def test_jensen_shannon_identical_zero():
    p = torch.tensor([0.25, 0.25, 0.5], dtype=torch.float64)
    assert abs(jensen_shannon(p, p)) < 1e-12


def test_merge_and_attach_grids_smoke():
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        # Minimal shard artifacts
        write_csv(
            run_dir / "activation_capture_summary_shard0.csv",
            [
                {
                    "sample_id": "natural_000",
                    "prompt_family": "natural",
                    "split": "discovery",
                    "phase": "prefill",
                    "decode_step": -1,
                    "layer_idx": 4,
                    "module_name": "model.layers.4.mlp.down_proj",
                    "projection": "down_proj",
                    "num_tokens": 2,
                    "num_elements": 128,
                }
            ],
        )
        write_csv(
            run_dir / "activation_group_residual_shard0.csv",
            [
                {
                    "sample_id": "natural_000",
                    "split": "discovery",
                    "phase": "prefill",
                    "decode_step": -1,
                    "layer_idx": 4,
                    "projection": "down_proj",
                    "group_idx": 0,
                    "num_tokens": 2,
                    "rms_delta": 0.1,
                }
            ],
        )
        pts = _cat_point_chunks(
            [
                {
                    "x_in": torch.zeros(4),
                    "a_nvfp4": torch.zeros(4),
                    "a_hif4": torch.zeros(4),
                    "delta_a": torch.zeros(4),
                    "layer_idx": torch.zeros(4, dtype=torch.int32),
                    "projection_id": torch.zeros(4, dtype=torch.uint8),
                    "phase_id": torch.zeros(4, dtype=torch.uint8),
                    "decode_step": torch.full((4,), -1, dtype=torch.int16),
                    "token_position": torch.zeros(4, dtype=torch.int16),
                    "channel_idx": torch.arange(4, dtype=torch.int32),
                    "nv_payload": torch.zeros(4),
                    "hf_payload": torch.zeros(4),
                    "nv_local_scale": torch.ones(4),
                    "hf_local_scale": torch.ones(4),
                    "sample_id": torch.zeros(4, dtype=torch.int32),
                }
            ],
            ["natural_000"],
        )
        torch.save(pts, run_dir / "activation_viz_points_shard0.pt")
        torch.save(
            {
                "entries": [
                    {
                        "sample_id": "natural_000",
                        "split": "discovery",
                        "phase": "prefill",
                        "decode_step": -1,
                        "layer_idx": 4,
                        "projection": "down_proj",
                        "map": torch.zeros(2, 1),
                    }
                ]
            },
            run_dir / "activation_token_group_maps_shard0.pt",
        )
        summary = merge_activation_viz_shards(run_dir)
        assert (run_dir / "activation_capture_summary.csv").is_file()
        assert (run_dir / "activation_viz_points.pt").is_file()
        assert (run_dir / "activation_viz_summary.json").is_file()
        assert (run_dir / "ax3_nvfp4_full_internal_grid.pt").is_file()
        assert summary["theoretical_grids"]["source"] in {
            "consolidated_ax3",
            "rebuilt",
            "rebuilt_missing_json_fields",
        }
        # Direct attach also works on empty dir
        with tempfile.TemporaryDirectory() as td2:
            info = attach_theoretical_grids(Path(td2))
            assert info["nvfp4_num_unique"] and info["hif4_num_unique"]
