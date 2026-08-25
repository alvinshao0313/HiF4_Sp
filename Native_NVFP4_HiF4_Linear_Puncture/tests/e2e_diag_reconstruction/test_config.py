from __future__ import annotations

import pytest

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import (
    E2ETrainConfig,
    build_train_parser,
    parse_log2_clamp,
    parse_train_args,
    validate_train_config,
)


def test_parse_log2_clamp():
    assert parse_log2_clamp("none") is None
    assert parse_log2_clamp("-4,4") == (-4.0, 4.0)
    assert parse_log2_clamp("-3,2") == (-3.0, 2.0)
    with pytest.raises(ValueError):
        parse_log2_clamp("-4")
    with pytest.raises(ValueError):
        parse_log2_clamp("4,-4")


def test_fusable_rejects_rot_then_diag():
    cfg = E2ETrainConfig.for_test(diag_mode="fusable", rot_order="rot_then_diag")
    with pytest.raises(ValueError, match="fusable"):
        validate_train_config(cfg)


def test_fusable_rejects_linear_independent():
    cfg = E2ETrainConfig.for_test(diag_mode="fusable", diag_train_scope="linear_independent")
    with pytest.raises(ValueError, match="linear_independent"):
        validate_train_config(cfg)


def test_for_test_defaults_to_s1k_original():
    cfg = E2ETrainConfig.for_test()
    assert cfg.calib_source == "s1k_original"


def test_default_hyperparameters():
    parser = build_train_parser()
    args = parser.parse_args(["--output_dir", "/tmp/out"])
    cfg = parse_train_args(["--output_dir", "/tmp/out"])
    assert args.diag_lr == 5e-3
    assert cfg.diag_lr == 5e-3
    assert cfg.diag_epochs == 20
    assert cfg.diag_scheduler == "cosine"
    assert cfg.diag_log2_clamp == (-4.0, 4.0)
    assert cfg.calib_nsamples == 128
    assert cfg.calib_val_nsamples == 32
    assert cfg.calib_seed == 42
    assert cfg.diag_batch_size == 4
    assert cfg.teacher_max_new_tokens == 32768
    assert cfg.diag_mode == "fusable"
    assert cfg.use_r64 is False
    assert cfg.rot_order == "diag_then_rot"
    assert cfg.diag_train_scope == "layer_joint"
    assert cfg.recon_loss == "block_delta_nmse"
    assert cfg.attn_aux_loss_weight == 0.0
    assert cfg.mlp_aux_loss_weight == 0.0
    assert cfg.calib_source == "s1k_original"
    assert cfg.teacher_trace_policy == "all"
    assert cfg.start_layer == 0
    assert cfg.end_layer == 47
    assert cfg.calib_cache_dir == ""
    assert cfg.fusable_diag_components == "all"
    assert cfg.calib_input_mode == "progressive_student"
    assert cfg.layer_rollback == "on"
    assert cfg.loss_rollback == "inherit"
    assert cfg.router_rollback == "inherit"
    assert cfg.router_align_loss_weight == 0.0


def test_calib_cache_dir_enters_to_dict():
    cfg = parse_train_args(
        ["--output_dir", "/tmp/out", "--calib_cache_dir", "/tmp/shared_calib"]
    )
    d = cfg.to_dict()
    assert d["calib_cache_dir"] == "/tmp/shared_calib"


def test_router_alignment_and_rollback_cli_overrides():
    cfg = parse_train_args(
        [
            "--output_dir",
            "/tmp/out",
            "--loss_rollback",
            "on",
            "--router_rollback",
            "off",
            "--router_align_loss_weight",
            "0.5",
        ]
    )
    assert cfg.loss_rollback == "on"
    assert cfg.router_rollback == "off"
    assert cfg.router_align_loss_weight == 0.5


def test_online_rejects_router_alignment_loss():
    cfg = E2ETrainConfig.for_test(diag_mode="online", router_align_loss_weight=0.5)
    with pytest.raises(ValueError, match="only valid for diag_mode=fusable"):
        validate_train_config(cfg)


def test_online_rejects_partial_fusable_components():
    cfg = E2ETrainConfig.for_test(diag_mode="online", fusable_diag_components="qkv")
    with pytest.raises(ValueError, match="fusable_diag_components=all"):
        validate_train_config(cfg)


def test_illegal_layer_range():
    with pytest.raises(ValueError, match="layer range"):
        validate_train_config(E2ETrainConfig.for_test(start_layer=5, end_layer=4))
    with pytest.raises(ValueError, match="layer range"):
        validate_train_config(E2ETrainConfig.for_test(start_layer=-1, end_layer=1))
    with pytest.raises(ValueError, match="layer range"):
        validate_train_config(E2ETrainConfig.for_test(start_layer=0, end_layer=48))


def test_non_positive_sample_counts():
    with pytest.raises(ValueError, match="calib_nsamples"):
        validate_train_config(E2ETrainConfig.for_test(calib_nsamples=0))
    with pytest.raises(ValueError, match="calib_val_nsamples"):
        validate_train_config(E2ETrainConfig.for_test(calib_val_nsamples=-3))


def test_non_positive_batch_size():
    with pytest.raises(ValueError, match="diag_batch_size"):
        validate_train_config(E2ETrainConfig.for_test(diag_batch_size=0))


def test_window_dataset_rejects_non_positive_seqlen():
    with pytest.raises(ValueError, match="calib_seqlen"):
        validate_train_config(
            E2ETrainConfig.for_test(calib_source="wikitext2", calib_seqlen=0)
        )
    with pytest.raises(ValueError, match="calib_seqlen"):
        validate_train_config(E2ETrainConfig.for_test(calib_source="c4", calib_seqlen=-1))


def test_s1k_allows_default_seqlen_without_using_it_as_truncation():
    cfg = E2ETrainConfig.for_test(calib_source="s1k_teacher_cot", calib_seqlen=1024)
    validate_train_config(cfg)


def test_moe_online_rejects_linear_independent():
    cfg = E2ETrainConfig.for_test(
        diag_mode="online",
        rot_order="rot_then_diag",
        diag_train_scope="linear_independent",
    )
    with pytest.raises(ValueError, match="Qwen3 MoE"):
        validate_train_config(cfg)


def test_train_cli_dispatches_qwen3_moe_to_lazy_trainer(monkeypatch, tmp_path):
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.cli import train as train_cli

    calls = {}
    monkeypatch.setattr(train_cli, "_require_cuda", lambda: "cuda")
    monkeypatch.setattr(
        train_cli,
        "train_qwen3_moe_lazy",
        lambda cfg, device: calls.update({"model_type": cfg.model_type, "device": device}),
    )
    train_cli.main(["--output_dir", str(tmp_path), "--end_layer", "0"])
    assert calls == {"model_type": "qwen3_moe", "device": "cuda"}
