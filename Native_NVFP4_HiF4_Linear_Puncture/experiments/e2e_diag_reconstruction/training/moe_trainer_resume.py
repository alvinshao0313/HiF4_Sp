"""Resume-aware wrapper for the canonical Qwen3-MoE trainer.

The layer optimization remains implemented only in ``moe_trainer``. This module
restores and replays the saved prefix, then injects that reconstructed hidden
cache into the canonical trainer and merges old/new compact artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.artifact import (
    assert_resume_artifacts,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import (
    E2ETrainConfig,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.batching import (
    DynamicCalibrationCollator,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.calibration import (
    build_or_load_calibration,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training import (
    moe_trainer as _base,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.moe_resume import (
    load_resume_prefix_records,
    replay_moe_prefix,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot
from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import ensure_dir, read_json


def _load_tokenizer(snapshot: Path):
    loader = getattr(AutoTokenizer, "from_" + "pretrained")
    return loader(str(snapshot), trust_remote_code=True)


def _assert_resume_runtime_config(cfg: E2ETrainConfig) -> None:
    """Lock fields that materially change the resumed optimization trajectory."""
    prev = read_json(Path(cfg.output_dir) / "config.json")
    current = cfg.to_dict()
    keys = (
        "model_path",
        "model_type",
        "num_layers",
        "num_experts",
        "top_k",
        "kv_cache_dtype",
        "native_has_online_rotation",
        "attn_aux_loss_weight",
        "mlp_aux_loss_weight",
        "diag_batch_size",
        "calib_seqlen",
        "teacher_trace_policy",
        "teacher_max_attempts",
        "teacher_max_new_tokens",
    )
    for key in keys:
        if prev.get(key) != current.get(key):
            raise ValueError(
                f"resume trajectory field {key}={prev.get(key)!r} != current {current.get(key)!r}"
            )


def train_qwen3_moe_lazy(cfg: E2ETrainConfig, device: torch.device) -> None:
    """Run a fresh canonical train or resume from an exact adopted prefix."""

    if cfg.start_layer == 0:
        _base.train_qwen3_moe_lazy(cfg, device)
        return

    snapshot = Path(resolve_local_snapshot(cfg.model_path))
    out_dir = ensure_dir(cfg.output_dir)
    assert_resume_artifacts(cfg)
    _assert_resume_runtime_config(cfg)

    tokenizer = _load_tokenizer(snapshot)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_samples, val_samples = build_or_load_calibration(
        cfg,
        tokenizer,
        None,
        Path(cfg.calib_cache_dir) if cfg.calib_cache_dir else out_dir,
    )
    collator = DynamicCalibrationCollator(int(tokenizer.pad_token_id))
    all_samples = list(train_samples) + list(val_samples)

    x_cache = _base.build_initial_moe_hidden_cache(
        snapshot,
        all_samples,
        collator,
        device,
        cfg.diag_batch_size,
    )
    prefix_records, prefix_summaries = load_resume_prefix_records(cfg)
    x_cache = replay_moe_prefix(
        cfg,
        snapshot,
        all_samples,
        collator,
        x_cache,
        device,
        prefix_records,
        prefix_summaries,
        propagate_fn=_base._propagate,
    )

    original_build_cache = _base.build_initial_moe_hidden_cache
    original_build_calibration = _base.build_or_load_calibration
    original_save_conversion = _base.save_conversion_artifact
    original_write_json = _base.write_json

    def resumed_build_cache(*_args: Any, **_kwargs: Any):
        return x_cache

    def resumed_build_calibration(*_args: Any, **_kwargs: Any):
        return train_samples, val_samples

    def resumed_save_conversion(*, cfg, layer_records, out_dir):
        merged = dict(prefix_records)
        merged.update(layer_records)
        return original_save_conversion(cfg=cfg, layer_records=merged, out_dir=out_dir)

    def resumed_write_json(path, payload):
        target = Path(path)
        if target == out_dir / "config.json":
            return None
        if target == out_dir / "summary.json" and isinstance(payload, dict):
            new_layers = list(payload.get("layers", []))
            payload = {**payload, "layers": [*prefix_summaries, *new_layers]}
        return original_write_json(path, payload)

    _base.build_initial_moe_hidden_cache = resumed_build_cache
    _base.build_or_load_calibration = resumed_build_calibration
    _base.save_conversion_artifact = resumed_save_conversion
    _base.write_json = resumed_write_json
    try:
        _base.train_qwen3_moe_lazy(cfg, device)
    finally:
        _base.build_initial_moe_hidden_cache = original_build_cache
        _base.build_or_load_calibration = original_build_calibration
        _base.save_conversion_artifact = original_save_conversion
        _base.write_json = original_write_json
