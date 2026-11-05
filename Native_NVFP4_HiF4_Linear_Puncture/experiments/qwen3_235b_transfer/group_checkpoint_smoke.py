"""Bounded canonical Group layer transition and epoch-checkpoint validation."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_model_spec import Qwen3MoeModelSpec
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction import train as canonical_train
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.artifact import load, sha256
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.config import Config


class _CheckpointWritten(Exception):
    """Layer 1's real first trained epoch was saved; end this bounded run."""


def _finite(value: Any, label: str) -> None:
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"non-finite checkpoint tensor: {label}")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite checkpoint scalar: {label}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite(item, f"{label}[{index}]")


def run_checkpoint_smoke(cfg: Config, record: dict) -> None:
    """Train one epoch in each of layers 0 and 1, then validate saved state.

    The caller installs the execution driver and owns the already-created output
    directory and status record. This exercises a real layer transition and
    checkpoint saving/loading; it does not resume training or claim exact
    continuation across optimizer updates.
    """
    cfg.validate()
    if (cfg.epochs, cfg.calib_nsamples, cfg.calib_val_nsamples, cfg.batch_size) != (1, 4, 4, 4):
        raise ValueError("checkpoint smoke requires one epoch and full batches of 4 train/4 validation samples")
    if cfg.matrix_sharing != "group" or cfg.router_loss != "top_mass":
        raise ValueError("checkpoint smoke requires the Group-top-mass recipe")
    out = Path(cfg.output_dir).resolve()
    if not out.is_dir():
        raise FileNotFoundError(f"caller must create the isolated output directory: {out}")
    checkpoint = out / "resume.pt"
    original_save = canonical_train.atomic_save
    stopped = False

    def save_and_stop(value, path):
        original_save(value, path)
        if (Path(path).resolve() == checkpoint and isinstance(value, dict)
                and value.get("layer") == 1 and value.get("next_epoch") == 1):
            raise _CheckpointWritten()

    canonical_train.atomic_save = save_and_stop
    try:
        try:
            canonical_train.train(cfg)
        except _CheckpointWritten:
            stopped = True
    finally:
        canonical_train.atomic_save = original_save
    if not stopped:
        raise RuntimeError("canonical training did not stop at layer 1's first saved epoch")

    progress = load(checkpoint)
    required = {"layer", "next_epoch", "hidden", "parameters", "optimizer", "scheduler", "best"}
    if set(progress) != required or progress["layer"] != 1 or progress["next_epoch"] != 1:
        raise ValueError("unexpected layer-1 first-epoch checkpoint fields or position")
    _finite(progress, "resume")
    training = load(Path(cfg.calib_cache_dir) / "calibration/train.pt")
    validation = load(Path(cfg.calib_cache_dir) / "calibration/val.pt")
    samples = training + validation
    if len(training) != 4 or len(validation) != 4 or len({s.sample_id for s in samples}) != 8:
        raise ValueError("checkpoint calibration must contain 8 distinct full samples")
    spec = Qwen3MoeModelSpec.from_snapshot(Path(cfg.model_path))
    hidden = progress["hidden"]
    if set(hidden) != {s.sample_id for s in samples}:
        raise ValueError("checkpoint hidden sample coverage differs from calibration")
    for sample in samples:
        tensor = hidden[sample.sample_id]
        if (tensor.device.type != "cpu" or tensor.dtype != torch.bfloat16
                or tuple(tensor.shape) != (sample.input_ids.numel(), spec.hidden_size)):
            raise ValueError(f"checkpoint hidden geometry changed: {sample.sample_id}")

    parameters = progress["parameters"]
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("checkpoint has no learned parameters")
    if not {"input_norm", "moe_norm", "matrices.router"}.issubset(parameters):
        raise ValueError("checkpoint is missing learned norm or router parameters")
    if any(not torch.is_tensor(value) or value.device.type != "cpu" for value in parameters.values()):
        raise ValueError("learned parameter checkpoint must contain CPU tensors")
    optimizer = progress["optimizer"]
    parameter_ids = [index for group in optimizer["param_groups"] for index in group["params"]]
    if len(parameter_ids) != len(parameters) or len(set(parameter_ids)) != len(parameter_ids):
        raise ValueError("Adam parameter groups do not cover learned parameters exactly once")
    states = optimizer["state"]
    if not states or not set(states).issubset(parameter_ids):
        raise ValueError("Adam state is absent or refers to unknown parameters")
    parameter_by_id = dict(zip(parameter_ids, parameters.values()))
    for index, state in states.items():
        if not {"step", "exp_avg", "exp_avg_sq"}.issubset(state) or float(state["step"]) != 1.0:
            raise ValueError(f"Adam parameter {index} must have exactly one update")
        for name in ("exp_avg", "exp_avg_sq"):
            if state[name].shape != parameter_by_id[index].shape or state[name].device.type != "cpu":
                raise ValueError(f"Adam moment shape/device mismatch: {index}.{name}")
    scheduler = progress["scheduler"]
    if scheduler.get("last_epoch") != 1 or scheduler.get("T_max") != 1:
        raise ValueError("scheduler position does not match the one-epoch smoke budget")
    best = progress["best"]
    if not isinstance(best, dict) or set(best) != {"epoch", "metrics", "parameters"} or best["epoch"] != 0:
        raise ValueError("checkpoint must select the actual trained epoch 0")
    if set(best["parameters"]) != set(parameters):
        raise ValueError("selected and current parameter coverage differs")
    if any(not torch.equal(value, best["parameters"][name]) for name, value in parameters.items()):
        raise ValueError("one-epoch selected parameters differ from the saved current parameters")
    expected_validation_tokens = sum(int(sample.loss_mask.bool().sum()) for sample in validation)
    if best["metrics"].get("tokens") != expected_validation_tokens:
        raise ValueError("selected validation token coverage differs from the full input samples")

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("completed_layers") != [0]:
        raise ValueError("the saved manifest must contain exactly completed layer 0")
    layer0 = out / "layers/000"
    selected0 = load(layer0 / "selected.pt")
    metrics0 = json.loads((layer0 / "metrics.json").read_text(encoding="utf-8"))
    epoch0 = json.loads((layer0 / "epochs/000.json").read_text(encoding="utf-8"))
    _finite(selected0, "layer_0.selected")
    _finite(metrics0, "layer_0.metrics")
    _finite(epoch0, "layer_0.epoch")
    if (selected0.get("epoch") != 0 or selected0.get("metrics") != metrics0
            or epoch0.get("epoch") != 0 or epoch0.get("validation") != metrics0):
        raise ValueError("completed layer 0's selected epoch and metrics disagree")
    if set(selected0.get("parameters", {})) != set(parameters):
        raise ValueError("completed layer 0's selected parameter coverage differs")
    expected_training_tokens = sum(int(sample.loss_mask.bool().sum()) for sample in training)
    if (metrics0.get("tokens") != expected_validation_tokens
            or epoch0.get("train", {}).get("tokens") != expected_training_tokens):
        raise ValueError("completed layer 0's train/validation token coverage differs")

    record["checkpoint_validation"] = {
        "status": "passed", "scope": "canonical_layer_transition_and_epoch_checkpoint_save_and_reload",
        "layer": 1, "next_epoch": 1, "completed_layers": [0], "hidden_samples": len(hidden),
        "hidden_lengths": {s.sample_id: s.input_ids.numel() for s in samples},
        "learned_parameters": len(parameters), "adam_parameter_states": len(states),
        "all_present_adam_steps": 1, "scheduler_last_epoch": scheduler["last_epoch"],
        "selected_epoch": best["epoch"], "selected_validation": best["metrics"],
        "layer_0_selected_epoch": selected0["epoch"], "layer_0_selected_validation": metrics0,
        "all_checkpoint_values_finite": True, "resume_execution_tested": False,
        "checkpoint": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(checkpoint),
    }
