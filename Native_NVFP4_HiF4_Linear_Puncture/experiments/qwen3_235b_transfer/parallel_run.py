"""Prepare isolated E4 continuations and verify their compact saved artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
from typing import Any

import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.artifact import (
    layer_artifact_dir, load_conversion_state,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import (
    E2ETrainConfig, parse_log2_clamp, validate_train_config,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.calibration import (
    _check_cached_manifest,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.artifact import (
    load, sha256, write_json,
)
from .config import diag_config, source_snapshot
from .fetch_model import REVISION


CALIBRATION_FILES = ("train.pt", "val.pt", "train_manifest.json", "val_manifest.json")
LAYER_FILES = ("best_diag.pt", "candidate_best_diag.pt", "candidate_metrics.json",
               "metrics.json", "train_log.jsonl")
LOCATION_FIELDS = {"output_dir", "calib_cache_dir", "start_layer", "end_layer"}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _compatible(previous: dict, current: dict, *, label: str) -> None:
    old = {key: value for key, value in previous.items() if key not in LOCATION_FIELDS}
    new = {key: value for key, value in current.items() if key not in LOCATION_FIELDS}
    if old != new:
        changed = sorted(key for key in old.keys() | new.keys() if old.get(key) != new.get(key))
        raise ValueError(f"{label} changes algorithm, data, or budget fields: {changed}")


def _prefix(reference_run_root: Path, smoke: bool, prefix_output: Path | None) -> Path:
    if prefix_output is not None:
        return prefix_output.resolve()
    if smoke:
        raise ValueError("a resumed smoke run requires explicit prefix_output")
    return reference_run_root.resolve() / "e4"


def make_config(
    reference_run_root: Path, output: Path, *, smoke: bool, start_layer: int,
    end_layer: int, prefix_output: Path | None = None,
) -> E2ETrainConfig:
    """Read the fixed original recipe without creating or changing any files."""
    reference_run_root = reference_run_root.resolve()
    original_dir = reference_run_root / ("smoke/e4" if smoke else "e4")
    original = _read(original_dir / "config.json")
    _compatible(original, diag_config(reference_run_root, smoke=smoke).to_dict(), label="source recipe")
    values = dict(original)
    values.update(output_dir=str(output.resolve()), calib_cache_dir=str(output.resolve() / "data"),
                  start_layer=start_layer, end_layer=end_layer)
    values["diag_log2_clamp"] = parse_log2_clamp(values["diag_log2_clamp"])
    cfg = E2ETrainConfig(**values)
    validate_train_config(cfg)
    if cfg.model_path != str(source_snapshot()):
        raise ValueError("continuation must use the pinned 235B source snapshot")
    if start_layer:
        prefix = _prefix(reference_run_root, smoke, prefix_output)
        _compatible(_read(prefix / "config.json"), cfg.to_dict(), label="saved prefix")
    return cfg


def _finite(value: Any, label: str) -> None:
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"non-finite tensor: {label}")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite scalar: {label}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite(item, f"{label}[{index}]")


def _steps(cfg: E2ETrainConfig) -> int:
    return cfg.diag_epochs * math.ceil(cfg.calib_nsamples / cfg.diag_batch_size)


def _layer_metrics(directory: Path, layer: int, expected_steps: int) -> dict:
    for filename in LAYER_FILES:
        if not (directory / filename).is_file():
            raise FileNotFoundError(directory / filename)
    metrics = _read(directory / "metrics.json")
    candidate = _read(directory / "candidate_metrics.json")
    if metrics.get("layer_id") != layer or candidate.get("layer_id") != layer:
        raise ValueError(f"incorrect layer ID in {directory}")
    if metrics.get("train_steps") != expected_steps:
        raise ValueError(f"{directory}: train_steps={metrics.get('train_steps')} != {expected_steps}")
    _finite(metrics, str(directory / "metrics.json"))
    _finite(candidate, str(directory / "candidate_metrics.json"))
    return metrics


def _copy_verified(source: Path, target: Path, output: Path) -> dict:
    before = sha256(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if sha256(target) != before or sha256(source) != before:
        raise ValueError(f"source changed or copy mismatch: {source}")
    return {"source": str(source.resolve()), "destination": str(target.relative_to(output)),
            "sha256": before, "bytes": target.stat().st_size}


def _source_evidence(reference_run_root: Path) -> dict:
    validation_path = reference_run_root / "source_validation.json"
    status_path = reference_run_root / "e4/status.json"
    validation = _read(validation_path)
    status = _read(status_path)
    source = status["source"]
    for record in (validation, source):
        if (record.get("revision") != REVISION
                or record.get("source_snapshot") != str(source_snapshot())
                or record.get("full_model_download_complete") is not True
                or not record.get("verified_shards_sha256")):
            raise ValueError("reference source evidence is not the complete pinned 235B snapshot")
    if validation["verified_shards_sha256"] != source["verified_shards_sha256"]:
        raise ValueError("reference source validation and running E4 source hashes disagree")
    return {"source_snapshot": str(source_snapshot()), "revision": REVISION,
            "weight_verification": "reuse_original_full_validation_and_e4_source_record",
            "source_validation": str(validation_path), "source_validation_sha256": sha256(validation_path),
            "e4_status": str(status_path), "e4_status_sha256": sha256(status_path),
            "reference_status": status.get("status"),
            "verified_shards_sha256": validation["verified_shards_sha256"]}


def prepare_run(
    reference_run_root: Path, output: Path, *, smoke: bool, start_layer: int,
    end_layer: int, prefix_output: Path | None = None,
) -> E2ETrainConfig:
    """Copy stable calibration and completed layers into a strictly new output."""
    reference_run_root, output = reference_run_root.resolve(), output.resolve()
    cfg = make_config(reference_run_root, output, smoke=smoke, start_layer=start_layer,
                      end_layer=end_layer, prefix_output=prefix_output)
    source_evidence = _source_evidence(reference_run_root)
    original_dir = reference_run_root / ("smoke/e4" if smoke else "e4")
    calibration = reference_run_root / ("smoke/data/calibration" if smoke else "data/calibration")
    for split in ("train", "val"):
        _check_cached_manifest(calibration / f"{split}_manifest.json", cfg, split)
    prefix = _prefix(reference_run_root, smoke, prefix_output) if start_layer else None
    if prefix is not None:
        prefix_config = _read(prefix / "config.json")
        prefix_calibration = Path(prefix_config["calib_cache_dir"]) / "calibration"
        for filename in CALIBRATION_FILES:
            if sha256(prefix_calibration / filename) != sha256(calibration / filename):
                raise ValueError(f"prefix calibration differs: {filename}")
        for layer in range(start_layer):
            _layer_metrics(layer_artifact_dir(prefix, layer), layer, _steps(cfg))
    output.mkdir(parents=True, exist_ok=False)
    copied = [_copy_verified(original_dir / "config.json", output / "source_config.json", output)]
    for filename in CALIBRATION_FILES:
        copied.append(_copy_verified(calibration / filename, output / "data/calibration" / filename, output))
    if prefix is not None:
        copied.append(_copy_verified(prefix / "config.json", output / "prefix_config.json", output))
        for layer in range(start_layer):
            for filename in LAYER_FILES:
                copied.append(_copy_verified(layer_artifact_dir(prefix, layer) / filename,
                                             layer_artifact_dir(output, layer) / filename, output))
    write_json(cfg.to_dict(), output / "config.json")
    write_json({"reference_run_root": str(reference_run_root), "output": str(output),
                "smoke_only": smoke, "prefix_output": str(prefix) if prefix is not None else None,
                "completed_prefix": list(range(start_layer)), "start_layer": start_layer,
                "end_layer": end_layer, "expected_steps_per_layer": _steps(cfg),
                "source": source_evidence, "copied_files": copied,
                "prepared_at": datetime.now(timezone.utc).isoformat()}, output / "provenance.json")
    return cfg


def _same_snapshot(actual: dict, expected: dict, label: str) -> None:
    if actual.keys() != expected.keys():
        raise ValueError(f"DIAG tensor keys differ: {label}")
    for name, tensor in expected.items():
        other = actual[name]
        if (not torch.is_tensor(tensor) or not torch.is_tensor(other)
                or tensor.dtype != other.dtype or tensor.shape != other.shape
                or not torch.equal(tensor, other)):
            raise ValueError(f"DIAG save/reload mismatch: {label}.{name}")


def validate_completed(cfg: E2ETrainConfig) -> dict:
    """Verify full prefix coverage, exact compact saves, and unchanged copied inputs."""
    output = Path(cfg.output_dir)
    provenance = _read(output / "provenance.json")
    for record in provenance["copied_files"]:
        for path in (Path(record["source"]), output / record["destination"]):
            if sha256(path) != record["sha256"]:
                raise ValueError(f"copied input or prefix changed: {path}")
    _compatible(_read(output / "config.json"), cfg.to_dict(), label="completed run")
    final_path = output / "checkpoint/final_model/conversion_state.pt"
    artifact = load_conversion_state(final_path)
    manifest = _read(final_path.parent / "manifest.json")
    summaries = _read(output / "summary.json")["layers"]
    expected = list(range(cfg.end_layer + 1))
    if artifact.get("num_layers") != cfg.num_layers or set(artifact["layers"]) != {str(i) for i in expected}:
        raise ValueError("final artifact has incomplete or extra layer coverage")
    if manifest.get("layer_ids") != expected or manifest.get("n_layers") != len(expected):
        raise ValueError("final manifest layer coverage differs")
    if [row.get("layer_id") for row in summaries] != expected:
        raise ValueError("summary has missing, reordered, or duplicate layers")
    _finite(artifact, "final_artifact")
    _finite(summaries, "summary")
    for layer in expected:
        directory = layer_artifact_dir(output, layer)
        metrics = _layer_metrics(directory, layer, _steps(cfg))
        if summaries[layer] != metrics:
            raise ValueError(f"summary differs from saved metrics for layer {layer}")
        saved = artifact["layers"][str(layer)]
        adopted = load(directory / "best_diag.pt")
        candidate = load(directory / "candidate_best_diag.pt")
        _finite(adopted, f"layer_{layer}.adopted")
        _finite(candidate, f"layer_{layer}.candidate")
        _same_snapshot(saved["adopted_z"], adopted, f"layer_{layer}.adopted")
        _same_snapshot(saved["z"], adopted, f"layer_{layer}.z")
        _same_snapshot(saved["candidate_z"], candidate, f"layer_{layer}.candidate")
        for key in ("accepted", "rollback", "best_epoch", "candidate_best_epoch",
                    "loss_rollback_applied", "router_rollback_applied"):
            if saved[key] != metrics[key]:
                raise ValueError(f"artifact metric mismatch: layer {layer}, {key}")
    return {"status": "passed", "layer_ids": expected, "num_layers": cfg.num_layers,
            "new_layer_ids": list(range(cfg.start_layer, cfg.end_layer + 1)),
            "expected_steps_per_layer": _steps(cfg), "copied_inputs_and_prefix_unchanged": True,
            "adopted_and_candidate_reload_exact": True, "all_scalars_and_tensors_finite": True,
            "final_artifact": str(final_path), "final_artifact_sha256": sha256(final_path)}
