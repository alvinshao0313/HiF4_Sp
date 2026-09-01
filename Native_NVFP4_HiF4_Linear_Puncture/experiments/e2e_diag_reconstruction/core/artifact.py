"""Compact conversion overlay: metadata + learned z, no full 8B copy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import (
    DEFAULT_MODEL_PATH,
    E2ETrainConfig,
    ROUTER_ALIGN_TEMPERATURE,
    ROUTER_ALIGN_TYPE,
    resolve_rollback_enabled,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.fold import (
    fold_fusable_layer_inplace,
    freeze_online_layer_for_eval,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.semantic_hif4 import (
    iter_switchable_linears,
    load_layer_diag_snapshot,
    upgrade_semantic_model_inplace,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import ensure_dir, load_pt, write_json

SCHEMA_VERSION = 1
LEGACY_MOE_SCHEMA_VERSION = 2
MOE_SCHEMA_VERSION = 3
DIAG_ARTIFACT_VARIANTS = ("adopted", "candidate")
CORE_KEYS = (
    "diag_mode",
    "use_r64",
    "rot_order",
    "calib_source",
    "calib_seed",
    "calib_input_mode",
)


def select_layer_diag(record: dict[str, Any], variant: str = "adopted") -> dict[str, torch.Tensor]:
    if variant not in DIAG_ARTIFACT_VARIANTS:
        raise ValueError(f"invalid artifact diag variant {variant!r}")
    if variant == "candidate":
        candidate = record.get("candidate_z")
        if candidate is None:
            raise ValueError("artifact does not contain candidate_z; rerun training with schema v3")
        return candidate
    adopted = record.get("adopted_z")
    if adopted is not None:
        return adopted
    return record["z"]


def save_conversion_artifact(
    *,
    cfg: E2ETrainConfig,
    layer_records: dict[int, dict[str, Any]],
    out_dir: str | Path,
) -> Path:
    ckpt_dir = ensure_dir(Path(out_dir) / "checkpoint" / "final_model")
    layers_ser = {}
    for idx, rec in sorted(layer_records.items()):
        adopted_z = rec.get("adopted_z", rec["z"])
        candidate_z = rec.get("candidate_z", adopted_z)
        layers_ser[str(idx)] = {
            "accepted": bool(rec["accepted"]),
            "rollback": bool(rec["rollback"]),
            "best_epoch": int(rec["best_epoch"]),
            "candidate_best_epoch": int(rec.get("candidate_best_epoch", rec["best_epoch"])),
            "loss_rollback_applied": bool(rec.get("loss_rollback_applied", False)),
            "router_rollback_applied": bool(rec.get("router_rollback_applied", False)),
            "candidate_z": candidate_z,
            "adopted_z": adopted_z,
            "z": adopted_z,
        }
    is_moe = cfg.model_type == "qwen3_moe"
    schema_version = MOE_SCHEMA_VERSION if is_moe else SCHEMA_VERSION
    state = {
        "schema_version": schema_version,
        "source_model": cfg.model_path,
        "model_type": cfg.model_type,
        "num_layers": cfg.num_layers,
        "num_experts": cfg.num_experts,
        "diag_mode": cfg.diag_mode,
        "use_r64": cfg.use_r64,
        "rot_order": cfg.rot_order,
        "calib_source": cfg.calib_source,
        "calib_seed": cfg.calib_seed,
        "calib_input_mode": cfg.calib_input_mode,
        "calib_nsamples": cfg.calib_nsamples,
        "calib_val_nsamples": cfg.calib_val_nsamples,
        "kv_cache_dtype": cfg.kv_cache_dtype,
        "fusable_diag_components": cfg.fusable_diag_components,
        "layer_rollback": cfg.layer_rollback,
        "loss_rollback": cfg.loss_rollback,
        "router_rollback": cfg.router_rollback,
        "resolved_loss_rollback_enabled": resolve_rollback_enabled(cfg.layer_rollback, cfg.loss_rollback),
        "resolved_router_rollback_enabled": resolve_rollback_enabled(cfg.layer_rollback, cfg.router_rollback),
        "router_align_type": ROUTER_ALIGN_TYPE,
        "router_align_temperature": ROUTER_ALIGN_TEMPERATURE,
        "router_align_loss_weight": cfg.router_align_loss_weight,
        "optimizer": cfg.optimizer,
        "weight_decay": cfg.weight_decay,
        "diag_scheduler": cfg.diag_scheduler,
        "diag_lr": cfg.diag_lr,
        "artifact_diag_variants": list(DIAG_ARTIFACT_VARIANTS),
        "layers": layers_ser,
    }
    path = ckpt_dir / "conversion_state.pt"
    torch.save(state, path)
    manifest = {
        "schema_version": schema_version,
        "source_model": cfg.model_path,
        "model_type": cfg.model_type,
        "num_layers": cfg.num_layers,
        "num_experts": cfg.num_experts,
        "diag_mode": cfg.diag_mode,
        "use_r64": cfg.use_r64,
        "rot_order": cfg.rot_order,
        "calib_source": cfg.calib_source,
        "calib_seed": cfg.calib_seed,
        "calib_input_mode": cfg.calib_input_mode,
        "calib_nsamples": cfg.calib_nsamples,
        "calib_val_nsamples": cfg.calib_val_nsamples,
        "kv_cache_dtype": cfg.kv_cache_dtype,
        "fusable_diag_components": cfg.fusable_diag_components,
        "layer_rollback": cfg.layer_rollback,
        "loss_rollback": cfg.loss_rollback,
        "router_rollback": cfg.router_rollback,
        "resolved_loss_rollback_enabled": resolve_rollback_enabled(cfg.layer_rollback, cfg.loss_rollback),
        "resolved_router_rollback_enabled": resolve_rollback_enabled(cfg.layer_rollback, cfg.router_rollback),
        "router_align_type": ROUTER_ALIGN_TYPE,
        "router_align_temperature": ROUTER_ALIGN_TEMPERATURE,
        "router_align_loss_weight": cfg.router_align_loss_weight,
        "optimizer": cfg.optimizer,
        "weight_decay": cfg.weight_decay,
        "diag_scheduler": cfg.diag_scheduler,
        "diag_lr": cfg.diag_lr,
        "artifact_diag_variants": list(DIAG_ARTIFACT_VARIANTS),
        "n_layers": len(layers_ser),
        "layer_ids": sorted(int(k) for k in layers_ser),
        "accepted_layers": [
            int(k) for k, v in layers_ser.items() if v["accepted"]
        ],
        "rollback_layers": [
            int(k) for k, v in layers_ser.items() if v["rollback"]
        ],
    }
    write_json(ckpt_dir / "manifest.json", manifest)
    return path


def load_conversion_state(path: str | Path) -> dict[str, Any]:
    state = load_pt(path, map_location="cpu")
    if int(state.get("schema_version", -1)) not in {
        SCHEMA_VERSION,
        LEGACY_MOE_SCHEMA_VERSION,
        MOE_SCHEMA_VERSION,
    }:
        raise ValueError(f"unsupported conversion schema_version={state.get('schema_version')}")
    return state


def apply_conversion_state(
    model: nn.Module,
    state: dict[str, Any],
    *,
    diag_variant: str = "adopted",
) -> None:
    if not iter_switchable_linears(model):
        cfg = E2ETrainConfig.for_test(
            model_path=state["source_model"],
            diag_mode=state["diag_mode"],
            use_r64=bool(state["use_r64"]),
            rot_order=state["rot_order"],
        )
        upgrade_semantic_model_inplace(model, cfg)
    for key, layer in enumerate(model.model.layers):
        rec = state["layers"].get(str(key))
        if rec is None:
            continue
        load_layer_diag_snapshot(layer, select_layer_diag(rec, diag_variant))
        if state["diag_mode"] == "fusable":
            fold_fusable_layer_inplace(layer, layer.diag_state, bool(state["use_r64"]))
        else:
            freeze_online_layer_for_eval(
                layer,
                layer.diag_state,
                bool(state["use_r64"]),
                state["rot_order"],
            )


def load_and_apply_conversion_artifact(
    model: nn.Module,
    artifact_path: str | Path,
    *,
    diag_variant: str = "adopted",
) -> dict[str, Any]:
    state = load_conversion_state(artifact_path)
    apply_conversion_state(model, state, diag_variant=diag_variant)
    return state


def layer_artifact_dir(output_dir: str | Path, layer_id: int) -> Path:
    return Path(output_dir) / "layers" / f"layer_{layer_id:02d}"


def save_layer_artifacts(
    output_dir: str | Path,
    layer_id: int,
    *,
    z: dict[str, torch.Tensor],
    metrics: dict[str, Any],
    train_log: list[dict[str, Any]],
    candidate_z: dict[str, torch.Tensor] | None = None,
    candidate_metrics: dict[str, Any] | None = None,
) -> Path:
    d = ensure_dir(layer_artifact_dir(output_dir, layer_id))
    torch.save(z, d / "best_diag.pt")
    if candidate_z is not None:
        torch.save(candidate_z, d / "candidate_best_diag.pt")
    if candidate_metrics is not None:
        write_json(d / "candidate_metrics.json", candidate_metrics)
    write_json(d / "metrics.json", metrics)
    with (d / "train_log.jsonl").open("w", encoding="utf-8") as f:
        import json

        for row in train_log:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return d


def load_layer_diag_file(output_dir: str | Path, layer_id: int) -> dict[str, torch.Tensor]:
    path = layer_artifact_dir(output_dir, layer_id) / "best_diag.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_pt(path, map_location="cpu")


def assert_resume_artifacts(cfg: E2ETrainConfig) -> None:
    if cfg.start_layer <= 0:
        return
    out = Path(cfg.output_dir)
    cfg_path = out / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"start_layer>0 requires existing {cfg_path}")
    import json

    prev = json.loads(cfg_path.read_text(encoding="utf-8"))
    current = {
        "diag_mode": cfg.diag_mode,
        "use_r64": cfg.use_r64,
        "rot_order": cfg.rot_order,
        "diag_train_scope": cfg.diag_train_scope,
        "recon_loss": cfg.recon_loss,
        "diag_lr": cfg.diag_lr,
        "diag_epochs": cfg.diag_epochs,
        "diag_scheduler": cfg.diag_scheduler,
        "optimizer": cfg.optimizer,
        "weight_decay": cfg.weight_decay,
        "diag_log2_clamp": cfg.to_dict()["diag_log2_clamp"],
        "calib_source": cfg.calib_source,
        "calib_seed": cfg.calib_seed,
        "calib_nsamples": cfg.calib_nsamples,
        "calib_val_nsamples": cfg.calib_val_nsamples,
        "calib_input_mode": cfg.calib_input_mode,
        "fusable_diag_components": cfg.fusable_diag_components,
        "router_align_loss_weight": cfg.router_align_loss_weight,
    }
    for key, val in current.items():
        prev_val = prev.get(key, 0.0 if key == "router_align_loss_weight" else None)
        if prev_val != val:
            raise ValueError(
                f"resume core field {key}={prev_val!r} != current {val!r}"
            )
    prev_master = str(prev.get("layer_rollback", "on"))
    prev_loss = str(prev.get("loss_rollback", "inherit"))
    prev_router = str(prev.get("router_rollback", "inherit"))
    if resolve_rollback_enabled(prev_master, prev_loss) != resolve_rollback_enabled(
        cfg.layer_rollback, cfg.loss_rollback
    ):
        raise ValueError("resume changes effective loss rollback policy")
    if resolve_rollback_enabled(prev_master, prev_router) != resolve_rollback_enabled(
        cfg.layer_rollback, cfg.router_rollback
    ):
        raise ValueError("resume changes effective router rollback policy")
    for i in range(cfg.start_layer):
        d = layer_artifact_dir(out, i)
        for name in ("best_diag.pt", "metrics.json", "train_log.jsonl"):
            p = d / name
            if not p.is_file():
                raise FileNotFoundError(f"missing layer artifact {p}")


def default_source_model() -> str:
    return DEFAULT_MODEL_PATH
