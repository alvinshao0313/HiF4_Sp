"""Layer-joint and online linear-independent DIAG training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import (
    E2ETrainConfig,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.fold import (
    fold_fusable_layer_inplace,
    freeze_online_layer_for_eval,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.semantic_hif4 import (
    ALL_PROJS,
    SwitchableNVHiF4Linear,
    clamp_hit_ratios,
    diag_parameter_stats,
    enable_eval_weight_cache,
    load_layer_diag_snapshot,
    set_layer_runtime_mode,
    snapshot_layer_diag,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.batching import (
    DynamicCalibrationCollator,
    build_length_bucket_batches,
    build_validation_batches,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.calibration import (
    CalibrationSample,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.layer_runtime import (
    PreparedLayerCall,
    ProgressiveHiddenCache,
    TeacherTargetCache,
    capture_qwen3_pre_layer_call,
    run_decoder_layer,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.losses import (
    finalize_reconstruction_loss,
    layer_objective,
    masked_reconstruction_components,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.metrics import relative_l2

ONLINE_TRAIN_ORDER = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class LayerTrainResult:
    layer_id: int
    accepted: bool
    rollback: bool
    best_epoch: int
    identity_val_loss: float
    best_val_loss: float
    final_val_loss: float
    recovery_vs_identity: float
    metrics: dict[str, Any]
    snapshot: dict[str, torch.Tensor]
    train_log: list[dict[str, Any]]


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"non-finite {name}")


def _packed_to_device(packed: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(packed)
    for key in ("input_ids", "attention_mask", "loss_mask", "lengths"):
        out[key] = packed[key].to(device)
    return out


def _prepare_batch(
    model: nn.Module,
    batch: list[CalibrationSample],
    collator: DynamicCalibrationCollator,
    x_cache: ProgressiveHiddenCache,
    device: torch.device,
) -> tuple[dict[str, Any], PreparedLayerCall, torch.Tensor]:
    packed = _packed_to_device(collator(batch), device)
    prepared = capture_qwen3_pre_layer_call(model, packed["input_ids"], packed["attention_mask"])
    hidden, _ = x_cache.assemble(packed["sample_ids"], device)
    return packed, prepared, hidden


def _branch_hooks(layer: nn.Module) -> tuple[list, dict[str, torch.Tensor]]:
    buf: dict[str, torch.Tensor] = {}
    handles = []

    def attn_hook(_m, _inp, output):
        y = output[0] if isinstance(output, tuple) else output
        buf["attn"] = y

    def mlp_hook(_m, _inp, output):
        y = output[0] if isinstance(output, tuple) else output
        buf["mlp"] = y

    handles.append(layer.self_attn.register_forward_hook(attn_hook))
    handles.append(layer.mlp.register_forward_hook(mlp_hook))
    return handles, buf


def build_teacher_targets(
    *,
    model: nn.Module,
    layer: nn.Module,
    samples: list[CalibrationSample],
    collator: DynamicCalibrationCollator,
    x_cache: ProgressiveHiddenCache,
    device: torch.device,
    batch_size: int,
    capture_aux: bool,
    capture_linears: bool,
) -> TeacherTargetCache:
    cache = TeacherTargetCache()
    set_layer_runtime_mode(layer, "native_nvfp4")
    layer.eval()
    for batch in build_validation_batches(samples, batch_size):
        packed, prepared, hidden = _prepare_batch(model, batch, collator, x_cache, device)
        handles = []
        buf: dict[str, torch.Tensor] = {}
        lin_in: dict[str, torch.Tensor] = {}
        lin_out: dict[str, torch.Tensor] = {}
        if capture_aux:
            handles, buf = _branch_hooks(layer)
        if capture_linears:
            for m in layer.modules():
                if isinstance(m, SwitchableNVHiF4Linear):

                    def _pre(mod, args, kwargs, name=m.proj):
                        lin_in[name] = (args[0] if args else kwargs["hidden_states"]).detach()

                    def _fwd(mod, args, output, name=m.proj):
                        y = output[0] if isinstance(output, tuple) else output
                        lin_out[name] = y.detach()

                    handles.append(m.register_forward_pre_hook(_pre, with_kwargs=True))
                    handles.append(m.register_forward_hook(_fwd))
        with torch.no_grad():
            y = run_decoder_layer(layer, hidden, prepared)
        for h in handles:
            h.remove()
        delta = y - hidden
        lengths = packed["lengths"]
        for i, sample in enumerate(batch):
            n = int(lengths[i].item())
            cache.delta[sample.sample_id] = delta[i, :n].detach().to("cpu", torch.bfloat16).contiguous()
            cache.output[sample.sample_id] = y[i, :n].detach().to("cpu", torch.bfloat16).contiguous()
            if capture_aux:
                cache.attn[sample.sample_id] = buf["attn"][i, :n].detach().to("cpu", torch.bfloat16).contiguous()
                cache.mlp[sample.sample_id] = buf["mlp"][i, :n].detach().to("cpu", torch.bfloat16).contiguous()
            if capture_linears:
                cache.linear_in[sample.sample_id] = {
                    k: v[i, :n].detach().to("cpu", torch.bfloat16).contiguous() for k, v in lin_in.items()
                }
                cache.linear_out[sample.sample_id] = {
                    k: v[i, :n].detach().to("cpu", torch.bfloat16).contiguous() for k, v in lin_out.items()
                }
    return cache


def _assemble_teacher(
    cache: TeacherTargetCache,
    sample_ids: list[str],
    field: str,
    device: torch.device,
) -> torch.Tensor:
    src: dict[str, torch.Tensor] = getattr(cache, field)
    hs = [src[sid] for sid in sample_ids]
    tmax = max(h.shape[0] for h in hs)
    out = torch.zeros(len(hs), tmax, hs[0].shape[1], dtype=torch.bfloat16, device=device)
    for i, h in enumerate(hs):
        out[i, : h.shape[0]] = h.to(device)
    return out


def evaluate_layer_real_qdq(
    *,
    model: nn.Module,
    layer: nn.Module,
    samples: list[CalibrationSample],
    collator: DynamicCalibrationCollator,
    x_cache: ProgressiveHiddenCache,
    teacher: TeacherTargetCache,
    cfg: E2ETrainConfig,
    device: torch.device,
) -> tuple[float, float, float]:
    set_layer_runtime_mode(layer, "hif4_eval")
    layer.eval()
    total_num = torch.zeros((), dtype=torch.float64)
    total_den = torch.zeros((), dtype=torch.float64)
    with torch.no_grad():
        for batch in build_validation_batches(samples, cfg.diag_batch_size):
            packed, prepared, hidden = _prepare_batch(model, batch, collator, x_cache, device)
            y_h = run_decoder_layer(layer, hidden, prepared)
            y_n = _assemble_teacher(teacher, packed["sample_ids"], "output", device)
            if cfg.recon_loss == "block_delta_nmse":
                pred = y_h.float() - hidden.float()
                target = y_n.float() - hidden.float()
            else:
                pred = y_h
                target = y_n
            num, den = masked_reconstruction_components(
                pred, target, packed["loss_mask"], packed["attention_mask"], cfg.recon_loss
            )
            if not torch.isfinite(num) or not torch.isfinite(den):
                raise RuntimeError("non-finite validation reconstruction components")
            total_num = total_num + num.detach().cpu().double()
            total_den = total_den + den.detach().cpu().double()
    loss = finalize_reconstruction_loss(total_num, total_den, cfg.recon_loss)
    return float(loss.item()), float(total_num.item()), float(total_den.item())


def _train_one_epoch(
    *,
    model: nn.Module,
    layer: nn.Module,
    samples: list[CalibrationSample],
    collator: DynamicCalibrationCollator,
    x_cache: ProgressiveHiddenCache,
    teacher: TeacherTargetCache,
    cfg: E2ETrainConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
) -> tuple[float, int]:
    set_layer_runtime_mode(layer, "hif4_train_ste")
    layer.train()
    total_num = 0.0
    total_den = 0.0
    steps = 0
    batches = build_length_bucket_batches(samples, cfg.diag_batch_size, cfg.calib_seed + epoch)
    capture_aux = cfg.attn_aux_loss_weight > 0 or cfg.mlp_aux_loss_weight > 0
    for batch in batches:
        packed, prepared, hidden = _prepare_batch(model, batch, collator, x_cache, device)
        y_n = _assemble_teacher(teacher, packed["sample_ids"], "output", device)
        handles = []
        buf: dict[str, torch.Tensor] = {}
        if capture_aux:
            handles, buf = _branch_hooks(layer)
        optimizer.zero_grad(set_to_none=True)
        y_h = run_decoder_layer(layer, hidden, prepared)
        attn_h = buf.get("attn")
        mlp_h = buf.get("mlp")
        attn_n = (
            _assemble_teacher(teacher, packed["sample_ids"], "attn", device)
            if cfg.attn_aux_loss_weight > 0
            else None
        )
        mlp_n = (
            _assemble_teacher(teacher, packed["sample_ids"], "mlp", device)
            if cfg.mlp_aux_loss_weight > 0
            else None
        )
        loss, num, den = layer_objective(
            y_h=y_h,
            y_n=y_n,
            x=hidden,
            loss_mask=packed["loss_mask"],
            attention_mask=packed["attention_mask"],
            loss_type=cfg.recon_loss,
            attn_h=attn_h,
            attn_n=attn_n,
            mlp_h=mlp_h,
            mlp_n=mlp_n,
            attn_weight=cfg.attn_aux_loss_weight,
            mlp_weight=cfg.mlp_aux_loss_weight,
        )
        _require_finite("train loss", loss)
        loss.backward()
        for p in layer.diag_state.parameters():
            if p.grad is not None:
                _require_finite("diag grad", p.grad)
        optimizer.step()
        layer.diag_state.project_log2_clamp(cfg.diag_log2_clamp)
        if scheduler is not None:
            scheduler.step()
        for h in handles:
            h.remove()
        total_num += float(num.detach().item())
        total_den += float(den.detach().item())
        steps += 1
    train_loss = float(finalize_reconstruction_loss(total_num, total_den, cfg.recon_loss).item())
    return train_loss, steps


def _identity_state(layer: nn.Module) -> None:
    layer.diag_state.zero_all()


def apply_rollback_decision(
    cfg: E2ETrainConfig,
    layer: nn.Module,
    best_val_loss: float,
    identity_val_loss: float,
) -> tuple[bool, bool, bool]:
    would_rollback = best_val_loss >= identity_val_loss
    rollback_applied = (cfg.layer_rollback == "on") and would_rollback
    accepted_vs_identity = not would_rollback
    if rollback_applied:
        _identity_state(layer)
    return would_rollback, rollback_applied, accepted_vs_identity


def rollback_metric_fields(
    *,
    cfg: E2ETrainConfig,
    would_rollback: bool,
    rollback_applied: bool,
    accepted_vs_identity: bool,
) -> dict[str, Any]:
    return {
        "rollback_enabled": cfg.layer_rollback,
        "would_rollback": would_rollback,
        "rollback_applied": rollback_applied,
        "accepted_vs_identity": accepted_vs_identity,
        "accepted": accepted_vs_identity,
        "rollback": rollback_applied,
    }


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def train_layer_joint(
    *,
    model: nn.Module,
    layer_idx: int,
    cfg: E2ETrainConfig,
    train_samples: list[CalibrationSample],
    val_samples: list[CalibrationSample],
    collator: DynamicCalibrationCollator,
    x_cache: ProgressiveHiddenCache,
    device: torch.device,
) -> LayerTrainResult:
    layer = model.model.layers[layer_idx]
    capture_aux = cfg.attn_aux_loss_weight > 0 or cfg.mlp_aux_loss_weight > 0
    teacher_train = build_teacher_targets(
        model=model,
        layer=layer,
        samples=train_samples,
        collator=collator,
        x_cache=x_cache,
        device=device,
        batch_size=cfg.diag_batch_size,
        capture_aux=capture_aux,
        capture_linears=False,
    )
    teacher_val = build_teacher_targets(
        model=model,
        layer=layer,
        samples=val_samples,
        collator=collator,
        x_cache=x_cache,
        device=device,
        batch_size=cfg.diag_batch_size,
        capture_aux=capture_aux,
        capture_linears=False,
    )
    _identity_state(layer)
    set_layer_runtime_mode(layer, "hif4_eval")
    enable_eval_weight_cache(layer)
    identity_val_loss, identity_num, identity_den = evaluate_layer_real_qdq(
        model=model,
        layer=layer,
        samples=val_samples,
        collator=collator,
        x_cache=x_cache,
        teacher=teacher_val,
        cfg=cfg,
        device=device,
    )
    for m in layer.modules():
        if isinstance(m, SwitchableNVHiF4Linear):
            m.clear_weight_cache()

    params = [p for p in layer.diag_state.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=cfg.diag_lr, weight_decay=0.0)
    n_batches = len(build_length_bucket_batches(train_samples, cfg.diag_batch_size, cfg.calib_seed))
    if n_batches <= 0:
        raise RuntimeError("no training batches")
    scheduler = None
    if cfg.diag_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.diag_epochs * n_batches, eta_min=0.0
        )

    best_val = float("inf")
    best_epoch = -1
    best_snap = snapshot_layer_diag(layer)
    train_log: list[dict[str, Any]] = []
    train_steps = 0
    for epoch in range(cfg.diag_epochs):
        train_loss, steps = _train_one_epoch(
            model=model,
            layer=layer,
            samples=train_samples,
            collator=collator,
            x_cache=x_cache,
            teacher=teacher_train,
            cfg=cfg,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
        )
        train_steps += steps
        set_layer_runtime_mode(layer, "hif4_eval")
        enable_eval_weight_cache(layer)
        val_loss, _, _ = evaluate_layer_real_qdq(
            model=model,
            layer=layer,
            samples=val_samples,
            collator=collator,
            x_cache=x_cache,
            teacher=teacher_val,
            cfg=cfg,
            device=device,
        )
        for m in layer.modules():
            if isinstance(m, SwitchableNVHiF4Linear):
                m.clear_weight_cache()
        stats = diag_parameter_stats(layer.diag_state)
        train_log.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": _current_lr(optimizer),
                **stats,
            }
        )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_snap = snapshot_layer_diag(layer)

    load_layer_diag_snapshot(layer, best_snap)
    would_rollback, rollback_applied, accepted_vs_identity = apply_rollback_decision(
        cfg, layer, best_val, identity_val_loss
    )
    accepted = accepted_vs_identity
    rollback = rollback_applied
    final_snap = snapshot_layer_diag(layer)
    set_layer_runtime_mode(layer, "hif4_eval")
    enable_eval_weight_cache(layer)
    final_val, _, _ = evaluate_layer_real_qdq(
        model=model,
        layer=layer,
        samples=val_samples,
        collator=collator,
        x_cache=x_cache,
        teacher=teacher_val,
        cfg=cfg,
        device=device,
    )
    recovery = 0.0 if identity_val_loss == 0 else (identity_val_loss - final_val) / identity_val_loss
    stats = diag_parameter_stats(layer.diag_state)
    hits = clamp_hit_ratios(layer.diag_state, cfg.diag_log2_clamp)
    metrics = {
        "layer_id": layer_idx,
        "identity_val_loss": identity_val_loss,
        "identity_val_num": identity_num,
        "identity_val_den": identity_den,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        **rollback_metric_fields(
            cfg=cfg,
            would_rollback=would_rollback,
            rollback_applied=rollback_applied,
            accepted_vs_identity=accepted_vs_identity,
        ),
        "recovery_vs_identity": recovery,
        "final_val_loss": final_val,
        "train_steps": train_steps,
        **stats,
        **hits,
    }
    teacher_train.clear()
    teacher_val.clear()
    return LayerTrainResult(
        layer_id=layer_idx,
        accepted=accepted,
        rollback=rollback,
        best_epoch=best_epoch,
        identity_val_loss=identity_val_loss,
        best_val_loss=best_val,
        final_val_loss=final_val,
        recovery_vs_identity=recovery,
        metrics=metrics,
        snapshot=final_snap,
        train_log=train_log,
    )


def _proj_module(layer: nn.Module, proj: str) -> SwitchableNVHiF4Linear:
    parent = layer.self_attn if proj in ("q_proj", "k_proj", "v_proj", "o_proj") else layer.mlp
    mod = getattr(parent, proj)
    if not isinstance(mod, SwitchableNVHiF4Linear):
        raise TypeError(proj)
    return mod


def train_layer_linear_independent(
    *,
    model: nn.Module,
    layer_idx: int,
    cfg: E2ETrainConfig,
    train_samples: list[CalibrationSample],
    val_samples: list[CalibrationSample],
    collator: DynamicCalibrationCollator,
    x_cache: ProgressiveHiddenCache,
    device: torch.device,
) -> LayerTrainResult:
    if cfg.diag_mode != "online":
        raise ValueError("linear_independent is only defined for online")
    layer = model.model.layers[layer_idx]
    teacher_train = build_teacher_targets(
        model=model,
        layer=layer,
        samples=train_samples,
        collator=collator,
        x_cache=x_cache,
        device=device,
        batch_size=cfg.diag_batch_size,
        capture_aux=False,
        capture_linears=True,
    )
    teacher_val = build_teacher_targets(
        model=model,
        layer=layer,
        samples=val_samples,
        collator=collator,
        x_cache=x_cache,
        device=device,
        batch_size=cfg.diag_batch_size,
        capture_aux=False,
        capture_linears=True,
    )
    _identity_state(layer)
    set_layer_runtime_mode(layer, "hif4_eval")
    enable_eval_weight_cache(layer)
    identity_val_loss, identity_num, identity_den = evaluate_layer_real_qdq(
        model=model,
        layer=layer,
        samples=val_samples,
        collator=collator,
        x_cache=x_cache,
        teacher=teacher_val,
        cfg=cfg,
        device=device,
    )
    for m in layer.modules():
        if isinstance(m, SwitchableNVHiF4Linear):
            m.clear_weight_cache()

    z_params = dict(layer.diag_state.named_parameters())
    train_log: list[dict[str, Any]] = []
    train_steps = 0
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.semantic_hif4 import (
        ONLINE_Z_NAMES,
    )

    for proj in ONLINE_TRAIN_ORDER:
        z_name = ONLINE_Z_NAMES[proj]
        for name, p in z_params.items():
            p.requires_grad_(name == z_name)
        linear = _proj_module(layer, proj)
        opt = torch.optim.Adam([z_params[z_name]], lr=cfg.diag_lr, weight_decay=0.0)
        n_batches = len(build_length_bucket_batches(train_samples, cfg.diag_batch_size, cfg.calib_seed))
        sched = None
        if cfg.diag_scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=cfg.diag_epochs * n_batches, eta_min=0.0
            )
        best_local = float("inf")
        best_local_snap = snapshot_layer_diag(layer)
        for epoch in range(cfg.diag_epochs):
            linear.set_mode("hif4_train_ste")
            epoch_num = 0.0
            epoch_den = 0.0
            for batch in build_length_bucket_batches(train_samples, cfg.diag_batch_size, cfg.calib_seed + epoch):
                packed = _packed_to_device(collator(batch), device)
                xs = []
                ys = []
                tmax = int(packed["lengths"].max().item())
                for i, sample in enumerate(batch):
                    xs.append(teacher_train.linear_in[sample.sample_id][proj])
                    ys.append(teacher_train.linear_out[sample.sample_id][proj])
                x = torch.zeros(len(batch), tmax, xs[0].shape[-1], dtype=torch.bfloat16, device=device)
                y_n = torch.zeros(len(batch), tmax, ys[0].shape[-1], dtype=torch.bfloat16, device=device)
                for i, (xv, yv) in enumerate(zip(xs, ys)):
                    x[i, : xv.shape[0]] = xv.to(device)
                    y_n[i, : yv.shape[0]] = yv.to(device)
                opt.zero_grad(set_to_none=True)
                y_h = linear(x)
                num, den = masked_reconstruction_components(
                    y_h, y_n, packed["loss_mask"], packed["attention_mask"], "block_output_nmse"
                )
                loss = finalize_reconstruction_loss(num, den, "block_output_nmse")
                _require_finite("local train loss", loss)
                loss.backward()
                if z_params[z_name].grad is not None:
                    _require_finite("local diag grad", z_params[z_name].grad)
                opt.step()
                layer.diag_state.project_log2_clamp(cfg.diag_log2_clamp)
                if sched is not None:
                    sched.step()
                train_steps += 1
                epoch_num += float(num.detach().item())
                epoch_den += float(den.detach().item())
            local_loss = float(
                finalize_reconstruction_loss(epoch_num, epoch_den, "block_output_nmse").item()
            )
            if local_loss < best_local:
                best_local = local_loss
                best_local_snap = snapshot_layer_diag(layer)
            train_log.append(
                {
                    "proj": proj,
                    "epoch": epoch,
                    "local_train_loss": local_loss,
                    "lr": _current_lr(opt),
                }
            )
        load_layer_diag_snapshot(layer, best_local_snap)

    for p in z_params.values():
        p.requires_grad_(True)
    set_layer_runtime_mode(layer, "hif4_eval")
    enable_eval_weight_cache(layer)
    best_val, _, _ = evaluate_layer_real_qdq(
        model=model,
        layer=layer,
        samples=val_samples,
        collator=collator,
        x_cache=x_cache,
        teacher=teacher_val,
        cfg=cfg,
        device=device,
    )
    would_rollback, rollback_applied, accepted_vs_identity = apply_rollback_decision(
        cfg, layer, best_val, identity_val_loss
    )
    accepted = accepted_vs_identity
    rollback = rollback_applied
    final_snap = snapshot_layer_diag(layer)
    final_val, _, _ = evaluate_layer_real_qdq(
        model=model,
        layer=layer,
        samples=val_samples,
        collator=collator,
        x_cache=x_cache,
        teacher=teacher_val,
        cfg=cfg,
        device=device,
    )
    recovery = 0.0 if identity_val_loss == 0 else (identity_val_loss - final_val) / identity_val_loss
    stats = diag_parameter_stats(layer.diag_state)
    hits = clamp_hit_ratios(layer.diag_state, cfg.diag_log2_clamp)
    metrics = {
        "layer_id": layer_idx,
        "identity_val_loss": identity_val_loss,
        "identity_val_num": identity_num,
        "identity_val_den": identity_den,
        "best_val_loss": best_val,
        "best_epoch": cfg.diag_epochs - 1,
        **rollback_metric_fields(
            cfg=cfg,
            would_rollback=would_rollback,
            rollback_applied=rollback_applied,
            accepted_vs_identity=accepted_vs_identity,
        ),
        "recovery_vs_identity": recovery,
        "final_val_loss": final_val,
        "train_steps": train_steps,
        "train_scope": "linear_independent",
        **stats,
        **hits,
    }
    teacher_train.clear()
    teacher_val.clear()
    return LayerTrainResult(
        layer_id=layer_idx,
        accepted=accepted,
        rollback=rollback,
        best_epoch=int(metrics["best_epoch"]),
        identity_val_loss=identity_val_loss,
        best_val_loss=best_val,
        final_val_loss=final_val,
        recovery_vs_identity=recovery,
        metrics=metrics,
        snapshot=final_snap,
        train_log=train_log,
    )


def run_fold_gate(
    *,
    model: nn.Module,
    layer: nn.Module,
    samples: list[CalibrationSample],
    collator: DynamicCalibrationCollator,
    x_cache: ProgressiveHiddenCache,
    device: torch.device,
    batch_size: int,
) -> float:
    set_layer_runtime_mode(layer, "hif4_eval")
    unfolded: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for batch in build_validation_batches(samples, batch_size):
            packed, prepared, hidden = _prepare_batch(model, batch, collator, x_cache, device)
            y = run_decoder_layer(layer, hidden, prepared)
            for i, sample in enumerate(batch):
                n = int(packed["lengths"][i].item())
                unfolded[sample.sample_id] = y[i, :n].float().cpu()
    fold_fusable_layer_inplace(layer, layer.diag_state, layer.self_attn.q_proj.use_r64)
    folded_err = []
    folded_ref = []
    with torch.no_grad():
        for batch in build_validation_batches(samples, batch_size):
            packed, prepared, hidden = _prepare_batch(model, batch, collator, x_cache, device)
            y = run_decoder_layer(layer, hidden, prepared)
            for i, sample in enumerate(batch):
                n = int(packed["lengths"][i].item())
                a = y[i, :n].float().cpu()
                b = unfolded[sample.sample_id]
                folded_err.append(a)
                folded_ref.append(b)
    rel = relative_l2(torch.cat([t.reshape(-1) for t in folded_err]), torch.cat([t.reshape(-1) for t in folded_ref]))
    if rel >= 1e-5:
        raise RuntimeError(f"fusable fold gate failed: relative L2={rel} >= 1e-5")
    return rel


def propagate_accepted_layer(
    *,
    model: nn.Module,
    layer: nn.Module,
    samples: list[CalibrationSample],
    collator: DynamicCalibrationCollator,
    x_cache: ProgressiveHiddenCache,
    device: torch.device,
    batch_size: int,
    finalize: Callable[[nn.Module], None],
) -> ProgressiveHiddenCache:
    finalize(layer)
    nxt = ProgressiveHiddenCache()
    layer.eval()
    with torch.no_grad():
        for batch in build_validation_batches(samples, batch_size):
            packed, prepared, hidden = _prepare_batch(model, batch, collator, x_cache, device)
            y = run_decoder_layer(layer, hidden, prepared)
            for i, sample in enumerate(batch):
                n = int(packed["lengths"][i].item())
                nxt.store(sample.sample_id, y[i, :n], n)
    x_cache.clear()
    return nxt


def finalize_layer_runtime(cfg: E2ETrainConfig, layer: nn.Module) -> None:
    if cfg.diag_mode == "fusable":
        return
    freeze_online_layer_for_eval(layer, layer.diag_state, cfg.use_r64, cfg.rot_order)
