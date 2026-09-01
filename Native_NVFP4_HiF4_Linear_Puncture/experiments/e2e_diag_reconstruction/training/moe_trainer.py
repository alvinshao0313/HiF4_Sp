"""Lazy layerwise training orchestration for Qwen3-MoE."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.artifact import (
    save_conversion_artifact,
    save_layer_artifacts,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import (
    E2ETrainConfig,
    ROUTER_ALIGN_TEMPERATURE,
    ROUTER_ALIGN_TYPE,
    resolve_rollback_enabled,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import (
    load_qwen3_moe_layer_state,
    release_qwen3_moe_layer_state,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_fold import (
    router_alignment_kl,
    router_compensation_topk_gate,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import (
    MoEFusableDiagState,
    NativeQwen3MoELayerRuntime,
    StudentQwen3MoELayerRuntime,
    StudentStepCache,
    build_moe_diag_state,
    summarize_expert_coverage,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.batching import (
    DynamicCalibrationCollator,
    build_length_bucket_batches,
    build_validation_batches,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.calibration import (
    CalibrationSample,
    build_or_load_calibration,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.layer_runtime import (
    ProgressiveHiddenCache,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.losses import (
    finalize_reconstruction_loss,
    masked_reconstruction_components,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.moe_layer_runtime import (
    build_qwen3_moe_layer_call,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot
from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import ensure_dir, write_json


def _load_tensor(snapshot: Path, key: str) -> torch.Tensor:
    import json

    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shard = index["weight_map"][key]
    with safe_open(str(snapshot / shard), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def build_initial_moe_hidden_cache(
    snapshot: Path,
    samples: list[CalibrationSample],
    collator: DynamicCalibrationCollator,
    device: torch.device,
    batch_size: int,
) -> ProgressiveHiddenCache:
    embed = _load_tensor(snapshot, "model.embed_tokens.weight").to(device=device, dtype=torch.bfloat16)
    cache = ProgressiveHiddenCache()
    for batch in build_validation_batches(samples, batch_size):
        packed = collator(batch)
        ids = packed["input_ids"].to(device)
        hidden = embed[ids]
        for i, sample in enumerate(batch):
            n = int(packed["lengths"][i].item())
            cache.store(sample.sample_id, hidden[i, :n], n)
    del embed
    return cache


def _teacher_targets(
    runtime: NativeQwen3MoELayerRuntime,
    snapshot: Path,
    samples: list[CalibrationSample],
    collator: DynamicCalibrationCollator,
    x_cache: ProgressiveHiddenCache,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    targets: dict[str, torch.Tensor] = {}
    runtime.eval()
    for batch in build_validation_batches(samples, batch_size):
        packed = collator(batch)
        sample_ids = [s.sample_id for s in batch]
        hidden, _ = x_cache.assemble(sample_ids, device)
        call = build_qwen3_moe_layer_call(str(snapshot), hidden)
        with torch.no_grad():
            y = runtime(
                hidden,
                attention_mask=None,
                position_embeddings=call.position_embeddings,
            ).output
        for i, sample in enumerate(batch):
            n = int(packed["lengths"][i].item())
            targets[sample.sample_id] = y[i, :n].detach().cpu().to(torch.bfloat16)
    return targets


def _assemble_targets(targets: dict[str, torch.Tensor], sample_ids: list[str], device: torch.device) -> torch.Tensor:
    hs = [targets[sid] for sid in sample_ids]
    tmax = max(int(h.shape[0]) for h in hs)
    hidden = int(hs[0].shape[1])
    out = torch.zeros(len(hs), tmax, hidden, dtype=torch.bfloat16, device=device)
    for i, h in enumerate(hs):
        out[i, : h.shape[0]] = h.to(device)
    return out


def _eval_student(
    runtime: StudentQwen3MoELayerRuntime,
    snapshot: Path,
    samples: list[CalibrationSample],
    collator: DynamicCalibrationCollator,
    x_cache: ProgressiveHiddenCache,
    targets: dict[str, torch.Tensor],
    cfg: E2ETrainConfig,
    device: torch.device,
) -> tuple[float, float, float, float, torch.Tensor, dict[str, int]]:
    total_num = torch.zeros((), dtype=torch.float64)
    total_den = torch.zeros((), dtype=torch.float64)
    router_kl_weighted_sum = 0.0
    router_kl_tokens = 0
    counts = torch.zeros(runtime.state.spec.num_experts, dtype=torch.long)
    qdq_calls: dict[str, int] = {}
    runtime.eval()
    for batch in build_validation_batches(samples, cfg.diag_batch_size):
        packed = collator(batch)
        sample_ids = [s.sample_id for s in batch]
        hidden, _ = x_cache.assemble(sample_ids, device)
        target = _assemble_targets(targets, sample_ids, device)
        call = build_qwen3_moe_layer_call(str(snapshot), hidden)
        cache = StudentStepCache.new()
        with torch.no_grad():
            out = runtime(
                hidden,
                attention_mask=None,
                position_embeddings=call.position_embeddings,
                step_cache=cache,
                use_ste=False,
            )
        pred = out.output.float() - hidden.float()
        tgt = target.float() - hidden.float()
        num, den = masked_reconstruction_components(
            pred, tgt, packed["loss_mask"].to(device), packed["attention_mask"].to(device), cfg.recon_loss
        )
        total_num += num.cpu().double()
        total_den += den.cpu().double()
        if cfg.diag_mode == "fusable":
            if out.router_input is None:
                raise RuntimeError("fusable validation requires router_input")
            valid_router_inputs = torch.cat(
                [
                    out.router_input[i, : int(packed["lengths"][i].item())]
                    for i in range(len(batch))
                ],
                dim=0,
            )
            with torch.no_grad():
                batch_router_kl = router_alignment_kl(
                    valid_router_inputs, runtime.state.router_weight, runtime.diag_state,
                    temperature=ROUTER_ALIGN_TEMPERATURE,
                )
            router_kl_weighted_sum += float(batch_router_kl.item()) * int(valid_router_inputs.shape[0])
            router_kl_tokens += int(valid_router_inputs.shape[0])
        counts += out.per_expert_routed_token_count.cpu()
        for key, value in cache.weight_qdq_calls_by_proj.items():
            qdq_calls[key] = qdq_calls.get(key, 0) + value
    loss = finalize_reconstruction_loss(total_num, total_den, cfg.recon_loss)
    router_kl = router_kl_weighted_sum / max(router_kl_tokens, 1)
    return float(loss.item()), router_kl, float(total_num.item()), float(total_den.item()), counts, qdq_calls


def _collect_router_gate_inputs(
    runtime: StudentQwen3MoELayerRuntime,
    snapshot: Path,
    samples: list[CalibrationSample],
    collator: DynamicCalibrationCollator,
    x_cache: ProgressiveHiddenCache,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    runtime.eval()
    for batch in build_validation_batches(samples, batch_size):
        packed = collator(batch)
        sample_ids = [s.sample_id for s in batch]
        hidden, _ = x_cache.assemble(sample_ids, device)
        call = build_qwen3_moe_layer_call(str(snapshot), hidden)
        with torch.no_grad():
            _, router_input = runtime.forward_to_router_input(
                hidden,
                attention_mask=None,
                position_embeddings=call.position_embeddings,
                step_cache=StudentStepCache.new(),
                use_ste=False,
            )
        for i, _sample in enumerate(batch):
            n = int(packed["lengths"][i].item())
            chunks.append(router_input[i, :n].detach())
    if not chunks:
        raise RuntimeError("router gate requires validation data")
    return torch.cat(chunks, dim=0)


def _adopt_candidate_snapshot(
    candidate: dict[str, torch.Tensor],
    *,
    loss_rollback_applied: bool,
    router_rollback_applied: bool,
) -> dict[str, torch.Tensor]:
    """Build the final DIAG without ever mutating the saved candidate."""
    adopted = {name: value.clone() for name, value in candidate.items()}
    if loss_rollback_applied:
        return {name: torch.zeros_like(value) for name, value in adopted.items()}
    if router_rollback_applied:
        if "z_gu" not in adopted:
            raise RuntimeError("router rollback requires fusable z_gu")
        adopted["z_gu"].zero_()
    return adopted


def _train_layer(
    runtime: StudentQwen3MoELayerRuntime,
    snapshot: Path,
    train_samples: list[CalibrationSample],
    collator: DynamicCalibrationCollator,
    x_cache: ProgressiveHiddenCache,
    targets: dict[str, torch.Tensor],
    cfg: E2ETrainConfig,
    device: torch.device,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    scheduler,
) -> int:
    steps = 0
    runtime.train()
    for batch in build_length_bucket_batches(train_samples, cfg.diag_batch_size, cfg.calib_seed + epoch):
        packed = collator(batch)
        sample_ids = [s.sample_id for s in batch]
        hidden, _ = x_cache.assemble(sample_ids, device)
        target = _assemble_targets(targets, sample_ids, device)
        call = build_qwen3_moe_layer_call(str(snapshot), hidden)
        cache = StudentStepCache.new()
        optimizer.zero_grad(set_to_none=True)
        out = runtime(
            hidden,
            attention_mask=None,
            position_embeddings=call.position_embeddings,
            step_cache=cache,
            use_ste=True,
        )
        pred = out.output.float() - hidden.float()
        tgt = target.float() - hidden.float()
        num, den = masked_reconstruction_components(
            pred, tgt, packed["loss_mask"].to(device), packed["attention_mask"].to(device), cfg.recon_loss
        )
        loss = finalize_reconstruction_loss(num, den, cfg.recon_loss)
        if cfg.diag_mode == "fusable" and cfg.router_align_loss_weight > 0:
            if out.router_input is None:
                raise RuntimeError("fusable router alignment requires router_input")
            valid_router_inputs = torch.cat(
                [
                    out.router_input[i, : int(packed["lengths"][i].item())]
                    for i in range(len(batch))
                ],
                dim=0,
            )
            router_kl = router_alignment_kl(
                valid_router_inputs,
                runtime.state.router_weight,
                runtime.diag_state,
                temperature=ROUTER_ALIGN_TEMPERATURE,
            )
            loss = loss + float(cfg.router_align_loss_weight) * router_kl
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite MoE layer train loss")
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        runtime.diag_state.clamp_log2_(cfg.diag_log2_clamp)
        steps += 1
    return steps


def _propagate(
    runtime,
    snapshot: Path,
    samples: list[CalibrationSample],
    collator: DynamicCalibrationCollator,
    x_cache: ProgressiveHiddenCache,
    device: torch.device,
    batch_size: int,
    *,
    student: bool,
) -> ProgressiveHiddenCache:
    nxt = ProgressiveHiddenCache()
    for batch in build_validation_batches(samples, batch_size):
        packed = collator(batch)
        sample_ids = [s.sample_id for s in batch]
        hidden, _ = x_cache.assemble(sample_ids, device)
        call = build_qwen3_moe_layer_call(str(snapshot), hidden)
        with torch.no_grad():
            if student:
                y = runtime(
                    hidden,
                    attention_mask=None,
                    position_embeddings=call.position_embeddings,
                    step_cache=StudentStepCache.new(),
                    use_ste=False,
                ).output
            else:
                y = runtime(
                    hidden,
                    attention_mask=None,
                    position_embeddings=call.position_embeddings,
                ).output
        for i, sample in enumerate(batch):
            n = int(packed["lengths"][i].item())
            nxt.store(sample.sample_id, y[i, :n], n)
    return nxt


def train_qwen3_moe_lazy(cfg: E2ETrainConfig, device: torch.device) -> None:
    snapshot = Path(resolve_local_snapshot(cfg.model_path))
    out_dir = ensure_dir(cfg.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=True)
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
    x_cache = build_initial_moe_hidden_cache(snapshot, all_samples, collator, device, cfg.diag_batch_size)
    write_json(out_dir / "config.json", cfg.to_dict())
    layer_records: dict[int, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []

    for layer_idx in range(cfg.start_layer, cfg.end_layer + 1):
        state = load_qwen3_moe_layer_state(snapshot, layer_idx, device)
        try:
            native = NativeQwen3MoELayerRuntime(state).to(device).eval()
            train_targets = _teacher_targets(native, snapshot, train_samples, collator, x_cache, device, cfg.diag_batch_size)
            val_targets = _teacher_targets(native, snapshot, val_samples, collator, x_cache, device, cfg.diag_batch_size)
            diag_state = build_moe_diag_state(state.spec, cfg.diag_mode).to(device)
            gu_active = True
            if isinstance(diag_state, MoEFusableDiagState):
                diag_state.configure_fusable_components(cfg.fusable_diag_components)
                gu_active = bool(diag_state.z_gu.requires_grad)
            student = StudentQwen3MoELayerRuntime(
                state,
                diag_state,
                use_r64=cfg.use_r64,
                rot_order=cfg.rot_order,
            ).to(device)
            identity_loss, identity_router_kl, identity_num, identity_den, val_counts, qdq_calls = _eval_student(
                student, snapshot, val_samples, collator, x_cache, val_targets, cfg, device
            )
            params = [p for p in diag_state.parameters() if p.requires_grad]
            if not params:
                raise RuntimeError(
                    f"no trainable DIAG params for fusable_diag_components={cfg.fusable_diag_components!r}"
                )
            if cfg.optimizer != "AdamW":
                raise ValueError(f"unsupported MoE optimizer={cfg.optimizer!r}")
            optimizer = torch.optim.AdamW(params, lr=cfg.diag_lr, weight_decay=float(cfg.weight_decay))
            n_batches = len(build_length_bucket_batches(train_samples, cfg.diag_batch_size, cfg.calib_seed))
            if n_batches <= 0:
                raise RuntimeError("no training batches")
            scheduler = None
            if cfg.diag_scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cfg.diag_epochs * n_batches, eta_min=0.0
                )
            identity_objective = identity_loss + float(cfg.router_align_loss_weight) * identity_router_kl
            best_loss = identity_loss
            best_router_kl = identity_router_kl
            best_objective = identity_objective
            best_epoch = -1
            best_snapshot = diag_state.snapshot()
            train_steps = 0
            for epoch in range(cfg.diag_epochs):
                train_steps += _train_layer(
                    student,
                    snapshot,
                    train_samples,
                    collator,
                    x_cache,
                    train_targets,
                    cfg,
                    device,
                    epoch,
                    optimizer,
                    scheduler,
                )
                val_loss, val_router_kl, _num, _den, val_counts, qdq_calls = _eval_student(
                    student, snapshot, val_samples, collator, x_cache, val_targets, cfg, device
                )
                val_objective = val_loss + float(cfg.router_align_loss_weight) * val_router_kl
                if val_objective < best_objective:
                    best_loss = val_loss
                    best_router_kl = val_router_kl
                    best_objective = val_objective
                    best_epoch = epoch
                    best_snapshot = diag_state.snapshot()
            diag_state.load_snapshot(best_snapshot)
            candidate_best_snapshot = diag_state.snapshot()
            candidate_best_loss = float(best_loss)
            candidate_best_router_kl = float(best_router_kl)
            candidate_best_objective = float(best_objective)
            candidate_best_epoch = int(best_epoch)
            loss_rollback_enabled = resolve_rollback_enabled(cfg.layer_rollback, cfg.loss_rollback)
            router_rollback_enabled = resolve_rollback_enabled(cfg.layer_rollback, cfg.router_rollback)
            loss_would_rollback = candidate_best_loss >= identity_loss

            candidate_gate_stats: dict[str, float | int] = {}
            router_gate_input_tokens = 0
            if cfg.diag_mode == "fusable":
                hidden_val = _collect_router_gate_inputs(
                    student,
                    snapshot,
                    val_samples,
                    collator,
                    x_cache,
                    device,
                    cfg.diag_batch_size,
                )
                router_gate_input_tokens = int(hidden_val.shape[0])
                candidate_gate_stats = router_compensation_topk_gate(
                    hidden_val,
                    state.router_weight,
                    diag_state,  # type: ignore[arg-type]
                    top_k=state.spec.top_k,
                )
            router_would_rollback = (
                cfg.diag_mode == "fusable"
                and gu_active
                and int(candidate_gate_stats.get("topk_mismatches", 0)) != 0
            )

            loss_rollback_applied = loss_rollback_enabled and loss_would_rollback
            router_rollback_applied = (
                not loss_rollback_applied
                and router_rollback_enabled
                and router_would_rollback
            )
            adopted_snapshot = _adopt_candidate_snapshot(
                candidate_best_snapshot,
                loss_rollback_applied=loss_rollback_applied,
                router_rollback_applied=router_rollback_applied,
            )

            diag_state.load_snapshot(adopted_snapshot)
            adopted_loss, adopted_router_kl, _adopted_num, _adopted_den, val_counts, qdq_calls = _eval_student(
                student, snapshot, val_samples, collator, x_cache, val_targets, cfg, device
            )
            adopted_objective = adopted_loss + float(cfg.router_align_loss_weight) * adopted_router_kl
            gate_stats: dict[str, float | int] = {}
            if cfg.diag_mode == "fusable":
                adopted_hidden_val = _collect_router_gate_inputs(
                    student,
                    snapshot,
                    val_samples,
                    collator,
                    x_cache,
                    device,
                    cfg.diag_batch_size,
                )
                gate_stats = router_compensation_topk_gate(
                    adopted_hidden_val,
                    state.router_weight,
                    diag_state,  # type: ignore[arg-type]
                    top_k=state.spec.top_k,
                )
                if router_rollback_applied and int(gate_stats["topk_mismatches"]) != 0:
                    raise RuntimeError(
                        f"D_GU-only router rollback did not restore BF16 router top-k: {gate_stats}"
                    )

            rollback = loss_rollback_applied or router_rollback_applied
            router_gate_rollback = router_rollback_applied
            best_snapshot = adopted_snapshot
            best_loss = float(adopted_loss)
            coverage = summarize_expert_coverage(
                train_counts=torch.zeros_like(val_counts),
                val_counts=val_counts,
                step_cache=StudentStepCache(transformed_weight_qdq={}, weight_qdq_calls_by_proj=qdq_calls),
            )
            metrics = {
                "layer_id": layer_idx,
                "identity_val_loss": identity_loss,
                "identity_router_kl": identity_router_kl,
                "identity_objective": identity_objective,
                "identity_val_num": identity_num,
                "identity_val_den": identity_den,
                "best_val_loss": best_loss,
                "best_epoch": best_epoch,
                "candidate_best_val_loss": candidate_best_loss,
                "candidate_best_router_kl": candidate_best_router_kl,
                "candidate_best_objective": candidate_best_objective,
                "candidate_best_epoch": candidate_best_epoch,
                "adopted_val_loss": best_loss,
                "adopted_router_kl": adopted_router_kl,
                "adopted_objective": adopted_objective,
                "loss_rollback_enabled": loss_rollback_enabled,
                "loss_would_rollback": loss_would_rollback,
                "loss_rollback_applied": loss_rollback_applied,
                "router_rollback_enabled": router_rollback_enabled,
                "router_would_rollback": router_would_rollback,
                "router_rollback_applied": router_rollback_applied,
                "router_align_type": ROUTER_ALIGN_TYPE,
                "router_align_temperature": ROUTER_ALIGN_TEMPERATURE,
                "router_align_loss_weight": cfg.router_align_loss_weight,
                "accepted": not rollback,
                "rollback": rollback,
                "router_gate_rollback": router_gate_rollback,
                "router_gate_input_count": router_gate_input_tokens,
                "train_steps": train_steps,
                "active_experts_val": sorted(coverage.active_experts_val),
                "never_routed_experts": coverage.never_routed_experts,
                "min_routed_tokens": coverage.min_routed_tokens,
                "median_routed_tokens": coverage.median_routed_tokens,
                "max_routed_tokens": coverage.max_routed_tokens,
                "weight_qdq_calls_by_proj": coverage.weight_qdq_calls_by_proj,
                **{f"candidate_router_{k}": v for k, v in candidate_gate_stats.items()},
                **{f"router_{k}": v for k, v in gate_stats.items()},
            }
            candidate_metrics = {
                "layer_id": layer_idx,
                "identity_val_loss": identity_loss,
                "identity_router_kl": identity_router_kl,
                "identity_objective": identity_objective,
                "candidate_best_val_loss": candidate_best_loss,
                "candidate_best_router_kl": candidate_best_router_kl,
                "candidate_best_objective": candidate_best_objective,
                "candidate_best_epoch": candidate_best_epoch,
                "loss_gate_pass": candidate_best_loss < identity_loss,
                "router_gate_input_count": router_gate_input_tokens,
                "loss_rollback_enabled": loss_rollback_enabled,
                "router_rollback_enabled": router_rollback_enabled,
                "router_would_rollback": router_would_rollback,
                "router_align_type": ROUTER_ALIGN_TYPE,
                "router_align_temperature": ROUTER_ALIGN_TEMPERATURE,
                "router_align_loss_weight": cfg.router_align_loss_weight,
                **{f"router_{k}": v for k, v in candidate_gate_stats.items()},
            }
            save_layer_artifacts(
                out_dir,
                layer_idx,
                z=best_snapshot,
                metrics=metrics,
                train_log=[],
                candidate_z=candidate_best_snapshot,
                candidate_metrics=candidate_metrics,
            )
            layer_records[layer_idx] = {
                "accepted": not rollback,
                "rollback": rollback,
                "best_epoch": best_epoch,
                "candidate_best_epoch": candidate_best_epoch,
                "candidate_z": candidate_best_snapshot,
                "adopted_z": best_snapshot,
                "z": best_snapshot,
                "loss_rollback_applied": loss_rollback_applied,
                "router_rollback_applied": router_rollback_applied,
            }
            summaries.append(metrics)
            if cfg.calib_input_mode == "teacher" or loss_rollback_applied:
                x_cache = _propagate(native, snapshot, all_samples, collator, x_cache, device, cfg.diag_batch_size, student=False)
            else:
                x_cache = _propagate(student, snapshot, all_samples, collator, x_cache, device, cfg.diag_batch_size, student=True)
        finally:
            release_qwen3_moe_layer_state(state)
    save_conversion_artifact(cfg=cfg, layer_records=layer_records, out_dir=out_dir)
    write_json(out_dir / "summary.json", {"layers": summaries})
