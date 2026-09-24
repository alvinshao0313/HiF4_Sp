"""Train one independent progressive error-cancellation run."""
from __future__ import annotations

import fcntl
import gc
import json
import math
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import (
    load_qwen3_moe_layer_state,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import (
    NativeQwen3MoELayerRuntime,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.batching import (
    DynamicCalibrationCollator,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.moe_layer_runtime import (
    build_qwen3_moe_layer_call,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.student import Student
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.transforms import fold_initial_diag
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot
from .artifact import KIND, SCHEMA_VERSION, atomic_save, cpu_tree, load, read_json, sha256, write_json
from .config import Config, train_args
from .data import Dataset, assemble, batches, ensure_dataset, initial_hidden, valid_mask
from .directions import build_direction, compact_direction, load_direction
from .losses import (absolute_mse, batch_cosine, decompose_errors, direction_metrics,
                     masked_router_loss, router_metrics, summarize_error)


def _require_hif4():
    if Path(sys.prefix).name != "hif4":
        raise RuntimeError("run this experiment in the hif4 conda environment")


def _snapshot_dict(cache):
    return {sid: cache.get(sid).detach().cpu().clone() for sid in cache._hidden}


@torch.no_grad()
def native_targets(runtime, snapshot, samples, hidden, collator, device, batch_size):
    outputs, routers = {}, {}
    runtime.eval()
    for batch in batches(samples, batch_size):
        x, _ = assemble(hidden, batch, device)
        call = build_qwen3_moe_layer_call(str(snapshot), x)
        out = runtime(x, attention_mask=call.attention_mask,
                      position_embeddings=call.position_embeddings)
        for i, sample in enumerate(batch):
            n = int(sample.input_ids.numel())
            outputs[sample.sample_id] = out.output[i, :n].detach().cpu().to(torch.bfloat16)
            routers[sample.sample_id] = out.router_logits.reshape(x.shape[0], x.shape[1], -1)[i, :n].detach().cpu()
    return outputs, routers


@torch.no_grad()
def propagate_student(student, snapshot, samples, hidden, device, batch_size):
    result = {}
    student.eval()
    for batch in batches(samples, batch_size):
        x, _ = assemble(hidden, batch, device)
        call = build_qwen3_moe_layer_call(str(snapshot), x)
        out = student(x, position_embeddings=call.position_embeddings, use_ste=False)
        for i, sample in enumerate(batch):
            result[sample.sample_id] = out.output[i, :sample.input_ids.numel()].detach().cpu().to(torch.bfloat16)
    return result


@torch.no_grad()
def native_on_student_inputs(runtime, snapshot, samples, hidden, device, batch_size):
    result = {}
    runtime.eval()
    for batch in batches(samples, batch_size):
        x, _ = assemble(hidden, batch, device)
        call = build_qwen3_moe_layer_call(str(snapshot), x)
        out = runtime(x, attention_mask=call.attention_mask,
                      position_embeddings=call.position_embeddings)
        for i, sample in enumerate(batch):
            result[sample.sample_id] = out.output[i, :sample.input_ids.numel()].detach().cpu().to(torch.bfloat16)
    return result


def _direction_batch(direction, batch, target_len, device):
    if direction is None:
        return None
    hidden = int(direction[batch[0].sample_id].shape[-1])
    out = torch.zeros(len(batch), target_len, hidden, dtype=torch.float32, device=device)
    for i, sample in enumerate(batch):
        value = direction[sample.sample_id]
        n = min(int(value.shape[0]), target_len)
        out[i, :n] = value[:n].to(device)
    return out


def run_epoch(student, cfg, snapshot, samples, hidden, targets, router_targets,
              direction, collator, device, *, epoch, optimizer=None, scheduler=None):
    training = optimizer is not None
    student.train(training)
    # A layer's direction mapping is frozen at its boundary.  Keep training
    # batch membership fixed across epochs so a shuffled null control does not
    # silently change its sample correspondence between steps.
    epoch_batches = batches(samples, cfg.batch_size, training=training,
                            seed=cfg.calib_seed)
    total_mse = total_router = total_direction = 0.0
    direction_active = 0
    tokens = 0
    metric_sums = {}
    for batch_index, batch in enumerate(epoch_batches):
        x, lengths = assemble(hidden, batch, device)
        valid = (torch.arange(x.shape[1], device=device).unsqueeze(0) < lengths.unsqueeze(1))
        target, _ = assemble(targets, batch, device)
        teacher_router, _ = assemble(router_targets, batch, device)
        call = build_qwen3_moe_layer_call(str(snapshot), x)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            out = student(x, position_embeddings=call.position_embeddings, use_ste=training)
            mse = absolute_mse(out.output, target, valid)
            aux_logits = student.router_aux_logits(out.pre_moe_norm) if training else out.router_logits
            router = masked_router_loss(aux_logits, teacher_router, valid,
                                        kind=cfg.router_loss, top_k=cfg.router_top_k,
                                        temperature=cfg.router_temperature)
            direction_value = out.output.float() - target.detach().float()
            dir_loss = out.output.new_zeros((), dtype=torch.float32)
            active = False
            if direction is not None and cfg.lambda_direction > 0:
                d = _direction_batch(direction, batch, x.shape[1], device)
                dir_loss, active, _, _ = batch_cosine(direction_value, d, valid)
            objective = mse + cfg.router_loss_weight * router + cfg.lambda_direction * dir_loss
            if not torch.isfinite(objective):
                raise RuntimeError("nonfinite progressive objective")
            if training:
                objective.backward()
                grads = [p.grad for p in student.parameters() if p.grad is not None]
                if not grads or any(not torch.isfinite(g).all() for g in grads):
                    raise RuntimeError("missing or nonfinite student gradient")
                optimizer.step()
                scheduler.step()
        count = int(valid.sum())
        total_mse += float(mse.detach()) * count
        total_router += float(router.detach()) * count
        if active:
            direction_active += 1
            total_direction += float((dir_loss - 1.).detach()) * count
        tokens += count
        for key, value in router_metrics(out.router_logits, teacher_router, valid,
                                         top_k=student.base.spec.top_k).items():
            metric_sums[key] = metric_sums.get(key, 0.) + value * count
        print(json.dumps({"event": "batch", "layer": student.base.layer_idx,
                          "epoch": epoch, "train": training, "batch": batch_index + 1,
                          "batches": len(epoch_batches), "tokens": count,
                          "mse": float(mse.detach()), "router": float(router.detach()),
                          "direction_cosine": float((dir_loss - 1.).detach()) if active else None}),
              flush=True)
        del out, objective, mse, router, aux_logits, dir_loss
    if not tokens:
        raise RuntimeError("empty progressive epoch")
    mse_value = total_mse / tokens
    router_value = total_router / tokens
    return {"mse": mse_value, "router_loss": router_value,
            "select_score": mse_value + cfg.router_loss_weight * router_value,
            "direction_cosine": total_direction / tokens if direction_active else None,
            "direction_active_batches": direction_active,
            "direction_total_batches": len(epoch_batches), "tokens": tokens,
            "direction_coverage": direction_active / len(epoch_batches) if epoch_batches else 0.,
            "finite": True,
            **{key: value / tokens for key, value in metric_sums.items()}}


@torch.no_grad()
def layer_diagnostics(student, native, snapshot, samples, student_hidden, native_hidden,
                      previous_error, targets, direction, device, batch_size):
    native_student = native_on_student_inputs(native, snapshot, samples, student_hidden, device, batch_size)
    rows = {}
    for sample in samples:
        sid = sample.sample_id
        x = student_hidden[sid].unsqueeze(0).to(device)
        call = build_qwen3_moe_layer_call(str(snapshot), x)
        student_out = student(x, position_embeddings=call.position_embeddings, use_ste=False).output[0].cpu()
        target = targets[sid]
        valid = torch.ones(1, target.shape[0], dtype=torch.bool)
        values = decompose_errors(student_out.unsqueeze(0), target.unsqueeze(0),
                                  native_student[sid].unsqueeze(0), valid)
        current_summary = summarize_error(values["cumulative"], valid)
        previous = previous_error.get(sid) if previous_error is not None else None
        previous_l2_sq = float(previous.float().square().sum()) if previous is not None else 0.
        row = {"cumulative": current_summary,
               "propagated": summarize_error(values["propagated"], valid),
               "local": summarize_error(values["local"], valid),
               "delta_cumulative_l2_sq": float(values["cumulative"].square().sum()) - previous_l2_sq}
        if direction is not None:
            aligned_direction = _direction_batch(
                direction, [sample], int(target.shape[0]), target.device,
            )
            row["direction"] = direction_metrics(values["cumulative"], aligned_direction, valid)
            p = values["propagated"][valid].reshape(-1)
            q = values["local"][valid].reshape(-1)
            row["cosine_local_propagated"] = float((p @ q) / (p.norm() * q.norm())) if p.norm() and q.norm() else None
        rows[sid] = row
    return rows


def _write_protocol(out, cfg, dataset):
    write_json({"kind": KIND, "schema_version": SCHEMA_VERSION,
                "config": cfg.to_dict(), "dataset_protocol": dataset.protocol,
                "dataset_protocol_sha256": sha256(dataset.root / "protocol.json"),
                "status": "COMPLETE"}, out / "protocol.json")


def _train_locked(cfg, out, device, resume):
    dataset = ensure_dataset(out / "dataset", cfg)
    _write_protocol(out, cfg, dataset)
    samples = dataset.split("train") + dataset.split("val") + dataset.split("holdout")
    tokenizer = AutoTokenizer.from_pretrained(str(dataset.snapshot), trust_remote_code=True)
    collator = DynamicCalibrationCollator(tokenizer.pad_token_id)
    dataset_snapshot = Path(dataset.snapshot)
    snapshot = (dataset_snapshot.resolve() if dataset_snapshot.is_dir()
                else Path(resolve_local_snapshot(str(dataset_snapshot))).resolve())
    init_path = Path(cfg.init_artifact).resolve()
    if resume:
        state = load(out / "resume.pt")
        manifest = read_json(out / "manifest.json")
        if manifest["config"] != cfg.to_dict() or manifest["dataset_protocol_sha256"] != sha256(dataset.root / "protocol.json"):
            raise RuntimeError("resume protocol differs")
        initialization = load(out / "initialization.pt")
    else:
        if (out / "manifest.json").exists() or (out / "resume.pt").exists():
            raise FileExistsError("run exists; use --resume")
        from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.artifact import load_initialization
        initialization = load_initialization(init_path, cfg.model_path)
        atomic_save(initialization, out / "initialization.pt")
        init = initial_hidden(snapshot, samples, collator, device, cfg.batch_size)
        initial_dict = _snapshot_dict(init)
        state = {"layer_index": 0, "next_epoch": 0,
                 "student_hidden": initial_dict, "native_hidden": initial_dict,
                 "direction": None, "direction_manifest": None}
        manifest = {"kind": KIND, "schema_version": SCHEMA_VERSION,
                    "config": cfg.to_dict(), "source_snapshot": str(snapshot),
                    "init_sha256": sha256(init_path), "dataset_protocol_sha256": sha256(dataset.root / "protocol.json"),
                    "completed_layers": []}
        atomic_save(state, out / "resume.pt")
        write_json(manifest, out / "manifest.json")
    layer_ids = cfg.layer_ids
    while state["layer_index"] < len(layer_ids):
        layer = layer_ids[state["layer_index"]]
        print(json.dumps({"event": "layer_start", "layer": layer}), flush=True)
        student_hidden = state["student_hidden"]
        native_hidden = state["native_hidden"]
        # Keep the layer input snapshots immutable for direction/JVP provenance.
        source_student_hidden = {k: v.clone() for k, v in student_hidden.items()}
        source_native_hidden = {k: v.clone() for k, v in native_hidden.items()}
        native_state = load_qwen3_moe_layer_state(snapshot, layer, device)
        native = NativeQwen3MoELayerRuntime(native_state).to(device).eval()
        targets, routers = native_targets(native, snapshot, samples, native_hidden,
                                          collator, device, cfg.batch_size)
        base = fold_initial_diag(native_state, initialization[str(layer)])
        student = Student(base, cfg.matrix_sharing).to(device)
        optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=0.)
        train_samples = dataset.split("train")
        val_samples = dataset.split("val")
        direction = state.get("direction")
        if direction is not None:
            direction = {k: v.float() for k, v in direction.items()}
        direction_manifest = state.get("direction_manifest")
        layer_dir = out / f"layers/L{layer:02d}"
        if direction is None and layer > 0 and cfg.method != "baseline":
            layer_dir.mkdir(parents=True, exist_ok=True)
            cache_dir = layer_dir / "direction"
            if (cache_dir / "manifest.json").exists():
                direction, direction_manifest = load_direction(cache_dir)
                if direction_manifest["source_layer"] != layer or direction_manifest["method"] != cfg.method:
                    raise RuntimeError("existing direction cache does not match current layer protocol")
            else:
                if cache_dir.exists():
                    raise RuntimeError("incomplete direction cache; refusing silent regeneration")
                direction, direction_manifest = build_direction(
                    method=cfg.method, source_layer=layer, runtime=native,
                    native_hidden=native_hidden, student_hidden=student_hidden,
                    samples=samples, snapshot=snapshot, device=device,
                    output_dir=cache_dir, shuffle_seed=cfg.shuffle_seed,
                    batch_groups={
                        "train": batches(dataset.split("train"), cfg.batch_size,
                                         training=True, seed=cfg.calib_seed),
                        "val": batches(dataset.split("val"), cfg.batch_size),
                        "holdout": batches(dataset.split("holdout"), cfg.batch_size),
                    })
            write_json(direction_manifest, layer_dir / "direction_manifest.json")
        next_epoch = int(state.get("next_epoch", 0))
        n_batches = len(batches(train_samples, cfg.batch_size, training=True, seed=cfg.calib_seed))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs * n_batches, eta_min=0.)
        if "parameters" in state:
            student.learned.load_state_dict(state["parameters"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            best = state["best"]
        else:
            initial_val = run_epoch(student, cfg, snapshot, val_samples, student_hidden, targets, routers,
                                    direction, collator, device, epoch=-1)
            write_json(initial_val, out / f"layers/L{layer:02d}/initial_metrics.json")
            best = None
        for epoch in range(next_epoch, cfg.epochs):
            train_metrics = run_epoch(student, cfg, snapshot, train_samples, student_hidden, targets, routers,
                                      direction, collator, device, epoch=epoch, optimizer=optimizer, scheduler=scheduler)
            val_metrics = run_epoch(student, cfg, snapshot, val_samples, student_hidden, targets, routers,
                                    direction, collator, device, epoch=epoch)
            if best is None or val_metrics["select_score"] < best["select_score"]:
                best = {"epoch": epoch, "select_score": val_metrics["select_score"],
                        "train_metrics": train_metrics, "metrics": val_metrics,
                        "parameters": student.learned.snapshot()}
            write_json({"epoch": epoch, "train": train_metrics, "validation": val_metrics},
                       out / f"layers/L{layer:02d}/epochs/{epoch:03d}.json")
            state.update(layer_index=state["layer_index"], next_epoch=epoch + 1,
                         student_hidden=student_hidden, native_hidden=native_hidden,
                         direction=direction, direction_manifest=direction_manifest,
                         parameters=student.learned.snapshot(), optimizer=cpu_tree(optimizer.state_dict()),
                         scheduler=scheduler.state_dict(), best=best)
            atomic_save(state, out / "resume.pt")
        if best is None:
            raise RuntimeError("no layer checkpoint selected")
        student.learned.load_state_dict(best["parameters"], strict=True)
        layer_dir = out / f"layers/L{layer:02d}"
        atomic_save(best, layer_dir / "selected.pt")
        holdout_metrics = run_epoch(student, cfg, snapshot, dataset.split("holdout"),
                                    student_hidden, targets, routers, direction, collator,
                                    device, epoch=-2)
        write_json({"layer": layer, "selected_epoch": best["epoch"],
                    "train": best["train_metrics"], "validation": best["metrics"],
                    "holdout": holdout_metrics,
                    "direction_manifest": direction_manifest,
                    "finite": all(x.get("finite", False) for x in
                                   (best["train_metrics"], best["metrics"], holdout_metrics))},
                   layer_dir / "metrics.json")
        previous_error = {sid: source_student_hidden[sid].float() - source_native_hidden[sid].float()
                          for sid in source_student_hidden}
        diag = {}
        for name, split in (("train", dataset.split("train")), ("val", dataset.split("val")),
                            ("holdout", dataset.split("holdout"))):
            diag[name] = layer_diagnostics(student, native, snapshot, split, student_hidden,
                                           native_hidden, previous_error, targets, direction,
                                           device, cfg.batch_size)
        write_json({"layer": layer, "method": cfg.method, "lambda_direction": cfg.lambda_direction,
                    "direction_manifest": direction_manifest, "samples": diag},
                   layer_dir / "metrics_diagnostics.json")
        next_student = propagate_student(student, snapshot, samples, student_hidden, device, cfg.batch_size)
        next_native = targets
        # The next layer receives a fresh boundary cache.  Its direction is
        # built once at the start of that layer and remains frozen across all
        # epochs; layer 0 intentionally has no direction term.
        next_direction = None
        next_direction_manifest = None
        state = {"layer_index": state["layer_index"] + 1, "next_epoch": 0,
                 "student_hidden": next_student, "native_hidden": next_native,
                 "direction": next_direction, "direction_manifest": next_direction_manifest}
        manifest["completed_layers"] = layer_ids[:state["layer_index"]]
        atomic_save(state, out / "resume.pt")
        write_json(manifest, out / "manifest.json")
        if direction is not None and direction_manifest is not None:
            compact_direction(layer_dir / "direction", direction, direction_manifest)
        del student, optimizer, scheduler, native, native_state, base, targets, routers
        gc.collect()
        torch.cuda.empty_cache()
    write_json({"status": "COMPLETE", "layers": layer_ids,
                "method": cfg.method, "lambda_direction": cfg.lambda_direction}, out / "summary.json")


def train(cfg: Config, *, resume=False):
    cfg.validate()
    _require_hif4()
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA")
    torch.manual_seed(cfg.calib_seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(cfg.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "train.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _train_locked(cfg, out, torch.device("cuda:0"), resume)


if __name__ == "__main__":
    config, should_resume = train_args()
    train(config, resume=should_resume)
