"""Layerwise training with independent progressive inputs and atomic epoch resume."""
from __future__ import annotations

import fcntl
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import NativeQwen3MoELayerRuntime
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.calibration import build_or_load_calibration
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.batching import (
    DynamicCalibrationCollator, build_length_bucket_batches, build_validation_batches,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.moe_layer_runtime import build_qwen3_moe_layer_call
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.moe_trainer import build_initial_moe_hidden_cache
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot
from .artifact import (KIND, SCHEMA_VERSION, atomic_save, cpu_tree, load, load_initialization,
                       load_manifest, sha256, write_json, source_fingerprint, baseline_provenance)
from .config import Config, train_args
from .losses import masked_router_loss, router_metrics
from .student import Student, causal_mask
from .transforms import fold_initial_diag
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import _rms_norm


def assemble(values, samples, device):
    items = [values[s.sample_id] for s in samples]
    result = torch.zeros(len(items), max(x.shape[0] for x in items), items[0].shape[-1],
                         dtype=items[0].dtype, device=device)
    for i, value in enumerate(items):
        result[i, :value.shape[0]] = value.to(device)
    return result


@torch.no_grad()
def teacher_targets(native, snapshot, samples, collator, hidden, device, batch_size):
    outputs, routers, attention_outputs = {}, {}, {}
    for batch in build_validation_batches(samples, batch_size):
        x = assemble(hidden, batch, device)
        call = build_qwen3_moe_layer_call(str(snapshot), x)
        normed = _rms_norm(x, native.state.input_layernorm_weight, native.rms_norm_eps)
        after_attention = x + native.attention_projections(normed, causal_mask(x), call.position_embeddings).o
        normed = _rms_norm(after_attention, native.state.post_attention_layernorm_weight, native.rms_norm_eps)
        out = native.routed_moe(normed)
        output = after_attention + out.output
        logits = out.router_logits.reshape(x.shape[0], x.shape[1], -1)
        for i, sample in enumerate(batch):
            n = sample.input_ids.numel()
            outputs[sample.sample_id] = output[i, :n].cpu().contiguous()
            routers[sample.sample_id] = logits[i, :n].cpu().contiguous()
            attention_outputs[sample.sample_id] = after_attention[i, :n].cpu().contiguous()
    return outputs, routers, attention_outputs


def run_epoch(student, cfg, snapshot, samples, collator, hidden, targets, router_targets, attention_targets,
              device, *, epoch, optimizer=None, scheduler=None):
    training = optimizer is not None
    student.train(training)
    batches = (build_length_bucket_batches(samples, cfg.batch_size, cfg.calib_seed + epoch)
               if training else build_validation_batches(samples, cfg.batch_size))
    total_num = total_den = router_sum = 0.0
    tokens = 0
    metrics_sum = {}
    expert_counts = torch.zeros(student.base.spec.num_experts, dtype=torch.long)
    started = time.monotonic()
    attention_num = attention_den = 0.0
    lora_energy = {"attention": 0.0, "moe": 0.0}
    residual_energy = {"attention": 0.0, "moe": 0.0}
    max_allocated = []
    update_records = []
    for batch_index, batch in enumerate(batches):
        packed = collator(batch)
        valid = (packed["attention_mask"].bool() & packed["loss_mask"].bool()).to(device)
        if not valid.any():
            raise ValueError("empty effective loss mask")
        x = assemble(hidden, batch, device)
        target = assemble(targets, batch, device)
        teacher_logits = assemble(router_targets, batch, device)
        attention_target = assemble(attention_targets, batch, device)
        call = build_qwen3_moe_layer_call(str(snapshot), x)
        if training:
            optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.set_grad_enabled(training):
            out = student(x, position_embeddings=call.position_embeddings, use_ste=training)
            pred_delta = out.output[valid].float() - x[valid].float()
            target_delta = target[valid].float() - x[valid].float()
            num = (pred_delta - target_delta).square().sum()
            den = target_delta.square().sum()
            if den <= 0:
                raise RuntimeError("zero reconstruction target energy")
            aux_logits = student.router_aux_logits(out.pre_moe_norm) if training else out.router_logits
            router = masked_router_loss(aux_logits, teacher_logits, valid, kind=cfg.router_loss,
                                        top_k=cfg.router_top_k, temperature=cfg.router_temperature)
            objective = num / den + cfg.router_loss_weight * router
            if not torch.isfinite(objective):
                raise RuntimeError("non-finite objective")
            if training:
                objective.backward()
                grads = [p.grad for p in student.parameters() if p.grad is not None]
                if not grads or not torch.stack([g.norm() for g in grads]).isfinite().all():
                    raise RuntimeError("missing or non-finite gradients")
                update_records.append({
                    "lora_gradient_norms": {n: float(p.grad.norm()) for n, p in student.learned.named_parameters()
                                             if "lora_" in n and p.requires_grad and p.grad is not None},
                    "group_gradient_norm": float(student.learned.matrices["q_proj"].grad.norm()),
                })
                optimizer.step()
                scheduler.step()
                del grads
        with torch.no_grad():
            attention_num += float((out.pre_moe_norm[valid].float() - attention_target[valid].float()).square().sum())
            attention_den += float((attention_target[valid].float() - x[valid].float()).square().sum())
            for branch, delta, residual in (("attention", out.attention_lora, x), ("moe", out.moe_lora, out.pre_moe_norm)):
                lora_energy[branch] += float(delta[valid].float().square().sum())
                residual_energy[branch] += float(residual[valid].float().square().sum())
        count = int(valid.sum())
        total_num += float(num.detach())
        total_den += float(den.detach())
        router_sum += float(router.detach()) * count
        tokens += count
        measures = router_metrics(out.router_logits, teacher_logits, valid, top_k=student.base.spec.top_k)
        for key, value in measures.items():
            metrics_sum[key] = metrics_sum.get(key, 0.0) + value * count
        selected = out.router_logits.detach()[valid].topk(student.base.spec.top_k, dim=-1).indices
        expert_counts += torch.bincount(selected.reshape(-1), minlength=expert_counts.numel()).cpu()
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        max_allocated.append(peak)
        print(json.dumps({"event": "batch", "layer": student.base.layer_idx, "epoch": epoch,
                          "train": training, "batch": batch_index + 1, "batches": len(batches),
                          "tokens": count, "objective": float(objective.detach()),
                          "peak_allocated_bytes": peak,
                          "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
        del out, objective, num, den, router, aux_logits, pred_delta, target_delta, x, target, attention_target, teacher_logits, delta, residual
    if tokens == 0:
        raise ValueError("empty epoch")
    reconstruction = total_num / total_den
    routing = router_sum / tokens
    return {"reconstruction_nmse": reconstruction, "router_loss": routing,
            "objective": reconstruction + cfg.router_loss_weight * routing, "tokens": tokens,
            **{k: v / tokens for k, v in metrics_sum.items()},
            "attention_residual_nmse": attention_num / attention_den,
            "moe_residual_nmse": reconstruction,
            "lora_output_energy": lora_energy,
            "lora_to_residual_energy": {k: lora_energy[k] / max(residual_energy[k], 1e-30) for k in lora_energy},
            "lora_parameter_norms": {k: float(p.detach().norm()) for k, p in student.learned.named_parameters() if "lora_" in k},
            "batch_peak_allocated_bytes": max_allocated,
            "updates": update_records,
            "expert_token_counts": expert_counts.tolist()}


@torch.no_grad()
def propagate(student, snapshot, samples, hidden, device, batch_size):
    result = {}
    for batch in build_validation_batches(samples, batch_size):
        x = assemble(hidden, batch, device)
        call = build_qwen3_moe_layer_call(str(snapshot), x)
        y = student(x, position_embeddings=call.position_embeddings, use_ste=False).output
        for i, sample in enumerate(batch):
            result[sample.sample_id] = y[i, :sample.input_ids.numel()].cpu().contiguous()
    return result


def train(cfg: Config, *, resume=False, smoke_layers=0):
    cfg.validate()
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA in the hif4 environment")
    torch.manual_seed(cfg.calib_seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    out = Path(cfg.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "train.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _train_locked(cfg, out, device, resume=resume, smoke_layers=smoke_layers)


def _train_locked(cfg, out, device, *, resume, smoke_layers=0):
    snapshot = Path(resolve_local_snapshot(cfg.model_path)).resolve()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        raise ValueError("source tokenizer must define pad_token_id")
    collator = DynamicCalibrationCollator(tokenizer.pad_token_id)
    train_samples, val_samples = build_or_load_calibration(cfg, tokenizer, None, cfg.calib_cache_dir)
    if smoke_layers:
        if smoke_layers not in (1, 2) or cfg.epochs != 2:
            raise ValueError("smoke requires 1-2 layers and 2 epochs")
        ordered = sorted(train_samples, key=lambda s: s.input_ids.numel())
        # Retain the full longest batch so short testing covers the formal
        # peak shape; use two epochs for repeated updates and validation return.
        train_samples = ordered[-4:]
        ordered = sorted(val_samples, key=lambda s: s.input_ids.numel())
        val_samples = ordered[:1] + ordered[-3:]
    layer_count = smoke_layers or 48
    samples = train_samples + val_samples
    if len({s.sample_id for s in samples}) != len(samples):
        raise ValueError("train and validation IDs must be disjoint")
    if resume:
        manifest = load_manifest(out)
        if manifest["config"] != cfg.to_dict():
            raise ValueError("resume configuration differs from the saved configuration")
        if manifest["source_snapshot"] != str(snapshot):
            raise ValueError("resume source snapshot changed")
        if manifest["smoke_layers"] != smoke_layers or manifest["source_fingerprint"] != source_fingerprint():
            raise ValueError("resume smoke settings or source fingerprint changed")
        initialization = load(out / "initialization.pt")
        progress = load(out / "resume.pt")
        # resume.pt is the commit point, so repair only derived manifest metadata.
        manifest["completed_layers"] = list(range(progress["layer"]))
        write_json(manifest, out / "manifest.json")
    else:
        if (out / "manifest.json").exists() or (out / "resume.pt").exists():
            raise FileExistsError("run already exists; use --resume")
        initialization = load_initialization(cfg.init_artifact, cfg.model_path)
        atomic_save(initialization, out / "initialization.pt")
        manifest = {"kind": KIND, "schema_version": SCHEMA_VERSION, "config": cfg.to_dict(),
                    "source_snapshot": str(snapshot), "init_sha256": sha256(cfg.init_artifact),
                    "completed_layers": [], "smoke_layers": smoke_layers,
                    "source_fingerprint": source_fingerprint(), "baseline": baseline_provenance(),
                    "sample_ids": {"train": [s.sample_id for s in train_samples], "validation": [s.sample_id for s in val_samples]}}
        cache = build_initial_moe_hidden_cache(snapshot, samples, collator, device, cfg.batch_size)
        hidden = {s.sample_id: cache.get(s.sample_id) for s in samples}
        del cache
        progress = {"layer": 0, "next_epoch": 0, "hidden": hidden}
        atomic_save(progress, out / "resume.pt")
        write_json(manifest, out / "manifest.json")
    for layer in range(progress["layer"], layer_count):
        print(json.dumps({"event": "layer_start", "layer": layer}), flush=True)
        hidden = progress["hidden"]
        native_state = load_qwen3_moe_layer_state(snapshot, layer, device)
        native = NativeQwen3MoELayerRuntime(native_state).to(device).eval()
        targets, router_targets, attention_targets = teacher_targets(native, snapshot, samples, collator, hidden, device, cfg.batch_size)
        base = fold_initial_diag(native_state, initialization[str(layer)])
        # Layer-specific seeds make each shared LoRA initialization identical
        # across ablations, independently of previous layers or resume.
        torch.manual_seed(cfg.calib_seed + layer)
        student = Student(base, cfg.matrix_sharing, lora_mode=cfg.lora_mode,
                          lora_rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha).to(device)
        del native, native_state
        optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=cfg.lr, weight_decay=0.0)
        batches = len(build_length_bucket_batches(train_samples, cfg.batch_size, cfg.calib_seed))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs * batches, eta_min=0.0)
        next_epoch = progress["next_epoch"]
        if "parameters" in progress:
            student.learned.load_state_dict(progress["parameters"], strict=True)
            optimizer.load_state_dict(progress["optimizer"])
            scheduler.load_state_dict(progress["scheduler"])
            best = progress["best"]
        else:
            baseline = run_epoch(student, cfg, snapshot, val_samples, collator, hidden, targets,
                                 router_targets, attention_targets, device, epoch=-1)
            write_json(baseline, out / f"layers/{layer:03d}/initial_metrics.json")
            best = None  # Epoch zero is a diagnostic, never an implicit rollback.
        for epoch in range(next_epoch, cfg.epochs):
            training = run_epoch(student, cfg, snapshot, train_samples, collator, hidden, targets,
                                 router_targets, attention_targets, device, epoch=epoch, optimizer=optimizer, scheduler=scheduler)
            validation = run_epoch(student, cfg, snapshot, val_samples, collator, hidden, targets,
                                   router_targets, attention_targets, device, epoch=epoch)
            if best is None or validation["objective"] < best["metrics"]["objective"]:
                best = {"epoch": epoch, "metrics": validation, "parameters": student.learned.snapshot()}
            write_json({"epoch": epoch, "train": training, "validation": validation},
                       out / f"layers/{layer:03d}/epochs/{epoch:03d}.json")
            progress = {"layer": layer, "next_epoch": epoch + 1, "hidden": hidden,
                        "parameters": student.learned.snapshot(), "optimizer": cpu_tree(optimizer.state_dict()),
                        "scheduler": scheduler.state_dict(), "best": best}
            atomic_save(progress, out / "resume.pt")
            print(json.dumps({"event": "epoch_complete", "layer": layer, "epoch": epoch,
                              "validation": validation["objective"]}), flush=True)
        if best is None:
            raise RuntimeError("no trained epoch selected")
        student.learned.load_state_dict(best["parameters"], strict=True)
        atomic_save(best, out / f"layers/{layer:03d}/selected.pt")
        write_json(best["metrics"], out / f"layers/{layer:03d}/metrics.json")
        # Compare selected group baseline on this run's SAME input, not the
        # older run's different progressive hidden states.
        selected_parameters = student.learned.snapshot()
        baseline_parameters = load(Path(manifest["baseline"]["path"]) / f"layers/{layer:03d}/selected.pt")["parameters"]
        comparison_parameters = dict(selected_parameters)
        comparison_parameters.update(baseline_parameters)
        for branch in ("attention", "moe"):
            comparison_parameters[f"{branch}_lora_B"] = torch.zeros_like(comparison_parameters[f"{branch}_lora_B"])
        student.learned.load_state_dict(comparison_parameters, strict=True)
        group_metrics = run_epoch(student, cfg, snapshot, val_samples, collator, hidden, targets,
                                  router_targets, attention_targets, device, epoch=-2)
        write_json({"group_on_same_input": group_metrics, "selected": best["metrics"],
                    "reconstruction_nmse_delta": best["metrics"]["reconstruction_nmse"] - group_metrics["reconstruction_nmse"]},
                   out / f"layers/{layer:03d}/baseline_comparison.json")
        student.learned.load_state_dict(selected_parameters, strict=True)
        del selected_parameters, baseline_parameters, comparison_parameters
        next_hidden = propagate(student, snapshot, samples, hidden, device, cfg.batch_size)
        progress = {"layer": layer + 1, "next_epoch": 0, "hidden": next_hidden}
        atomic_save(progress, out / "resume.pt")
        manifest["completed_layers"] = list(range(layer + 1))
        write_json(manifest, out / "manifest.json")
        del student, optimizer, scheduler, base, targets, router_targets, attention_targets, best, hidden
        gc.collect()
        torch.cuda.empty_cache()
    write_json({"complete": True, "layers": [json.loads((out / f"layers/{i:03d}/metrics.json").read_text())
                                              for i in range(layer_count)], "smoke_only": bool(smoke_layers)}, out / "summary.json")


if __name__ == "__main__":
    config, should_resume = train_args()
    train(config, resume=should_resume)
