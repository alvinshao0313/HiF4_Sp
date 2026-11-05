"""Isolated real-layer E4 data-parallel feasibility and timing measurements.

This probe never resumes or changes a formal run. Its prepared teacher outputs,
global padding, optimizer and scheduler are shared by all comparison modes.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import NativeQwen3MoELayerRuntime, StudentQwen3MoELayerRuntime, StudentStepCache, build_moe_diag_state
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.batching import DynamicCalibrationCollator, build_length_bucket_batches
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.layer_runtime import ProgressiveHiddenCache
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.losses import finalize_reconstruction_loss, masked_reconstruction_components
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.moe_layer_runtime import build_qwen3_moe_layer_call
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.moe_trainer import _assemble_targets, _teacher_targets, _train_layer, build_initial_moe_hidden_cache
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.artifact import atomic_save, load, write_json
from .config import diag_config, source_snapshot
from .execution import SavedTensorsOnGpus


def initialize(device: torch.device) -> None:
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def prepare(run_root: Path, output: Path, selection: str) -> None:
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    initialize(device)
    cfg = diag_config(run_root)
    snapshot = source_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    collator = DynamicCalibrationCollator(tokenizer.pad_token_id)
    samples = load(run_root / "data/calibration/train.pt")
    batches = build_length_bucket_batches(samples, cfg.diag_batch_size, cfg.calib_seed)
    choose = max if selection == "longest" else min
    selected = choose(batches, key=lambda batch: max(s.input_ids.numel() for s in batch))
    if len(selected) != 4:
        raise ValueError("probe requires one full global batch of four real samples")
    cache = build_initial_moe_hidden_cache(snapshot, selected, collator, device, 4)
    state = load_qwen3_moe_layer_state(snapshot, 0, device)
    native = NativeQwen3MoELayerRuntime(state).to(device).eval()
    targets = _teacher_targets(native, snapshot, selected, collator, cache, device, 4)
    bundle = {"samples": selected, "hidden": {s.sample_id: cache.get(s.sample_id) for s in selected},
              "targets": targets, "pad_token_id": tokenizer.pad_token_id}
    atomic_save(bundle, output / "batch.pt")
    digest = hashlib.sha256()
    with (output / "batch.pt").open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    write_json({"selection": selection, "source": str(snapshot), "layer": 0,
                "samples": [{"id": s.sample_id, "length": s.input_ids.numel()} for s in selected],
                "global_batch_size": 4, "batch_sha256": digest.hexdigest(),
                "scheduler_total_steps": cfg.diag_epochs * len(batches),
                "prepared_at": datetime.now(timezone.utc).isoformat()}, output / "batch.json")


def packed_step(bundle: dict, cfg, epoch: int, device: torch.device):
    samples = bundle["samples"]
    batches = build_length_bucket_batches(samples, 4, cfg.calib_seed + epoch)
    if len(batches) != 1:
        raise ValueError("probe selection must form exactly one batch")
    batch = batches[0]
    packed = DynamicCalibrationCollator(bundle["pad_token_id"])(batch)
    cache = ProgressiveHiddenCache()
    for sid, value in bundle["hidden"].items():
        cache.store(sid, value, value.shape[0])
    ids = [s.sample_id for s in batch]
    hidden, _ = cache.assemble(ids, device)
    target = _assemble_targets(bundle["targets"], ids, device)
    return batch, packed, cache, hidden, target


def cpu_evidence(diag, optimizer, scheduler) -> dict:
    return {"gradients": {name: None if p.grad is None else p.grad.detach().cpu().clone()
                          for name, p in diag.named_parameters()},
            "parameters": diag.snapshot(),
            "optimizer_steps": {name: float(optimizer.state[p]["step"])
                                for name, p in diag.named_parameters() if p in optimizer.state},
            "scheduler_step": scheduler.last_epoch}


def execute(run_root: Path, prepared: Path, output: Path, mode: str, steps: int) -> None:
    rank = int(os.environ.get("RANK", "0"))
    distributed = mode == "parallel"
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
        initialize(device)
        dist.init_process_group("nccl", device_id=device)
        if dist.get_world_size() != 4:
            raise ValueError("the parallel probe uses exactly four ranks")
    else:
        device = torch.device("cuda:0")
        initialize(device)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    if distributed:
        dist.barrier()
    record = {"status": "running", "mode": mode, "rank": rank,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "steps": []}
    status_path = output / f"rank_{rank}.json"
    write_json(record, status_path)
    try:
        cfg = diag_config(run_root)
        snapshot = source_snapshot()
        bundle = load(prepared / "batch.pt")
        state = load_qwen3_moe_layer_state(snapshot, 0, device)
        diag = build_moe_diag_state(state.spec, cfg.diag_mode).to(device)
        diag.configure_fusable_components(cfg.fusable_diag_components)
        runtime = StudentQwen3MoELayerRuntime(state, diag, use_r64=cfg.use_r64,
                                             rot_order=cfg.rot_order).to(device).train()
        if distributed:
            from .parallel_execution import broadcast_parameters, e4_parallel_step
            broadcast_parameters(diag)
        optimizer = torch.optim.AdamW([p for p in diag.parameters() if p.requires_grad],
                                      lr=cfg.diag_lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=640, eta_min=0.0)
        initial = diag.snapshot()
        atomic_save(initial, output / f"initial_{rank}.pt")
        storage = SavedTensorsOnGpus() if mode == "reference" else nullcontext()
        measured_devices = list(range(3)) if mode == "reference" else [device.index]
        for step in range(steps):
            gc.collect()
            data_started = time.monotonic()
            batch, packed, cache, hidden, target = packed_step(bundle, cfg, step, torch.device("cpu"))
            if distributed:
                hidden = hidden[rank:rank + 1].to(device)
                target = target[rank:rank + 1].to(device)
                loss_mask = packed["loss_mask"][rank:rank + 1].to(device)
                attention_mask = packed["attention_mask"][rank:rank + 1].to(device)
                dist.barrier()
            elif mode in ("reference", "resident"):
                # Observe the pre-update loss outside the timed canonical step.
                # No dropout is active; discard all diagnostic GPU tensors so
                # its peak and duplicate hidden copies do not pollute timing.
                with torch.no_grad():
                    h = hidden.to(device)
                    t = target.to(device)
                    call = build_qwen3_moe_layer_call(str(snapshot), h)
                    result = runtime(h, attention_mask=None, position_embeddings=call.position_embeddings,
                                     step_cache=StudentStepCache.new(), use_ste=True)
                    num, den = masked_reconstruction_components(
                        result.output.float() - h.float(), t.float() - h.float(),
                        packed["loss_mask"].to(device), packed["attention_mask"].to(device), cfg.recon_loss)
                    captured = {"loss": float(finalize_reconstruction_loss(num, den, cfg.recon_loss)),
                                "numerator": float(num), "denominator": float(den)}
                    del h, t, call, result, num, den
            else:
                hidden, target = hidden.to(device), target.to(device)
            for measured in measured_devices:
                torch.cuda.synchronize(measured)
                torch.cuda.reset_peak_memory_stats(measured)
            started = time.monotonic()
            # The canonical trainer includes its own collate/assembly/H2D.
            # Include the candidate's corresponding work, without counting
            # the reference-only diagnostic forward in its measured time.
            input_seconds = started - data_started if distributed else 0.0
            with storage:
                if distributed:
                    metrics = e4_parallel_step(runtime, hidden, target, loss_mask, attention_mask,
                                               snapshot, cfg, optimizer, scheduler)
                elif mode in ("reference", "resident"):
                    count = _train_layer(runtime, snapshot, bundle["samples"],
                                         DynamicCalibrationCollator(bundle["pad_token_id"]), cache,
                                         bundle["targets"], cfg, device, step, optimizer, scheduler)
                    if count != 1:
                        raise ValueError("expected one canonical optimizer update")
                    metrics = captured
                elif mode == "partition_reference":
                    # Serial microbatches isolate batch-shape rounding from distributed SUM.
                    optimizer.zero_grad(set_to_none=True)
                    loss_mask = packed["loss_mask"].to(device)
                    attention_mask = packed["attention_mask"].to(device)
                    _, denominator = masked_reconstruction_components(
                        target.float() - hidden.float(), target.float() - hidden.float(),
                        loss_mask, attention_mask, cfg.recon_loss)
                    numerator = torch.zeros((), device=device)
                    for i in range(4):
                        h = hidden[i:i + 1]
                        call = build_qwen3_moe_layer_call(str(snapshot), h)
                        result = runtime(h, attention_mask=None, position_embeddings=call.position_embeddings,
                                         step_cache=StudentStepCache.new(), use_ste=True)
                        num, _ = masked_reconstruction_components(
                            result.output.float() - h.float(), target[i:i + 1].float() - h.float(),
                            loss_mask[i:i + 1], attention_mask[i:i + 1], cfg.recon_loss)
                        numerator += num.detach()
                        finalize_reconstruction_loss(num, denominator, cfg.recon_loss).backward()
                        del result, num, call
                    optimizer.step()
                    scheduler.step()
                    diag.clamp_log2_(cfg.diag_log2_clamp)
                    metrics = {"loss": float(finalize_reconstruction_loss(numerator, denominator, cfg.recon_loss)),
                               "numerator": float(numerator), "denominator": float(denominator)}
                else:
                    raise ValueError(mode)
            for measured in measured_devices:
                torch.cuda.synchronize(measured)
            elapsed = time.monotonic() - started + input_seconds
            evidence = cpu_evidence(diag, optimizer, scheduler)
            if any(not torch.isfinite(v).all() for v in evidence["gradients"].values() if v is not None):
                raise ValueError("nonfinite parameter gradient")
            atomic_save(evidence, output / f"step_{step}_rank_{rank}.pt")
            row = {"step": step, "elapsed_seconds": elapsed, "metrics": metrics,
                   "included_input_seconds": input_seconds,
                   "sample_ids": packed["sample_ids"], "global_padded_length": int(packed["input_ids"].shape[1]),
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                   "gpu_peaks": {str(i): {"allocated": torch.cuda.max_memory_allocated(i),
                                          "reserved": torch.cuda.max_memory_reserved(i)}
                                 for i in measured_devices}}
            record["steps"].append(row)
            print(json.dumps({"rank": rank, **row}), flush=True)
            write_json(record, status_path)
            del hidden, target, cache, evidence
        record["status"] = "complete"
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(record, status_path)
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--mode", required=True, choices=("prepare", "reference", "resident", "partition_reference", "parallel"))
    parser.add_argument("--selection", choices=("shortest", "longest"), default="longest")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.run_root.resolve(), args.output.resolve(), args.selection)
    else:
        if args.prepared is None or args.steps < 1:
            parser.error("execution needs --prepared and positive --steps")
        execute(args.run_root.resolve(), args.prepared.resolve(), args.output.resolve(), args.mode, args.steps)
