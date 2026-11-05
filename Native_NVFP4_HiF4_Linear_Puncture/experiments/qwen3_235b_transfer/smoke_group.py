"""Exercise the real first 235B layer, longest full batches and Group export.

Consumes the actual one-layer DIAG smoke artifact. It never presents that
partial artifact as a complete 94-layer initialization or inference model.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.artifact import select_layer_diag
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import NativeQwen3MoELayerRuntime
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.batching import DynamicCalibrationCollator, build_length_bucket_batches
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.calibration import build_or_load_calibration
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.moe_materialize import _state_to_tensors
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.moe_trainer import build_initial_moe_hidden_cache
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.artifact import atomic_save, load, write_json
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.student import Student
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.train import run_epoch, teacher_targets, propagate
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.transforms import fold_initial_diag, projection_weights, transformed_state
from .config import diag_config, group_config, source_snapshot
from .execution import SavedTensorsOnGpus, memory_peaks, reset_memory_peaks


def smoke(run_root: Path) -> None:
    e4 = Path(diag_config(run_root, smoke=True).output_dir)
    if json.loads((e4 / "status.json").read_text())["status"] != "complete":
        raise ValueError("the real DIAG smoke must finish before the Group smoke")
    initial = load(e4 / "checkpoint/final_model/conversion_state.pt")
    if (initial["source_model"] != str(source_snapshot()) or initial["num_layers"] != 94
            or set(initial["layers"]) != {"0"}):
        raise ValueError("expected the actual 235B layer-0 DIAG smoke artifact")
    cfg = dataclasses.replace(group_config(run_root),
                              output_dir=str(run_root / "smoke/group_top_mass"),
                              calib_cache_dir=str(run_root / "smoke/data"),
                              epochs=2, calib_nsamples=4, calib_val_nsamples=4)
    cfg.validate()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    write_json(cfg.to_dict(), out / "config.json")
    storage = SavedTensorsOnGpus()
    record = {"status": "running", "smoke_only": True, "layer": 0,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "execution": storage.metadata()}
    write_json(record, out / "status.json")
    try:
        with storage:
            execute(cfg, initial, out, record)
        record["status"] = "complete"
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["execution"] = storage.metadata()
        record["gpu_memory_peaks"] = memory_peaks()
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(record, out / "status.json")


def execute(cfg, initial, out, record):
    device = torch.device("cuda:0")
    torch.manual_seed(cfg.calib_seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reset_memory_peaks()
    snapshot = source_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    collator = DynamicCalibrationCollator(tokenizer.pad_token_id)
    training, validation = build_or_load_calibration(cfg, tokenizer, None, cfg.calib_cache_dir)
    samples = training + validation
    cache = build_initial_moe_hidden_cache(snapshot, samples, collator, device, cfg.batch_size)
    hidden = {s.sample_id: cache.get(s.sample_id) for s in samples}
    del cache
    native_state = load_qwen3_moe_layer_state(snapshot, 0, device)
    native = NativeQwen3MoELayerRuntime(native_state).to(device).eval()
    targets, routers = teacher_targets(native, snapshot, samples, collator, hidden, device, cfg.batch_size)
    base = fold_initial_diag(native_state, select_layer_diag(initial["layers"]["0"], "adopted"))
    del native, native_state
    gc.collect()
    student = Student(base, cfg.matrix_sharing).to(device)
    frozen = dict(projection_weights(base))
    frozen.update(input_norm=base.input_layernorm_weight, moe_norm=base.post_attention_layernorm_weight,
                  q_norm=base.q_norm_weight, k_norm=base.k_norm_weight)
    if any(weight.requires_grad for weight in frozen.values()):
        raise ValueError("base checkpoint weights must stay frozen")
    versions = {name: weight._version for name, weight in frozen.items()}
    initial_parameters = student.learned.snapshot()
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=0.0)
    batches = len(build_length_bucket_batches(training, cfg.batch_size, cfg.calib_seed))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs * batches, eta_min=0.0)
    baseline = run_epoch(student, cfg, snapshot, validation, collator, hidden, targets, routers, device, epoch=-1)
    write_json(baseline, out / "initial_metrics.json")
    best = None
    for epoch in range(cfg.epochs):
        train_metrics = run_epoch(student, cfg, snapshot, training, collator, hidden, targets, routers,
                                  device, epoch=epoch, optimizer=optimizer, scheduler=scheduler)
        val_metrics = run_epoch(student, cfg, snapshot, validation, collator, hidden, targets, routers,
                                device, epoch=epoch)
        write_json({"train": train_metrics, "validation": val_metrics}, out / f"epochs/{epoch:03d}.json")
        if best is None or val_metrics["objective"] < best["metrics"]["objective"]:
            best = {"epoch": epoch, "metrics": val_metrics, "parameters": student.learned.snapshot()}
    if versions != {name: weight._version for name, weight in frozen.items()}:
        raise ValueError("a frozen base tensor was modified")
    changed = [name for name, tensor in student.learned.snapshot().items()
               if not torch.equal(tensor, initial_parameters[name])]
    if not changed or "moe_norm" not in changed or not any(name.startswith("matrices.") for name in changed):
        raise ValueError("expected both Group and norm parameters to update")
    atomic_save(best, out / "selected.pt")
    reloaded = load(out / "selected.pt")
    if any(not torch.equal(value, reloaded["parameters"][name]) for name, value in best["parameters"].items()):
        raise ValueError("saved selected parameters changed on reload")
    student.learned.load_state_dict(reloaded["parameters"], strict=True)
    restored = run_epoch(student, cfg, snapshot, validation, collator, hidden, targets, routers, device, epoch=cfg.epochs)
    if not math.isclose(restored["objective"], best["metrics"]["objective"], rel_tol=1e-5, abs_tol=1e-8):
        raise ValueError("selected-checkpoint reload changed the validation objective")
    next_hidden = propagate(student, snapshot, samples, hidden, device, cfg.batch_size)
    if any(next_hidden[s.sample_id].shape != (s.input_ids.numel(), 4096)
           or not torch.isfinite(next_hidden[s.sample_id]).all() for s in samples):
        raise ValueError("progressive propagation changed sample lengths or produced nonfinite output")
    atomic_save(next_hidden, out / "next_hidden.pt")
    # Exercise the same transform, QDQ, tensor names and storage used by full export.
    with torch.no_grad():
        final = transformed_state(base, student.learned)
        tensors = _state_to_tensors(final)
    shard = out / "layer_000.safetensors"
    save_file(tensors, str(shard))
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(tensors):
            raise ValueError("exported tensor coverage differs")
        for name, tensor in tensors.items():
            if not torch.equal(handle.get_tensor(name), tensor.cpu()):
                raise ValueError(f"exported tensor changed on reload: {name}")
    record.update(changed_parameter_count=len(changed), selected_epoch=best["epoch"],
                  baseline=baseline, selected=best["metrics"], restored=restored,
                  export_bytes=shard.stat().st_size,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True, type=Path)
    smoke(parser.parse_args().run_root.resolve())
