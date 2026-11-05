"""Native-start matrix_norm with fixed validation and explicit short budgets."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from ..e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from ..e2e_diag_reconstruction.core.moe_semantic_hif4 import NativeQwen3MoELayerRuntime
from ..e2e_diag_reconstruction.data.batching import DynamicCalibrationCollator, build_length_bucket_batches
from ..e2e_diag_reconstruction.training.moe_trainer import build_initial_moe_hidden_cache
from ..module_design_ablation.student import Student
from ..non_equivalent_reconstruction.train import propagate, run_epoch, teacher_targets
from .artifacts import atomic_save, load, read_json, sha256, state_digest, validate_root, verify_sources, write_json
from .config import BATCH_SIZE, CANDIDATES, LAYERS, LR, NATIVE, SEED
from .prepare import samples_for


@contextmanager
def _timed(timings, name, device=None):
    if device is not None:
        torch.cuda.synchronize(device)
    started = time.monotonic()
    try:
        yield
    finally:
        if device is not None:
            torch.cuda.synchronize(device)
        timings[name] = time.monotonic() - started


def training_spec(candidate, qualification):
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown candidate: {candidate!r}")
    definition = CANDIDATES[candidate]
    epochs = 2 if qualification else definition["epochs"]
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("training epochs must be a positive integer")
    return definition["selection"], epochs, 2 if qualification else LAYERS


def selected_samples(training, validation, *, qualification):
    if not qualification:
        return training, validation
    if min(len(training), len(validation)) < BATCH_SIZE:
        raise ValueError("qualification requires one full training and validation batch")
    # Keep the original samples and complete token/mask tensors at formal batch size.
    return tuple(sorted(values, key=lambda s: s.input_ids.numel(), reverse=True)[:BATCH_SIZE]
                 for values in (training, validation))


def epoch_config():
    return SimpleNamespace(batch_size=BATCH_SIZE, calib_seed=SEED, router_loss="top_mass",
                           router_top_k=8, router_temperature=1., router_loss_weight=0.)


def train(root, candidate, *, qualification=False):
    wall_started = time.monotonic()
    root = validate_root(root)
    verify_sources(root)
    selection, epochs, layers = training_spec(candidate, qualification)
    training, validation, hashes = samples_for(root, selection)
    training, validation = selected_samples(training, validation, qualification=qualification)
    if not torch.cuda.is_available():
        raise RuntimeError("matrix_norm training requires CUDA in the hif4 environment")
    output = root / ("qualification" if qualification else "training") / candidate
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(NATIVE, local_files_only=True)
    collator = DynamicCalibrationCollator(tokenizer.pad_token_id)
    cfg = epoch_config()
    samples = training + validation
    batches = len(build_length_bucket_batches(training, BATCH_SIZE, SEED))
    manifest = {
        "kind": "matrix_norm_efficiency_training", "complete": False,
        "candidate": candidate, "selection": selection, "qualification": qualification,
        "initialization": str(NATIVE), "variant": "matrix_norm", "epochs": epochs, "layers": layers,
        "batch_size": BATCH_SIZE, "seed": SEED, "learning_rate": LR,
        "optimizer": {"name": "AdamW", "weight_decay": 0.},
        "scheduler": {"name": "CosineAnnealingLR", "T_max": epochs * batches, "eta_min": 0.},
        "objective": "block_delta_nmse_on_student_prefix", "router_loss_weight": 0.,
        "calibration_sha256": hashes, "source_hashes": read_json(root / "source_hashes.json"),
        "sample_ids": {"train": [s.sample_id for s in training], "val": [s.sample_id for s in validation]},
        "completed_layers": [], "timings": {},
    }
    write_json(manifest, output / "manifest.json")
    preparation_seconds = time.monotonic() - wall_started
    timings = manifest["timings"]
    layer_metrics = []
    # Same main boundary as module_design_ablation: start immediately before embedding.
    optimization_started = time.monotonic()
    with _timed(timings, "initial_hidden_cache_seconds", device):
        cache = build_initial_moe_hidden_cache(NATIVE, samples, collator, device, BATCH_SIZE)
        hidden = {s.sample_id: cache.get(s.sample_id) for s in samples}
        del cache
    manifest["runtime"] = {
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_device_count": torch.cuda.device_count(), "training_gpu_count": 1,
        "training_device": str(device), "device_name": torch.cuda.get_device_name(device),
    }
    for layer in range(layers):
        layer_started = time.monotonic()
        layer_times = {}
        torch.cuda.reset_peak_memory_stats(device)
        print(f"training {candidate} layer {layer + 1}/{layers}", flush=True)
        with _timed(layer_times, "load_native_seconds", device):
            base = load_qwen3_moe_layer_state(NATIVE, layer, device)
        with _timed(layer_times, "teacher_targets_seconds", device):
            teacher = NativeQwen3MoELayerRuntime(base).to(device).eval()
            targets, logits = teacher_targets(teacher, NATIVE, samples, collator, hidden, device, BATCH_SIZE)
            del teacher
        with _timed(layer_times, "initialize_student_seconds", device):
            student = Student(base, "matrix_norm").to(device)
            names = [name for name, p in student.named_parameters() if p.requires_grad]
            if (not names or any(not n.startswith(("learned.matrices.", "learned.norms.")) or "router" in n
                                 for n in names)):
                raise ValueError(f"wrong matrix_norm trainable parameters: {names}")
            frozen_before = state_digest({"router": base.router_weight, **dict(student.learned.named_buffers())})
            initial_parameters = student.learned.snapshot()
            initial_state = state_digest(initial_parameters)
            optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * batches, eta_min=0.)
        with _timed(layer_times, "initial_validation_seconds", device):
            initial = run_epoch(student, cfg, NATIVE, validation, collator, hidden, targets, logits,
                                device, epoch=-1)
        write_json(initial, output / f"layers/{layer:03d}/initial_metrics.json")
        best = None
        epoch_times = []
        for epoch in range(epochs):
            measured = {}
            with _timed(measured, "training_seconds", device):
                t = run_epoch(student, cfg, NATIVE, training, collator, hidden, targets, logits, device,
                              epoch=epoch, optimizer=optimizer, scheduler=scheduler)
            with _timed(measured, "validation_seconds", device):
                v = run_epoch(student, cfg, NATIVE, validation, collator, hidden, targets, logits,
                              device, epoch=epoch)
            with _timed(measured, "selection_seconds", device):
                if best is None or v["reconstruction_nmse"] < best["metrics"]["reconstruction_nmse"]:
                    best = {"epoch": epoch + 1, "metrics": v, "parameters": student.learned.snapshot()}
            epoch_times.append(measured)
            write_json({"epoch": epoch + 1, "train": t, "validation": v, "timings": measured},
                       output / f"layers/{layer:03d}/epochs/{epoch + 1:03d}.json")
        if best is None:
            raise RuntimeError("no trained epoch selected")
        with _timed(layer_times, "save_reload_seconds", device):
            selected = output / f"layers/{layer:03d}/selected.pt"
            atomic_save(best, selected)
            student.learned.load_state_dict(load(selected)["parameters"], strict=True)
            selected_state = student.learned.snapshot()
            if state_digest(selected_state) != state_digest(best["parameters"]):
                raise RuntimeError("selected state differs after reload")
            if initial_state == state_digest(selected_state):
                raise RuntimeError("matrix_norm parameters did not update")
            frozen_after = state_digest({"router": base.router_weight, **dict(student.learned.named_buffers())})
            if frozen_before != frozen_after:
                raise RuntimeError("matrix_norm frozen state changed")
            family_changed = {family: any(not torch.equal(initial_parameters[name], value)
                                         for name, value in selected_state.items() if name.startswith(family + "."))
                              for family in ("matrices", "norms")}
        with _timed(layer_times, "propagate_seconds", device):
            hidden = propagate(student, NATIVE, samples, hidden, device, BATCH_SIZE)
            if any(not torch.isfinite(value).all() for value in hidden.values()):
                raise RuntimeError("invalid propagated layer output")
        metric = {**best["metrics"], "layer": layer, "selected_epoch": best["epoch"],
                  "selected_sha256": sha256(selected), "parameter_family_changed": family_changed,
                  "frozen_state_unchanged": True, "selected_reload_exact": True,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                  "timings": layer_times, "epoch_timings": epoch_times}
        with _timed(layer_times, "cleanup_seconds", device):
            del student, base, targets, logits, optimizer, scheduler, best, initial_parameters, selected_state
            gc.collect()
            torch.cuda.empty_cache()
        metric["seconds"] = time.monotonic() - layer_started
        layer_metrics.append(metric)
        write_json(metric, output / f"layers/{layer:03d}/metrics.json")
        manifest["completed_layers"].append(layer)
        manifest["optimization_seconds"] = time.monotonic() - optimization_started
        write_json(manifest, output / "manifest.json")
    optimization_seconds = time.monotonic() - optimization_started
    with _timed(timings, "final_source_verification_seconds"):
        verify_sources(root)
    # The original trainer checks source hashes before recording its final duration.
    legacy_comparable_seconds = time.monotonic() - optimization_started
    with _timed(timings, "final_input_verification_seconds"):
        _, _, final_hashes = samples_for(root, selection)
        if hashes != final_hashes:
            raise RuntimeError("calibration inputs changed during training")
    manifest.update({"complete": True, "preparation_seconds": preparation_seconds,
                     "optimization_seconds": optimization_seconds,
                     "legacy_comparable_seconds": legacy_comparable_seconds,
                     "seconds": legacy_comparable_seconds, "wall_seconds": time.monotonic() - wall_started})
    write_json(manifest, output / "manifest.json")
    write_json({**manifest, "layer_metrics": layer_metrics}, output / "summary.json")
    return output


def training_dir(root, candidate, *, qualification=False):
    directory = Path(root) / ("qualification" if qualification else "training") / candidate
    manifest = read_json(directory / "manifest.json")
    _, epochs, layers = training_spec(candidate, qualification)
    if (not manifest["complete"] or manifest.get("qualification", False) != qualification
            or manifest.get("candidate") != candidate or manifest.get("epochs") != epochs
            or manifest.get("completed_layers") != list(range(layers))):
        raise ValueError("training checkpoint is incomplete or differs from the requested candidate")
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--qualification", action="store_true")
    args = parser.parse_args()
    train(args.run, args.candidate, qualification=args.qualification)
