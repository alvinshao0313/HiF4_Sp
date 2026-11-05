"""Independent native-start training; no legacy E4 initialization is read."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from ..e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from ..e2e_diag_reconstruction.core.moe_semantic_hif4 import NativeQwen3MoELayerRuntime
from ..e2e_diag_reconstruction.data.batching import DynamicCalibrationCollator, build_length_bucket_batches
from ..e2e_diag_reconstruction.data.calibration import build_or_load_calibration
from ..e2e_diag_reconstruction.training.moe_trainer import build_initial_moe_hidden_cache
from ..non_equivalent_reconstruction.artifact import atomic_save, load, sha256, write_json
from ..non_equivalent_reconstruction.train import teacher_targets, run_epoch, propagate
from .config import Config, parameter_groups
from .student import Student


def source_hashes():
    package = Path(__file__).resolve().parent
    paths = [package / name for name in ("train.py", "config.py", "parameters.py", "student.py", "rotation.py", "variants.py")]
    paths += [package.parent / "non_equivalent_reconstruction" / name
              for name in ("train.py", "student.py", "losses.py", "transforms.py")]
    return {str(path): sha256(path) for path in sorted(paths)}


def train(cfg: Config):
    cfg.validate()
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA in hif4")
    torch.manual_seed(cfg.calib_seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    out = Path(cfg.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    snapshot = Path(cfg.model_path).resolve()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if tokenizer.pad_token_id is None:
        raise ValueError("native tokenizer must define a padding token")
    collator = DynamicCalibrationCollator(tokenizer.pad_token_id)
    # Read the existing split, with the original source/count/seed checks.
    for split in ("train", "val"):
        if not (Path(cfg.calib_cache_dir) / "calibration" / f"{split}.pt").is_file():
            raise FileNotFoundError("this experiment requires the existing full S1K calibration cache")
    train_samples, val_samples = build_or_load_calibration(cfg, tokenizer, None, cfg.calib_cache_dir)
    if cfg.phase == "smoke":
        # Preserve formal batch and worst sequence lengths; shorten only steps.
        train_samples = sorted(train_samples, key=lambda sample: sample.input_ids.numel(), reverse=True)[:4]
        val_samples = sorted(val_samples, key=lambda sample: sample.input_ids.numel(), reverse=True)[:4]
    samples = train_samples + val_samples
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("calibration train/validation sample IDs overlap")
    manifest = {
        "kind": "module_design_ablation", "schema_version": 1, "config": cfg.to_dict(),
        "source_snapshot": str(snapshot), "source_hashes": source_hashes(),
        "initialization": "native_norms; unit_DIAG; ordinary_R64; router_I64; no_E4",
        "calibration_hashes": {
            split: sha256(Path(cfg.calib_cache_dir) / "calibration" / f"{split}.pt")
            for split in ("train", "val")},
        "sample_ids": {"train": [s.sample_id for s in train_samples], "validation": [s.sample_id for s in val_samples]},
        "completed_layers": [], "complete": False,
    }
    write_json(manifest, out / "manifest.json")
    start = time.monotonic()
    cache = build_initial_moe_hidden_cache(snapshot, samples, collator, device, cfg.batch_size)
    hidden = {s.sample_id: cache.get(s.sample_id) for s in samples}
    del cache
    for layer in range(cfg.layers):
        layer_start = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        print(json.dumps({"event": "layer_start", "layer": layer, "variant": cfg.variant}), flush=True)
        base = load_qwen3_moe_layer_state(snapshot, layer, device)
        teacher = NativeQwen3MoELayerRuntime(base).to(device).eval()
        targets, router_targets = teacher_targets(
            teacher, snapshot, samples, collator, hidden, device, cfg.batch_size,
        )
        del teacher
        student = Student(base, cfg.variant).to(device)
        groups = parameter_groups(student, cfg.lr_multiplier)
        optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
        if student.learned.diag is not None:
            optimizer.register_step_post_hook(lambda *_: student.learned.diag.clamp_log2_((-4.0, 4.0)))
        batches = len(build_length_bucket_batches(train_samples, cfg.batch_size, cfg.calib_seed))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs * batches, eta_min=0.0)
        initial = run_epoch(student, cfg, snapshot, val_samples, collator, hidden, targets,
                            router_targets, device, epoch=-1)
        write_json({**initial, "parameters": sum(p.numel() for p in student.parameters()),
                    "learning_rates": {g["name"]: g["lr"] for g in optimizer.param_groups}},
                   out / f"layers/{layer:03d}/initial_metrics.json")
        best = None
        for epoch in range(cfg.epochs):
            training = run_epoch(student, cfg, snapshot, train_samples, collator, hidden, targets,
                                 router_targets, device, epoch=epoch, optimizer=optimizer, scheduler=scheduler)
            validation = run_epoch(student, cfg, snapshot, val_samples, collator, hidden, targets,
                                   router_targets, device, epoch=epoch)
            if best is None or validation["objective"] < best["metrics"]["objective"]:
                best = {"epoch": epoch, "metrics": validation, "parameters": student.learned.snapshot()}
            write_json({"epoch": epoch, "train": training, "validation": validation,
                        "elapsed_seconds": time.monotonic() - layer_start},
                       out / f"layers/{layer:03d}/epochs/{epoch:03d}.json")
        if best is None:
            raise RuntimeError("no trained epoch selected")
        selected_path = out / f"layers/{layer:03d}/selected.pt"
        atomic_save(best, selected_path)
        selected = load(selected_path)
        student.learned.load_state_dict(selected["parameters"], strict=True)
        for name, value in student.learned.state_dict().items():
            if not torch.equal(value.cpu(), best["parameters"][name]):
                raise RuntimeError(f"selected parameter changed on reload: {name}")
        constraint = {}
        if student.learned.rotations:
            with torch.no_grad():
                errors = []
                for rotation in student.learned.rotations.values():
                    q = rotation()
                    errors.append((q @ q.transpose(-1, -2) - rotation.identity).abs().max())
                error = float(torch.stack(errors).max())
                if not 0 <= error <= 1e-5:
                    raise RuntimeError(f"selected FP32 rotation violates the orthogonality tolerance: {error}")
                constraint["orthogonality_max_abs"] = error
                del q, errors
        write_json({**best["metrics"], "selected_epoch": best["epoch"],
                    **constraint, "seconds": time.monotonic() - layer_start,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}, out / f"layers/{layer:03d}/metrics.json")
        hidden = propagate(student, snapshot, samples, hidden, device, cfg.batch_size)
        manifest["completed_layers"].append(layer)
        write_json(manifest, out / "manifest.json")
        del student, optimizer, scheduler, groups, base, targets, router_targets, best, selected
        gc.collect()
        torch.cuda.empty_cache()
    if manifest["source_hashes"] != source_hashes():
        raise RuntimeError("training source changed during this run")
    manifest["complete"] = True
    manifest["seconds"] = time.monotonic() - start
    write_json(manifest, out / "manifest.json")
    write_json({"complete": True, "phase": cfg.phase, "seconds": manifest["seconds"],
                "layers": [json.loads((out / f"layers/{i:03d}/metrics.json").read_text()) for i in range(cfg.layers)]},
               out / "summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--phase", required=True, choices=("smoke", "screen", "formal"))
    parser.add_argument("--lr_multiplier", type=float, default=1.0)
    train(Config(**vars(parser.parse_args())))
