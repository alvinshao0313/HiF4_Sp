"""Full-trajectory joint training with epoch-zero selection and a fresh formal optimizer."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from ..e2e_diag_reconstruction.data.batching import (
    DynamicCalibrationCollator, build_length_bucket_batches, build_validation_batches,
)
from ..e2e_diag_reconstruction.training.moe_layer_runtime import build_qwen3_moe_layer_call
from .artifacts import atomic_save, load, read_json, selected_samples, state_digest, verify, write_json
from .cache import FixedCache
from .config import NATIVE, PROTOCOL
from .losses import output_loss
from .model import LastHead


def optimizer_for(model, lr):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                             betas=(.9, .999), eps=1e-8, weight_decay=0.)


def check_update(model, optimizer):
    groups = {"last_matrix": 0., "last_norm": 0., "final_norm": 0., "head_matrix": 0.}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if not p.requires_grad or p.dtype != torch.float32 or p.grad.dtype != torch.float32 or not torch.isfinite(p.grad).all():
            raise RuntimeError(f"invalid gradient: {name}")
        group = ("last_matrix" if ".matrices." in name else "last_norm" if ".norms." in name else name)
        groups[group] += float(p.grad.square().sum())
    if any(not value > 0 for value in groups.values()):
        raise RuntimeError(f"missing joint gradient path: {groups}")
    for state in optimizer.state.values():
        for key in ("exp_avg", "exp_avg_sq"):
            if state[key].dtype != torch.float32 or not torch.isfinite(state[key]).all():
                raise RuntimeError("AdamW moments must be finite FP32")
    return groups


def epoch(model, samples, cache, collator, *, number, optimizer=None, label="formal"):
    device = model.head_base.device
    training = optimizer is not None
    model.train(training)
    batches = (build_length_bucket_batches(samples, 4, 42 + number) if training
               else build_validation_batches(samples, 4))
    numerator, denominator = 0., 0
    started = time.monotonic()
    for index, batch in enumerate(batches):
        x, targets = cache.batch(batch, device)
        packed = collator(batch)
        valid = (packed["attention_mask"].bool() & packed["loss_mask"].bool()).to(device)
        call = build_qwen3_moe_layer_call(str(NATIVE), x)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            hidden = model(x, call.position_embeddings, use_ste=training)[valid]
            fused = model.fused_head()
            objective = output_loss(hidden, fused, targets)
            if not torch.isfinite(objective):
                raise RuntimeError("non-finite vocabulary KL")
            if training:
                objective.backward()
                optimizer.step()
                gradient_norms = check_update(model, optimizer)
        count = int(valid.sum())
        numerator += float(objective.detach()) * count
        denominator += count
        print(json.dumps({"event": "batch", "phase": label, "epoch": number, "train": training,
                          "batch": index + 1, "batches": len(batches), "tokens": count,
                          "KL": float(objective.detach()), "seconds": round(time.monotonic() - started, 2),
                          "peak_allocated": torch.cuda.max_memory_allocated(),
                          **({"gradient_squared_norms": gradient_norms} if training else {})}), flush=True)
        del x, hidden, fused, objective, targets, call, packed, valid
    return {"KL": numerator / denominator, "tokens": denominator, "seconds": time.monotonic() - started}


def trial(model, initial, cache, collator, train, val, directory, *, lr, epochs, label):
    directory.mkdir(parents=True, exist_ok=False)
    model.restore(initial)
    optimizer = optimizer_for(model, lr)
    if optimizer.state:
        raise RuntimeError("a trial must start with a new optimizer")
    baseline = epoch(model, val, cache, collator, number=0, label=label)
    best = {"epoch": 0, "metrics": baseline, "parameters": model.snapshot(), "lr": lr}
    atomic_save(best, directory / "selected.pt")
    write_json(baseline, directory / "epochs/000.json")
    for number in range(1, epochs + 1):
        t = epoch(model, train, cache, collator, number=number, optimizer=optimizer, label=label)
        v = epoch(model, val, cache, collator, number=number, label=label)
        if v["KL"] < best["metrics"]["KL"]:
            best = {"epoch": number, "metrics": v, "parameters": model.snapshot(), "lr": lr}
            atomic_save(best, directory / "selected.pt")
        write_json({"train": t, "validation": v, "selected_epoch": best["epoch"]}, directory / f"epochs/{number:03d}.json")
        print(json.dumps({"event": "epoch_complete", "phase": label, "epoch": number,
                          "validation_KL": v["KL"], "selected_epoch": best["epoch"]}), flush=True)
    if label == "longest_batch_qualification":
        atomic_save({"epoch": epochs, "parameters": model.snapshot(), "lr": lr}, directory / "last_updated.pt")
    model.restore(load(directory / "selected.pt")["parameters"])
    if state_digest(model.snapshot()) != state_digest(best["parameters"]):
        raise RuntimeError("selected checkpoint did not reload exactly")
    write_json({"complete": True, "epochs": epochs, "lr": lr, "selected_epoch": best["epoch"],
                "validation": best["metrics"]}, directory / "manifest.json")
    del optimizer
    return best


def train(root, *, qualification=False):
    root, m = verify(root)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    training, validation = selected_samples(m, qualification)
    directory = root / ("qualification/training" if qualification else "training")
    directory.mkdir(parents=True, exist_ok=False)
    cache = FixedCache(root, m)
    collator = DynamicCalibrationCollator(AutoTokenizer.from_pretrained(NATIVE, local_files_only=True).pad_token_id)
    model = LastHead(m, device)
    frozen = model.frozen_digest()
    initial = model.snapshot()
    torch.cuda.reset_peak_memory_stats()
    if qualification:
        from .validation import real_checks
        real_checks(model, cache, collator, training, validation, directory)
        model.restore(initial)
        chosen = trial(model, initial, cache, collator, training, validation, directory / "formal",
                       lr=3e-5, epochs=2, label="longest_batch_qualification")
    else:
        candidates = []
        for lr in PROTOCOL["learning_rates"]:
            chosen = trial(model, initial, cache, collator, training, validation, directory / f"lr_{lr:g}",
                           lr=lr, epochs=3, label=f"screen_{lr:g}")
            candidates.append((chosen["metrics"]["KL"], lr))
            del chosen
            gc.collect()
            torch.cuda.empty_cache()
        lr = min(candidates)[1]
        write_json({"learning_rate": lr, "candidates": candidates, "criterion": "validation_token_mean_KL"}, directory / "selection.json")
        chosen = trial(model, initial, cache, collator, training, validation, directory / "formal",
                       lr=lr, epochs=20, label="formal")
    atomic_save(chosen, directory / "selected.pt")
    if model.frozen_digest() != frozen:
        raise RuntimeError("a frozen last-layer or base head tensor changed")
    write_json({"complete": True, "qualification": qualification,
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "frozen_tensors_unchanged": True, "FP32_optimizer_verified": True,
                "selected_epoch": chosen["epoch"], "validation": chosen["metrics"],
                "peak_allocated": torch.cuda.max_memory_allocated(),
                "peak_reserved": torch.cuda.max_memory_reserved(),
                "sample_lengths": {"train": [s.input_ids.numel() for s in training],
                                   "val": [s.input_ids.numel() for s in validation]}}, directory / "manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--qualification", action="store_true")
    args = parser.parse_args()
    train(args.run, qualification=args.qualification)
