"""Equal-budget loss comparison; development selection runs after calibration training."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from ..e2e_diag_reconstruction.data.batching import DynamicCalibrationCollator, build_length_bucket_batches, build_validation_batches
from ..e2e_diag_reconstruction.training.moe_layer_runtime import build_qwen3_moe_layer_call
from ..reasoning_trace_last_head.model import LastHead
from ..reasoning_trace_last_head.training import check_update, optimizer_for
from .artifacts import atomic_save, load, read_json, selected_samples, state_digest, verify, write_json
from .cache import FixedCache
from .config import CANDIDATES, NATIVE, PROTOCOL
from .losses import ce_positions, output_loss


def batch_loss(model, samples, cache, collator, objective, *, training):
    device = model.head_base.device
    values, packed = cache.batch(samples, device), collator(samples)
    x = values["student"]
    call = build_qwen3_moe_layer_call(str(NATIVE), x)
    out = model(x, call.position_embeddings, use_ste=training)
    fused = model.fused_head()  # Exactly once for this complete batch.
    if objective == "full_kl":
        valid = (packed["attention_mask"].bool() & packed["loss_mask"].bool()).to(device)
        hidden = out[valid]
        loss = output_loss(hidden, fused, objective=objective,
                           teacher_hidden=values["teacher_hidden"][valid], teacher_head=model.head_base)
    else:
        valid, labels = ce_positions(packed)
        hidden = out[:, :-1][valid.to(device)]
        loss = output_loss(hidden, fused, objective=objective, labels=labels.to(device))
    return loss, len(hidden)


def epoch(model, samples, cache, collator, objective, *, number, label, optimizer=None):
    training = optimizer is not None
    model.train(training)
    batches = (build_length_bucket_batches(samples, 4, 42 + number)
               if training and label != "qualification" else build_validation_batches(samples, 4))
    numerator, denominator = 0., 0
    started = time.monotonic()
    for index, batch in enumerate(batches):
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss, count = batch_loss(model, batch, cache, collator, objective, training=training)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite distillation loss")
            if training:
                loss.backward()
                optimizer.step()
                gradients = check_update(model, optimizer)
        numerator += float(loss.detach()) * count
        denominator += count
        print(json.dumps({"event": "batch", "objective": objective, "phase": label, "epoch": number,
                          "train": training, "batch": index + 1, "batches": len(batches), "tokens": count,
                          "loss": float(loss.detach()), "seconds": round(time.monotonic() - started, 2),
                          "peak_allocated": torch.cuda.max_memory_allocated(),
                          **({"gradient_squared_norms": gradients} if training else {})}), flush=True)
        del loss
    return {"loss": numerator / denominator, "tokens": denominator, "seconds": time.monotonic() - started}


def trial(model, initial, cache, collator, objective, train, val, directory, lr, epochs, label):
    directory.mkdir(parents=True, exist_ok=False)
    model.restore(initial)
    if state_digest(model.snapshot()) != state_digest(initial):
        raise RuntimeError("trial did not reset all trainable parameters to original B")
    optimizer = optimizer_for(model, lr)
    if optimizer.state:
        raise RuntimeError("trial inherited optimizer state")
    baseline = epoch(model, val, cache, collator, objective, number=0, label=label)
    best = {"epoch": 0, "metrics": baseline, "parameters": model.snapshot(), "lr": lr}
    atomic_save(best, directory / "selected.pt")
    write_json(baseline, directory / "epochs/000.json")
    for number in range(1, epochs + 1):
        t = epoch(model, train, cache, collator, objective, number=number, label=label, optimizer=optimizer)
        v = epoch(model, val, cache, collator, objective, number=number, label=label)
        if v["loss"] < best["metrics"]["loss"]:
            best = {"epoch": number, "metrics": v, "parameters": model.snapshot(), "lr": lr}
            atomic_save(best, directory / "selected.pt")
        write_json({"train": t, "validation": v, "selected_epoch": best["epoch"]}, directory / f"epochs/{number:03d}.json")
        print(json.dumps({"event": "epoch_complete", "objective": objective, "phase": label,
                          "epoch": number, "validation_loss": v["loss"], "selected_epoch": best["epoch"]}), flush=True)
    if label == "qualification":
        atomic_save({"epoch": epochs, "parameters": model.snapshot(), "lr": lr}, directory / "last_updated.pt")
    model.restore(load(directory / "selected.pt")["parameters"])
    if state_digest(model.snapshot()) != state_digest(best["parameters"]):
        raise RuntimeError("selected joint checkpoint changed on save/load")
    write_json({"complete": True, "epochs": epochs, "lr": lr, "selected_epoch": best["epoch"],
                "validation": best["metrics"]}, directory / "manifest.json")
    return {"checkpoint": str(directory / "selected.pt"), "epoch": best["epoch"], "lr": lr,
            "validation_loss": best["metrics"]["loss"], "parameters_sha256": state_digest(best["parameters"])}


def train(root, objective, *, qualification=False):
    root, m = verify(root)
    if not qualification and m["selection_policy"] not in ("development", "validation"):
        raise ValueError("checkpoint selection preference must be fixed before formal training")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    train_samples, val = selected_samples(m, qualification)
    directory = root / ("qualification/training" if qualification else "training") / objective
    directory.mkdir(parents=True, exist_ok=False)
    model = LastHead(m, device)
    cache = FixedCache(root, m, objective)
    collator = DynamicCalibrationCollator(AutoTokenizer.from_pretrained(NATIVE, local_files_only=True).pad_token_id)
    frozen, initial = model.frozen_digest(), model.snapshot()
    atomic_save({"epoch": 0, "parameters": initial, "lr": 0.}, directory / "epoch_zero.pt")
    model.restore(load(directory / "epoch_zero.pt")["parameters"])
    if state_digest(model.snapshot()) != state_digest(initial):
        raise RuntimeError("epoch-zero exact reload failed")
    torch.cuda.reset_peak_memory_stats()
    if qualification:
        from .validation import real_checks
        real_checks(model, cache, collator, objective, train_samples, directory)
        result = trial(model, initial, cache, collator, objective, train_samples, val,
                       directory / "formal", 3e-5, 2, "qualification")
        pool = [result]
    else:
        pool = []
        for lr in PROTOCOL["learning_rates"]:
            pool.append(trial(model, initial, cache, collator, objective, train_samples, val,
                              directory / f"lr_{lr:g}", lr, 3, f"screen_{lr:g}"))
            gc.collect()
            torch.cuda.empty_cache()
        selected = min(pool, key=lambda r: (r["validation_loss"], r["lr"]))
        write_json(selected, directory / "learning_rate.json")
        result = trial(model, initial, cache, collator, objective, train_samples, val,
                       directory / "formal", selected["lr"], 20, "formal")
        pool.append(result)
    atomic_save(load(result["checkpoint"]), directory / "calibration_selected.pt")
    if model.frozen_digest() != frozen:
        raise RuntimeError("frozen model tensors changed")
    write_json({"complete": True, "qualification": qualification, "objective": objective,
                "candidate": CANDIDATES[objective], "pool": pool, "calibration_selected": result,
                "initial_parameters_sha256": state_digest(initial), "frozen_tensors_unchanged": True,
                "FP32_optimizer_verified": True, "epoch_zero_exact_reload": True,
                "trainable_parameters": 40638464, "peak_allocated": torch.cuda.max_memory_allocated(),
                "peak_reserved": torch.cuda.max_memory_reserved(),
                "sample_lengths": {"train": [s.input_ids.numel() for s in train_samples],
                                   "val": [s.input_ids.numel() for s in val]}}, directory / "manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--objective", choices=CANDIDATES, required=True)
    parser.add_argument("--qualification", action="store_true")
    args = parser.parse_args()
    train(args.run, args.objective, qualification=args.qualification)
