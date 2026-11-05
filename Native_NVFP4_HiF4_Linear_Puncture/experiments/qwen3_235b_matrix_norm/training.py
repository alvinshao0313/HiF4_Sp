"""Four sample replicas, one global NMSE update, native-initialized matrix_norm."""
from __future__ import annotations

import argparse
from datetime import timedelta
import gc
import os
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from ..e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from ..e2e_diag_reconstruction.core.moe_model_spec import Qwen3MoeModelSpec
from ..e2e_diag_reconstruction.core.moe_semantic_hif4 import NativeQwen3MoELayerRuntime
from ..e2e_diag_reconstruction.data.batching import DynamicCalibrationCollator, build_length_bucket_batches, build_validation_batches
from ..e2e_diag_reconstruction.training.moe_layer_runtime import build_qwen3_moe_layer_call
from ..e2e_diag_reconstruction.training.moe_trainer import build_initial_moe_hidden_cache
from ..module_design_ablation.student import Student
from ..non_equivalent_reconstruction.train import assemble, propagate, run_epoch, teacher_targets
from ..qwen3_235b_transfer.group_parallel_step import sum_gradients
from .artifacts import atomic_save, load, now, read_json, sha256, state_digest, verify_sources, write_json
from .config import NATIVE, TRAINING
from .generate import calibration


def trainable(student):
    values = sorted((n, p) for n, p in student.named_parameters() if p.requires_grad)
    if not values or any(not n.startswith(("learned.matrices.", "learned.norms.")) or "router" in n for n, _ in values):
        raise ValueError("only matrix_norm matrices and two norms may be trained")
    return values


def frozen_digest(student):
    return state_digest({"router": student.base.router_weight, **dict(student.learned.named_buffers())})


def reconstruction(student, x, target, valid):
    call = build_qwen3_moe_layer_call(str(NATIVE), x)
    out = student(x, position_embeddings=call.position_embeddings, use_ste=True)
    predicted = out.output[valid].float() - x[valid].float()
    expected = target[valid].float() - x[valid].float()
    return (predicted - expected).square().sum(), expected.square().sum()


def parallel_step(student, x, target, valid, optimizer, scheduler):
    """Each rank contributes N_r / SUM(D_r); SUM gradients, never average ratios."""
    student.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autograd.graph.save_on_cpu(pin_memory=False):
        num, den = reconstruction(student, x, target, valid)
        totals = torch.stack((num.detach(), den.detach(), valid.sum().to(num.dtype)))
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if not torch.isfinite(totals).all() or totals[1] <= 0:
            raise RuntimeError("non-finite NMSE or zero global target energy")
        (num / totals[1]).backward()
    sum_gradients(trainable(student))
    optimizer.step()
    scheduler.step()
    return {"num": float(totals[0]), "den": float(totals[1]), "tokens": int(totals[2]),
            "objective": float(totals[0] / totals[1])}


def serial_step(student, x, target, valid, optimizer, denominator):
    """Same sample shapes/padding as the replicas; qualification reference only."""
    student.train()
    optimizer.zero_grad(set_to_none=True)
    numerator = 0.
    for i in range(4):
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            num, _ = reconstruction(student, x[i:i + 1], target[i:i + 1], valid[i:i + 1])
            (num / denominator).backward()
        numerator += float(num.detach())
    return numerator / denominator


def relative_error(first, second):
    numerator = denominator = 0.
    for name in first:
        a, b = first[name], second[name]
        if (a is None) != (b is None):
            raise RuntimeError(f"gradient-presence contract differs: {name}")
        if a is not None:
            numerator += float((a.double() - b.double()).square().sum())
            denominator += float(b.double().square().sum())
    return (numerator / max(denominator, 1e-30)) ** .5


def gradient_snapshot(student):
    return {n: None if p.grad is None else p.grad.detach().cpu().clone() for n, p in trainable(student)}


def optimizer_digest(student, optimizer, scheduler):
    tensors, metadata = {}, {}
    for name, p in trainable(student):
        tensors[name] = p.detach()
        if p.grad is not None:
            tensors[name + ".grad"] = p.grad
        metadata[name] = {"grad_none": p.grad is None, "keys": sorted(optimizer.state.get(p, {}))}
        for key, value in optimizer.state.get(p, {}).items():
            if torch.is_tensor(value):
                # Adam's step is a scalar; byte views require a dimension.
                tensors[name + "." + key] = value.reshape(1) if value.ndim == 0 else value
                metadata[name + "." + key + ".shape"] = list(value.shape)
            else:
                metadata[name + "." + key] = value
    return {"tensors": state_digest(tensors), "metadata": metadata, "scheduler": scheduler.state_dict()}


def assert_replicas(student, optimizer, scheduler, group):
    rows = [None] * 4
    dist.all_gather_object(rows, optimizer_digest(student, optimizer, scheduler), group=group)
    if any(row != rows[0] for row in rows):
        raise RuntimeError("parameters, gradients, Adam or scheduler differ across replicas")


def distribute(batch, hidden, targets, collator, device):
    """Only rank zero owns progressive hidden/teacher caches."""
    rank = dist.get_rank()
    if len(batch) != 4:
        raise ValueError("the approved global batch must contain four samples")
    seq = max(s.input_ids.numel() for s in batch)
    width = read_json(NATIVE / "config.json")["hidden_size"]
    global_values = None
    if rank == 0:
        packed = collator(batch)
        global_values = (assemble(hidden, batch, device), assemble(targets, batch, device),
                         (packed["attention_mask"].bool() & packed["loss_mask"].bool()).to(device))
    local = []
    for i, (shape, dtype) in enumerate((((1, seq, width), torch.bfloat16),
                                       ((1, seq, width), torch.bfloat16), ((1, seq), torch.bool))):
        value = torch.empty(shape, dtype=dtype, device=device)
        pieces = None if rank else [piece.contiguous() for piece in global_values[i].split(1)]
        dist.scatter(value, scatter_list=pieces, src=0)
        local.append(value)
    return (*local, global_values)


def train(root, *, qualification=False):
    root = calibration_root(root)
    verify_sources(root)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(hours=2))
    control = dist.new_group(backend="gloo", timeout=timedelta(hours=2))
    if dist.get_world_size() != 4:
        raise ValueError("matrix_norm training requires four sample replicas")
    try:
        _train(root, torch.device("cuda", local_rank), control, qualification=qualification)
    finally:
        dist.destroy_process_group()


def calibration_root(root):
    from .artifacts import verify_inputs
    root = verify_inputs(root)
    if not read_json(root / "isolation/audit.json")["complete"]:
        raise ValueError("calibration isolation is incomplete")
    return root


def _train(root, device, control, *, qualification):
    rank = dist.get_rank()
    training, validation = calibration(root)
    if qualification:
        # Select the actual longest epoch-zero batch; never truncate tokens.
        training = max(build_length_bucket_batches(training, 4, 42),
                       key=lambda b: (max(s.input_ids.numel() for s in b), sum(s.input_ids.numel() for s in b)))
        validation = max(build_validation_batches(validation, 4),
                         key=lambda b: (max(s.input_ids.numel() for s in b), sum(s.input_ids.numel() for s in b)))
    elif not read_json(root / "qualification/training/manifest.json")["complete"]:
        raise ValueError("longest-batch two-layer training qualification is required")
    spec = Qwen3MoeModelSpec.from_snapshot(NATIVE)
    output = root / ("qualification/training" if qualification else "training")
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(NATIVE, local_files_only=True)
    collator = DynamicCalibrationCollator(tokenizer.pad_token_id)
    samples = list(training) + list(validation)
    hidden = None
    if rank == 0:
        cache = build_initial_moe_hidden_cache(NATIVE, samples, collator, device, 4)
        hidden = {s.sample_id: cache.get(s.sample_id) for s in samples}
        del cache
    epochs, layers = (2, 2) if qualification else (TRAINING["epochs"], spec.num_layers)
    cfg = SimpleNamespace(batch_size=4, calib_seed=42, router_loss="top_mass", router_top_k=8,
                          router_temperature=1., router_loss_weight=0.)
    manifest = {"complete": False, "qualification": qualification, "source_snapshot": str(NATIVE),
                "training": TRAINING, "epochs": epochs, "layers": layers, "completed_layers": [],
                "calibration_sha256": read_json(root / "generation/manifest.json")["calibration_sha256"],
                "sample_lengths": {"train": [s.input_ids.numel() for s in training],
                                   "val": [s.input_ids.numel() for s in validation]}, "started_at": now()}
    started = time.monotonic()
    for layer in range(layers):
        torch.cuda.reset_peak_memory_stats(device)
        base = load_qwen3_moe_layer_state(NATIVE, layer, device)
        targets = logits = None
        if rank == 0:
            teacher = NativeQwen3MoELayerRuntime(base).to(device).eval()
            targets, logits = teacher_targets(teacher, NATIVE, samples, collator, hidden, device, 4)
            del teacher
        student = Student(base, "matrix_norm").to(device)
        trainable(student)
        frozen = frozen_digest(student)
        initial = state_digest(student.learned.snapshot())
        optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=0.)
        batches_per_epoch = len(build_length_bucket_batches(training, 4, 42))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * batches_per_epoch, eta_min=0.)
        reference = ref_optimizer = None
        if qualification and rank == 0:
            reference = Student(base, "matrix_norm").to(device)
            reference.learned.load_state_dict(student.learned.state_dict(), strict=True)
            ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=3e-4, weight_decay=0.)
        best = None
        for epoch in range(epochs):
            total_num = total_den = 0.
            for batch_index, batch in enumerate(build_length_bucket_batches(training, 4, 42 + epoch)):
                x, target, valid, global_values = distribute(batch, hidden, targets, collator, device)
                metrics = parallel_step(student, x, target, valid, optimizer, scheduler)
                if qualification:
                    assert_replicas(student, optimizer, scheduler, control)
                    if rank == 0:
                        # The reference uses the pre-step LR and one update per
                        # original batch, including Adam's existing moments.
                        ref_optimizer.param_groups[0]["lr"] = 3e-4 if epoch == 0 else previous_lr
                        ref_loss = serial_step(reference, *global_values, ref_optimizer, metrics["den"])
                        gradient_error = relative_error(gradient_snapshot(student), gradient_snapshot(reference))
                        ref_optimizer.step()
                        update_error = relative_error(dict(student.learned.snapshot()), dict(reference.learned.snapshot()))
                        loss_error = abs(ref_loss - metrics["objective"]) / max(abs(ref_loss), 1e-30)
                        if gradient_error > 1e-4 or update_error > 1e-5 or loss_error > 1e-4:
                            raise RuntimeError(f"sample-parallel reference differs: loss={loss_error}, gradient={gradient_error}, parameters={update_error}")
                        write_json({"loss_relative": loss_error, "gradient_relative_l2": gradient_error,
                                    "parameters_relative_l2": update_error,
                                    "replicas_exact": True, "reference": "same-padding sample-serial accumulation"},
                                   output / f"layers/{layer:03d}/reference_{epoch:03d}.json")
                        previous_lr = optimizer.param_groups[0]["lr"]
                total_num += metrics["num"]
                total_den += metrics["den"]
                if rank == 0:
                    print({"event": "batch", "layer": layer, "epoch": epoch + 1, "batch": batch_index + 1,
                           "batches": batches_per_epoch, **metrics, "seconds": time.monotonic() - started}, flush=True)
                del x, target, valid, global_values
            if rank == 0:
                validation_metrics = run_epoch(student, cfg, NATIVE, validation, collator, hidden, targets, logits,
                                               device, epoch=epoch)
                if best is None or validation_metrics["reconstruction_nmse"] < best["metrics"]["reconstruction_nmse"]:
                    best = {"epoch": epoch + 1, "metrics": validation_metrics, "parameters": student.learned.snapshot()}
                write_json({"train": {"reconstruction_nmse": total_num / total_den}, "validation": validation_metrics},
                           output / f"layers/{layer:03d}/epochs/{epoch + 1:03d}.json")
            dist.barrier()
        assert_replicas(student, optimizer, scheduler, control)
        if frozen_digest(student) != frozen:
            raise RuntimeError("frozen router or native buffers changed")
        if state_digest(student.learned.snapshot()) == initial:
            raise RuntimeError("matrix_norm parameters did not update")
        selected = output / f"layers/{layer:03d}/selected.pt"
        if rank == 0:
            atomic_save(best, selected)
        dist.barrier()
        saved = load(selected)
        student.learned.load_state_dict(saved["parameters"], strict=True)
        if state_digest(student.learned.snapshot()) != state_digest(saved["parameters"]):
            raise RuntimeError("selected parameters changed after save/reload")
        if rank == 0:
            hidden = propagate(student, NATIVE, samples, hidden, device, 4)
            if any(not torch.isfinite(x).all() for x in hidden.values()):
                raise RuntimeError("non-finite selected-layer propagation")
            manifest["completed_layers"].append(layer)
            manifest["seconds"] = time.monotonic() - started
            write_json({**saved["metrics"], "selected_epoch": saved["epoch"], "frozen_router_sha256": frozen,
                        "peak_memory_bytes_rank0": torch.cuda.max_memory_allocated(device),
                        "selected_sha256": sha256(selected)}, output / f"layers/{layer:03d}/metrics.json")
            write_json(manifest, output / "manifest.json")
        del student, base, optimizer, scheduler, best, saved, reference, ref_optimizer, targets, logits
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
    verify_sources(root)
    if rank == 0:
        write_json({**manifest, "complete": True, "finished_at": now()}, output / "manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--qualification", action="store_true")
    train(**vars(parser.parse_args()))
