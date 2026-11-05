"""Router-only top-mass KL on detached actual student inputs; epoch zero allowed."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from ..e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from ..e2e_diag_reconstruction.core.moe_semantic_hif4 import NativeQwen3MoELayerRuntime, _rms_norm
from ..e2e_diag_reconstruction.data.batching import build_length_bucket_batches, build_validation_batches
from ..e2e_diag_reconstruction.training.moe_layer_runtime import build_qwen3_moe_layer_call
from ..module_design_ablation.student import Student
from ..non_equivalent_reconstruction.losses import masked_router_loss
from ..non_equivalent_reconstruction.train import assemble, propagate, teacher_targets
from .artifacts import atomic_save, load, read_json, sha256, state_digest, validate_root, verify_sources, write_json
from .config import MODELS, NATIVE, PROTOCOL
from .prepare import require_isolation
from .training import context, samples_for, training_dir


class RouterStudent(Student):
    def __init__(self, base, parameters):
        super().__init__(base, "matrix_norm")
        self.learned.load_state_dict(parameters, strict=True)
        self.requires_grad_(False)
        # The actual gate, without a router transform or additional factors.
        self.router_master = nn.Parameter(base.router_weight.detach().float().clone())
        if [n for n, p in self.named_parameters() if p.requires_grad] != ["router_master"]:
            raise RuntimeError("only the actual router gate may be trained")

    def router(self, normed):
        return F.linear(normed.to(torch.bfloat16), self.router_master.to(torch.bfloat16))


@torch.no_grad()
def cache_inputs(student, samples, hidden, device):
    inputs = {}
    for batch in build_validation_batches(samples, 4):
        x = assemble(hidden, batch, device)
        call = build_qwen3_moe_layer_call(str(NATIVE), x)
        out = student(x, position_embeddings=call.position_embeddings, use_ste=False)
        normed = _rms_norm(out.pre_moe_norm, student.learned.moe_norm, student.eps).to(torch.bfloat16)
        if not torch.equal(student.router(normed).reshape_as(out.router_logits), out.router_logits):
            raise RuntimeError("cached router input differs from the actual student router input")
        for i, sample in enumerate(batch):
            inputs[sample.sample_id] = normed[i, :sample.input_ids.numel()].detach().cpu().contiguous()
    return inputs


def cached_epoch(student, samples, collator, inputs, targets, device, *, epoch, optimizer=None):
    training = optimizer is not None
    batches = (build_length_bucket_batches(samples, 4, 42 + epoch) if training else build_validation_batches(samples, 4))
    numerator, tokens = 0., 0
    for batch in batches:
        packed = collator(batch)
        valid = (packed["attention_mask"].bool() & packed["loss_mask"].bool()).to(device)
        x, t = assemble(inputs, batch, device), assemble(targets, batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = student.router(x)
            loss = masked_router_loss(logits, t, valid, kind="top_mass", top_k=8, temperature=1.)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite router KL")
            if training:
                loss.backward()
                grad = student.router_master.grad
                if grad is None or not torch.isfinite(grad).all():
                    raise RuntimeError("invalid actual-router gradient")
                optimizer.step()
        count = int(valid.sum())
        numerator += float(loss.detach()) * count
        tokens += count
    if not tokens:
        raise ValueError("empty router epoch")
    return {"kl": numerator / tokens, "valid_tokens": tokens}


def cache_equivalence(student, batch, collator, hidden, targets, inputs, device):
    """Compare the cache with the full real-layer path, including the gate gradient."""
    x = assemble(hidden, batch, device)
    call = build_qwen3_moe_layer_call(str(NATIVE), x)
    packed = collator(batch)
    mask = (packed["attention_mask"].bool() & packed["loss_mask"].bool()).to(device)
    teacher = assemble(targets, batch, device)
    student.zero_grad(set_to_none=True)
    out = student(x, position_embeddings=call.position_embeddings, use_ste=False)
    direct = masked_router_loss(out.router_logits, teacher, mask, kind="top_mass", top_k=8, temperature=1.)
    direct.backward()
    grad = student.router_master.grad.detach().clone()
    student.zero_grad(set_to_none=True)
    cached = masked_router_loss(student.router(assemble(inputs, batch, device)), teacher, mask,
                                kind="top_mass", top_k=8, temperature=1.)
    cached.backward()
    # Identical BF16 gate operands, loss, and reduction are expected here.
    if not torch.equal(direct.detach(), cached.detach()) or not torch.equal(grad, student.router_master.grad):
        raise RuntimeError("router cache changed the KL or FP32 master gradient")
    student.zero_grad(set_to_none=True)
    return {"kl": float(cached.detach()), "gradient_max": float(grad.abs().max()), "equal": True}


def fit(student, training, validation, collator, inputs, targets, device, *, lr, epochs, output):
    frozen = state_digest(student.learned.state_dict())
    optimizer = torch.optim.AdamW([student.router_master], lr=lr, weight_decay=0.)
    baseline = cached_epoch(student, validation, collator, inputs, targets, device, epoch=0)
    best = {"epoch": 0, "metrics": baseline, "router_master": student.router_master.detach().cpu().clone()}
    write_json(baseline, output / "epochs/000.json")
    for epoch in range(1, epochs + 1):
        t = cached_epoch(student, training, collator, inputs, targets, device, epoch=epoch - 1, optimizer=optimizer)
        v = cached_epoch(student, validation, collator, inputs, targets, device, epoch=epoch)
        if v["kl"] < best["metrics"]["kl"]:
            best = {"epoch": epoch, "metrics": v, "router_master": student.router_master.detach().cpu().clone()}
        write_json({"train": t, "validation": v, "lr": lr}, output / f"epochs/{epoch:03d}.json")
    if frozen != state_digest(student.learned.state_dict()):
        raise RuntimeError("non-router state changed during alignment")
    atomic_save(best, output / "selected.pt")
    loaded = load(output / "selected.pt")
    with torch.no_grad():
        student.router_master.copy_(loaded["router_master"].to(device))
    if not torch.equal(student.router_master.detach().cpu(), best["router_master"]):
        raise RuntimeError("router save/reload mismatch")
    return best


def align(root, *, qualification=False, candidate=None):
    root = validate_root(root)
    if not qualification:
        require_isolation(root)
    verify_sources(root)
    selection = read_json(root / "development/selection.json") if not qualification else {"winner": candidate}
    candidate = selection["winner"]
    if candidate not in MODELS[:2]:
        raise ValueError("invalid development selection")
    origin = training_dir(root, candidate, qualification=qualification)
    if not qualification:
        exported = root / "models" / candidate / "export.json"
        if sha256(exported) != selection["export_sha256"][candidate]:
            raise ValueError("winner checkpoint changed after development selection")
        export_manifest = read_json(exported)
        if any(sha256(origin / f"layers/{i:03d}/selected.pt") != export_manifest["selected_sha256"][str(i)] for i in range(48)):
            raise ValueError("winner training state differs from its evaluated export")
    training, validation, hashes = samples_for(root, candidate)
    if qualification:
        training = sorted(training, key=lambda s: s.input_ids.numel(), reverse=True)[:4]
        validation = sorted(validation, key=lambda s: s.input_ids.numel(), reverse=True)[:4]
    samples = training + validation
    output = root / ("qualification/router" if qualification else "router")
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    collator, device, hidden = context(samples)
    lr = None
    manifest = {"complete": False, "winner": candidate, "calibration_sha256": hashes,
                "origin": str(origin), "completed_layers": [], "qualification": qualification}
    write_json(manifest, output / "manifest.json")
    for layer in range(2 if qualification else 48):
        base = load_qwen3_moe_layer_state(NATIVE, layer, device)
        teacher = NativeQwen3MoELayerRuntime(base).to(device).eval()
        _, targets = teacher_targets(teacher, NATIVE, samples, collator, hidden, device, 4)
        del teacher
        parameters = load(origin / f"layers/{layer:03d}/selected.pt")["parameters"]
        student = RouterStudent(base, parameters).to(device)
        reference = Student(base, "matrix_norm").to(device)
        reference.learned.load_state_dict(parameters, strict=True)
        with torch.no_grad():
            batch = build_validation_batches(validation, 4)[-1]
            x = assemble(hidden, batch, device)
            call = build_qwen3_moe_layer_call(str(NATIVE), x)
            reference_out = reference(x, position_embeddings=call.position_embeddings, use_ste=False)
            initial_out = student(x, position_embeddings=call.position_embeddings, use_ste=False)
            if not torch.equal(reference_out.output, initial_out.output) or not torch.equal(reference_out.router_logits, initial_out.router_logits):
                raise RuntimeError("router zero-step model differs from its initialization checkpoint")
        del reference, reference_out, initial_out
        inputs = cache_inputs(student, samples, hidden, device)
        equivalence = cache_equivalence(student, batch, collator, hidden, targets, inputs, device)
        write_json(equivalence, output / f"layers/{layer:03d}/cache_equivalence.json")
        if layer == 0:
            initial = student.router_master.detach().clone()
            screen = []
            for rate in PROTOCOL["router"]["learning_rates"]:
                with torch.no_grad():
                    student.router_master.copy_(initial)
                best = fit(student, training, validation, collator, inputs, targets, device, lr=rate,
                           epochs=1 if qualification else 3, output=output / f"screen/{rate:g}")
                screen.append({"lr": rate, "validation_kl": best["metrics"]["kl"], "selected_epoch": best["epoch"]})
            lr = min(screen, key=lambda item: (item["validation_kl"], item["lr"]))["lr"]
            write_json({"trials": screen, "lr": lr}, output / "learning_rate.json")
            with torch.no_grad():
                student.router_master.copy_(initial)
            del initial
        best = fit(student, training, validation, collator, inputs, targets, device, lr=lr,
                   epochs=2 if qualification else 5, output=output / f"layers/{layer:03d}")
        hidden = propagate(student, NATIVE, samples, hidden, device, 4)
        manifest["completed_layers"].append(layer)
        manifest["lr"] = lr
        write_json(manifest, output / "manifest.json")
        print(json.dumps({"event": "router_layer", "layer": layer, "epoch": best["epoch"], "kl": best["metrics"]["kl"]}), flush=True)
        del student, base, inputs, targets, best, parameters
        gc.collect()
        torch.cuda.empty_cache()
    verify_sources(root)
    manifest["complete"] = True
    write_json(manifest, output / "manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--qualification", action="store_true")
    parser.add_argument("--candidate", choices=MODELS[:2])
    args = parser.parse_args()
    align(args.run, qualification=args.qualification, candidate=args.candidate)
