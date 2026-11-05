"""Independent native/student prefixes and cached BF16 gate-only AdamW."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from ..e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from ..e2e_diag_reconstruction.core.moe_semantic_hif4 import NativeQwen3MoELayerRuntime
from ..e2e_diag_reconstruction.data.batching import build_length_bucket_batches, build_validation_batches
from ..e2e_diag_reconstruction.training.moe_layer_runtime import build_qwen3_moe_layer_call
from ..non_equivalent_reconstruction.losses import masked_router_loss, router_metrics
from ..non_equivalent_reconstruction.train import assemble, propagate, teacher_targets
from ..reasoning_trace_calibration.router import cache_inputs
from ..reasoning_trace_calibration.training import context, samples_for
from .artifacts import atomic_save, load, read_json, state_digest, verify, write_json
from .config import NATIVE, ORIGIN, PROTOCOL
from .student import RouterStudent, load_layer


def optimizer_for(student, lr):
    student.check_trainable()
    return torch.optim.AdamW([student.router_weight], lr=lr, betas=(.9, .999), eps=1e-8,
                            weight_decay=0., foreach=False, fused=False)


def optimizer_contract(student, optimizer):
    parameter = student.router_weight
    if parameter.grad is None or parameter.grad.dtype != torch.bfloat16:
        raise RuntimeError("router gradient must be BF16")
    state = optimizer.state[parameter]
    if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
        raise RuntimeError("unexpected optimizer state or hidden master parameter")
    if any(state[k].dtype != torch.bfloat16 or state[k].shape != parameter.shape
           for k in ("exp_avg", "exp_avg_sq")):
        raise RuntimeError("AdamW moments must be BF16")
    if len(optimizer.param_groups) != 1 or optimizer.param_groups[0]["params"] != [parameter]:
        raise RuntimeError("optimizer contains a non-router parameter")
    return {"weight": str(parameter.dtype), "gradient": str(parameter.grad.dtype),
            "exp_avg": str(state["exp_avg"].dtype), "exp_avg_sq": str(state["exp_avg_sq"].dtype),
            "fp32_master": False, "step_counter": str(state["step"].dtype)}


def epoch(student, samples, collator, inputs, targets, device, *, index, optimizer=None):
    batches = (build_length_bucket_batches(samples, 4, 42 + index) if optimizer is not None
               else build_validation_batches(samples, 4))
    sums, tokens, contract = {}, 0, None
    for batch in batches:
        packed = collator(batch)
        valid = (packed["attention_mask"].bool() & packed["loss_mask"].bool()).to(device)
        x, teacher = assemble(inputs, batch, device), assemble(targets, batch, device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            logits = student.router(x)
            loss = masked_router_loss(logits, teacher, valid, kind="top_mass", top_k=8, temperature=1.)
            if loss.dtype != torch.float32 or not torch.isfinite(loss):
                raise RuntimeError("invalid FP32 top-mass KL")
            if optimizer is not None:
                loss.backward()
                if student.router_weight.grad is None or not torch.isfinite(student.router_weight.grad).all():
                    raise RuntimeError("non-finite or missing BF16 router gradient")
                optimizer.step()
                contract = optimizer_contract(student, optimizer)
        with torch.no_grad():
            stats = router_metrics(logits, teacher, valid, top_k=8)
        count = int(valid.sum())
        for key, value in {"kl": float(loss.detach()), **stats}.items():
            sums[key] = sums.get(key, 0.) + value * count
        tokens += count
    if not tokens:
        raise ValueError("no valid calibration tokens")
    return {**{k: v / tokens for k, v in sums.items()}, "valid_tokens": tokens,
            **({"optimizer_contract": contract} if contract is not None else {})}


def cache_equivalence(student, batch, collator, hidden, targets, inputs, device):
    x = assemble(hidden, batch, device)
    call = build_qwen3_moe_layer_call(str(NATIVE), x)
    packed = collator(batch)
    valid = (packed["attention_mask"].bool() & packed["loss_mask"].bool()).to(device)
    teacher = assemble(targets, batch, device)
    student.zero_grad(set_to_none=True)
    out = student(x, position_embeddings=call.position_embeddings, use_ste=False)
    direct = masked_router_loss(out.router_logits, teacher, valid, kind="top_mass", top_k=8, temperature=1.)
    direct.backward()
    gradient = student.router_weight.grad.detach().clone()
    student.zero_grad(set_to_none=True)
    cached = masked_router_loss(student.router(assemble(inputs, batch, device)), teacher, valid,
                                kind="top_mass", top_k=8, temperature=1.)
    cached.backward()
    if not torch.equal(direct.detach(), cached.detach()) or not torch.equal(gradient, student.router_weight.grad):
        raise RuntimeError("cache changed the KL or BF16 gate gradient")
    student.zero_grad(set_to_none=True)
    return {"equal": True, "kl": float(cached.detach()), "gradient_dtype": str(gradient.dtype),
            "gradient_max": float(gradient.abs().max())}


def fit(student, train, val, collator, inputs, targets, device, *, lr, epochs, output):
    optimizer = optimizer_for(student, lr)
    baseline = epoch(student, val, collator, inputs, targets, device, index=0)
    initial = student.router_weight.detach().clone()
    best = {"epoch": 0, "metrics": baseline, "router_weight": initial.cpu().clone()}
    write_json(baseline, output / "epochs/000.json")
    updated = 0
    for i in range(1, epochs + 1):
        training = epoch(student, train, collator, inputs, targets, device, index=i - 1, optimizer=optimizer)
        validation = epoch(student, val, collator, inputs, targets, device, index=i)
        updated = max(updated, int((student.router_weight.detach() != initial).sum()))
        if validation["kl"] < best["metrics"]["kl"]:
            best = {"epoch": i, "metrics": validation,
                    "router_weight": student.router_weight.detach().cpu().clone()}
        write_json({"train": training, "validation": validation, "lr": lr, "updated_elements": updated},
                   output / f"epochs/{i:03d}.json")
        print(json.dumps({"event": "router_epoch", "layer": student.base.layer_idx,
                          "epoch": i, "lr": lr, "train_kl": training["kl"], "val_kl": validation["kl"],
                          "updated_elements": updated}), flush=True)
    best["updated_elements"] = updated
    atomic_save(best, output / "selected.pt")
    loaded = load(output / "selected.pt")
    if loaded["router_weight"].dtype != torch.bfloat16:
        raise RuntimeError("checkpoint router is not BF16")
    with torch.no_grad():
        student.router_weight.copy_(loaded["router_weight"].to(device))
    if not torch.equal(student.router_weight.detach().cpu(), best["router_weight"]):
        raise RuntimeError("BF16 router save/reload mismatch")
    return best


def align(root, *, qualification=False):
    root, manifest = verify(root)
    train, val, hashes = samples_for(Path(manifest["source_run"]), ORIGIN)
    if hashes != manifest["calibration_sha256"]:
        raise ValueError("B calibration changed")
    if qualification:
        train = sorted(train, key=lambda s: s.input_ids.numel(), reverse=True)[:4]
        val = sorted(val, key=lambda s: s.input_ids.numel(), reverse=True)[:4]
    samples = train + val
    output = root / ("qualification/training" if qualification else "training")
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    collator, device, initial_hidden = context(samples)
    # The embedding is identical in B and native; subsequent prefixes are separate.
    native_hidden, student_hidden = dict(initial_hidden), dict(initial_hidden)
    del initial_hidden
    record = {"complete": False, "qualification": qualification, "completed_layers": [],
              "calibration_sha256": hashes, "prefixes": "independent_native_and_student",
              "source_hashes": read_json(root / "source_hashes.json"),
              "lengths": {"train": [s.input_ids.numel() for s in train], "val": [s.input_ids.numel() for s in val]}}
    write_json(record, output / "manifest.json")
    started, lr = time.monotonic(), None
    for layer in range(2 if qualification else 48):
        print(json.dumps({"event": "native_prefix", "layer": layer}), flush=True)
        native_state = load_qwen3_moe_layer_state(NATIVE, layer, device)
        teacher = NativeQwen3MoELayerRuntime(native_state).to(device).eval()
        next_native, targets = teacher_targets(teacher, NATIVE, samples, collator, native_hidden, device, 4)
        del teacher, native_state, native_hidden
        native_hidden = next_native
        student = RouterStudent(load_layer(manifest["initialization"], layer, device)).to(device)
        frozen = student.frozen_digest()
        if not torch.equal(student.router_weight.detach(), student.base.router_weight):
            raise RuntimeError("zero-step gate differs from B")
        inputs = cache_inputs(student, samples, student_hidden, device)
        if qualification:
            checked = cache_equivalence(student, train, collator, student_hidden, targets, inputs, device)
            write_json(checked, output / f"layers/{layer:03d}/cache_equivalence.json")
        if layer == 0:
            initial = student.router_weight.detach().clone()
            trials = []
            for rate in PROTOCOL["learning_rates"]:
                with torch.no_grad():
                    student.router_weight.copy_(initial)
                best = fit(student, train, val, collator, inputs, targets, device, lr=rate,
                           epochs=1 if qualification else 3, output=output / f"screen/{rate:g}")
                trials.append({"lr": rate, "validation_kl": best["metrics"]["kl"], "epoch": best["epoch"]})
            lr = min(trials, key=lambda x: (x["validation_kl"], x["lr"]))["lr"]
            write_json({"lr": lr, "trials": trials}, output / "learning_rate.json")
            with torch.no_grad():
                student.router_weight.copy_(initial)
            del initial
        best = fit(student, train, val, collator, inputs, targets, device, lr=lr,
                   epochs=2 if qualification else 5, output=output / f"layers/{layer:03d}")
        if qualification and not best["updated_elements"]:
            raise RuntimeError("BF16 gate failed to update during qualification")
        if student.frozen_digest() != frozen:
            raise RuntimeError("a non-router B tensor changed")
        student_hidden = propagate(student, NATIVE, samples, student_hidden, device, 4)
        if any(not torch.isfinite(v).all() for d in (native_hidden, student_hidden) for v in d.values()):
            raise RuntimeError("non-finite propagated prefix")
        if any(native_hidden[k].data_ptr() == student_hidden[k].data_ptr() for k in native_hidden):
            raise RuntimeError("teacher and student prefix caches alias")
        if qualification:
            write_json({"native_output_sha256": state_digest(native_hidden),
                        "student_output_sha256": state_digest(student_hidden),
                        "distinct_prefixes": True, "non_router_unchanged": True,
                        "trainable": [n for n, p in student.named_parameters() if p.requires_grad]},
                       output / f"layers/{layer:03d}/contracts.json")
        record["completed_layers"].append(layer)
        record.update(lr=lr, seconds=time.monotonic() - started, peak_memory_bytes=torch.cuda.max_memory_allocated())
        write_json(record, output / "manifest.json")
        print(json.dumps({"event": "router_layer", "layer": layer, "selected_epoch": best["epoch"],
                          "kl": best["metrics"]["kl"], "seconds": record["seconds"]}), flush=True)
        del student, inputs, targets, best
        gc.collect()
        torch.cuda.empty_cache()
    verify(root)
    record["complete"] = True
    write_json(record, output / "manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--qualification", action="store_true")
    args = parser.parse_args()
    align(args.run, qualification=args.qualification)
