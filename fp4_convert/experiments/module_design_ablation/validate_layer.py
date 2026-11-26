"""Real native layer diagnostics; this is not the formal longest-batch test."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from ..e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from ..e2e_diag_reconstruction.core.moe_semantic_hif4 import NativeQwen3MoELayerRuntime
from ..e2e_diag_reconstruction.training.moe_trainer import _load_tensor
from ..e2e_diag_reconstruction.training.moe_layer_runtime import build_qwen3_moe_layer_call
from ..non_equivalent_reconstruction.artifact import atomic_save, load, write_json
from ..non_equivalent_reconstruction.losses import masked_router_loss
from ..non_equivalent_reconstruction.student import Student as PreviousStudent, causal_mask
from .config import CALIB_CACHE, NATIVE_SNAPSHOT, parameter_groups
from .student import Student
from .variants import VARIANTS


def validate(output):
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = load_qwen3_moe_layer_state(NATIVE_SNAPSHOT, 0, "cuda:0")
    sample = torch.load(CALIB_CACHE / "calibration/train.pt", weights_only=False)[0]
    ids = sample.input_ids[:64]
    embedding = _load_tensor(NATIVE_SNAPSHOT, "model.embed_tokens.weight")
    hidden = embedding[ids].unsqueeze(0).to(device="cuda:0", dtype=torch.bfloat16)
    del embedding
    call = build_qwen3_moe_layer_call(str(NATIVE_SNAPSHOT), hidden)
    mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
    with torch.no_grad():
        teacher = NativeQwen3MoELayerRuntime(base).to(hidden.device).eval()
        target = teacher(hidden, attention_mask=causal_mask(hidden), position_embeddings=call.position_embeddings)
        teacher_logits = target.router_logits.reshape(*hidden.shape[:2], -1)
        previous = PreviousStudent(base, "group")
        expected = previous(hidden, position_embeddings=call.position_embeddings, use_ste=False).output
        del teacher, previous
    den = (target.output.float() - hidden.float()).square().sum()
    records = []
    for name, variant in VARIANTS.items():
        if variant.is_reference:
            continue
        student = Student(base, name)
        params = dict(student.named_parameters())
        initial_norm = student.learned.input_norm.detach().clone()
        with torch.no_grad():
            initial = student(hidden, position_embeddings=call.position_embeddings, use_ste=False).output
        if name in {"current", "current_kl"}:
            if not torch.equal(initial, expected):
                raise AssertionError("native-start current parameterization differs from previous student")
        row = {"variant": name, "trainable_parameters": sum(p.numel() for p in params.values()),
               "initial_nmse": float((initial.float() - target.output.float()).square().sum() / den)}
        if variant.needs_training:
            optimizer = torch.optim.AdamW(parameter_groups(student, 1.0), weight_decay=0)
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                result = student(hidden, position_embeddings=call.position_embeddings)
                aux = student.router_aux_logits(result.pre_moe_norm)
                router_loss = masked_router_loss(aux, teacher_logits, mask)
                if step == 0:
                    grads = (torch.autograd.grad(router_loss, tuple(params.values()), allow_unused=True,
                                                 retain_graph=True) if router_loss.requires_grad else [None] * len(params))
                    reached = {key: float(grad.float().norm()) for key, grad in zip(params, grads) if grad is not None}
                    allowed = {"learned.norms.moe", "learned.matrices.router", "learned.diag.z_gu"}
                    if set(reached) - allowed:
                        raise AssertionError(f"router auxiliary gradient escaped its boundary: {set(reached) - allowed}")
                    row["router_aux_gradient_norms"] = reached
                    del grads
                nmse = (result.output.float() - target.output.float()).square().sum() / den
                loss = nmse + variant.kl_weight * router_loss
                loss.backward()
                gradients = {key: parameter.grad for key, parameter in params.items() if parameter.grad is not None}
                if not gradients or any(not grad.isfinite().all() for grad in gradients.values()):
                    raise AssertionError("missing or nonfinite main gradients")
                row[f"step_{step}_nmse"] = float(nmse.detach())
                optimizer.step()
                if student.learned.diag is not None:
                    student.learned.diag.clamp_log2_()
                del result, aux, router_loss, loss, nmse, gradients
            if variant.transform == "rotation":
                errors = []
                with torch.no_grad():
                    for rotation in student.learned.rotations.values():
                        q = rotation()
                        errors.append((q @ q.transpose(-1, -2) - rotation.identity).abs().max())
                error = float(torch.stack(errors).max())
                if error > 1e-5:
                    raise AssertionError(f"orthogonality error {error}")
                row["orthogonality_max_abs"] = error
                del q, errors
            if not variant.train_norm and variant.transform != "diag":
                if not torch.equal(initial_norm, student.learned.input_norm):
                    raise AssertionError("frozen norm changed")
            saved = student.learned.snapshot()
            atomic_save(saved, out / f"{name}.pt")
            student.learned.load_state_dict(load(out / f"{name}.pt"), strict=True)
            if any(not torch.equal(value.cpu(), saved[key]) for key, value in student.learned.state_dict().items()):
                raise AssertionError("save/reload changed parameters")
            del optimizer, saved
        records.append(row)
        write_json(row, out / f"{name}.json")
        print(row, flush=True)
        del student, params, initial, initial_norm
        gc.collect()
        torch.cuda.empty_cache()
    write_json({"passed": True, "sample_id": sample.sample_id, "tokens": ids.numel(),
                "full_shape_training_verified": False, "runtime_verified": False,
                "records": records}, out / "validation.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    validate(**vars(parser.parse_args()))
