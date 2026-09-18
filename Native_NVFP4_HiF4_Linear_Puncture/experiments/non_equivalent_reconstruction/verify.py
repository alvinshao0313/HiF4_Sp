"""Real checkpoint initialization, gradient, export and vLLM operator checks."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_fold import fold_fusable_moe_layer_state
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import (
    NativeQwen3MoELayerRuntime, build_moe_diag_state, _rms_norm,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.moe_layer_runtime import build_qwen3_moe_layer_call
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.moe_trainer import _load_tensor
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot
from .artifact import load, load_initialization, write_json
from .config import DEFAULT_INIT, LEGACY_RESULTS
from .losses import masked_router_loss
from .student import Student, causal_mask, quantize
from .transforms import fold_initial_diag, projection_weights, transformed_state


def relative_error(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def verify(output, init_artifact=str(DEFAULT_INIT), layer=0):
    from vllm.model_executor.layers.fused_moe.experts.hif4_emulation_moe import _apply_hif4_triton_moe
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    model_path = "nvidia/Qwen3-30B-A3B-NVFP4"
    snapshot = Path(resolve_local_snapshot(model_path))
    initialization = load_initialization(init_artifact, model_path)
    state = load_qwen3_moe_layer_state(snapshot, layer, "cuda")
    base = fold_initial_diag(state, initialization[str(layer)])
    diag = build_moe_diag_state(state.spec, "fusable").cuda()
    diag.load_snapshot(initialization[str(layer)])
    reference = fold_fusable_moe_layer_state(state, diag, use_r64=True)
    samples = load(LEGACY_RESULTS / "shared_calibration/s1k_original_n128_v32_seed42/calibration/train.pt")
    lengths = sorted(s.input_ids.numel() for s in samples)
    embedding = _load_tensor(snapshot, "model.embed_tokens.weight").cuda()
    x = embedding[samples[0].input_ids[:32].cuda()].unsqueeze(0).to(torch.bfloat16)
    del embedding
    # For nonzero layers this is an operator probe, not a calibration estimate.
    call = build_qwen3_moe_layer_call(str(snapshot), x)
    with torch.no_grad():
        teacher = NativeQwen3MoELayerRuntime(state).cuda()(x, attention_mask=causal_mask(x),
                                                       position_embeddings=call.position_embeddings)
    report = {"layer": layer, "train_max_tokens": lengths[-1], "train_median_tokens": lengths[len(lengths)//2]}
    previous = None
    for sharing in ("linear", "group"):
        student = Student(base, sharing).cuda()
        with torch.no_grad():
            transformed = transformed_state(base, student.learned)
            differences = 0
            max_master_error = 0.0
            for (name, w), (other, expected) in zip(projection_weights(transformed), projection_weights(reference)):
                assert name == other
                max_master_error = max(max_master_error, float((w - expected).abs().max()))
                a, b = (w, expected) if name == "router" else (quantize(w, False), quantize(expected, False))
                differences += int((a != b).sum())
            if differences:
                raise AssertionError(f"{sharing}: {differences} E4 initialization quantized values differ")
            out = student(x, position_embeddings=call.position_embeddings, use_ste=False)
            if previous is not None:
                torch.testing.assert_close(out.output, previous, atol=0, rtol=0)
            previous = out.output
        teacher_logits = teacher.router_logits.reshape_as(out.router_logits)
        aux = student.router_aux_logits(out.pre_moe_norm)
        loss = masked_router_loss(aux, teacher_logits, torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
        loss.backward()
        gradient_names = {name for name, p in student.learned.named_parameters() if p.grad is not None}
        assert gradient_names == {"moe_norm", "matrices.router"}, gradient_names
        student.zero_grad(set_to_none=True)
        trained = student(x, position_embeddings=call.position_embeddings, use_ste=True)
        rec = (trained.output.float() - teacher.output.float()).square().mean()
        rec.backward()
        assert student.learned.input_norm.grad is not None
        assert student.learned.matrices["q_proj"].grad is not None
        assert all(torch.isfinite(p.grad).all() for p in student.parameters() if p.grad is not None)
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=0)
        optimizer.step()
        with torch.no_grad():
            del transformed
            transformed = transformed_state(base, student.learned)
            normed = _rms_norm(out.pre_moe_norm, student.learned.moe_norm, student.eps).reshape(-1, state.spec.hidden_size)
            moe, logits, _ = student.moe(normed, use_ste=False)
            weights, ids = logits.float().softmax(-1).topk(state.spec.top_k, -1)
            weights = weights / weights.sum(-1, keepdim=True)
            w13 = torch.stack([torch.cat([quantize(e.gate_proj, False), quantize(e.up_proj, False)])
                               for e in transformed.experts])
            w2 = torch.stack([quantize(e.down_proj, False) for e in transformed.experts])
            deployed = _apply_hif4_triton_moe(normed.contiguous(), w13, w2, weights.contiguous(),
                                             ids.to(torch.int32).contiguous(), use_r64=True)
            error = relative_error(moe, deployed)
            torch.testing.assert_close(moe, deployed, rtol=0, atol=0)
            report[sharing] = {"e4_quantized_mismatches": differences, "max_master_error": max_master_error,
                               "router_gradient_names": sorted(gradient_names), "vllm_moe_rel_l2": error,
                               "trainable_parameters": sum(p.numel() for p in student.parameters())}
        del student, optimizer, transformed, trained, aux, loss, rec, w13, w2, deployed
        gc.collect()
        torch.cuda.empty_cache()
    write_json(report, output)
    print(report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--init_artifact", default=str(DEFAULT_INIT))
    parser.add_argument("--layer", type=int, default=0)
    verify(**vars(parser.parse_args()))
