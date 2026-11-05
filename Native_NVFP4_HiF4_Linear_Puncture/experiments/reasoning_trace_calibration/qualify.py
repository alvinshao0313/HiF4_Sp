"""Actual longest-batch qualification and small real-TP2 exported-model inference."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .artifacts import atomic_save, bind_sources, load, read_json, state_digest, write_json
from .config import GENERATION, MODELS, NATIVE, OLD_TRAINING


def router_contracts(root):
    """Independent implementation check; does not train a formal candidate."""
    from ..e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
    from ..e2e_diag_reconstruction.core.moe_semantic_hif4 import NativeQwen3MoELayerRuntime
    from ..e2e_diag_reconstruction.training.moe_layer_runtime import build_qwen3_moe_layer_call
    from ..module_design_ablation.student import Student
    from ..non_equivalent_reconstruction.train import assemble, propagate, teacher_targets
    from ..non_equivalent_reconstruction.transforms import projection_weights
    from .router import RouterStudent, cache_equivalence, cache_inputs, cached_epoch
    from .training import context, samples_for
    output = Path(root) / "qualification/router_contracts"
    output.mkdir(parents=True, exist_ok=False)
    train, val, _ = samples_for(root, MODELS[0])
    train = sorted(train, key=lambda s: s.input_ids.numel(), reverse=True)[:4]
    val = sorted(val, key=lambda s: s.input_ids.numel(), reverse=True)[:4]
    samples = train + val
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    collator, device, hidden = context(samples)
    base = load_qwen3_moe_layer_state(NATIVE, 0, device)
    teacher = NativeQwen3MoELayerRuntime(base).to(device).eval()
    _, targets = teacher_targets(teacher, NATIVE, samples, collator, hidden, device, 4)
    del teacher
    parameters = load(OLD_TRAINING / "layers/000/selected.pt")["parameters"]
    student = RouterStudent(base, parameters).to(device)
    original = Student(base, "matrix_norm").to(device)
    original.learned.load_state_dict(parameters, strict=True)
    with torch.no_grad():
        x = assemble(hidden, train, device)
        call = build_qwen3_moe_layer_call(str(NATIVE), x)
        a = original(x, position_embeddings=call.position_embeddings, use_ste=False)
        b = student(x, position_embeddings=call.position_embeddings, use_ste=False)
        if not torch.equal(a.output, b.output) or not torch.equal(a.router_logits, b.router_logits):
            raise RuntimeError("zero-step actual layer mismatch")
    del original, a, b
    inputs = cache_inputs(student, samples, hidden, device)
    equality = cache_equivalence(student, train, collator, hidden, targets, inputs, device)
    frozen = state_digest({**dict(projection_weights(base)), **student.learned.state_dict()})
    initial = student.router_master.detach().clone()
    optimizer = torch.optim.AdamW([student.router_master], lr=3e-5, weight_decay=0.)
    for epoch in range(2):
        cached_epoch(student, train, collator, inputs, targets, device, epoch=epoch, optimizer=optimizer)
    if torch.equal(initial, student.router_master):
        raise RuntimeError("actual FP32 router failed to update")
    if frozen != state_digest({**dict(projection_weights(base)), **student.learned.state_dict()}):
        raise RuntimeError("frozen base/norm/transforms changed")
    validation = cached_epoch(student, val, collator, inputs, targets, device, epoch=2)
    atomic_save(student.router_master.detach().cpu(), output / "router.pt")
    if not torch.equal(load(output / "router.pt"), student.router_master.detach().cpu()):
        raise RuntimeError("router save/reload failed")
    propagated = propagate(student, NATIVE, samples, hidden, device, 4)
    if any(not torch.isfinite(v).all() for v in propagated.values()):
        raise RuntimeError("invalid full student layer propagation")
    write_json({"passed": True, "cache_equivalence": equality, "validation": validation,
                "frozen_sha256": frozen, "train_lengths": [s.input_ids.numel() for s in train],
                "validation_lengths": [s.input_ids.numel() for s in val],
                "trainable": [n for n, p in student.named_parameters() if p.requires_grad],
                "peak_memory_bytes": torch.cuda.max_memory_allocated(), "scope": "real native layer0 and existing matrix_norm state; implementation qualification only"},
               output / "metrics.json")


def inference(root, candidate, *, qualification):
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from .generate import close_engine, engine, prompt
    root = Path(root)
    output = root / ("qualification/inference" if qualification else "inference") / candidate
    output.mkdir(parents=True, exist_ok=False)
    model = NATIVE if candidate == "native_nvfp4" else root / ("qualification/models" if qualification else "models") / candidate
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    rows = [{"domain": "math", "question": "Compute 2 + 2."},
            {"domain": "code", "question": "Read two integers from standard input and print their sum."}]
    ids = [prompt(tokenizer, row) for row in rows]
    # Compress decode iterations only; retain the full engine context, KV dtype,
    # TP2 and allocation/batch limits. These outputs never become calibration.
    sampling = {**GENERATION, "max_tokens": 64}
    llm = engine(model)
    try:
        results = llm.generate([{"prompt_token_ids": tokens} for tokens in ids], SamplingParams(**sampling))
        saved = []
        for tokens, result in zip(ids, results):
            response = result.outputs[0]
            if (not response.token_ids or len(response.token_ids) > 64
                    or response.finish_reason not in {"stop", "length"}
                    or any(token < 0 or token >= len(tokenizer) for token in response.token_ids)):
                raise RuntimeError("TP2 prefill/decode produced invalid output tokens or termination")
            saved.append({"prompt_token_ids": tokens, "generated_token_ids": list(response.token_ids),
                          "text": response.text, "finish_reason": response.finish_reason})
        write_json({"passed": True, "tensor_parallel_size": 2, "sampling": sampling, "responses": saved,
                    "scope": "real prefill and up to 64 decode steps; full engine resource settings; no task-accuracy claim"}, output / "metrics.json")
    finally:
        close_engine(llm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--router-contracts", action="store_true")
    parser.add_argument("--candidate", choices=(*MODELS, "native_nvfp4"))
    parser.add_argument("--qualification", action="store_true")
    args = parser.parse_args()
    if args.router_contracts:
        router_contracts(args.run)
    else:
        inference(args.run, args.candidate, qualification=args.qualification)
