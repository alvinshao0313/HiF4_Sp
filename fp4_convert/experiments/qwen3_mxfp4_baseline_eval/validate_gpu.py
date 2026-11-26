"""Validate the newly used direct TP2 local K=384 against the existing reference."""
import argparse
from pathlib import Path

import torch

from ..qwen3_mxfp4.format import activation_reference, decode_scales, dequantize
from ..qwen3_mxfp4.runtime import routed_experts
from ..qwen3_mxfp4.validate_gpu import error_record, gemm_reference, measure, native_swiglu_reference, validate_qdq
from ..qwen3_mxfp4_dualscale_tp2.model_io import load_experts
from .common import file_hash, now, read, save


def validate(root):
    root = Path(root).resolve()
    torch.set_num_threads(8)
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    tensors = {k: v.cuda() for k, v in load_experts(read(root / "manifest.json")["models"]["direct"], outer=False).items()}
    w13, w2 = tensors["w13_weight"], tensors["w2_weight"]
    s13, s2 = decode_scales(tensors["w13_weight_scale"]), decode_scales(tensors["w2_weight_scale"])
    dense13 = dequantize(w13, tensors["w13_weight_scale"])
    dense2 = dequantize(w2, tensors["w2_weight_scale"])
    ones = torch.ones(128, device="cuda")
    report = {"status": "running", "at": now(), "scope": "Real layer0 TP2 rank0, all128 experts; TP communication checked by full-model probes",
              "manifest_sha256": file_hash(root / "manifest.json"), "cases": []}
    try:
        for count in (1, 256, 8192):
            x = torch.randn(count, 2048, device="cuda", dtype=torch.bfloat16)
            selected = ((torch.arange(count, device="cuda")[:, None] * 7 + torch.arange(8, device="cuda")) % 128).int().contiguous()
            weights = torch.rand(count, 8, device="cuda")
            weights /= weights.sum(-1, keepdim=True)
            def apply():
                return routed_experts(x, w13, w2, s13, s2, ones, weights, selected, "direct")
            actual, states = routed_experts(x, w13, w2, s13, s2, ones, weights, selected, "direct", diagnostics=True)
            first = gemm_reference(activation_reference(x, "direct"), dense13, selected)
            activated = native_swiglu_reference(first)
            contributions = gemm_reference(activation_reference(activated, "direct"), dense2, selected, weights)
            expected = contributions.float().sum(1).bfloat16()
            case = {"tokens": count, "gate_up_qdq": validate_qdq(x, "direct"),
                    "down_qdq": validate_qdq(states["activated"], "direct"),
                    "first_gemm": error_record(states["first"], first),
                    "swiglu": error_record(states["activated"], activated),
                    "weighted_second_gemm": error_record(states["second"], contributions),
                    "moe": error_record(actual, expected, contributions.float().abs().sum(1)),
                    "performance": measure(apply, count, warmup=3, repeats=10)}
            report["cases"].append(case)
            save(root / "direct_tp2_operators.json", report)
            print(f"Direct TP2 tokens={count}: numerical checks passed", flush=True)
            del states, actual, first, activated, contributions, expected
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save(root / "direct_tp2_operators.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    validate(parser.parse_args().root)
