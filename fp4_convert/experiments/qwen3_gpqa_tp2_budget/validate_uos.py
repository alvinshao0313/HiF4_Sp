"""Real two-rank UOS expert slicing, QDQ, BF16 GEMM and SUM validation."""
import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist

from ..qwen3_mxfp4.artifacts import save
from ..qwen3_mxfp4.validate_gpu import error_record, gemm_reference, measure, native_swiglu_reference
from ..qwen3_mxfp4_dualscale_tp2.model_io import load_experts
from ..qwen3_mxfp4_dualscale_tp2.validate_communication import setup
from ..qwen3_mxfp4_uos.format import VARIANT, activation_reference, decode_scales, dequantize
from ..qwen3_mxfp4_uos.runtime import routed_experts
from ..qwen3_mxfp4_uos.validate_gpu import validate_qdq


def validate(model, output):
    rank = setup()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    report = {"status": "running", "rank": rank, "variant": VARIANT, "tests": [],
              "implementation_change": "none; existing UOS runtime supports K=384", "tp_size": 2}
    try:
        tensors = {k: v.cuda() for k, v in load_experts(model, rank=rank, outer=False).items()}
        w13, w2 = tensors["w13_weight"], tensors["w2_weight"]
        s13, s2 = decode_scales(tensors["w13_weight_scale"]), decode_scales(tensors["w2_weight_scale"])
        d13, d2 = dequantize(w13, tensors["w13_weight_scale"]), dequantize(w2, tensors["w2_weight_scale"])
        ones = torch.ones(128, device="cuda")
        for count in (1, 256, 8192):
            torch.manual_seed(1234)
            x = torch.randn(count, 2048, device="cuda", dtype=torch.bfloat16)
            selected = ((torch.arange(count, device="cuda")[:, None] * 7 + torch.arange(8, device="cuda")) % 128).int().contiguous()
            weights = torch.softmax(torch.randn(count, 8, device="cuda"), -1)
            for value in (x, selected, weights):
                dist.broadcast(value, src=0)
            actual, states = routed_experts(x, w13, w2, s13, s2, ones, weights, selected, VARIANT, diagnostics=True)
            qdq = [validate_qdq(x, VARIANT), validate_qdq(states["activated"], VARIANT)]
            first = gemm_reference(states["a1"], d13, selected)
            second = gemm_reference(states["a2"], d2, selected, weights)
            first_error, second_error = error_record(states["first"], first), error_record(states["second"], second)
            swiglu = error_record(states["activated"], native_swiglu_reference(states["first"]))
            contributions = gemm_reference(activation_reference(native_swiglu_reference(first), VARIANT), d2, selected, weights)
            expected, amplitude = contributions.float().sum(1).bfloat16(), contributions.float().abs().sum(1)
            local_error = error_record(actual, expected, amplitude)
            for value in (actual, expected, amplitude):
                dist.all_reduce(value)
            sum_error = error_record(actual, expected, amplitude)
            report["tests"].append({"tokens": count, "qdq": qdq, "first_gemm": first_error,
                                    "second_gemm": second_error, "swiglu": swiglu,
                                    "local_moe": local_error, "tp_sum": sum_error,
                                    "performance": measure(lambda: routed_experts(x, w13, w2, s13, s2, ones,
                                                       weights, selected, VARIANT), count)})
            print(f"rank={rank} UOS TP2 tokens={count} passed", flush=True)
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save(Path(output).with_name(f"uos_operators_rank{rank}.json"), report)
    reports = [None, None] if rank == 0 else None
    dist.gather_object(report, reports, dst=0)
    if rank == 0:
        save(output, {"status": "passed", "ranks": reports})
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    validate(args.model, args.output)
