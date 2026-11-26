"""Real TP2 dual-scale GEMM/MoE checks and identical-scope timing comparisons."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist
from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import moe_align_block_size, try_get_optimal_moe_config

from .artifacts import code_hashes, save
from ..qwen3_mxfp4_dualscale.format import group_size, decode_scales, mx_dequantize, normalized_reference
from .kernels import activation_qdq, packed_gemm
from .model_io import load_experts
from .runtime import routed_experts
from .validate_communication import setup
from ..qwen3_mxfp4.kernels import activation_qdq as direct_qdq
from functools import partial
load_direct = partial(load_experts, outer=False)
from ..qwen3_mxfp4.runtime import routed_experts as direct_moe
from ..qwen3_mxfp4.validate_gpu import device_state, error_record, measure, native_swiglu_reference


def gather_k(local):
    pieces = [torch.empty_like(local) for _ in range(2)]
    dist.all_gather(pieces, local.contiguous())
    return torch.cat(pieces, -1)


def reference_gemm(a, dense, ta, tw, selected, group, offset=0, weights=None):
    """Independent FP32 reference of BF16 operands, never the inference path."""
    rows, topk = selected.shape
    result = torch.empty((rows, topk, dense.shape[1]), dtype=torch.bfloat16, device=a.device)
    amplitude = torch.empty_like(result, dtype=torch.float32)
    k = a.shape[-1]
    for eid in range(dense.shape[0]):
        row, slot = torch.where(selected == eid)
        if row.numel() == 0:
            continue
        ids = row if weights is None else row * topk + slot
        values = torch.zeros((row.numel(), dense.shape[1]), device=a.device)
        magnitude = torch.zeros_like(values)
        for g in range(offset // group, (offset + k + group - 1) // group):
            begin, end = max(g * group - offset, 0), min((g + 1) * group - offset, k)
            left, right = a[ids, begin:end].float(), dense[eid, :, begin:end].float()
            scale = ta[ids, g, None] * tw[eid, :, g][None, :]
            values = values + (left @ right.T) * scale
            magnitude = magnitude + (left.abs() @ right.abs().T) * scale.abs()
        if weights is not None:
            values *= weights[row, slot, None]
            magnitude *= weights[row, slot, None].abs()
        result[row, slot], amplitude[row, slot] = values.bfloat16(), magnitude
    return result, amplitude


def global_down_reference(activated, group, rank):
    # Gathering is validation-only; production communicates maxima only.
    normalized, outer = normalized_reference(gather_k(activated).cpu(), group)
    return normalized[:, rank*384:(rank+1)*384].contiguous().cuda(), outer.cuda()


def mma_record(kernel):
    ptx = kernel.asm["ptx"]
    instructions = sorted({line.strip() for line in ptx.splitlines() if "mma.sync" in line})
    if not instructions or any(".bf16.bf16." not in line or ".tf32." in line for line in instructions):
        raise AssertionError(f"Expected BF16 Tensor Core MMA, found {instructions}")
    return {"ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(), "mma_instructions": instructions}


def with_sum(fn):
    output = fn()
    dist.all_reduce(output, op=dist.ReduceOp.SUM)
    return output


def validate(model, direct_model, output):
    rank = setup()
    torch.backends.cuda.matmul.allow_tf32 = False
    variant = json.loads((Path(model) / "conversion.json").read_text())["variant"]
    group = group_size(variant)
    report = {"status": "running", "variant": variant, "rank": rank, "source_sha256": code_hashes(),
              "moe": [], "performance": [], "gemm_precision": "BF16 operands, FP32 accumulation and outer-scale rescaling",
              "reference_precision": "Independent FP32 matmul of BF16 operands, TF32 disabled"}
    if rank == 0:
        report["gpu_state_before"] = device_state()
    try:
        tensors = {name: value.cuda() for name, value in load_experts(model, rank=rank).items()}
        w13, w2 = tensors["w13_weight"], tensors["w2_weight"]
        s13, s2 = decode_scales(tensors["w13_weight_scale"]), decode_scales(tensors["w2_weight_scale"])
        t13, t2 = tensors["w13_weight_scale_fp32"], tensors["w2_weight_scale_fp32"]
        dense13 = mx_dequantize(w13, tensors["w13_weight_scale"])
        dense2 = mx_dequantize(w2, tensors["w2_weight_scale"])
        old = {name: value.cuda() for name, value in load_direct(direct_model, rank=rank).items()}
        old_s13, old_s2 = decode_scales(old["w13_weight_scale"]), decode_scales(old["w2_weight_scale"])
        ones = torch.ones(128, device="cuda")
        for count in (1, 256, 8384):
            torch.manual_seed(1234)
            x = torch.randn(count, 2048, dtype=torch.bfloat16, device="cuda") * .2
            selected = torch.rand(count, 128, device="cuda").topk(8, -1).indices.to(torch.int32)
            weights = torch.softmax(torch.randn(count, 8, device="cuda"), -1)
            for item in (x, selected, weights):
                copies = [torch.empty_like(item) for _ in range(2)]
                dist.all_gather(copies, item)
                if any(not torch.equal(item, copy) for copy in copies):
                    raise AssertionError("TP ranks must share identical token and route ordering")
            del copies
            def candidate():
                return routed_experts(x, w13, w2, s13, s2, t13, t2, weights, selected, variant,
                                      rank=rank, process_group=dist.group.WORLD)
            def baseline():
                return direct_moe(x, old["w13_weight"], old["w2_weight"], old_s13, old_s2,
                                  ones, weights, selected, "direct")
            actual, states = routed_experts(x, w13, w2, s13, s2, t13, t2, weights, selected, variant,
                                            diagnostics=True, rank=rank, process_group=dist.group.WORLD)
            compiled = [mma_record(kernel) for kernel in states["compiled_kernels"]]
            # CPU reference preserves correctly rounded FP32 division by six;
            # PyTorch CUDA scalar division may multiply by a rounded reciprocal.
            a1, ta1 = (value.cuda() for value in normalized_reference(x.cpu(), group))
            torch.testing.assert_close(states["a1"], a1, rtol=0, atol=0)
            torch.testing.assert_close(states["ta1"], ta1, rtol=0, atol=0)
            first, amplitude = reference_gemm(a1, dense13, ta1, t13, selected, group)
            first_error = error_record(states["first"], first, amplitude)
            activation_error = error_record(states["activated"], native_swiglu_reference(states["first"]))
            a2, ta2 = global_down_reference(states["activated"], group, rank)
            torch.testing.assert_close(states["a2"], a2, rtol=0, atol=0)
            torch.testing.assert_close(states["ta2"], ta2, rtol=0, atol=0)
            second, amplitude2 = reference_gemm(a2, dense2, ta2, t2, selected, group, rank*384, weights)
            second_error = error_record(states["second"], second, amplitude2)
            a2_full, ta2_full = global_down_reference(native_swiglu_reference(first), group, rank)
            contributions, _ = reference_gemm(a2_full, dense2, ta2_full, t2, selected, group, rank*384, weights)
            expected = contributions.float().sum(1).bfloat16()
            contribution_amplitude = contributions.float().abs().sum(1)
            full_error = error_record(actual, expected, contribution_amplitude)
            dist.all_reduce(actual, op=dist.ReduceOp.SUM)
            dist.all_reduce(expected, op=dist.ReduceOp.SUM)
            dist.all_reduce(contribution_amplitude, op=dist.ReduceOp.SUM)
            sum_error = error_record(actual, expected, contribution_amplitude)
            report["moe"].append({"tokens": count, "first_gemm": first_error, "native_swiglu": activation_error,
                "weighted_second_gemm": second_error, "end_to_end_local": full_error, "tp_sum": sum_error,
                "gate_down_qdq_exact": True, "route_order_identical": True, "compiled_kernels": compiled})
            config = try_get_optimal_moe_config(w13.shape, w2.shape, 8, None, count, block_shape=None)
            ids, experts, padded = moe_align_block_size(selected, config["BLOCK_SIZE_M"], 128, None)
            qdq, gemms = [], []
            for boundary, operands, options in (("gate_up", x, {}), ("down", states["activated"],
                    {"global_k": 768, "offset": rank*384, "process_group": dist.group.WORLD})):
                for label, fn in (("direct", lambda a=operands: direct_qdq(a, "direct")),
                        (variant, lambda a=operands, kw=options: activation_qdq(a, group, **kw))):
                    dist.barrier()
                    qdq.append({"boundary": boundary, "operator": label, **measure(fn, operands.shape[0])})
            for boundary, fn in (("gate_up", lambda: packed_gemm(states["a1"], w13, s13, states["ta1"], t13,
                    states["first"], ids, experts, padded, 8, None, False, config, group)),
                    ("down", lambda: packed_gemm(states["a2"], w2, s2, states["ta2"], t2,
                    states["second"], ids, experts, padded, 1, weights, True, config, group, rank*384))):
                dist.barrier()
                gemms.append({"boundary": boundary, **measure(fn, count)})
            maximum = torch.zeros((count*8, (768+group-1)//group), device="cuda")
            dist.barrier()
            max_time = measure(lambda: dist.all_reduce(maximum, op=dist.ReduceOp.MAX), count)
            timings = {}
            for label, fn in (("direct_local", baseline), ("candidate_local", candidate),
                              ("direct_with_output_sum", lambda: with_sum(baseline)),
                              ("candidate_with_output_sum", lambda: with_sum(candidate))):
                dist.barrier()
                timings[label] = measure(fn, count)
            report["performance"].append({"tokens": count, "qdq": qdq, "gemm": gemms,
                "max_collective": max_time, "max_buffer_bytes": maximum.numel()*4, "moe": timings,
                "timing_scope": "local includes candidate MAX; with_output_sum also includes original TP SUM"})
            save(Path(output).with_name(Path(output).stem + f"_rank{rank}.json"), report)
            if rank == 0:
                print(f"{variant}: TP2 numerical/MMA/performance passed tokens={count}", flush=True)
            del states, actual, first, second, contributions, amplitude, amplitude2, expected
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save(Path(output).with_name(Path(output).stem + f"_rank{rank}.json"), report)
    gathered = [None]*2 if rank == 0 else None
    dist.gather_object(report, gathered, dst=0)
    if rank == 0:
        save(output, {"status": "passed", "variant": variant, "source_sha256": code_hashes(),
                      "ranks": gathered, "gpu_state_after": device_state()})
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--direct-model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    validate(args.model, args.direct_model, args.output)
