"""Actual TP4 expert shapes: numerical checks and software-emulation timings."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import subprocess

import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType, nvfp4_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import Nvfp4QuantizationEmulationTritonExperts
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import ref_nvfp4_quant_dequant

from .artifacts import code_hashes, save
from .convert import SOURCE
from .format import activation_reference, decode_scales, dequantize, quantize, rotate, unpack_codes
from .fht32 import fht32_reference
from .kernels import activation_qdq
from .model_io import load_experts
from .runtime import routed_experts


def device_state():
    return subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu", "--format=csv"],
                          check=True, text=True, capture_output=True).stdout


def measure(fn, tokens, warmup=5, repeats=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated()
    durations = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        durations.append(start.elapsed_time(end))
        del result
    return {"median_ms": statistics.median(durations), "min_ms": min(durations), "max_ms": max(durations),
            "p10_ms": sorted(durations)[repeats//10], "p90_ms": sorted(durations)[repeats*9//10],
            "tokens_per_second": tokens * 1000 / statistics.median(durations),
            "resident_allocated_bytes": resident, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "incremental_peak_bytes": torch.cuda.max_memory_allocated() - resident,
            "samples_ms": durations, "warmup": warmup, "repeats": repeats}


def error_record(actual, expected, amplitude=None):
    delta = actual.float() - expected.float()
    denominator = expected.float().square().sum().clamp_min(1e-30)
    nmse = (delta.square().sum() / denominator).item()
    amplitude = expected.float().abs() if amplitude is None else amplitude
    bound = 4 * torch.finfo(torch.bfloat16).eps * amplitude + 1e-5
    scaled = (delta.abs() / bound).amax().item()
    if not torch.isfinite(actual).all() or nmse >= 1e-4 or scaled > 1:
        raise AssertionError(f"BF16 validation failed: NMSE={nmse}; amplitude-bound ratio={scaled}")
    return {"nmse": nmse, "max_abs": delta.abs().max().item(), "max_error_over_amplitude_bound": scaled}


def validate_qdq(x, variant):
    got, codes, scales, transformed = activation_qdq(x, variant, diagnostics=True)
    reference_transformed = fht32_reference(x) if variant == "fht32" else rotate(x, variant)
    rotation_delta = transformed - reference_transformed
    torch.testing.assert_close(transformed, reference_transformed, rtol=2e-6, atol=2e-6)
    # Quantization is checked EXACTLY at the kernel's FP32 rotation boundary;
    # matrix-multiply vs Hadamard accumulation differences are measured above.
    packed, expected_scales = quantize(transformed)
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
    torch.testing.assert_close(codes, unpack_codes(packed), rtol=0, atol=0)
    torch.testing.assert_close(got, dequantize(packed, expected_scales), rtol=0, atol=0)
    return {"shape": list(x.shape), "scale_code_reconstruction_exact": True,
            "rotation_max_abs": rotation_delta.abs().max().item(),
            "rotation_nmse": (rotation_delta.square().sum() / reference_transformed.square().sum().clamp_min(1e-30)).item()}


def gemm_reference(a, dense, selected, weights=None):
    """FP32 accumulation of BF16 operands; route multiply before BF16 cast."""
    rows, topk = selected.shape
    n = dense.shape[1]
    output = torch.empty((rows, topk, n), dtype=torch.bfloat16, device=a.device)
    for eid in range(dense.shape[0]):
        row, slot = torch.where(selected == eid)
        if row.numel() == 0:
            continue
        operands = a[row] if weights is None else a[row * topk + slot]
        accum = operands.float() @ dense[eid].float().T
        if weights is not None:
            accum *= weights[row, slot, None]
        output[row, slot] = accum.bfloat16()
    return output


def native_swiglu_reference(first):
    # vLLM activation_kernels.cu casts silu_kernel/packed_silu_kernel to
    # scalar_t/packed_t BEFORE multiplication by up. Preserve both BF16 rounds.
    gate, up = first.view(-1, first.shape[-1]).chunk(2, -1)
    return torch.nn.functional.silu(gate.float()).bfloat16() * up


def native_callable(tensors, count):
    # Exercise the existing native expert class itself, including its workspace
    # reuse and native NVFP4 QDQ. No synthetic stand-in for the baseline kernel.
    parallel = replace(FusedMoEParallelConfig.make_no_parallel(), tp_size=4, tp_rank=0)
    moe = FusedMoEConfig(num_experts=128, experts_per_token=8, hidden_dim=2048,
        intermediate_size_per_partition=192, num_local_experts=128, num_logical_experts=128,
        activation=MoEActivation.SILU, device="cuda", routing_method=RoutingMethodType.Default,
        moe_parallel_config=parallel, in_dtype=torch.bfloat16)
    quant = nvfp4_moe_quant_config(g1_alphas=tensors["g1"], g2_alphas=tensors["g2"],
        a1_gscale=tensors["a1"], a2_gscale=tensors["a2"],
        w1_scale=tensors["w13_weight_scale"].float(), w2_scale=tensors["w2_weight_scale"].float(),
        is_nvfp4_scale_swizzled=False)
    expert = Nvfp4QuantizationEmulationTritonExperts(moe, quant)
    output = torch.empty(count, 2048, dtype=torch.bfloat16, device="cuda")
    workspace13 = torch.empty(count * 8 * 192, dtype=torch.bfloat16, device="cuda")
    workspace2 = torch.empty(count * 8 * 2048, dtype=torch.bfloat16, device="cuda")
    def apply(x, weights, selected):
        expert.apply(output, x, tensors["w13_weight"], tensors["w2_weight"], weights, selected,
                     MoEActivation.SILU, 128, None, None, None, workspace13, workspace2, None, False)
        return output
    return apply


def validate(model, output):
    torch.set_num_threads(8)
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    model, output = Path(model), Path(output)
    manifest = json.loads((model / "conversion.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Conversion is not verified")
    variant = manifest["variant"]
    result = {"status": "running", "variant": variant, "model": str(model), "seed": 1234,
              "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
              "gpu_state_before": device_state(), "source_sha256": code_hashes(), "qdq": [], "moe": [], "performance": [],
              "scope": "real layer-0 TP4 rank-0 slices, all 128 experts; software QDQ/BF16 GEMM"}
    save(output, result)
    try:
        tensors = {key: value.cuda() for key, value in load_experts(model).items()}
        native = {key: value.cuda() for key, value in load_experts(SOURCE, native=True).items()}
        w13, w2 = tensors["w13_weight"], tensors["w2_weight"]
        s13, s2 = decode_scales(tensors["w13_weight_scale"]), decode_scales(tensors["w2_weight_scale"])
        dense13, dense2 = dequantize(w13, tensors["w13_weight_scale"]), dequantize(w2, tensors["w2_weight_scale"])
        ones = torch.ones(128, device="cuda")
        edges = torch.zeros(8, 64, device="cuda", dtype=torch.bfloat16)
        edges[0, :16] = torch.tensor([7., .25, .75, 1.25, 1.75, 2.5, 3.5, 5., -7., -.25, -.75, -1.25, -1.75, -2.5, -3.5, -5.], device="cuda")
        for row, exponent in enumerate((-127, -126, -80, -1, 0, 30), 1):
            edges[row] = 2. ** exponent
        edges[-1] = -0.
        result["qdq"].append(validate_qdq(edges, "direct"))
        if variant == "fht32":
            # Basis vectors catch ordering, sign, normalization and K32 boundaries.
            result["qdq"].append(validate_qdq(torch.eye(32, device="cuda", dtype=torch.bfloat16), variant))
            result["qdq"].append(validate_qdq(torch.ones(2, 192, device="cuda", dtype=torch.bfloat16), variant))
        for count in (1, 256, 8192):
            x = torch.randn(count, 2048, device="cuda", dtype=torch.bfloat16)
            if count > 1:
                x[0].zero_()
            # Every row has eight distinct routes; all experts are exercised at
            # batch/prefill scale. Routes and weights are fixed across timings.
            selected = ((torch.arange(count, device="cuda")[:, None] * 7 + torch.arange(8, device="cuda")) % 128).int().contiguous()
            weights = torch.rand(count, 8, device="cuda")
            weights /= weights.sum(-1, keepdim=True)
            actual, states = routed_experts(x, w13, w2, s13, s2, ones, weights, selected, variant, diagnostics=True)
            result["qdq"].append(validate_qdq(x, variant))
            result["qdq"].append(validate_qdq(states["activated"], variant))
            # Isolate packed GEMM correctness from activation quantization error.
            first_ref = gemm_reference(states["a1"], dense13, selected)
            first_error = error_record(states["first"], first_ref)
            second_ref = gemm_reference(states["a2"], dense2, selected, weights)
            second_error = error_record(states["second"], second_ref)
            a1_ref = activation_reference(x, variant)
            full_first = gemm_reference(a1_ref, dense13, selected)
            activation_error = error_record(states["activated"], native_swiglu_reference(states["first"]))
            activated_ref = native_swiglu_reference(full_first)
            a2_ref = activation_reference(activated_ref, variant)
            contributions = gemm_reference(a2_ref, dense2, selected, weights)
            expected = contributions.float().sum(1).bfloat16()
            full_error = error_record(actual, expected, contributions.float().abs().sum(1))
            result["moe"].append({"tokens": count, "first_gemm": first_error, "native_swiglu": activation_error,
                                  "weighted_second_gemm": second_error, "end_to_end": full_error})
            baseline = native_callable(native, count)
            qdq_cases = []
            for label, a, global_scale in (("gate_up", x, native["a1"]), ("down", states["activated"], native["a2"])):
                cases = [("native_nvfp4", lambda a=a, g=global_scale: ref_nvfp4_quant_dequant(a, g, 16)),
                                 ("direct_mxfp4", lambda a=a: activation_qdq(a, "direct")),
                                 ("r64_mxfp4", lambda a=a: activation_qdq(a, "r64"))]
                if variant == "fht32":
                    cases.append(("fht32_mxfp4", lambda a=a: activation_qdq(a, "fht32")))
                for name, fn in cases:
                    qdq_cases.append({"boundary": label, "operator": name, **measure(fn, a.shape[0])})
            result["performance"].append({"tokens": count, "qdq": qdq_cases,
                "native_moe": measure(lambda: baseline(x, weights, selected), count),
                "candidate_moe": measure(lambda: routed_experts(x, w13, w2, s13, s2, ones, weights, selected, variant), count)})
            save(output, result)
            print(f"{variant}: validated and timed tokens={count}", flush=True)
            del states, baseline, actual, contributions, expected, full_first, first_ref, second_ref
        result.update(status="passed", gpu_state_after=device_state())
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save(output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    validate(args.model, args.output)
