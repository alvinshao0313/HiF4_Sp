"""UOS c7.25 numerical and performance checks using the unchanged TP4 reference protocol."""
import argparse
import json
from pathlib import Path

import torch
from ..qwen3_mxfp4.validate_gpu import (device_state, measure, error_record, gemm_reference,
    native_swiglu_reference)
from ..qwen3_mxfp4.kernels import activation_qdq as base_activation_qdq
from ..qwen3_mxfp4.model_io import load_experts
from ..qwen3_mxfp4.runtime import routed_experts as base_routed_experts
from .artifacts import code_hashes, save
from .convert import SOURCE
from .format import activation_reference, decode_scales, dequantize, quantize, unpack_codes
from .test_format import threshold_values, scalar_reference
from .format import VARIANT
from .kernels import activation_qdq, activation_kernel
from .runtime import routed_experts


def validate_qdq(x, variant):
    got, codes, scales, transformed = activation_qdq(x, variant, diagnostics=True)
    # Use the CPU implementation whose scale and E2M1 encoding were checked
    # against scalar threshold search and nearest-distance RNE reference.
    packed, expected_scales = quantize(x.cpu())
    torch.testing.assert_close(transformed, x.float(), rtol=0, atol=0)
    torch.testing.assert_close(scales.cpu(), expected_scales, rtol=0, atol=0)
    torch.testing.assert_close(codes.cpu(), unpack_codes(packed), rtol=0, atol=0)
    torch.testing.assert_close(got.cpu(), dequantize(packed, expected_scales), rtol=0, atol=0)
    return {"shape": list(x.shape), "scale_code_reconstruction_exact": True}


def validate_fp32_boundaries():
    # Exercise the actual kernel with FP32 pointers to cover immediate FP32
    # neighbors; the formal runtime still requires BF16 activations.
    maxima = torch.cat((torch.tensor([0., 2.**-149, torch.finfo(torch.float32).max]), threshold_values()))
    x = torch.zeros(len(maxima), 32)
    x[:, 0] = maxima
    x[:, 1] = -maxima
    codes_ref, scales_ref = scalar_reference(x)
    packed, scales_cpu = quantize(x)
    torch.testing.assert_close(scales_cpu, scales_ref, rtol=0, atol=0)
    x = x.cuda()
    output = torch.empty_like(x, dtype=torch.bfloat16)
    codes = torch.empty_like(x, dtype=torch.uint8)
    scales = torch.empty(len(maxima), 1, dtype=torch.uint8, device="cuda")
    transformed = torch.empty_like(x)
    activation_kernel[(len(maxima), 1)](x, output, codes, scales, transformed, 32, True, BLOCK=32,
                                       num_warps=1, enable_fp_fusion=False)
    torch.testing.assert_close(scales.cpu(), scales_ref, rtol=0, atol=0)
    torch.testing.assert_close(codes.cpu(), codes_ref, rtol=0, atol=0)
    torch.testing.assert_close(output.cpu(), dequantize(packed, scales_cpu), rtol=0, atol=0)
    return {"groups": len(maxima), "scale_code_reconstruction_exact": True, "input_dtype": "float32 diagnostic only"}


def validate(model, output, direct_model):
    torch.set_num_threads(8)
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    model, output = Path(model), Path(output)
    manifest = json.loads((model / "conversion.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Conversion is not verified")
    variant = manifest["variant"]
    if variant != VARIANT:
        raise ValueError("Expected the UOS c7.25 checkpoint")
    result = {"status": "running", "variant": variant, "model": str(model), "seed": 1234,
              "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
              "gpu_state_before": device_state(), "source_sha256": code_hashes(), "qdq": [], "moe": [], "performance": [],
              "direct_model": str(direct_model),
              "scope": "real layer-0 TP4 rank-0 slices, all 128 experts; software QDQ/BF16 GEMM"}
    save(output, result)
    try:
        tensors = {key: value.cuda() for key, value in load_experts(model).items()}
        direct_manifest = json.loads((Path(direct_model) / "conversion.json").read_text())
        if (direct_manifest["status"] != "completed" or direct_manifest["variant"] != "direct"
                or manifest["source_shards"] != direct_manifest["source_shards"]):
            raise ValueError("Direct performance baseline must use the same original source")
        old = {key: value.cuda() for key, value in load_experts(direct_model).items()}
        old_s13, old_s2 = decode_scales(old["w13_weight_scale"]), decode_scales(old["w2_weight_scale"])
        result["fp32_boundaries"] = validate_fp32_boundaries()
        w13, w2 = tensors["w13_weight"], tensors["w2_weight"]
        s13, s2 = decode_scales(tensors["w13_weight_scale"]), decode_scales(tensors["w2_weight_scale"])
        dense13, dense2 = dequantize(w13, tensors["w13_weight_scale"]), dequantize(w2, tensors["w2_weight_scale"])
        ones = torch.ones(128, device="cuda")
        edges = torch.zeros(8, 64, device="cuda", dtype=torch.bfloat16)
        edges[0, :16] = torch.tensor([7., .25, .75, 1.25, 1.75, 2.5, 3.5, 5., -7., -.25, -.75, -1.25, -1.75, -2.5, -3.5, -5.], device="cuda")
        for row, exponent in enumerate((-127, -126, -80, -1, 0, 30), 1):
            edges[row] = 2. ** exponent
        edges[-1] = -0.
        result["qdq"].append(validate_qdq(edges, variant))
        boundary = torch.tensor(7.25, dtype=torch.bfloat16)
        neighbors = torch.stack((torch.nextafter(boundary, torch.tensor(0., dtype=torch.bfloat16)), boundary,
                                 torch.nextafter(boundary, torch.tensor(float("inf"), dtype=torch.bfloat16))))
        result["qdq"].append(validate_qdq(neighbors[:, None].expand(-1, 32).contiguous().cuda(), variant))
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
            qdq_cases = []
            for label, a in (("gate_up", x), ("down", states["activated"])):
                cases = [("direct_mxfp4", lambda a=a: base_activation_qdq(a, "direct")),
                         ("direct_uos_c7p25", lambda a=a: activation_qdq(a, variant))]
                for name, fn in cases:
                    qdq_cases.append({"boundary": label, "operator": name, **measure(fn, a.shape[0])})
            result["performance"].append({"tokens": count, "qdq": qdq_cases,
                "direct_moe": measure(lambda: base_routed_experts(x, old["w13_weight"], old["w2_weight"],
                    old_s13, old_s2, ones, weights, selected, "direct"), count),
                "candidate_moe": measure(lambda: routed_experts(x, w13, w2, s13, s2, ones, weights, selected, variant), count)})
            save(output, result)
            print(f"{variant}: validated and timed tokens={count}", flush=True)
            del states, actual, contributions, expected, full_first, first_ref, second_ref
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
    parser.add_argument("--direct-model", required=True)
    args = parser.parse_args()
    validate(args.model, args.output, args.direct_model)
