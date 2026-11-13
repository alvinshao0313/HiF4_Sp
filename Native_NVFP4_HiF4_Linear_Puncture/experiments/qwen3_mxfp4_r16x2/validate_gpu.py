"""R16x2 numerical and performance checks using the unchanged TP4 reference protocol."""
import argparse
import json
from pathlib import Path

import torch
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import ref_nvfp4_quant_dequant
from ..qwen3_mxfp4.validate_gpu import (device_state, measure, error_record, gemm_reference,
    native_swiglu_reference, native_callable, validate_qdq as base_validate_qdq)
from ..qwen3_mxfp4.kernels import activation_qdq as base_activation_qdq
from ..qwen3_mxfp4.model_io import load_experts
from .artifacts import code_hashes, save
from .convert import SOURCE
from .format import activation_reference, decode_scales, dequantize, quantize, unpack_codes
from .transform import r16x2_reference
from .kernels import activation_qdq
from .runtime import routed_experts


def validate_qdq(x, variant):
    got, codes, scales, transformed = activation_qdq(x, variant, diagnostics=True)
    reference_transformed = r16x2_reference(x)
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
        result["qdq"].append(base_validate_qdq(edges, "direct"))
        if variant == "r16x2":
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
                                 ("direct_mxfp4", lambda a=a: base_activation_qdq(a, "direct")),
                                 ("r64_mxfp4", lambda a=a: base_activation_qdq(a, "r64"))]
                cases.append(("fht32_mxfp4", lambda a=a: base_activation_qdq(a, "fht32")))
                if variant == "r16x2":
                    cases.append(("r16x2_mxfp4", lambda a=a: activation_qdq(a, "r16x2")))
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
