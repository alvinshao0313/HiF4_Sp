import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from HiFloat4.hif4_scale_threshold_optimization.src.quantizer import HiF4QuantConfig, quantize_hif4
from HiFloat4.hif4_scale_threshold_optimization.src.weight_search import search_weight_groups
from ..qwen3_mxfp4.format import nvfp4_to_bf16
from ..qwen3_mxfp4_dualscale.format import normalized_reference
from ..qwen3_mxfp4_uos.format import activation_reference
from .export import export
from .operands import operand
from .runtime import DualScaleLinear, PaperConfig, TargetLinear
from .scale_methods import CANDIDATES, GROUP_SIZE, METHOD_TARGET

NAME = "model.layers.0.self_attn.k_proj"
SHARD = "model-00001-of-00001.safetensors"


def projection_source(folder, name, tensors):
    # Real index and native ModelOpt scale layout accepted by the converter.
    folder.mkdir()
    values = {name + "." + k: v for k, v in tensors.items()}
    save_file(values, folder / SHARD)
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict.fromkeys(values, SHARD)}))
    (folder / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (folder / "hf_quant_config.json").write_text(json.dumps({"quantization": {"quant_algo": "NVFP4", "group_size": 16}}))


def check_projection(source, dest, name, native_weight, method, device):
    target = METHOD_TARGET[method]
    export(source, dest, target, method, None, device)
    conversion = json.loads((dest / "conversion.json").read_text())
    report = conversion["matrices"][name]
    config = PaperConfig(json.loads((dest / "config.json").read_text())["quantization_config"])
    tensors = load_file(dest / SHARD)
    saved = tensors[name + ".weight"]
    x = native_weight[[0, -1]].contiguous()
    if method in ("scale_search", "matrix_c"):
        direct = operand(x, "hif4")
        assert torch.equal(config.activation(x.to(device)).cpu(), direct)
        if method == "scale_search":
            reference = search_weight_groups(native_weight, budget="full", device="cpu").reconstruction
        else:
            errors = {}
            for c in CANDIDATES:
                candidate = quantize_hif4(native_weight, config=HiF4QuantConfig(s0_divisor=c))
                errors[c] = (native_weight.float() - candidate.reconstruction.float()).square().double().mean().item()
            assert report["candidate_mse"] == pytest.approx({str(c): err for c, err in errors.items()})
            selected = min(CANDIDATES, key=lambda c: (errors[c], abs(c - 7), c))
            assert report["selected_c"] == selected
            reference = quantize_hif4(native_weight, config=HiF4QuantConfig(s0_divisor=selected)).reconstruction
        ref_output = torch.nn.functional.linear(direct, reference)
        assert report["selected_sum_squared_error"] <= report["direct_sum_squared_error"]
    elif method == "uos":
        reference = activation_reference(native_weight)
        ref_output = torch.nn.functional.linear(activation_reference(x), reference)
    else:
        group = GROUP_SIZE[method]
        reference, sw = normalized_reference(native_weight, group)
        a, sa = normalized_reference(x, group)
        outer = tensors[name + ".weight_scale_fp32"]
        assert outer.dtype == torch.float32 and torch.equal(outer, sw)
        assert report["outer_fp32_bytes"] == sw.numel() * 4
        # Independent dot over the restored high-precision operands. The
        # production path instead scales each BF16 partial before summing.
        a = a.double() * sa.repeat_interleave(group, -1)[:, :x.shape[1]].double()
        w = reference.double() * sw.repeat_interleave(group, -1)[:, :x.shape[1]].double()
        ref_output = (a @ w.T).bfloat16()
    assert saved.dtype == torch.bfloat16 and torch.equal(saved, reference)
    layer = SimpleNamespace(weight=saved.to(device))
    if method in GROUP_SIZE:
        layer.weight_scale_fp32 = outer.to(device)
        linear = DualScaleLinear(config, name)
        linear.process_weights_after_loading(layer)
    else:
        linear = TargetLinear(config, name)
    out = linear.apply(layer, x.to(device)).cpu()
    torch.testing.assert_close(out, ref_output, rtol=0.008, atol=1.e-5)
    return {"shape": list(saved.shape), "method": method, "activation_rule": conversion["activation_rule"],
            "saved_reference_equal": True, "forward_max_abs_difference": (out.float()-ref_output.float()).abs().max().item(),
            "selected_c": report.get("selected_c"), "outer_fp32_bytes": report.get("outer_fp32_bytes")}


@pytest.mark.parametrize("method", METHOD_TARGET.keys())
def test_export_reload_forward(tmp_path, method):
    rng = torch.Generator().manual_seed(20261126)
    # Two output chunks, complete G64s/G32s and a partial last G256/G512.
    n, k = 257, 320
    values = {"weight": torch.randint(0, 256, (n, k//2), dtype=torch.uint8, generator=rng),
              "weight_scale": (torch.rand((n, k//16), generator=rng) * 8).to(torch.float8_e4m3fn),
              "weight_scale_2": torch.tensor(.015625), "input_scale": torch.tensor(.5)}
    values["weight"][0].zero_()
    source = tmp_path / "source"
    projection_source(source, NAME, values)
    w = nvfp4_to_bf16(values["weight"], values["weight_scale"], values["weight_scale_2"])
    check_projection(source, tmp_path / method, NAME, w, method, "cpu")
