#!/usr/bin/env python3
"""Task 10: Dense + MoE numerical puncture on a real NVFP4 checkpoint layer.

Compares EmulationNvFp4LinearKernel / Nvfp4QuantizationEmulationTritonExperts
against upstream-style reference math using v0.27.0 test tolerances.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[2]
VLLM_ROOT = REPO_ROOT / "3rdparty" / "vllm"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(VLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VLLM_ROOT))

DEFAULT_CKPT = (
    Path.home()
    / ".cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4"
    / "snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3"
)
DEFAULT_REPORT = (
    REPO_ROOT / "NVFP4/reports/vllm_v027_nvfp4_backport/qwen3_30b_puncture.md"
)

# Upstream test_nvfp4_emulation.py tolerances.
DENSE_ATOL = 0.0
DENSE_RTOL = 0.0
MOE_ATOL = 0.05
MOE_RTOL = 0.01
SEED = 0


def _load_keys(ckpt: Path, keys: list[str]) -> dict[str, torch.Tensor]:
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    by_file: dict[str, list[str]] = {}
    for k in keys:
        by_file.setdefault(weight_map[k], []).append(k)
    out: dict[str, torch.Tensor] = {}
    for shard, ks in by_file.items():
        with safe_open(str(ckpt / shard), framework="pt", device="cpu") as sf:
            for k in ks:
                out[k] = sf.get_tensor(k)
    return out


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a64 = a.detach().float().reshape(-1)
    b64 = b.detach().float().reshape(-1)
    diff = (a64 - b64).abs()
    denom = (a64 * a64).sum().clamp_min(1e-12)
    nmse = float(((a64 - b64).pow(2).sum() / denom).item())
    rel_l2 = float(torch.linalg.vector_norm(a64 - b64).item() / torch.linalg.vector_norm(a64).clamp_min(1e-12).item())
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rel_l2": rel_l2,
        "nmse": nmse,
    }


def _assert_close(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> bool:
    try:
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        return True
    except AssertionError:
        return False


def _pick_dense_prefix(ckpt: Path) -> str:
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())
    keys = set(index["weight_map"])
    suffixes = ("weight", "weight_scale", "weight_scale_2", "input_scale")
    cands: list[str] = []
    for k in keys:
        if not k.endswith(".weight") or ".mlp.experts." in k or ".mlp.gate." in k:
            continue
        prefix = k[: -len(".weight")]
        if all(f"{prefix}.{s}" in keys for s in suffixes):
            cands.append(prefix)

    def sort_key(p: str) -> tuple:
        parts = p.split(".")
        layer = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 10**9
        prefer = 0 if "self_attn.q_proj" in p else (1 if "self_attn" in p else 2)
        return (layer, prefer, p)

    if not cands:
        raise RuntimeError("no complete dense NVFP4 layer in index")
    return sorted(cands, key=sort_key)[0]


def _pick_moe_layer_experts(ckpt: Path, num_experts: int) -> tuple[int, list[int]]:
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())
    keys = set(index["weight_map"])
    for layer in range(0, 256):
        experts: list[int] = []
        for e in range(0, 512):
            base = f"model.layers.{layer}.mlp.experts.{e}"
            ok = True
            for proj in ("gate_proj", "up_proj", "down_proj"):
                for s in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
                    if f"{base}.{proj}.{s}" not in keys:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                experts.append(e)
            if len(experts) >= num_experts:
                return layer, experts
        if experts:
            # incomplete count — keep searching
            continue
    raise RuntimeError("no MoE layer with enough complete experts")


def run_dense(ckpt: Path, device: str) -> dict[str, Any]:
    from torch.nn.parameter import Parameter

    from vllm.config import set_current_vllm_config, VllmConfig
    from vllm.config.kernel import KernelConfig
    from vllm.model_executor.kernels.linear.nvfp4.emulation import (
        EmulationNvFp4LinearKernel,
    )
    from vllm.model_executor.kernels.linear.nvfp4.select import init_nvfp4_linear_kernel
    from vllm.model_executor.layers.quantization.utils import nvfp4_emulation_utils
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        run_nvfp4_emulations,
    )

    e2m1 = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
    )

    prefix = _pick_dense_prefix(ckpt)
    keys = [f"{prefix}.{s}" for s in ("weight", "weight_scale", "weight_scale_2", "input_scale")]
    tensors = _load_keys(ckpt, keys)
    weight = tensors[f"{prefix}.weight"]
    weight_scale = tensors[f"{prefix}.weight_scale"]
    weight_gs = tensors[f"{prefix}.weight_scale_2"].reshape(-1)[0].float()
    input_gs = tensors[f"{prefix}.input_scale"].reshape(-1)[0].float()

    n, k_packed = weight.shape
    k = k_packed * 2
    torch.manual_seed(SEED)
    x = torch.randn(4, k, dtype=torch.bfloat16)

    nvfp4_emulation_utils.kE2M1ToFloat_handle.val = e2m1.clone()
    if device == "cuda":
        nvfp4_emulation_utils.kE2M1ToFloat_handle.val = (
            nvfp4_emulation_utils.kE2M1ToFloat_handle.val.to(device)
        )
        x = x.to(device)
        weight = weight.to(device)
        weight_scale = weight_scale.to(device)

    input_global_scale = input_gs.to(torch.float32)
    weight_global_scale = weight_gs.to(torch.float32)
    input_global_scale_inv = (1.0 / input_global_scale).to(torch.float32)

    ref = run_nvfp4_emulations(
        x=x,
        input_global_scale=input_global_scale_inv,
        weight=weight,
        weight_scale_swizzled=weight_scale,
        weight_global_scale=weight_global_scale,
        swizzle=False,
    )

    with set_current_vllm_config(
        VllmConfig(kernel_config=KernelConfig(linear_backend="emulation"))
    ):
        kernel = init_nvfp4_linear_kernel()
        assert isinstance(kernel, EmulationNvFp4LinearKernel)
        layer = torch.nn.Module()
        layer.weight = Parameter(weight.clone(), requires_grad=False)
        layer.weight_scale = Parameter(weight_scale.clone(), requires_grad=False)
        layer.weight_global_scale = Parameter(
            torch.tensor(
                float(weight_global_scale),
                dtype=torch.float32,
                device=weight.device,
            ),
            requires_grad=False,
        )
        layer.input_global_scale_inv = Parameter(
            torch.tensor(
                float(input_global_scale_inv),
                dtype=torch.float32,
                device=weight.device,
            ),
            requires_grad=False,
        )
        kernel.process_weights_after_loading(layer)
        out = kernel.apply_weights(layer, x, bias=None)

    metrics = _metrics(out, ref)
    passed = _assert_close(out, ref, DENSE_ATOL, DENSE_RTOL) and bool(
        torch.isfinite(out).all()
    )
    return {
        "prefix": prefix,
        "weight_shape": list(weight.shape),
        "weight_dtype": str(weight.dtype).replace("torch.", ""),
        "act_shape": list(x.shape),
        "kernel": type(kernel).__name__,
        "python_path_ok_on_sm80": not nvfp4_emulation_utils._nvfp4_triton_fp8e4nv_supported(),
        "tolerance": {"atol": DENSE_ATOL, "rtol": DENSE_RTOL},
        "metrics": metrics,
        "passed": passed,
        "finite": bool(torch.isfinite(out).all() and torch.isfinite(ref).all()),
    }


def _torch_nvfp4_moe_reference(
    hidden_states,
    w1,
    w1_scale,
    w1_gscale,
    w2,
    w2_scale,
    w2_gscale,
    a1_gscale,
    a2_gscale,
    topk_weights,
    topk_ids,
):
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        dequantize_to_dtype,
        ref_nvfp4_quant_dequant,
    )

    num_tokens, hidden_dim = hidden_states.shape
    top_k = topk_ids.size(1)
    dtype = hidden_states.dtype
    w1_dq = dequantize_to_dtype(
        w1, w1_scale, w1_gscale, dtype=dtype, block_size=16, swizzle=False
    )
    w2_dq = dequantize_to_dtype(
        w2, w2_scale, w2_gscale, dtype=dtype, block_size=16, swizzle=False
    )
    hs_qdq = ref_nvfp4_quant_dequant(hidden_states, a1_gscale, block_size=16)
    out = torch.zeros_like(hidden_states)
    for t in range(num_tokens):
        acc = torch.zeros(hidden_dim, dtype=torch.float32, device=hidden_states.device)
        for k in range(top_k):
            e = int(topk_ids[t, k].item())
            gemm1 = torch.nn.functional.linear(hs_qdq[t], w1_dq[e])
            gate, up = gemm1.chunk(2, dim=-1)
            act = torch.nn.functional.silu(gate) * up
            act_qdq = ref_nvfp4_quant_dequant(
                act.unsqueeze(0), a2_gscale, block_size=16
            ).squeeze(0)
            gemm2 = torch.nn.functional.linear(act_qdq, w2_dq[e])
            acc = acc + gemm2.float() * float(topk_weights[t, k].item())
        out[t] = acc.to(dtype)
    return out


def run_moe(ckpt: Path, device: str, num_local_experts: int = 2, top_k: int = 2) -> dict[str, Any]:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
        nvfp4_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
        Nvfp4QuantizationEmulationTritonExperts,
    )
    from vllm.model_executor.layers.quantization.utils import nvfp4_emulation_utils

    e2m1 = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
    )

    layer_id, expert_ids = _pick_moe_layer_experts(ckpt, num_local_experts)
    keys: list[str] = []
    for e in expert_ids:
        base = f"model.layers.{layer_id}.mlp.experts.{e}"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for s in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
                keys.append(f"{base}.{proj}.{s}")
    tensors = _load_keys(ckpt, keys)

    gates_w, ups_w, downs_w = [], [], []
    gates_s, ups_s, downs_s = [], [], []
    gates_g, ups_g, downs_g = [], [], []
    gates_a, ups_a, downs_a = [], [], []
    for e in expert_ids:
        base = f"model.layers.{layer_id}.mlp.experts.{e}"
        gates_w.append(tensors[f"{base}.gate_proj.weight"])
        ups_w.append(tensors[f"{base}.up_proj.weight"])
        downs_w.append(tensors[f"{base}.down_proj.weight"])
        gates_s.append(tensors[f"{base}.gate_proj.weight_scale"])
        ups_s.append(tensors[f"{base}.up_proj.weight_scale"])
        downs_s.append(tensors[f"{base}.down_proj.weight_scale"])
        gates_g.append(tensors[f"{base}.gate_proj.weight_scale_2"].reshape(()).float())
        ups_g.append(tensors[f"{base}.up_proj.weight_scale_2"].reshape(()).float())
        downs_g.append(tensors[f"{base}.down_proj.weight_scale_2"].reshape(()).float())
        gates_a.append(tensors[f"{base}.gate_proj.input_scale"].reshape(()).float())
        ups_a.append(tensors[f"{base}.up_proj.input_scale"].reshape(()).float())
        downs_a.append(tensors[f"{base}.down_proj.input_scale"].reshape(()).float())

    # Fuse gate+up -> W13 like ModelOpt fused MoE loader.
    w1 = torch.stack(
        [torch.cat([g, u], dim=0) for g, u in zip(gates_w, ups_w)], dim=0
    )
    w1_scale = torch.stack(
        [torch.cat([g, u], dim=0) for g, u in zip(gates_s, ups_s)], dim=0
    )
    # process_weights_after_loading uses w13_weight_scale_2[:, 0] (gate column).
    w1_gscale = torch.stack(gates_g, dim=0)
    w2 = torch.stack(downs_w, dim=0)
    w2_scale = torch.stack(downs_s, dim=0)
    w2_gscale = torch.stack(downs_g, dim=0)

    # Activation scales: stack gate/up then take max() reciprocal (EMULATION path).
    a13_raw = torch.stack(
        [torch.stack([ga, ua]) for ga, ua in zip(gates_a, ups_a)], dim=0
    )  # [E, 2]
    a2_raw = torch.stack(downs_a, dim=0)
    a1_gscale = (1.0 / a13_raw.max().to(torch.float32)).reshape(())
    a2_gscale = (1.0 / a2_raw.max().to(torch.float32)).reshape(())

    hidden_dim = w2.shape[1]
    intermediate = gates_w[0].shape[0]
    num_experts = len(expert_ids)

    torch.manual_seed(SEED)
    num_tokens = 4
    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)
    topk_weights = torch.randn(num_tokens, top_k, dtype=torch.float32).softmax(dim=-1)
    topk_ids = torch.stack(
        [torch.randperm(num_experts)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)

    nvfp4_emulation_utils.kE2M1ToFloat_handle.val = e2m1.clone()
    if device == "cuda":
        nvfp4_emulation_utils.kE2M1ToFloat_handle.val = (
            nvfp4_emulation_utils.kE2M1ToFloat_handle.val.to(device)
        )
        to_dev = lambda t: t.to(device)  # noqa: E731
        w1, w1_scale, w1_gscale = to_dev(w1), to_dev(w1_scale), to_dev(w1_gscale)
        w2, w2_scale, w2_gscale = to_dev(w2), to_dev(w2_scale), to_dev(w2_gscale)
        a1_gscale, a2_gscale = to_dev(a1_gscale), to_dev(a2_gscale)
        hidden_states = to_dev(hidden_states)
        topk_weights, topk_ids = to_dev(topk_weights), to_dev(topk_ids)

    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden_dim,
        intermediate_size_per_partition=intermediate,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device=device,
        routing_method=RoutingMethodType.TopK,
        moe_backend="emulation",
    )
    quant_config = nvfp4_moe_quant_config(
        g1_alphas=w1_gscale.clone(),
        g2_alphas=w2_gscale.clone(),
        a1_gscale=a1_gscale.clone(),
        a2_gscale=a2_gscale.clone(),
        w1_scale=w1_scale.clone(),
        w2_scale=w2_scale.clone(),
    )
    experts = Nvfp4QuantizationEmulationTritonExperts(
        moe_config=moe_config, quant_config=quant_config
    )

    N = w1.size(1)
    K = hidden_dim
    ws13 = torch.zeros(
        num_tokens * top_k * max(intermediate, K),
        dtype=torch.bfloat16,
        device=device,
    )
    ws2 = torch.zeros(
        num_tokens * top_k * max(N, K), dtype=torch.bfloat16, device=device
    )
    output = torch.zeros(num_tokens, K, dtype=torch.bfloat16, device=device)
    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=ws13,
        workspace2=ws2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    ref = _torch_nvfp4_moe_reference(
        hidden_states,
        w1,
        w1_scale,
        w1_gscale,
        w2,
        w2_scale,
        w2_gscale,
        a1_gscale,
        a2_gscale,
        topk_weights,
        topk_ids,
    )
    metrics = _metrics(output, ref)
    passed = _assert_close(output, ref, MOE_ATOL, MOE_RTOL) and bool(
        torch.isfinite(output).all()
    )
    return {
        "layer": layer_id,
        "expert_ids": expert_ids,
        "w13_shape": list(w1.shape),
        "w2_shape": list(w2.shape),
        "top_k": top_k,
        "num_tokens": num_tokens,
        "experts_cls": type(experts).__name__,
        "python_path_ok_on_sm80": not nvfp4_emulation_utils._nvfp4_triton_fp8e4nv_supported(),
        "tolerance": {"atol": MOE_ATOL, "rtol": MOE_RTOL},
        "metrics": metrics,
        "passed": passed,
        "finite": bool(torch.isfinite(output).all() and torch.isfinite(ref).all()),
    }


def render_report(dense: dict[str, Any], moe: dict[str, Any], ckpt: Path, device: str) -> str:
    overall = dense["passed"] and moe["passed"]
    lines = [
        "# Qwen3-30B-A3B-NVFP4 Puncture (Task 10)",
        "",
        f"- Checkpoint: `{ckpt}`",
        f"- Device: `{device}`",
        f"- Overall: **{'PASS' if overall else 'FAIL'}**",
        "",
        "## Dense",
        "",
        f"- Layer prefix: `{dense['prefix']}`",
        f"- Weight: shape=`{dense['weight_shape']}`, dtype=`{dense['weight_dtype']}`",
        f"- Activation: BF16 seed={SEED}, shape=`{dense['act_shape']}`",
        f"- Kernel: `{dense['kernel']}` vs `run_nvfp4_emulations`",
        f"- SM80 Python fp8e4nv path expected: `{dense['python_path_ok_on_sm80']}`",
        f"- Tolerance: atol={dense['tolerance']['atol']}, rtol={dense['tolerance']['rtol']}",
        f"- Metrics: max_abs={dense['metrics']['max_abs']:.6g}, "
        f"mean_abs={dense['metrics']['mean_abs']:.6g}, "
        f"rel_l2={dense['metrics']['rel_l2']:.6g}, "
        f"nmse={dense['metrics']['nmse']:.6g}",
        f"- Finite: `{dense['finite']}`",
        f"- Result: **{'PASS' if dense['passed'] else 'FAIL'}**",
        "",
        "## MoE (W13 → silu-and-mul → W2)",
        "",
        f"- Layer `{moe['layer']}`, experts `{moe['expert_ids']}`",
        f"- W13 shape `{moe['w13_shape']}`, W2 shape `{moe['w2_shape']}`",
        f"- Tokens={moe['num_tokens']}, top_k={moe['top_k']}",
        f"- Experts class: `{moe['experts_cls']}` vs torch reference",
        f"- SM80 Python fp8e4nv path expected: `{moe['python_path_ok_on_sm80']}`",
        f"- Tolerance: atol={moe['tolerance']['atol']}, rtol={moe['tolerance']['rtol']}",
        f"- Metrics: max_abs={moe['metrics']['max_abs']:.6g}, "
        f"mean_abs={moe['metrics']['mean_abs']:.6g}, "
        f"rel_l2={moe['metrics']['rel_l2']:.6g}, "
        f"nmse={moe['metrics']['nmse']:.6g}",
        f"- Finite: `{moe['finite']}`",
        f"- Result: **{'PASS' if moe['passed'] else 'FAIL'}**",
        "",
        "## Gate",
        "",
    ]
    if overall:
        lines.append("- Dense and MoE both passed upstream tolerance → E2E smoke allowed.")
    else:
        lines.append("- STOP: do not run E2E smoke until puncture passes.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()

    ckpt = args.checkpoint.resolve()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA required for puncture", file=sys.stderr)
        return 2

    dense = run_dense(ckpt, args.device)
    print("DENSE:", json.dumps({k: dense[k] for k in ("prefix", "metrics", "passed")}, indent=2))
    moe = run_moe(ckpt, args.device, num_local_experts=args.num_experts, top_k=args.top_k)
    print("MOE:", json.dumps({k: moe[k] for k in ("layer", "expert_ids", "metrics", "passed")}, indent=2))

    report = render_report(dense, moe, ckpt, args.device)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote report: {args.report}")
    return 0 if dense["passed"] and moe["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
