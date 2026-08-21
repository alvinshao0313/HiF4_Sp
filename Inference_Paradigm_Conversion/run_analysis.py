#!/usr/bin/env python3
"""CLI entry for Inference Paradigm Conversion analysis stages."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

REPO_ROOT = Path("/home/shaoyuantian/program/HiF4_Sp")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Inference_Paradigm_Conversion.ipc_analysis.config import (  # noqa: E402
    LINEAR_PROJECTIONS,
    load_experiment_config,
    resolve_activation_scale_file,
    resolve_representative_layers,
)
from Inference_Paradigm_Conversion.ipc_analysis.formats.fingerprint import (  # noqa: E402
    list_safetensor_keys,
)
from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import (  # noqa: E402
    load_nvfp4_activation_scales,
    resolve_nvfp4_scale_for_module,
)
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import (  # noqa: E402
    atomic_write_json,
    ensure_dir,
    write_csv,
    write_text,
)  # write_text used by ledger + preflight
from Inference_Paradigm_Conversion.ipc_analysis.records import RunManifest  # noqa: E402


def _git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


def _dtype_name(dt: torch.dtype) -> str:
    return str(dt).replace("torch.", "")


def _projection_of(name: str) -> str | None:
    for p in LINEAR_PROJECTIONS:
        if name.endswith(f".{p}.weight") or f".{p}.weight" in name:
            return p
    return None


def _module_prefix_from_weight(weight_name: str) -> str:
    if weight_name.endswith(".weight"):
        return weight_name[: -len(".weight")]
    return weight_name


def run_preflight(config_path: Path, out_dir: Path) -> dict[str, Any]:
    cfg = load_experiment_config(config_path)
    ckpt = cfg.source_checkpoint_path()
    if not ckpt.is_dir():
        raise FileNotFoundError(f"source checkpoint not found: {ckpt}")

    with (ckpt / "config.json").open("r", encoding="utf-8") as f:
        model_cfg = json.load(f)

    num_layers = int(model_cfg["num_hidden_layers"])
    resolved_layers = resolve_representative_layers(num_layers)
    weight_map = list_safetensor_keys(ckpt)

    # Inventory weights
    module_rows: list[dict[str, Any]] = []
    proj_counts: Counter[str] = Counter()
    storage_dtypes: set[str] = set()
    non_div64: list[str] = []
    weight_tensor_count = 0
    sample_checked = 0

    # Group by shard for efficient reads
    shard_to_names: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        shard_to_names[shard].append(name)

    for shard_name, names in shard_to_names.items():
        shard_path = ckpt / shard_name
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for name in names:
                if not name.endswith(".weight"):
                    continue
                proj = _projection_of(name)
                if proj is None and "lm_head" not in name and "embed" not in name:
                    # still record other weights lightly
                    pass
                t = handle.get_tensor(name)
                dt = _dtype_name(t.dtype)
                storage_dtypes.add(dt)
                weight_tensor_count += 1
                shape = list(t.shape)
                k_ok = True
                if proj is not None:
                    proj_counts[proj] += 1
                    # weight layout [out, in]; K = in_features = shape[1]
                    if len(shape) != 2:
                        raise ValueError(f"{name} expected 2D, got {shape}")
                    if shape[1] % 64 != 0:
                        k_ok = False
                        non_div64.append(name)
                    if sample_checked < 3 and proj in LINEAR_PROJECTIONS:
                        if t.dtype != torch.bfloat16:
                            raise TypeError(
                                f"formal Linear weight must be BF16 storage, got {t.dtype} for {name}"
                            )
                        sample_checked += 1
                module_rows.append(
                    {
                        "tensor_name": name,
                        "projection": proj or "",
                        "shape0": shape[0] if shape else "",
                        "shape1": shape[1] if len(shape) > 1 else "",
                        "ndim": len(shape),
                        "dtype": dt,
                        "k_divisible_by_64": k_ok if proj else "",
                        "shard": shard_name,
                    }
                )

    # Activation scales via existing loader path
    scale_file = resolve_activation_scale_file(ckpt)
    scales = load_nvfp4_activation_scales(scale_file)
    scale_rows: list[dict[str, Any]] = []
    for key, val in sorted(scales.items()):
        scale_rows.append(
            {
                "scale_key": key,
                "dtype": _dtype_name(val.dtype),
                "numel": int(val.numel()),
                "value": float(val.item()),
            }
        )

    linear_weight_names = [
        r["tensor_name"]
        for r in module_rows
        if r["projection"] in LINEAR_PROJECTIONS
    ]
    mapped = []
    unmapped = []
    for wname in linear_weight_names:
        prefix = _module_prefix_from_weight(wname)
        try:
            resolve_nvfp4_scale_for_module(scales, prefix)
            mapped.append(prefix)
        except ValueError:
            unmapped.append(prefix)

    scale_key_set = set(scales.keys())
    expected_keys = {f"{p}.input_global_scale" for p in mapped}
    # also accept language_model alias forms already resolved
    extra_scale_keys = sorted(k for k in scale_key_set if not any(
        k.endswith(f".{proj}.input_global_scale") for proj in LINEAR_PROJECTIONS
    ))

    # Optional FP32 probe
    fp32_probe = cfg.optional_fp32_probe_path()
    fp32_probe_exists = fp32_probe.is_dir()
    storage_probe: dict[str, Any] = {"exists": fp32_probe_exists, "compared": False}
    if fp32_probe_exists:
        storage_probe["path"] = str(fp32_probe)
        storage_probe["compared"] = False  # optional; not blocking

    # Capability matrix
    caps: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": [],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        caps["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    try:
        import vllm  # type: ignore

        caps["vllm_version"] = getattr(vllm, "__version__", "present")
    except Exception as e:
        caps["vllm_version"] = f"unavailable: {e}"
    try:
        import flashinfer  # type: ignore

        caps["flashinfer"] = getattr(flashinfer, "__version__", "present")
    except Exception as e:
        caps["flashinfer"] = f"unavailable: {e}"
    try:
        from NVFP4.torch_fake import fake_quant_nvfp4_activation  # noqa: F401

        caps["nvfp4_fake_activation"] = True
    except Exception as e:
        caps["nvfp4_fake_activation"] = f"unavailable: {e}"
    try:
        from ChuanCi.nvfp4_hif4_torch import quantize_hif4  # noqa: F401

        caps["hif4_fake"] = True
    except Exception as e:
        caps["hif4_fake"] = f"unavailable: {e}"
    try:
        from Inference_Paradigm_Conversion.ipc_analysis.formats.mxfp8 import (  # noqa: F401
            quantize_mxfp8_activation,
        )

        caps["mxfp8_oracle"] = True
    except Exception as e:
        caps["mxfp8_oracle"] = f"unavailable: {e}"

    # Quantization metadata in config
    dequant_cfg = model_cfg.get("dequantization_config", {})
    hadamard_folded = bool(dequant_cfg.get("hadamard_folded", False))
    if cfg.model.hadamard_runtime != "disabled":
        raise RuntimeError("hadamard_runtime must remain disabled")
    if not hadamard_folded:
        # still proceed but flag — formal model should have folded hadamard
        hadamard_note = "config.dequantization_config.hadamard_folded is not true"
    else:
        hadamard_note = "hadamard_folded=true; runtime Hadamard disabled"

    # Tokenizer info
    tok_cfg_path = ckpt / "tokenizer_config.json"
    tokenizer_info: dict[str, Any] = {"present": tok_cfg_path.is_file()}
    if tok_cfg_path.is_file():
        with tok_cfg_path.open("r", encoding="utf-8") as f:
            tok_cfg = json.load(f)
        tokenizer_info["tokenizer_class"] = tok_cfg.get("tokenizer_class")
        tokenizer_info["model_max_length"] = tok_cfg.get("model_max_length")
    chat_template = (ckpt / "chat_template.jinja").is_file()

    if storage_dtypes != {"bfloat16"} and not storage_dtypes.issubset({"bfloat16"}):
        # Linear weights must be bf16; other tensors might differ — check Linear only
        linear_dtypes = {r["dtype"] for r in module_rows if r["projection"] in LINEAR_PROJECTIONS}
        if linear_dtypes != {"bfloat16"}:
            raise TypeError(f"Linear weight dtypes must be only bfloat16, got {linear_dtypes}")

    if unmapped:
        # For formal Qwen3-8B we expect full coverage of 7 projs; fail if any target Linear missing
        raise RuntimeError(
            f"Unmapped Linear modules without NVFP4 activation scale ({len(unmapped)}): "
            f"{unmapped[:10]}"
        )
    if non_div64:
        raise RuntimeError(
            f"Linear K dim not divisible by 64 ({len(non_div64)}): {non_div64[:10]}"
        )

    for proj in LINEAR_PROJECTIONS:
        if proj_counts[proj] != num_layers:
            raise RuntimeError(
                f"expected {num_layers} tensors for {proj}, got {proj_counts[proj]}"
            )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_f0_preflight"
    out = ensure_dir(out_dir / run_id)

    manifest = RunManifest(
        run_id=run_id,
        git_commit=_git_commit(cfg.repo_root),
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_version=torch.__version__,
        status="passed",
        notes="F0 preflight for Qwen3-8B NVFP4-QAT dequant BF16",
        source_checkpoint=str(ckpt),
        source_semantics=cfg.model.source_semantics,
        source_weight_dtype="bfloat16",
        source_weight_tensor_count=weight_tensor_count,
        num_hidden_layers=num_layers,
        resolved_representative_layers=resolved_layers,
        hadamard_runtime="disabled",
        nvfp4_activation_scale_file=str(scale_file),
        nvfp4_activation_scale_count=len(scales),
        matched_activation_module_count=len(mapped),
        seed=cfg.seed,
        path_ids=[
            "P1_semantic",
            "P1_runtime",
            "P2_matched_semantic",
            "P2_matched_runtime",
            "P2_deployment_semantic",
            "P2_deployment_runtime",
            "W_storage_probe",
        ],
        extra={
            "model_type": model_cfg.get("model_type"),
            "hidden_size": model_cfg.get("hidden_size"),
            "intermediate_size": model_cfg.get("intermediate_size"),
            "num_attention_heads": model_cfg.get("num_attention_heads"),
            "num_key_value_heads": model_cfg.get("num_key_value_heads"),
            "head_dim": model_cfg.get("head_dim"),
            "dequantization_config": dequant_cfg,
            "hadamard_note": hadamard_note,
            "projection_counts": dict(proj_counts),
            "unmapped_linear_names": unmapped,
            "extra_scale_keys": extra_scale_keys,
            "non_div64": non_div64,
            "fp32_storage_probe": storage_probe,
            "capabilities": caps,
            "tokenizer": tokenizer_info,
            "chat_template_present": chat_template,
            "config_path": str(config_path),
        },
    )

    preflight = {
        "manifest": manifest.to_dict(),
        "resolved_activation_scale_file": str(scale_file),
        "scale_tensor_count": len(scales),
        "scale_tensor_dtype": "float32",
        "scale_key_examples": sorted(scales.keys())[:8],
        "mapped_linear_count": len(mapped),
        "unmapped_linear_names": unmapped,
        "extra_scale_keys": extra_scale_keys,
        "resolved_representative_layers": {
            "early": resolved_layers[0],
            "middle": resolved_layers[1],
            "late": resolved_layers[2],
        },
        "capabilities": caps,
        "source_semantics_statement": (
            "本实验的 source weight 不是 packed NVFP4，也不是原始 BF16 模型。"
            "它是 NVFP4 QAT 权重已经反量化并以 BF16 保存后的数值模型。"
            "后续 W_NV 的 reference 就是该 checkpoint 当前保存的 BF16 数值。"
        ),
    }

    atomic_write_json(out / "preflight.json", preflight)
    atomic_write_json(out / "manifest.json", manifest.to_dict())
    write_csv(out / "module_inventory.csv", module_rows)
    write_csv(out / "activation_scale_inventory.csv", scale_rows)

    md = f"""# F0 Preflight — Qwen3-8B NVFP4-QAT

## Source semantics

本实验的 source weight 不是 packed NVFP4，也不是原始 BF16 模型。
它是 NVFP4 QAT 权重已经反量化并以 BF16 保存后的数值模型。
后续 W_NV 的 reference 就是该 checkpoint 当前保存的 BF16 数值。

- checkpoint: `{ckpt}`
- source_semantics: `{cfg.model.source_semantics}`
- storage dtype (Linear): `bfloat16`
- hadamard_runtime: `disabled` ({hadamard_note})

## Architecture

- model_type: `{model_cfg.get("model_type")}`
- num_hidden_layers: `{num_layers}`
- hidden_size: `{model_cfg.get("hidden_size")}`
- intermediate_size: `{model_cfg.get("intermediate_size")}`
- attention heads / kv heads / head_dim: `{model_cfg.get("num_attention_heads")}` / `{model_cfg.get("num_key_value_heads")}` / `{model_cfg.get("head_dim")}`

## Representative layers

- early = floor(L/8) = **{resolved_layers[0]}**
- middle = floor(L/2) = **{resolved_layers[1]}**
- late = L-2 = **{resolved_layers[2]}**

## Linear inventory

| projection | count |
|---|---|
"""
    for p in LINEAR_PROJECTIONS:
        md += f"| {p} | {proj_counts[p]} |\n"
    md += f"""
- weight_tensor_count: {weight_tensor_count}
- K dim not divisible by 64: {len(non_div64)}

## NVFP4 activation scales

- resolved_activation_scale_file: `{scale_file}`
- scale_tensor_count: {len(scales)}
- mapped_linear_count: {len(mapped)}
- unmapped_linear_names: {unmapped}
- extra_scale_keys: {extra_scale_keys}
- calibration (config): s1k-1.1 / 512 / seqlen=4096 / head slice

## Capabilities

```json
{json.dumps(caps, indent=2, ensure_ascii=False)}
```

## Status

**PASSED** — source dtype/shape/module mapping/activation scale mapping verified.
"""
    write_text(out / "preflight.md", md)

    # Also write a stable "latest" pointer for subsequent stages
    latest = out_dir / "latest_f0"
    if latest.is_symlink() or latest.exists():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            # directory leftover — replace pointer file
            pass
    try:
        if latest.exists():
            if latest.is_dir():
                write_text(latest / "RUN_ID", run_id)
            else:
                latest.unlink()
                write_text(out_dir / "latest_f0_run_id.txt", run_id)
        else:
            write_text(out_dir / "latest_f0_run_id.txt", run_id)
    except Exception:
        write_text(out_dir / "latest_f0_run_id.txt", run_id)
    write_text(out_dir / "latest_f0_run_id.txt", run_id)

    print(f"[F0] PASSED → {out}")
    print(f"[F0] representative layers: {resolved_layers}")
    print(f"[F0] activation scales: {len(scales)} @ {scale_file}")
    return preflight


def run_weight(args: argparse.Namespace) -> dict[str, Any]:
    from Inference_Paradigm_Conversion.ipc_analysis.analysis.weight_conversion import (
        run_weight_analysis,
    )

    cfg = load_experiment_config(args.config)
    ckpt = cfg.source_checkpoint_path()
    layers = None
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    elif args.representative_only:
        with (ckpt / "config.json").open("r", encoding="utf-8") as f:
            num_layers = int(json.load(f)["num_hidden_layers"])
        layers = resolve_representative_layers(num_layers)

    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_weight"
    )
    out = ensure_dir(Path(args.out_dir) / run_id)
    summary = run_weight_analysis(
        ckpt,
        out,
        device=args.device,
        layer_indices=layers,
        emit_all_group_records=args.emit_all_groups,
        max_group_records_per_tensor=(
            None if args.emit_all_groups else args.max_groups_per_tensor
        ),
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )
    print(f"[W] done → {out} shard={args.shard_id}/{args.num_shards}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="IPC analysis runner")
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "Inference_Paradigm_Conversion"
        / "configs"
        / "qwen3_8b_nvfp4_qat_formal.yaml",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "Inference_Paradigm_Conversion" / "results",
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight", help="F0 checkpoint/scale/capability preflight")

    w = sub.add_parser("weight", help="W0–W2 weight conversion analysis")
    w.add_argument("--device", type=str, default="cuda:0")
    w.add_argument("--run-id", type=str, default="")
    w.add_argument("--layers", type=str, default="", help="comma-separated layer indices")
    w.add_argument("--representative-only", action="store_true")
    w.add_argument("--shard-id", type=int, default=0)
    w.add_argument("--num-shards", type=int, default=1)
    w.add_argument("--emit-all-groups", action="store_true")
    w.add_argument("--max-groups-per-tensor", type=int, default=4096)

    al = sub.add_parser("repr-al", help="Representative-layer activation + linear analysis")
    al.add_argument("--device", type=str, default="cuda:0")
    al.add_argument("--run-id", type=str, default="")
    al.add_argument("--shard-id", type=int, default=0)
    al.add_argument("--num-shards", type=int, default=1)
    al.add_argument("--samples-per-family", type=int, default=32)
    al.add_argument("--max-seq-len", type=int, default=256)
    al.add_argument("--decode-steps", type=int, default=8)

    a2 = sub.add_parser("a2", help="A2 NVFP4/HiF4 activation internal-step counterfactuals")
    a2.add_argument("--device", type=str, default="cuda:0")
    a2.add_argument("--run-id", type=str, default="")
    a2.add_argument("--shard-id", type=int, default=0)
    a2.add_argument("--num-shards", type=int, default=1)
    a2.add_argument("--samples-per-family", type=int, default=8)
    a2.add_argument("--max-seq-len", type=int, default=128)
    a2.add_argument("--decode-steps", type=int, default=4)

    a5 = sub.add_parser("a5", help="A5 H2 activation distribution interventions")
    a5.add_argument("--device", type=str, default="cuda:0")
    a5.add_argument("--run-id", type=str, default="")
    a5.add_argument("--shard-id", type=int, default=0)
    a5.add_argument("--num-shards", type=int, default=1)
    a5.add_argument("--samples-per-family", type=int, default=4)
    a5.add_argument("--max-seq-len", type=int, default=128)
    a5.add_argument("--max-groups-per-module", type=int, default=64)

    l2 = sub.add_parser("l2", help="L2 Shapley phi_W/phi_A + FP64 audit on representative layers")
    l2.add_argument("--device", type=str, default="cuda:0")
    l2.add_argument("--run-id", type=str, default="")
    l2.add_argument("--shard-id", type=int, default=0)
    l2.add_argument("--num-shards", type=int, default=1)
    l2.add_argument("--samples-per-family", type=int, default=4)
    l2.add_argument("--max-seq-len", type=int, default=128)

    w3 = sub.add_parser("w3", help="W3 HiF4 counterfactual variants on representative layers")
    w3.add_argument("--device", type=str, default="cuda:0")
    w3.add_argument("--run-id", type=str, default="")
    w3.add_argument("--shard-id", type=int, default=0)
    w3.add_argument("--num-shards", type=int, default=1)

    w4 = sub.add_parser("w4", help="W4 16→64 dispersion causal interventions")
    w4.add_argument("--device", type=str, default="cuda:0")
    w4.add_argument("--run-id", type=str, default="")
    w4.add_argument("--shard-id", type=int, default=0)
    w4.add_argument("--num-shards", type=int, default=1)
    w4.add_argument("--max-groups-per-tensor", type=int, default=512)

    l3 = sub.add_parser("l3", help="L3/L4 weight NMSE vs output-aware predictors")
    l3.add_argument("--device", type=str, default="cuda:0")
    l3.add_argument("--run-id", type=str, default="")
    l3.add_argument("--shard-id", type=int, default=0)
    l3.add_argument("--num-shards", type=int, default=1)

    led = sub.add_parser("ledger", help="Build root-cause ledger from latest results")
    led.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "Inference_Paradigm_Conversion" / "results",
    )

    attn = sub.add_parser("attn", help="T1–T6 attention propagation on representative layers")
    attn.add_argument("--device", type=str, default="cuda:0")
    attn.add_argument("--run-id", type=str, default="")
    attn.add_argument("--shard-id", type=int, default=0)
    attn.add_argument("--num-shards", type=int, default=1)
    attn.add_argument("--samples-per-family", type=int, default=8)

    inj = sub.add_parser("inject", help="N network injection / oracle repair")
    inj.add_argument("--device", type=str, default="cuda:0")
    inj.add_argument("--run-id", type=str, default="")
    inj.add_argument("--shard-id", type=int, default=0)
    inj.add_argument("--num-shards", type=int, default=1)
    inj.add_argument(
        "--mode",
        type=str,
        default="n1_n2",
        choices=["n1_n2", "prefix_suffix", "oracle"],
    )

    gemm = sub.add_parser("gemm", help="G0–G5 GEMM semantic chain on synthetic tensors")
    gemm.add_argument("--device", type=str, default="cpu")
    gemm.add_argument("--run-id", type=str, default="")

    syn = sub.add_parser("synthetic", help="S1–S7 synthetic mechanism suite")
    syn.add_argument("--seeds", type=int, default=10)
    syn.add_argument("--run-id", type=str, default="")

    rep = sub.add_parser("report", help="Build aggregate md/html report from latest_* pointers")
    rep.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "Inference_Paradigm_Conversion" / "results",
    )

    ax = sub.add_parser("ax", help="AX1–AX4 activation incremental shard runner")
    ax.add_argument("--device", type=str, default="cuda:0")
    ax.add_argument("--run-id", type=str, default="")
    ax.add_argument("--shard-id", type=int, default=0)
    ax.add_argument("--num-shards", type=int, default=1)
    ax.add_argument("--split", type=str, default="discovery", choices=["discovery", "validation"])
    ax.add_argument("--phases", type=str, default="prefill", help="comma-separated: prefill,decode")
    ax.add_argument("--samples-per-family", type=int, default=8)
    ax.add_argument("--max-seq-len", type=int, default=128)
    ax.add_argument("--decode-steps", type=int, default=4)
    ax.add_argument("--max-tokens-prefill", type=int, default=32)
    ax.add_argument("--skip-ax1", action="store_true")
    ax.add_argument("--skip-ax2", action="store_true")
    ax.add_argument("--skip-ax3", action="store_true")
    ax.add_argument("--skip-ax4", action="store_true")
    ax.add_argument("--a2-run-dir", type=Path, default=None)

    axm = sub.add_parser("ax-merge", help="Merge AX shard CSVs, AX5 ranking + rules")
    axm.add_argument("--run-id", type=str, required=True)
    axm.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="overrides parent --out-dir when placed after ax-merge",
    )
    axm.add_argument("--a2-run-dir", type=Path, default=None)
    axm.add_argument("--skip-ax5-rules", action="store_true")

    axr = sub.add_parser("ax-report", help="Build AX Chinese report from existing run_id")
    axr.add_argument("--run-id", type=str, required=True)
    axr.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="overrides parent --out-dir when placed after ax-report",
    )

    av = sub.add_parser(
        "activation-viz",
        help="W4A4 online activation distribution + NVFP4→HiF4 residual viz shard",
    )
    av.add_argument("--device", type=str, default="cuda:0")
    av.add_argument("--run-id", type=str, default="")
    av.add_argument("--shard-id", type=int, default=0)
    av.add_argument("--num-shards", type=int, default=1)
    av.add_argument("--split", type=str, default="discovery", choices=["discovery", "validation"])
    av.add_argument("--samples-per-family", type=int, default=8)
    av.add_argument("--max-seq-len", type=int, default=256)
    av.add_argument("--decode-steps", type=int, default=8)
    av.add_argument("--max-point-samples-per-capture", type=int, default=1024)

    avm = sub.add_parser("activation-viz-merge", help="Merge activation-viz shards (no model load)")
    avm.add_argument("--run-id", type=str, required=True)
    avm.add_argument("--out-dir", type=Path, default=None)

    avr = sub.add_parser("activation-viz-report", help="Build activation-viz figures/report (no model)")
    avr.add_argument("--run-id", type=str, required=True)
    avr.add_argument("--out-dir", type=Path, default=None)
    avr.add_argument(
        "--discovery-run-id",
        type=str,
        default="",
        help="optional discovery run for stability JS when reporting validation",
    )

    return p


def run_repr_al(args: argparse.Namespace) -> dict[str, Any]:
    from Inference_Paradigm_Conversion.ipc_analysis.analysis.repr_pipeline import (
        run_repr_al_shard,
    )

    cfg = load_experiment_config(args.config)
    ckpt = cfg.source_checkpoint_path()
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_repr_al"
    )
    out = ensure_dir(Path(args.out_dir) / run_id)
    return run_repr_al_shard(
        ckpt,
        out,
        device=args.device,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        samples_per_family=args.samples_per_family,
        max_seq_len=args.max_seq_len,
        decode_steps=args.decode_steps,
    )


def run_a2(args: argparse.Namespace) -> dict[str, Any]:
    from Inference_Paradigm_Conversion.ipc_analysis.analysis.a2_repr_pipeline import (
        run_a2_repr_shard,
    )

    cfg = load_experiment_config(args.config)
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_a2"
    )
    out = ensure_dir(Path(args.out_dir) / run_id)
    return run_a2_repr_shard(
        cfg.source_checkpoint_path(),
        out,
        device=args.device,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        samples_per_family=args.samples_per_family,
        max_seq_len=args.max_seq_len,
        decode_steps=args.decode_steps,
    )


def run_a5(args: argparse.Namespace) -> dict[str, Any]:
    from Inference_Paradigm_Conversion.ipc_analysis.analysis.a5_repr_pipeline import (
        run_a5_repr_shard,
    )

    cfg = load_experiment_config(args.config)
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_a5"
    )
    out = ensure_dir(Path(args.out_dir) / run_id)
    return run_a5_repr_shard(
        cfg.source_checkpoint_path(),
        out,
        device=args.device,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        samples_per_family=args.samples_per_family,
        max_seq_len=args.max_seq_len,
        max_groups_per_module=args.max_groups_per_module,
    )


def run_l2(args: argparse.Namespace) -> dict[str, Any]:
    from Inference_Paradigm_Conversion.ipc_analysis.analysis.l2_shapley_pipeline import (
        run_l2_shapley_shard,
    )

    cfg = load_experiment_config(args.config)
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_l2"
    )
    out = ensure_dir(Path(args.out_dir) / run_id)
    return run_l2_shapley_shard(
        cfg.source_checkpoint_path(),
        out,
        device=args.device,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        samples_per_family=args.samples_per_family,
        max_seq_len=args.max_seq_len,
    )


def run_w3(args: argparse.Namespace) -> dict[str, Any]:
    from Inference_Paradigm_Conversion.ipc_analysis.analysis.w3_counterfactual import (
        run_w3_representative,
    )

    cfg = load_experiment_config(args.config)
    ckpt = cfg.source_checkpoint_path()
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_w3"
    )
    out = ensure_dir(Path(args.out_dir) / run_id)
    return run_w3_representative(
        ckpt,
        out,
        device=args.device,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )


def run_w4(args: argparse.Namespace) -> dict[str, Any]:
    from Inference_Paradigm_Conversion.ipc_analysis.analysis.w4_causal import run_w4_shard

    cfg = load_experiment_config(args.config)
    ckpt = cfg.source_checkpoint_path()
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_w4"
    )
    out = ensure_dir(Path(args.out_dir) / run_id)
    return run_w4_shard(
        ckpt,
        out,
        device=args.device,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        max_groups_per_tensor=args.max_groups_per_tensor,
    )


def run_l3(args: argparse.Namespace) -> dict[str, Any]:
    from Inference_Paradigm_Conversion.ipc_analysis.analysis.l3_predictors import run_l3_shard

    cfg = load_experiment_config(args.config)
    ckpt = cfg.source_checkpoint_path()
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_l3"
    )
    out = ensure_dir(Path(args.out_dir) / run_id)
    return run_l3_shard(
        ckpt,
        out,
        device=args.device,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )


def run_ledger(args: argparse.Namespace) -> dict[str, Any]:
    from Inference_Paradigm_Conversion.ipc_analysis.reporting.root_cause import build_ledger

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_ledger"
    out = ensure_dir(Path(args.out_dir) / run_id)
    ledger = build_ledger(args.results_root, out)
    print(f"[ledger] → {out} records={ledger['num_records']}")
    write_text(Path(args.out_dir) / "latest_ledger_run_id.txt", run_id)
    return ledger


def run_attn(args: argparse.Namespace) -> dict[str, Any]:
    from Inference_Paradigm_Conversion.ipc_analysis.analysis.attn_repr_pipeline import (
        run_attn_repr_shard,
    )

    cfg = load_experiment_config(args.config)
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_attn"
    )
    out = ensure_dir(Path(args.out_dir) / run_id)
    return run_attn_repr_shard(
        cfg.source_checkpoint_path(),
        out,
        device=args.device,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        samples_per_family=args.samples_per_family,
    )


def run_inject(args: argparse.Namespace) -> dict[str, Any]:
    from Inference_Paradigm_Conversion.ipc_analysis.analysis.injection_pipeline import (
        run_injection_shard,
    )

    cfg = load_experiment_config(args.config)
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_inject_{args.mode}"
    )
    out = ensure_dir(Path(args.out_dir) / run_id)
    return run_injection_shard(
        cfg.source_checkpoint_path(),
        out,
        device=args.device,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        mode=args.mode,
    )


def run_gemm(args: argparse.Namespace) -> dict[str, Any]:
    from Inference_Paradigm_Conversion.ipc_analysis.analysis.gemm_arithmetic import (
        gemm_chain_p1,
        gemm_chain_p2,
    )

    torch.manual_seed(20260810)
    device = torch.device(args.device)
    x = torch.randn(32, 256, dtype=torch.bfloat16, device=device)
    w = torch.randn(128, 256, device=device)
    scale = torch.tensor(64.0, device=device)
    p1 = gemm_chain_p1(x, w)
    p2 = gemm_chain_p2(x.cpu() if device.type != "cpu" else x, w.cpu(), scale.cpu())
    # keep p1 on device tensors already handled inside
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_gemm"
    )
    out = ensure_dir(Path(args.out_dir) / run_id)
    summary = {
        "run_id": run_id,
        "P1_output_nmse": p1["output"]["nmse"],
        "P2_output_nmse": p2["output"]["nmse"],
        "note": "format-semantic Oracle GEMM only; not real kernel perf",
    }
    atomic_write_json(out / "gemm_summary.json", summary)
    write_text(Path(args.out_dir) / "latest_gemm_run_id.txt", run_id)
    print(summary)
    return summary


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    from Inference_Paradigm_Conversion.ipc_analysis.reporting.report import build_report

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_report"
    out = ensure_dir(Path(args.out_dir) / run_id)
    summary = build_report(args.results_root, out)
    write_text(Path(args.out_dir) / "latest_report_run_id.txt", run_id)
    print(f"[report] → {out}/report.html figures={summary.get('figures')}")
    return summary


def run_ax(args: argparse.Namespace) -> dict[str, Any]:
    from Inference_Paradigm_Conversion.ipc_analysis.analysis.ax_pipeline import run_ax_shard

    cfg = load_experiment_config(args.config)
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_ax"
    )
    out = ensure_dir(Path(args.out_dir) / run_id)
    phases = tuple(p.strip() for p in args.phases.split(",") if p.strip())
    return run_ax_shard(
        cfg.source_checkpoint_path(),
        out,
        device=args.device,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        split=args.split,
        samples_per_family=args.samples_per_family,
        max_seq_len=args.max_seq_len,
        decode_steps=args.decode_steps,
        max_tokens_prefill=args.max_tokens_prefill,
        phases=phases,
        run_ax1=not args.skip_ax1,
        run_ax2=not args.skip_ax2,
        run_ax3=not args.skip_ax3,
        run_ax4=not args.skip_ax4,
        a2_run_dir=args.a2_run_dir,
    )


def run_ax_merge(args: argparse.Namespace) -> dict[str, Any]:
    from Inference_Paradigm_Conversion.ipc_analysis.analysis.ax_pipeline import merge_ax_shards

    out_dir = Path(args.out_dir) if getattr(args, "out_dir", None) else Path("Inference_Paradigm_Conversion/results")
    # subparser --out-dir may shadow parent; accept either
    run_dir = ensure_dir(out_dir / args.run_id)
    summary = merge_ax_shards(
        run_dir,
        a2_run_dir=args.a2_run_dir,
        run_ax5_rules=not args.skip_ax5_rules,
    )
    write_text(out_dir / "latest_ax_run_id.txt", args.run_id)
    print(f"[ax-merge] → {run_dir} ranking={summary.get('ranking_count')}")
    return summary


def run_ax_report(args: argparse.Namespace) -> dict[str, Any]:
    from Inference_Paradigm_Conversion.ipc_analysis.reporting.activation_incremental_report import (
        build_activation_incremental_report,
    )

    out_dir = Path(args.out_dir) if getattr(args, "out_dir", None) else Path("Inference_Paradigm_Conversion/results")
    run_dir = out_dir / args.run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"AX run dir not found: {run_dir}")
    summary = build_activation_incremental_report(run_dir)
    print(f"[ax-report] → {run_dir} figures={summary.get('figures')}")
    return summary


def run_activation_viz(args: argparse.Namespace) -> dict[str, Any]:
    from Inference_Paradigm_Conversion.ipc_analysis.analysis.activation_viz_pipeline import (
        run_activation_viz_shard,
    )

    cfg = load_experiment_config(args.config)
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_activation_viz_{args.split}"
    )
    out = ensure_dir(Path(args.out_dir) / run_id)
    summary = run_activation_viz_shard(
        cfg.source_checkpoint_path(),
        out,
        device=args.device,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        split=args.split,
        samples_per_family=args.samples_per_family,
        max_seq_len=args.max_seq_len,
        decode_steps=args.decode_steps,
        max_point_samples_per_capture=args.max_point_samples_per_capture,
    )
    write_text(Path(args.out_dir) / "latest_activation_viz_run_id.txt", run_id)
    print(
        f"[activation-viz] → {out} "
        f"prompts={summary.get('num_prompts')} rows={summary.get('num_summary_rows')}"
    )
    return summary


def run_activation_viz_merge(args: argparse.Namespace) -> dict[str, Any]:
    from Inference_Paradigm_Conversion.ipc_analysis.analysis.activation_viz_pipeline import (
        attach_theoretical_grids,
        merge_activation_viz_shards,
    )

    out_dir = Path(args.out_dir) if getattr(args, "out_dir", None) else Path(
        "Inference_Paradigm_Conversion/results"
    )
    run_dir = out_dir / args.run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"activation-viz run dir not found: {run_dir}")
    summary = merge_activation_viz_shards(run_dir)
    theory = attach_theoretical_grids(run_dir)
    summary["theory"] = {
        "nvfp4_unique": theory.get("nvfp4_full_stats", {}).get("num_unique_values"),
        "hif4_unique": theory.get("hif4_full_stats", {}).get("num_unique_values"),
        "note": theory.get("note"),
    }
    write_text(out_dir / "latest_activation_viz_run_id.txt", args.run_id)
    print(f"[activation-viz-merge] → {run_dir}")
    return summary


def run_activation_viz_report(args: argparse.Namespace) -> dict[str, Any]:
    from Inference_Paradigm_Conversion.ipc_analysis.reporting.activation_viz_report import (
        build_activation_viz_report,
    )

    out_dir = Path(args.out_dir) if getattr(args, "out_dir", None) else Path(
        "Inference_Paradigm_Conversion/results"
    )
    run_dir = out_dir / args.run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"activation-viz run dir not found: {run_dir}")
    discovery_dir = None
    if getattr(args, "discovery_run_id", None):
        discovery_dir = out_dir / args.discovery_run_id
    summary = build_activation_viz_report(run_dir, discovery_run_dir=discovery_dir)
    print(f"[activation-viz-report] → {run_dir} figures={len(summary.get('figures', []))}")
    return summary


def run_synthetic(args: argparse.Namespace) -> dict[str, Any]:
    from Inference_Paradigm_Conversion.ipc_analysis.analysis.synthetic_s_suite import (
        run_all_synthetic,
    )

    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_synthetic"
    )
    out = ensure_dir(Path(args.out_dir) / run_id)
    suite = run_all_synthetic(seeds=args.seeds)
    # persist without bulky per-row dumps in summary; keep rows in separate CSVs
    evidence = suite["evidence_summary"]
    for name, block in suite["experiments"].items():
        rows = block.get("rows", [])
        if rows:
            write_csv(out / f"{name.lower()}_rows.csv", rows)
        slim = {k: v for k, v in block.items() if k != "rows"}
        atomic_write_json(out / f"{name.lower()}_summary.json", slim)
    atomic_write_json(out / "synthetic_summary.json", {"run_id": run_id, **{k: evidence[k] for k in evidence}})
    write_text(Path(args.out_dir) / "latest_synthetic_run_id.txt", run_id)
    print(json.dumps({k: {"supports": v["supports_mechanism"], "H": v["hypothesis_id"]} for k, v in evidence.items()}, indent=2))
    return suite


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dispatch = {
        "preflight": lambda: run_preflight(args.config, args.out_dir),
        "weight": lambda: run_weight(args),
        "repr-al": lambda: run_repr_al(args),
        "a2": lambda: run_a2(args),
        "a5": lambda: run_a5(args),
        "l2": lambda: run_l2(args),
        "w3": lambda: run_w3(args),
        "w4": lambda: run_w4(args),
        "l3": lambda: run_l3(args),
        "ledger": lambda: run_ledger(args),
        "attn": lambda: run_attn(args),
        "inject": lambda: run_inject(args),
        "gemm": lambda: run_gemm(args),
        "synthetic": lambda: run_synthetic(args),
        "report": lambda: run_report(args),
        "ax": lambda: run_ax(args),
        "ax-merge": lambda: run_ax_merge(args),
        "ax-report": lambda: run_ax_report(args),
        "activation-viz": lambda: run_activation_viz(args),
        "activation-viz-merge": lambda: run_activation_viz_merge(args),
        "activation-viz-report": lambda: run_activation_viz_report(args),
    }
    if args.command not in dispatch:
        raise SystemExit(f"unknown command: {args.command}")
    dispatch[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
