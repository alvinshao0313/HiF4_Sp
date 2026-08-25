"""MMLU-Pro / AIME via repo-root main.py (vLLM + lighteval)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import (
    DEFAULT_MODEL_PATH,
    TARGET_LINEAR_COUNT,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.fold import (
    FusedDiagRMSNorm,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.semantic_hif4 import (
    ALL_PROJS,
    ATTN_PROJS,
    SwitchableNVHiF4Linear,
    quant_weight,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.semantic_model import (
    NativeNVFP4SemanticLinear,
    load_native_nvfp4_semantic_model,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.common import (
    REASONING_EVAL_NUM_GPUS,
    require_visible_cuda_count,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.lm_eval_vllm import (
    build_lm_eval_vllm_kwargs,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.moe_materialize import (
    materialize_moe_checkpoint,
    materialize_moe_identity_checkpoint,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot
from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import ensure_dir, write_json

REPO_ROOT = Path(__file__).resolve().parents[4]
MAIN_PY = REPO_ROOT / "main.py"
SHARED_VLLM_ROOT = (
    REPO_ROOT
    / "Native_NVFP4_HiF4_Linear_Puncture"
    / "results"
    / "e2e_diag_reconstruction"
    / "shared_vllm_qwen3_30b"
)
NVFP4_SCALES_FILENAME = "nvfp4_activation_scales.safetensors"
MLP_PROJS = tuple(p for p in ALL_PROJS if p not in ATTN_PROJS)


@dataclass(frozen=True)
class VllmEvalSpec:
    model_path: Path
    fake_act_quant: str
    materialized: bool
    hif4_runtime_spec_path: Path | None = None
    native_nvfp4: bool = False


def vllm_fake_act_for_variant(variant: str) -> str:
    if variant == "native_nvfp4":
        return "none"
    if variant in ("direct_hif4", "r64_only", "artifact"):
        return "none"
    raise ValueError(f"unknown variant {variant!r}")


def needs_materialized_checkpoint(variant: str) -> bool:
    return variant in ("direct_hif4", "r64_only", "artifact")


def _run_id(output_dir: Path) -> str:
    return output_dir.resolve().name


def _shared_moe_ckpt_dir(output_dir: Path, variant: str) -> Path:
    return SHARED_VLLM_ROOT / _run_id(output_dir) / variant


def _complete_materialized_moe_ckpt(path: Path) -> bool:
    index = path / "model.safetensors.index.json"
    sidecar = path / "hif4_runtime_spec.pt"
    return index.is_file() and sidecar.is_file()


def _snapshot_cache_key(snapshot: Path) -> str:
    digest = hashlib.sha256(str(snapshot.resolve()).encode("utf-8")).hexdigest()[:16]
    return f"{snapshot.name}_{digest}"


def _open_index_tensor(snapshot: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    shard = snapshot / weight_map[key]
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def extract_nvfp4_activation_scales(snapshot: Path) -> dict[str, torch.Tensor]:
    index_path = snapshot / "model.safetensors.index.json"
    weight_map: dict[str, str] = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    scales: dict[str, torch.Tensor] = {}
    for key in weight_map:
        if not key.endswith(".act_global_scale"):
            continue
        prefix = key[: -len(".act_global_scale")]
        out_key = f"{prefix}.input_global_scale"
        tensor = _open_index_tensor(snapshot, weight_map, key)
        if tensor.numel() != 1:
            raise ValueError(f"activation scale must be scalar: {key} shape={tuple(tensor.shape)}")
        scales[out_key] = tensor.reshape(()).to(torch.float32).contiguous()
    if len(scales) != TARGET_LINEAR_COUNT:
        raise RuntimeError(
            f"expected {TARGET_LINEAR_COUNT} NVFP4 activation scales, got {len(scales)}"
        )
    return scales


def _write_vllm_weight_index(model_dir: Path) -> None:
    """Exclude sidecar safetensors (e.g. activation scales) from vLLM weight loading."""
    weight_map: dict[str, str] = {}
    total_size = 0
    for path in sorted(model_dir.glob("model*.safetensors")):
        if path.name == NVFP4_SCALES_FILENAME:
            continue
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key == "__metadata__":
                    continue
                tensor = handle.get_tensor(key)
                weight_map[key] = path.name
                total_size += tensor.numel() * tensor.element_size()
    if not weight_map:
        raise FileNotFoundError(f"no model weight shards under {model_dir}")
    write_json(
        model_dir / "model.safetensors.index.json",
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
    )


def _link_snapshot_auxiliary_files(snapshot: Path, cache_dir: Path) -> None:
    skip = {
        "model.safetensors",
        "model.safetensors.index.json",
        NVFP4_SCALES_FILENAME,
        "config.json",
        "generation_config.json",
    }
    skip.update(path.name for path in cache_dir.iterdir() if path.is_file())
    for item in snapshot.iterdir():
        if item.name in skip or item.name.startswith("model-"):
            continue
        dst = cache_dir / item.name
        if dst.exists() or dst.is_symlink():
            continue
        os.symlink(item.resolve(), dst)


def _checkpoint_uses_fp_quant(snapshot: Path) -> bool:
    cfg = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    qc = cfg.get("quantization_config") or {}
    return qc.get("quant_method") == "fp_quant"


def _ensure_nvfp4_activation_scales(snapshot: Path) -> None:
    scales_path = snapshot / NVFP4_SCALES_FILENAME
    if scales_path.is_file():
        return
    save_file(extract_nvfp4_activation_scales(snapshot), str(scales_path))


def _unwrap_native_semantic_linears(model: nn.Module) -> dict[str, torch.Tensor]:
    scales: dict[str, torch.Tensor] = {}
    for layer in model.model.layers:
        for parent_name, projs in (("self_attn", ATTN_PROJS), ("mlp", MLP_PROJS)):
            parent = getattr(layer, parent_name)
            for proj in projs:
                mod = getattr(parent, proj)
                if not isinstance(mod, NativeNVFP4SemanticLinear):
                    raise TypeError(
                        f"{parent_name}.{proj} is {type(mod)}, expected NativeNVFP4SemanticLinear"
                    )
                scales[f"{mod.module_name}.input_global_scale"] = (
                    mod.input_global_scale.detach().cpu().to(torch.float32).reshape(()).contiguous()
                )
                standard = nn.Linear(
                    int(mod.weight.shape[1]),
                    int(mod.weight.shape[0]),
                    bias=mod.bias is not None,
                )
                with torch.no_grad():
                    standard.weight.copy_(mod.weight.detach().cpu())
                    if mod.bias is not None:
                        standard.bias.copy_(mod.bias.detach().cpu())
                setattr(parent, proj, standard)
    return scales


def materialize_vllm_bf16_checkpoint(snapshot: Path) -> Path:
    """Dequantize fp_quant checkpoints for vLLM on non-Blackwell GPUs."""
    if not _checkpoint_uses_fp_quant(snapshot):
        _ensure_nvfp4_activation_scales(snapshot)
        return snapshot

    cache_dir = ensure_dir(SHARED_VLLM_ROOT / f"{_snapshot_cache_key(snapshot)}_bf16_vllm")
    marker = cache_dir / "e2e_vllm_bf16_dequant.json"
    scales_path = cache_dir / NVFP4_SCALES_FILENAME
    index_path = cache_dir / "model.safetensors.index.json"
    if marker.is_file() and scales_path.is_file() and index_path.is_file():
        return cache_dir

    print(
        f"materializing vLLM BF16 checkpoint from fp_quant: {snapshot} -> {cache_dir}",
        flush=True,
    )
    model, _index = load_native_nvfp4_semantic_model(
        snapshot, device="cpu", dtype=torch.bfloat16
    )
    activation_scales = _unwrap_native_semantic_linears(model)
    if len(activation_scales) != TARGET_LINEAR_COUNT:
        raise RuntimeError(
            f"expected {TARGET_LINEAR_COUNT} NVFP4 activation scales, got {len(activation_scales)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=True)
    model.save_pretrained(str(cache_dir))
    tokenizer.save_pretrained(str(cache_dir))

    config_path = cache_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("quantization_config", None)
    config["dtype"] = "bfloat16"
    if "torch_dtype" in config:
        config["torch_dtype"] = "bfloat16"
    write_json(config_path, config)

    save_file(activation_scales, str(scales_path))
    _write_vllm_weight_index(cache_dir)
    _link_snapshot_auxiliary_files(snapshot, cache_dir)
    write_json(
        marker,
        {
            "source_snapshot": str(snapshot.resolve()),
            "activation_scale_count": len(activation_scales),
            "backend": "semantic_dequant_bf16",
        },
    )
    del model, tokenizer
    return cache_dir


def export_switchable_linear(mod: SwitchableNVHiF4Linear) -> nn.Linear:
    if mod._mode == "folded":
        if mod._folded_weight_fp32 is None:
            raise RuntimeError(f"{mod.module_name}: folded mode missing FP32 master weight")
        w_t = mod._folded_weight_fp32
        bias_src = mod._folded_bias_fp32
    else:
        w_t = mod.transformed_master_weight()
        bias_src = mod.bias
    w_h = quant_weight(w_t, use_ste=False).detach()
    linear = nn.Linear(
        int(w_h.shape[1]),
        int(w_h.shape[0]),
        bias=bias_src is not None,
        device=w_h.device,
        dtype=w_h.dtype,
    )
    with torch.no_grad():
        linear.weight.copy_(w_h)
        if bias_src is not None:
            linear.bias.copy_(bias_src.to(dtype=w_h.dtype))
    return linear


def materialize_eval_checkpoint(model, tokenizer, ckpt_dir: Path) -> Path:
    for layer in model.model.layers:
        for norm_name in ("input_layernorm", "post_attention_layernorm"):
            norm = getattr(layer, norm_name)
            if isinstance(norm, FusedDiagRMSNorm):
                base = norm.base
                with torch.no_grad():
                    base.weight.copy_(norm.weight.to(dtype=base.weight.dtype))
                setattr(layer, norm_name, base)
        for parent_name, projs in (("self_attn", ATTN_PROJS), ("mlp", MLP_PROJS)):
            parent = getattr(layer, parent_name)
            for proj in projs:
                mod = getattr(parent, proj)
                if not isinstance(mod, SwitchableNVHiF4Linear):
                    raise TypeError(f"{parent_name}.{proj} is {type(mod)}, expected SwitchableNVHiF4Linear")
                setattr(parent, proj, export_switchable_linear(mod))
        if hasattr(layer, "diag_state"):
            del layer.diag_state
    ensure_dir(ckpt_dir)
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    write_json(
        ckpt_dir / "e2e_vllm_materialize.json",
        {"backend": "vllm_materialize", "fake_act_quant": "hif4"},
    )
    return ckpt_dir


def resolve_vllm_eval_spec(
    *,
    variant: str,
    model_path: str,
    artifact_path: str | Path | None,
    artifact_diag_variant: str = "adopted",
    output_dir: Path,
    device: str,
) -> VllmEvalSpec:
    snapshot = resolve_local_snapshot(model_path)
    if variant == "native_nvfp4":
        return VllmEvalSpec(
            model_path=snapshot,
            fake_act_quant=vllm_fake_act_for_variant(variant),
            materialized=False,
            native_nvfp4=(variant == "native_nvfp4"),
        )
    if variant == "direct_hif4":
        ckpt_dir = _shared_moe_ckpt_dir(output_dir, variant)
        if not _complete_materialized_moe_ckpt(ckpt_dir):
            materialize_moe_identity_checkpoint(
                source_snapshot=snapshot,
                output_dir=ckpt_dir,
                layer_indices=None,
                use_r64=False,
            )
        return VllmEvalSpec(
            model_path=ckpt_dir,
            fake_act_quant="none",
            materialized=True,
            hif4_runtime_spec_path=ckpt_dir / "hif4_runtime_spec.pt",
        )
    if variant == "r64_only":
        ckpt_dir = _shared_moe_ckpt_dir(output_dir, variant)
        if not _complete_materialized_moe_ckpt(ckpt_dir):
            materialize_moe_identity_checkpoint(
                source_snapshot=snapshot,
                output_dir=ckpt_dir,
                layer_indices=None,
                use_r64=True,
            )
        return VllmEvalSpec(
            model_path=ckpt_dir,
            fake_act_quant="none",
            materialized=True,
            hif4_runtime_spec_path=ckpt_dir / "hif4_runtime_spec.pt",
        )
    if variant != "artifact":
        raise ValueError(f"unsupported variant {variant!r}")
    if artifact_path is None:
        raise ValueError("artifact variant requires artifact_path")
    materialized_variant = "artifact" if artifact_diag_variant == "adopted" else f"artifact_{artifact_diag_variant}"
    ckpt_dir = _shared_moe_ckpt_dir(output_dir, materialized_variant)
    if not _complete_materialized_moe_ckpt(ckpt_dir):
        materialize_moe_checkpoint(
            source_snapshot=snapshot,
            artifact_path=artifact_path,
            output_dir=ckpt_dir,
            diag_variant=artifact_diag_variant,
        )
    return VllmEvalSpec(
        model_path=ckpt_dir,
        fake_act_quant="none",
        materialized=True,
        hif4_runtime_spec_path=(ckpt_dir / "hif4_runtime_spec.pt"),
    )


def cleanup_materialized_eval_spec(spec: VllmEvalSpec) -> None:
    if not spec.materialized or os.environ.get("KEEP_EVAL_CKPT") == "1":
        return
    try:
        spec.model_path.relative_to(SHARED_VLLM_ROOT)
    except ValueError:
        return
    shutil.rmtree(spec.model_path, ignore_errors=True)


def _patch_lm_eval_transformers() -> None:
    import transformers

    orig = transformers.__class__.__getattr__

    def patched(self, name):  # noqa: ANN001
        if name == "AutoModelForVision2Seq":
            return transformers.AutoModelForImageTextToText
        return orig(self, name)

    transformers.__class__.__getattr__ = patched
    setattr(transformers, "AutoModelForVision2Seq", transformers.AutoModelForImageTextToText)


def run_arc_vllm(
    *,
    spec: VllmEvalSpec,
    output_dir: Path,
    max_model_len: int = 4096,
    max_num_batched_tokens: int = 4096,
) -> dict[str, Any]:
    _patch_lm_eval_transformers()
    from lm_eval import simple_evaluate

    kwargs = build_lm_eval_vllm_kwargs(
        model_path=str(spec.model_path),
        hif4_runtime_spec_path=(
            str(spec.hif4_runtime_spec_path)
            if spec.hif4_runtime_spec_path is not None
            else None
        ),
        native_nvfp4=spec.native_nvfp4,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    if spec.hif4_runtime_spec_path is not None:
        os.environ["HIF4_RUNTIME_SPEC_PATH"] = str(spec.hif4_runtime_spec_path.resolve())
    out = simple_evaluate(
        model="vllm",
        model_args=kwargs,
        tasks=["arc_easy", "arc_challenge"],
        num_fewshot=0,
        batch_size="auto",
    )
    scores = {}
    for task, task_result in out["results"].items():
        if not isinstance(task_result, dict):
            continue
        _key, value = _pick_metric(task_result)
        if value is not None:
            scores[task] = value
    payload = {
        "backend": "lm_eval_vllm",
        "tensor_parallel_size": 2,
        "kv_cache_dtype": "bfloat16",
        "enforce_eager": True,
        "scores": scores,
        "raw_results": out["results"],
    }
    if spec.hif4_runtime_spec_path is not None:
        payload["hif4_runtime_spec_path"] = str(spec.hif4_runtime_spec_path)
    ensure_dir(output_dir / "eval" / "arc")
    write_json(output_dir / "eval" / "arc" / "metrics.json", payload)
    return payload


def _pick_metric(task_result: dict[str, Any]) -> tuple[str, float | None]:
    for key in ("extractive_match", "exact_match", "acc", "acc,none", "qem"):
        if key in task_result and isinstance(task_result[key], (int, float)):
            return key, float(task_result[key])
    return "", None


def parse_main_py_results(output_dir: Path) -> dict[str, Any]:
    files = sorted(output_dir.rglob("results_*.json"))
    if not files:
        raise FileNotFoundError(f"no lighteval results_*.json under {output_dir}")
    payload = json.loads(files[-1].read_text(encoding="utf-8"))
    results = payload.get("results", payload)
    if not isinstance(results, dict):
        raise TypeError(f"unexpected results payload in {files[-1]}")
    return {"results": results, "results_file": str(files[-1])}


def run_main_py_lighteval(
    *,
    model_path: Path,
    output_dir: Path,
    datasets: str,
    max_samples: int | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    fake_act_quant: str,
    disable_thinking: bool,
    tensor_parallel_size: int = REASONING_EVAL_NUM_GPUS,
    hif4_runtime_spec_path: Path | None = None,
    native_nvfp4: bool = False,
) -> dict[str, Any]:
    require_visible_cuda_count(tensor_parallel_size)
    ensure_dir(output_dir)
    cmd = [
        sys.executable,
        str(MAIN_PY),
        "--model_path",
        str(model_path.resolve()),
        "--datasets",
        datasets,
        "--tensor_parallel_size",
        str(tensor_parallel_size),
        "--max_model_length",
        "32768",
        "--max_new_tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
        "--top_p",
        str(top_p),
        "--top_k",
        str(top_k),
        "--gpu_memory_utilization",
        "0.9",
        "--fake_act_quant",
        fake_act_quant,
        "--fake_act_quant_exclude",
        "lm_head",
        "--kv_cache_dtype",
        "bfloat16",
        "--enforce_eager",
        "--output_dir",
        str(output_dir.resolve()),
    ]
    if native_nvfp4:
        cmd.extend(["--linear_backend", "emulation", "--moe_backend", "emulation"])
    if hif4_runtime_spec_path is not None:
        cmd.extend(["--hif4_runtime_spec", str(hif4_runtime_spec_path.resolve())])
        cmd.extend(["--max_num_batched_tokens", "4096"])
    if max_samples is not None:
        cmd.extend(["--max_samples", str(max_samples)])
    if disable_thinking:
        cmd.append("--disable_thinking")
    if fake_act_quant == "nvfp4":
        cmd.append("--allow-deprecated-quantization")
    log_path = output_dir / "main_py.log"
    env = dict(os.environ)
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            check=False,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"main.py failed rc={proc.returncode}; see {log_path}")
    parsed = parse_main_py_results(output_dir)
    parsed["backend"] = "vllm_lighteval"
    parsed["fake_act_quant"] = fake_act_quant
    parsed["tensor_parallel_size"] = tensor_parallel_size
    parsed["kv_cache_dtype"] = "bfloat16"
    parsed["enforce_eager"] = True
    parsed["native_nvfp4"] = native_nvfp4
    if hif4_runtime_spec_path is not None:
        parsed["hif4_runtime_spec_path"] = str(hif4_runtime_spec_path)
    if max_samples is not None:
        parsed["max_samples"] = max_samples
    return parsed


def run_mmlu_pro_300_vllm(
    *,
    variant: str,
    output_dir: Path,
    model_path: str = DEFAULT_MODEL_PATH,
    artifact_path: str | Path | None = None,
    artifact_diag_variant: str = "adopted",
    device: str = "cuda",
) -> dict[str, Any]:
    spec = resolve_vllm_eval_spec(
        variant=variant,
        model_path=model_path,
        artifact_path=artifact_path,
        artifact_diag_variant=artifact_diag_variant,
        output_dir=output_dir,
        device=device,
    )
    eval_root = ensure_dir(output_dir / "eval" / "mmlu_pro")
    results = run_main_py_lighteval(
        model_path=spec.model_path,
        output_dir=eval_root / "vllm_run",
        datasets="mmlu_pro|0",
        max_samples=300,
        max_new_tokens=32768,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        fake_act_quant=spec.fake_act_quant,
        disable_thinking=True,
        hif4_runtime_spec_path=spec.hif4_runtime_spec_path,
        native_nvfp4=spec.native_nvfp4,
    )
    write_json(eval_root / "metrics.json", results)
    return results


def run_aime25_avg5_vllm(
    *,
    variant: str,
    output_dir: Path,
    model_path: str = DEFAULT_MODEL_PATH,
    artifact_path: str | Path | None = None,
    artifact_diag_variant: str = "adopted",
    device: str = "cuda",
) -> dict[str, Any]:
    spec = resolve_vllm_eval_spec(
        variant=variant,
        model_path=model_path,
        artifact_path=artifact_path,
        artifact_diag_variant=artifact_diag_variant,
        output_dir=output_dir,
        device=device,
    )
    eval_root = ensure_dir(output_dir / "eval" / "aime25")
    results = run_main_py_lighteval(
        model_path=spec.model_path,
        output_dir=eval_root / "vllm_run",
        datasets="aime25_avg5|0",
        max_samples=None,
        max_new_tokens=32768,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        fake_act_quant=spec.fake_act_quant,
        disable_thinking=False,
        hif4_runtime_spec_path=spec.hif4_runtime_spec_path,
        native_nvfp4=spec.native_nvfp4,
    )
    write_json(eval_root / "metrics.json", results)
    return results
