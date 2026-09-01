#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[3]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.config import (
    DEFAULT_FREE_RUN_MAX_NEW_TOKENS,
    DEFAULT_FREE_RUN_SAMPLES,
    DEFAULT_MODEL_PATH,
    DEFAULT_PHASEA_ROOT,
    REPO_ROOT,
    resolve_variant,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.vllm_runner import (
    resolve_vllm_eval_spec,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot

CHAT_TEMPLATE_NAME = "chat_template.jinja"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=["E0", "E1", "E2", "E3", "E4"])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--phasea_root", default=str(DEFAULT_PHASEA_ROOT))
    p.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--max_samples", type=int, default=DEFAULT_FREE_RUN_SAMPLES)
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_FREE_RUN_MAX_NEW_TOKENS)
    p.add_argument("--tensor_parallel_size", type=int, default=2)
    p.add_argument("--profile", choices=["greedy", "task_matched"], default="greedy")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--min_p", type=float, default=None)
    return p.parse_args()


def native_chat_template_path(model_id: str) -> Path:
    snapshot = resolve_local_snapshot(model_id)
    path = snapshot / CHAT_TEMPLATE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"native snapshot missing {CHAT_TEMPLATE_NAME}: {path}")
    return path


def ensure_chat_template_model_path(model_path: Path, native_template: Path, output_dir: Path) -> Path:
    """Phase-A HiF4 sidecars omit chat_template.jinja. Expose it via a local symlink view.

    Does not write into Phase-A / shared materialization directories.
    """
    model_path = Path(model_path).resolve()
    native_template = Path(native_template).resolve()
    if (model_path / CHAT_TEMPLATE_NAME).is_file():
        return model_path
    view = Path(output_dir).resolve() / "model_view"
    view.mkdir(parents=True, exist_ok=True)
    for item in model_path.iterdir():
        dst = view / item.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(item.resolve(), dst)
    dst = view / CHAT_TEMPLATE_NAME
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(native_template, dst)
    return view


def sampling_args(args: argparse.Namespace) -> tuple[float, float, int, float]:
    if args.profile == "greedy":
        default = (0.0, 1.0, -1, 0.0)
    else:
        # Matches the current Phase-A MMLU-Pro runner. Explicit CLI values always win.
        default = (0.6, 0.95, 20, 0.0)
    return (
        default[0] if args.temperature is None else args.temperature,
        default[1] if args.top_p is None else args.top_p,
        default[2] if args.top_k is None else args.top_k,
        default[3] if args.min_p is None else args.min_p,
    )


def main() -> None:
    args = parse_args()
    variant = resolve_variant(args.variant)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    phasea_root = Path(args.phasea_root)
    artifact_path = variant.artifact_path(phasea_root)
    # Reuse the exact Phase-A materialization key (E1_direct_hif4 / E2_r64_only /
    # E3_fusable / E4_fusable_r64) so this diagnostic does not duplicate large
    # vLLM checkpoints or accidentally drift to a different runtime sidecar.
    resolver_output_dir = variant.phasea_run_dir(phasea_root)
    spec = resolve_vllm_eval_spec(
        variant=variant.eval_variant,
        model_path=args.model_path,
        artifact_path=artifact_path,
        artifact_diag_variant="adopted",
        output_dir=resolver_output_dir,
        device="cuda",
    )
    runtime_abi_version = None
    if spec.hif4_runtime_spec_path is not None:
        runtime_spec = torch.load(spec.hif4_runtime_spec_path, map_location="cpu", weights_only=False)
        runtime_abi_version = int(runtime_spec.get("runtime_abi_version", -1))
        if runtime_abi_version != 3:
            raise RuntimeError(
                f"{args.variant} requires optimized HiF4 runtime ABI 3, got {runtime_abi_version}: "
                f"{spec.hif4_runtime_spec_path}"
            )
    temperature, top_p, top_k, min_p = sampling_args(args)
    native_template = native_chat_template_path(args.model_path)
    capture_model_path = ensure_chat_template_model_path(spec.model_path, native_template, output_dir)
    manifest = {
        "variant": args.variant,
        "profile": args.profile,
        "eval_variant": variant.eval_variant,
        "model_path": str(capture_model_path),
        "source_model_path": str(spec.model_path),
        "chat_template_injected": str(capture_model_path) != str(Path(spec.model_path).resolve()),
        "artifact_path": str(artifact_path) if artifact_path is not None else None,
        "hif4_runtime_spec_path": str(spec.hif4_runtime_spec_path) if spec.hif4_runtime_spec_path is not None else None,
        "runtime_abi_version": runtime_abi_version,
        "native_nvfp4": bool(spec.native_nvfp4),
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "max_samples": int(args.max_samples),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "min_p": float(min_p),
        "kv_cache_dtype": "bfloat16",
        "enforce_eager": True,
    }
    (output_dir / "capture_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    cmd = [
        sys.executable,
        str(
            REPO_ROOT
            / "Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/capture_main.py"
        ),
        "--model_path",
        str(capture_model_path),
        "--datasets",
        "mmlu_pro|0",
        "--tensor_parallel_size",
        str(args.tensor_parallel_size),
        "--max_model_length",
        "40960",
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(temperature),
        "--top_p",
        str(top_p),
        "--top_k",
        str(top_k),
        "--min_p",
        str(min_p),
        "--gpu_memory_utilization",
        "0.9",
        "--fake_act_quant",
        spec.fake_act_quant,
        "--fake_act_quant_exclude",
        "lm_head",
        "--kv_cache_dtype",
        "bfloat16",
        "--enforce_eager",
        "--max_samples",
        str(args.max_samples),
        "--output_dir",
        str(output_dir),
    ]
    if spec.native_nvfp4:
        cmd.extend(["--linear_backend", "emulation", "--moe_backend", "emulation"])
    if spec.hif4_runtime_spec_path is not None:
        cmd.extend(["--hif4_runtime_spec", str(spec.hif4_runtime_spec_path)])
        cmd.extend(["--max_num_batched_tokens", "4096"])
    env = dict(os.environ)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


if __name__ == "__main__":
    main()
