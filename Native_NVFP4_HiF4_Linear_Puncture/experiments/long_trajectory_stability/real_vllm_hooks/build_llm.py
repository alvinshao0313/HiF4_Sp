from __future__ import annotations

import os
from pathlib import Path

import torch
from vllm import LLM

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.vllm_runner import (
    resolve_vllm_eval_spec,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PHASEA_ROOT,
    resolve_variant,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.forced_trajectory import (
    ForcedTrajectoryLogitsProcessor,
)


def resolve_real_vllm_spec(
    variant_name: str,
    *,
    model_path: str = DEFAULT_MODEL_PATH,
    phasea_root: Path = DEFAULT_PHASEA_ROOT,
):
    variant = resolve_variant(variant_name)
    artifact_path = variant.artifact_path(phasea_root)
    resolver_output_dir = variant.phasea_run_dir(phasea_root)
    spec = resolve_vllm_eval_spec(
        variant=variant.eval_variant,
        model_path=model_path,
        artifact_path=artifact_path,
        artifact_diag_variant="adopted",
        output_dir=resolver_output_dir,
        device="cuda",
    )
    runtime_abi_version = None
    if spec.hif4_runtime_spec_path is not None:
        runtime_spec = torch.load(
            spec.hif4_runtime_spec_path, map_location="cpu", weights_only=False
        )
        runtime_abi_version = int(runtime_spec.get("runtime_abi_version", -1))
        if runtime_abi_version != 3:
            raise RuntimeError(
                f"{variant_name} requires HiF4 runtime ABI 3, got {runtime_abi_version}: "
                f"{spec.hif4_runtime_spec_path}"
            )
    return variant, spec, runtime_abi_version


def build_real_vllm(
    variant_name: str,
    *,
    model_path: str = DEFAULT_MODEL_PATH,
    phasea_root: Path = DEFAULT_PHASEA_ROOT,
    gpu_memory_utilization: float = 0.90,
    max_num_seqs: int = 1,
    max_num_batched_tokens: int | None = None,
    seed: int | None = None,
    enable_forced_trajectory_processor: bool = True,
) -> tuple[LLM, dict]:
    """Build real vLLM.

    Default ``max_num_seqs=1`` is for mechanism probes (isolated/hooks/puncture).
    Formal benchmark alignment with Phase-A E0 uses ``max_num_seqs=128``.
    """
    variant, spec, runtime_abi_version = resolve_real_vllm_spec(
        variant_name, model_path=model_path, phasea_root=phasea_root
    )
    # Real vLLM apply_model() RPCs callables to TP workers via msgpack.
    # Custom hook install/begin/flush functions require the documented pickle
    # fallback; this does not change model math or production kernels.
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    additional_config: dict = {}
    max_num_seqs = int(max_num_seqs)
    if max_num_seqs < 1:
        raise ValueError("max_num_seqs must be >= 1")
    kwargs: dict = {
        "model": str(spec.model_path),
        "trust_remote_code": True,
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "max_model_len": 40960,
        "dtype": "auto",
        "enforce_eager": True,
        "kv_cache_dtype": "bfloat16",
        "enable_prefix_caching": False,
        "max_num_seqs": max_num_seqs,
    }
    if seed is not None:
        kwargs["seed"] = int(seed)
    if enable_forced_trajectory_processor:
        kwargs["logits_processors"] = [ForcedTrajectoryLogitsProcessor]
    if spec.native_nvfp4:
        kwargs.update({"linear_backend": "emulation", "moe_backend": "emulation"})
    if spec.hif4_runtime_spec_path is not None:
        runtime_spec_path = str(Path(spec.hif4_runtime_spec_path).resolve())
        os.environ["HIF4_RUNTIME_SPEC_PATH"] = runtime_spec_path
        additional_config["hif4_runtime_spec_path"] = runtime_spec_path
        # Mechanism path historically used 4096; formal Phase-A E0 used 2048.
        kwargs["max_num_batched_tokens"] = (
            int(max_num_batched_tokens) if max_num_batched_tokens is not None else 4096
        )
        # Current formal ABI-3 path selects the Triton MoE backend when a HiF4
        # runtime spec is present.
        kwargs["moe_backend"] = "triton"
    elif max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
    if additional_config:
        kwargs["additional_config"] = additional_config

    manifest = {
        "variant": variant_name,
        "eval_variant": variant.eval_variant,
        "model_path": str(spec.model_path),
        "artifact_path": str(variant.artifact_path(phasea_root))
        if variant.uses_artifact
        else None,
        "hif4_runtime_spec_path": str(spec.hif4_runtime_spec_path)
        if spec.hif4_runtime_spec_path is not None
        else None,
        "runtime_abi_version": runtime_abi_version,
        "native_nvfp4": bool(spec.native_nvfp4),
        "tensor_parallel_size": 2,
        "kv_cache_dtype": "bfloat16",
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": kwargs.get("max_num_batched_tokens"),
        "seed": seed,
        "max_model_len": 40960,
        "gpu_memory_utilization": float(gpu_memory_utilization),
    }
    return LLM(**kwargs), manifest
