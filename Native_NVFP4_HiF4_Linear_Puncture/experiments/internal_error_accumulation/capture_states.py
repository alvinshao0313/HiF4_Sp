"""E0/E1 actual-path capture for calibration prefix states."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from vllm.inputs import TokensPrompt

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.build_llm import (
    build_real_vllm,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.forced_trajectory import (
    make_forced_sampling_params,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.hook_spec import (
    build_probe_map,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.worker_hooks import (
    BeginSampleOp,
    InstallHooksOp,
    flush_sample,
    remove_hooks,
)

from .config import DEFAULT_MODEL_PATH, DEFAULT_PHASEA_ROOT
from .run_state import atomic_write_json, read_jsonl


def capture_variant_states(
    *,
    variant: str,
    cohort_path: Path,
    output_root: Path,
    model_path: str = DEFAULT_MODEL_PATH,
    phasea_root: Path = DEFAULT_PHASEA_ROOT,
    capture_level: str = "core_qkv",
    sample_keys: list[str] | None = None,
    gpu_memory_utilization: float = 0.90,
) -> dict[str, Any]:
    states = read_jsonl(cohort_path)
    if sample_keys is not None:
        wanted = set(sample_keys)
        states = [s for s in states if s["sample_key"] in wanted]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    llm, runtime_manifest = build_real_vllm(
        variant,
        model_path=model_path,
        phasea_root=Path(phasea_root),
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=1,
    )
    install = llm.apply_model(InstallHooksOp(variant, str(output_root / "hooks"), capture_level))
    if len(install) != 2:
        raise RuntimeError(f"expected 2 TP install replies, got {install}")
    results = []
    try:
        for state in states:
            key = state["sample_key"]
            prompt = [int(x) for x in state["prompt_token_ids"]]
            forced = [int(x) for x in state["forced_token_ids"]]
            decode_index = int(state["decode_index"])
            if decode_index != len(forced) - 1:
                raise RuntimeError(
                    f"{key}: decode_index={decode_index} must equal len(forced)-1={len(forced)-1}"
                )
            if int(forced[-1]) != int(state["target_token_id"]):
                raise RuntimeError(f"{key}: forced final token != target_token_id")
            probe_map = build_probe_map(len(prompt), [{"decode_index": decode_index}])
            begin = llm.apply_model(
                BeginSampleOp(sample_key=key, prompt_len=len(prompt), probe_abs_to_decode=probe_map)
            )
            if len(begin) != 2:
                raise RuntimeError(f"begin_sample replies={begin}")
            params = make_forced_sampling_params(
                forced,
                max_tokens=len(forced),
                sample_key=key,
                variant=variant,
                probe_decode_indices=[decode_index],
                logits_root=str(output_root / "raw_logits"),
            )
            outputs = llm.generate(
                [TokensPrompt(prompt_token_ids=prompt)],
                [params],
                use_tqdm=False,
            )
            generated = [int(x) for x in outputs[0].outputs[0].token_ids]
            if generated != forced:
                raise RuntimeError(
                    f"forced trajectory mismatch for {key}: generated={generated[:8]}... "
                    f"expected={forced[:8]}..."
                )
            flush = llm.apply_model(flush_sample)
            if len(flush) != 2:
                raise RuntimeError(f"flush replies={flush}")
            for row in flush:
                row["sha256"] = hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
            results.append(
                {
                    "sample_key": key,
                    "variant": variant,
                    "prompt_len": len(prompt),
                    "decode_index": decode_index,
                    "prefix_length_j": state["prefix_length_j"],
                    "target_token_id": state["target_token_id"],
                    "forced_exact": True,
                    "flush": flush,
                }
            )
            atomic_write_json(output_root / "per_state" / f"{key}.json", results[-1])
    finally:
        llm.apply_model(remove_hooks)
    manifest = {
        "variant": variant,
        "capture_level": capture_level,
        "runtime": runtime_manifest,
        "n_states": len(results),
        "install": install,
        "results": results,
    }
    atomic_write_json(output_root / "capture_manifest.json", manifest)
    return manifest


def load_rank_records(hooks_root: Path, variant: str, sample_key: str, rank: int) -> list[dict]:
    path = Path(hooks_root) / variant / sample_key / f"rank{rank}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "records" in payload:
        return list(payload["records"])
    if isinstance(payload, list):
        return payload
    raise RuntimeError(f"unexpected capture payload at {path}")


def load_raw_logits(logits_root: Path, variant: str, sample_key: str, decode_index: int, rank: int = 0) -> torch.Tensor:
    path = Path(logits_root) / variant / sample_key / f"rank{rank}_decode{decode_index}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["logits"]
