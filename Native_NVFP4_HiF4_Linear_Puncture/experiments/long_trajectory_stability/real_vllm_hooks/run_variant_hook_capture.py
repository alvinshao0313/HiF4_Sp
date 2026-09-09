#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm import SamplingParams
from vllm.inputs import TokensPrompt

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PHASEA_ROOT,
)
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
    inspect_hook_state,
    remove_hooks,
)


DEFAULT_SMOKE_PLAN = (
    REPO_ROOT
    / "Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/"
    "trajectory_stability_smoke/analysis/probe_plan.json"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=["E0", "E1", "E2", "E3", "E4"])
    p.add_argument("--output_root", required=True)
    p.add_argument("--probe_plan", default=str(DEFAULT_SMOKE_PLAN))
    p.add_argument(
        "--mode",
        required=True,
        choices=["forced_logits_only", "forced_core", "greedy_none", "greedy_core",
                 "forced_feature_scan", "greedy_feature_scan", "forced_core_qkv"],
    )
    p.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--phasea_root", default=str(DEFAULT_PHASEA_ROOT))
    p.add_argument("--sample_keys", nargs="*", default=None)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--smoke_focus", action="store_true")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    return p.parse_args()


def load_samples(path: Path, sample_keys: list[str] | None, smoke_focus: bool) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = list(payload["samples"])
    if sample_keys:
        wanted = set(sample_keys)
        rows = [row for row in rows if str(row["prompt_key"]) in wanted]
        missing = wanted - {str(row["prompt_key"]) for row in rows}
        if missing:
            raise RuntimeError(f"sample keys missing from probe plan: {sorted(missing)}")
    if smoke_focus:
        wanted = {"n159_c94220492", "n346_c216392826"}
        rows = [row for row in rows if str(row["prompt_key"]) in wanted]
        if {str(row["prompt_key"]) for row in rows} != wanted:
            raise RuntimeError("smoke probe plan must contain both n159 and n346")
        for row in rows:
            key = str(row["prompt_key"])
            indices = [10, 63] if key.startswith("n159_") else [12, 24, 35, 63]
            indices = [idx for idx in indices if idx < len(row["output_ids"])]
            row["positions"] = [
                {"decode_index": idx, "bin": "smoke_focus", "reasons": ["real_vllm_hook_smoke"]}
                for idx in indices
            ]
            row["max_required_decode_index"] = max(indices)
    if not rows:
        raise RuntimeError("no samples selected")
    return rows


def greedy_params(max_tokens: int) -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        max_tokens=int(max_tokens),
    )


def run_one(llm, args: argparse.Namespace, sample: dict, hooks_enabled: bool) -> dict:
    key = str(sample["prompt_key"])
    input_ids = [int(x) for x in sample["input_ids"]]
    output_ids = [int(x) for x in sample["output_ids"]]
    probe_decode_indices = sorted(int(x["decode_index"]) for x in sample["positions"])
    required_tokens = max(probe_decode_indices) + 1
    if args.max_tokens is not None:
        required_tokens = min(required_tokens, int(args.max_tokens))
    if required_tokens <= max(probe_decode_indices):
        raise RuntimeError(
            f"max_tokens={required_tokens} does not reach last probe={max(probe_decode_indices)}"
        )
    probe_decode_indices = [idx for idx in probe_decode_indices if idx < required_tokens]
    probe_map = build_probe_map(len(input_ids), [{"decode_index": idx} for idx in probe_decode_indices])

    if hooks_enabled:
        begin = llm.apply_model(
            BeginSampleOp(
                sample_key=key,
                prompt_len=len(input_ids),
                probe_abs_to_decode=probe_map,
            )
        )
        if len(begin) != 2:
            raise RuntimeError(f"expected two TP begin_sample replies, got {begin}")

    if args.mode.startswith("forced_"):
        logit_probes = [] if args.mode == "forced_feature_scan" else probe_decode_indices
        params = make_forced_sampling_params(
            output_ids,
            max_tokens=required_tokens,
            sample_key=key,
            variant=args.variant,
            probe_decode_indices=logit_probes,
            logits_root=str(Path(args.output_root).resolve() / "raw_logits"),
        )
    else:
        params = greedy_params(required_tokens)

    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=input_ids)],
        [params],
        use_tqdm=False,
    )
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise RuntimeError("expected exactly one vLLM request/output")
    generated = [int(x) for x in outputs[0].outputs[0].token_ids]
    expected_forced = output_ids[:required_tokens]
    forced_exact = generated == expected_forced if args.mode.startswith("forced_") else None
    if args.mode.startswith("forced_") and not forced_exact:
        first = next(
            (i for i, (a, b) in enumerate(zip(generated, expected_forced)) if a != b),
            min(len(generated), len(expected_forced)),
        )
        raise RuntimeError(
            f"forced trajectory mismatch for {key}: first={first} "
            f"generated_len={len(generated)} expected_len={len(expected_forced)}"
        )

    flush = None
    if hooks_enabled:
        flush = llm.apply_model(flush_sample)
        if len(flush) != 2:
            raise RuntimeError(f"expected two TP flush replies, got {flush}")
        for row in flush:
            row["sha256"] = hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()

    return {
        "sample_key": key,
        "input_ids": input_ids,
        "canonical_output_ids_sha256": hashlib.sha256(json.dumps(output_ids, separators=(",", ":")).encode()).hexdigest(),
        "prompt_len": len(input_ids),
        "required_tokens": required_tokens,
        "probe_decode_indices": probe_decode_indices,
        "generated_ids": generated,
        "expected_forced_ids": expected_forced if args.mode.startswith("forced_") else None,
        "forced_exact": forced_exact,
        "flush": flush,
    }


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    plan_sha256 = hashlib.sha256(Path(args.probe_plan).read_bytes()).hexdigest()
    samples = load_samples(Path(args.probe_plan), args.sample_keys, args.smoke_focus)
    llm, runtime_manifest = build_real_vllm(
        args.variant,
        model_path=args.model_path,
        phasea_root=Path(args.phasea_root),
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    hooks_enabled = args.mode in {"forced_core", "greedy_core", "forced_feature_scan",
                                 "greedy_feature_scan", "forced_core_qkv"}
    capture_level = ("feature_scan" if args.mode.endswith("feature_scan") else
                     "core_qkv" if args.mode.endswith("core_qkv") else "core")
    hook_install = None
    if hooks_enabled:
        hook_install = llm.apply_model(
            InstallHooksOp(
                variant=args.variant,
                output_root=str(out_root / "hooks"),
                capture_level=capture_level,
            )
        )
        if len(hook_install) != 2:
            raise RuntimeError(f"expected two TP hook install replies, got {hook_install}")
        for row in hook_install:
            if int(row["num_layers"]) != 48 or int(row["world_size"]) != 2:
                raise RuntimeError(f"invalid hook install reply: {row}")

    sample_results = []
    for sample in samples:
        sample_results.append(run_one(llm, args, sample, hooks_enabled))

    hook_final = None
    hook_remove = None
    if hooks_enabled:
        hook_final = llm.apply_model(inspect_hook_state)
        hook_remove = llm.apply_model(remove_hooks)

    if hashlib.sha256(Path(args.probe_plan).read_bytes()).hexdigest() != plan_sha256:
        raise RuntimeError("probe plan changed during capture")
    manifest = {
        "schema_version": 1,
        "variant": args.variant,
        "mode": args.mode,
        "runtime": runtime_manifest,
        "probe_plan": str(Path(args.probe_plan).resolve()),
        "probe_plan_sha256": plan_sha256,
        "smoke_focus": bool(args.smoke_focus),
        "hook_install": hook_install,
        "hook_final": hook_final,
        "hook_remove": hook_remove,
        "samples": sample_results,
    }
    manifest_path = out_root / f"{args.variant}_{args.mode}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
