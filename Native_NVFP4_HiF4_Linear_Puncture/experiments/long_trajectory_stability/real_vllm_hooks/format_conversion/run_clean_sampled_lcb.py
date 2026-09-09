#!/usr/bin/env python3
"""Clean controlled sampled LCB free-run aligned to formal Phase-A E0.

Formal benchmark shape: exact E0 raw-cache input_ids, max_num_seqs=128, seed=1234.
``max_num_seqs=1`` is reserved for mechanism probes elsewhere.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
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
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import (
    read_jsonl,
    write_jsonl,
)

# Aligned to Phase-A E0 formal LCB results model_config.
CLEAN_PROTOCOL = {
    "dataset": "lcb:codegeneration_v6|0",
    "thinking": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "max_new_tokens": 38912,
    "tensor_parallel_size": 2,
    "kv_cache_dtype": "bfloat16",
    "max_model_length": 40960,
    "enforce_eager": True,
    "max_num_seqs": 128,
    "max_num_batched_tokens": 2048,
    "enable_prefix_caching": False,
    "speculative_decoding": False,
    "seed_policy": "phasea_e0_global_seed",
    "seed": 1234,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True, choices=["E0", "E1", "E2", "E3", "E4"])
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--prompt_manifest", required=True, type=Path)
    p.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--phasea_root", default=str(DEFAULT_PHASEA_ROOT))
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max_samples", type=int, default=0, help="0 means all manifest rows")
    return p.parse_args()


def sampled_params(max_new_tokens: int) -> SamplingParams:
    return SamplingParams(
        temperature=float(CLEAN_PROTOCOL["temperature"]),
        top_p=float(CLEAN_PROTOCOL["top_p"]),
        top_k=int(CLEAN_PROTOCOL["top_k"]),
        min_p=float(CLEAN_PROTOCOL["min_p"]),
        max_tokens=int(max_new_tokens),
        # Per-request seed left unset: Phase-A E0 used global LLM seed=1234 only.
    )


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def validate_completed_rows(rows: list[dict], source_rows: list[dict], max_new_tokens: int) -> None:
    existing = {str(row["prompt_key"]): row for row in rows}
    if len(existing) != len(rows):
        raise RuntimeError("duplicate clean output prompt_key")
    for row in rows:
        for field in ("input_ids", "output_ids"):
            ids = row.get(field)
            if not isinstance(ids, list) or not ids or not all(type(x) is int and x >= 0 for x in ids):
                raise RuntimeError(f"incomplete clean trajectory {row['prompt_key']}: invalid {field}")
        if (
            type(row.get("output_len")) is not int
            or row["output_len"] != len(row["output_ids"])
            or row["output_len"] > max_new_tokens
            or not isinstance(row.get("raw_text"), str)
            or type(row.get("seed")) is not int
        ):
            raise RuntimeError(f"incomplete clean trajectory {row['prompt_key']}: metadata differs")
    for source in source_rows:
        old = existing.get(str(source["prompt_key"]))
        if old is not None and old["input_ids"] != source["input_ids"]:
            raise RuntimeError(f"clean resume input_ids mismatch: {source['prompt_key']}")


def write_completion_metadata(out: Path, rows: list[dict], protocol: dict,
                              runtime_manifest: dict, prompt_source: str) -> None:
    lengths = [int(row["output_len"]) for row in rows]
    _atomic_json(
        out.with_suffix(".meta.json"),
        {
            "schema_version": 2,
            "execution_shape": "formal_benchmark_max_num_seqs_128",
            "variant": protocol["variant"],
            "prompt_source": str(Path(prompt_source).resolve()),
            "num_trajectories": len(rows),
            "min_output_len": min(lengths),
            "max_output_len": max(lengths),
            "clean_protocol": CLEAN_PROTOCOL,
            "runtime": runtime_manifest,
        },
    )


def main() -> None:
    args = parse_args()
    if args.variant == "E0":
        raise SystemExit(
            "formal LCB E0 must be reused from audited Phase-A results; "
            "do not re-run E0 benchmark with this driver"
        )
    source_rows = read_jsonl(args.prompt_manifest)
    if args.max_samples and args.max_samples > 0:
        source_rows = source_rows[: int(args.max_samples)]
    if not source_rows:
        raise RuntimeError("empty clean prompt manifest")
    if any(row.get("prompt_source") != "E0_formal_raw_cache_exact_input_ids" for row in source_rows):
        raise RuntimeError("clean sampled LCB requires exact E0 raw-cache input_ids manifest")

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    resume_path = out.with_suffix(".resume.json")
    protocol = {
        "variant": args.variant,
        "model_path": args.model_path,
        "phasea_root": str(Path(args.phasea_root).resolve()),
        "prompt_manifest": str(Path(args.prompt_manifest).resolve()),
        "clean_protocol": CLEAN_PROTOCOL,
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "build_llm_sha256": hashlib.sha256(
            Path(__file__).resolve().parents[1].joinpath("build_llm.py").read_bytes()
        ).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    rows: list[dict] = []
    saved_runtime = None
    if args.resume and out.exists():
        saved = json.loads(resume_path.read_text()) if resume_path.exists() else {}
        if saved.get("schema_version") != 1 or saved.get("protocol") != protocol:
            raise RuntimeError("clean resume manifest differs from requested protocol")
        saved_runtime = saved.get("runtime")
        if not isinstance(saved_runtime, dict) or not saved_runtime:
            raise RuntimeError("clean resume missing runtime manifest")
        rows = read_jsonl(out)
        validate_completed_rows(rows, source_rows, int(CLEAN_PROTOCOL["max_new_tokens"]))
        existing = {str(row["prompt_key"]): row for row in rows}
        source_rows = [row for row in source_rows if str(row["prompt_key"]) not in existing]
    if not source_rows:
        write_completion_metadata(out, rows, protocol, saved_runtime, args.prompt_manifest)
        print(f"[clean {args.variant}] all requested trajectories already complete", flush=True)
        return
    validate_completed_rows(rows, source_rows, int(CLEAN_PROTOCOL["max_new_tokens"]))
    out.with_suffix(".meta.json").unlink(missing_ok=True)

    llm, runtime_manifest = build_real_vllm(
        args.variant,
        model_path=args.model_path,
        phasea_root=Path(args.phasea_root),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=int(CLEAN_PROTOCOL["max_num_seqs"]),
        max_num_batched_tokens=int(CLEAN_PROTOCOL["max_num_batched_tokens"]),
        seed=int(CLEAN_PROTOCOL["seed"]),
        enable_forced_trajectory_processor=False,
    )
    if runtime_manifest.get("max_num_seqs") != 128:
        raise RuntimeError(f"execution shape lock failed: {runtime_manifest}")
    if saved_runtime is not None and saved_runtime != runtime_manifest:
        raise RuntimeError("clean resume actual runtime differs from previous trajectories")
    _atomic_json(resume_path, {"schema_version": 1, "protocol": protocol, "runtime": runtime_manifest})

    batch_size = int(CLEAN_PROTOCOL["max_num_seqs"])
    params = sampled_params(int(CLEAN_PROTOCOL["max_new_tokens"]))
    global_seed = int(CLEAN_PROTOCOL["seed"])
    done = len(rows)
    total = done + len(source_rows)
    for start in range(0, len(source_rows), batch_size):
        batch = source_rows[start : start + batch_size]
        prompts = [TokensPrompt(prompt_token_ids=[int(x) for x in s["input_ids"]]) for s in batch]
        outputs = llm.generate(prompts, [params] * len(batch), use_tqdm=False)
        if len(outputs) != len(batch):
            raise RuntimeError(f"unexpected vLLM batch cardinality: {len(outputs)} != {len(batch)}")
        for source, out_item in zip(batch, outputs):
            if len(out_item.outputs) != 1:
                raise RuntimeError(f"unexpected n_outputs for {source['prompt_key']}")
            candidate = out_item.outputs[0]
            output_ids = [int(x) for x in candidate.token_ids]
            if not output_ids:
                raise RuntimeError(f"empty clean output for {source['prompt_key']}")
            rows.append(
                {
                    "prompt_key": str(source["prompt_key"]),
                    "doc_id": str(source["doc_id"]),
                    "input_ids": [int(x) for x in source["input_ids"]],
                    "output_ids": output_ids,
                    "output_len": len(output_ids),
                    "raw_text": candidate.text,
                    "seed": global_seed,
                    "finish_reason": getattr(candidate, "finish_reason", None),
                    "variant": args.variant,
                }
            )
            done += 1
            print(
                f"[clean {args.variant}] {done}/{total} doc={source['doc_id']} "
                f"{source['prompt_key']} seed={global_seed} output_len={len(output_ids)}",
                flush=True,
            )
        write_jsonl(out, rows)
        _atomic_json(resume_path, {"schema_version": 1, "protocol": protocol, "runtime": runtime_manifest})

    write_completion_metadata(out, rows, protocol, runtime_manifest, args.prompt_manifest)
    print(f"[clean {args.variant}] COMPLETE n={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
