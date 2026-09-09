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
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import (
    read_jsonl,
    write_jsonl,
)


DEFAULT_PROMPT_SOURCE = (
    REPO_ROOT
    / "Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/"
    "trajectory_stability_formal/normalized/E0.jsonl"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=["E0", "E1", "E2", "E3", "E4"])
    p.add_argument("--output", required=True)
    p.add_argument("--prompt_source", default=str(DEFAULT_PROMPT_SOURCE))
    p.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--phasea_root", default=str(DEFAULT_PHASEA_ROOT))
    p.add_argument("--max_samples", type=int, default=64)
    p.add_argument("--max_new_tokens", type=int, default=16384)
    p.add_argument("--sample_keys", nargs="*", default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def greedy_params(max_new_tokens: int) -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        max_tokens=int(max_new_tokens),
    )


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def validate_completed_rows(rows: list[dict], source_rows: list[dict], max_new_tokens: int) -> None:
    existing = {str(row["prompt_key"]): row for row in rows}
    if len(existing) != len(rows):
        raise RuntimeError("duplicate isolated output prompt_key")
    if len({str(row["prompt_key"]) for row in source_rows}) != len(source_rows):
        raise RuntimeError("duplicate isolated source prompt_key")
    for row in rows:
        for field in ("input_ids", "output_ids"):
            ids = row.get(field)
            if not isinstance(ids, list) or not ids or not all(type(x) is int and x >= 0 for x in ids):
                raise RuntimeError(f"incomplete isolated trajectory {row['prompt_key']}: invalid {field}")
        if (type(row.get("output_len")) is not int or row["output_len"] != len(row["output_ids"])
                or row["output_len"] > max_new_tokens or not isinstance(row.get("raw_text"), str)):
            raise RuntimeError(f"incomplete isolated trajectory {row['prompt_key']}: output metadata differs")
    for source in source_rows:
        old = existing.get(str(source["prompt_key"]))
        if old is not None and old["input_ids"] != source["input_ids"]:
            raise RuntimeError(f"isolated resume input_ids mismatch: {source['prompt_key']}")


def write_completion_metadata(out: Path, rows: list[dict], protocol: dict,
                              runtime_manifest: dict, prompt_source: str) -> None:
    if not rows or not isinstance(runtime_manifest, dict) or not runtime_manifest:
        raise RuntimeError("isolated completion metadata requires rows and persisted runtime manifest")
    lengths = [int(row["output_len"]) for row in rows]
    _atomic_json(out.with_suffix(".meta.json"), {
        "schema_version": 1,
        "execution_shape": "single_request_max_num_seqs_1",
        "variant": protocol["variant"],
        "prompt_source": str(Path(prompt_source).resolve()),
        "num_trajectories": len(rows),
        "min_output_len": min(lengths),
        "max_output_len": max(lengths),
        "max_new_tokens": protocol["max_new_tokens"],
        "runtime": runtime_manifest,
    })


def main() -> None:
    args = parse_args()
    source_rows = read_jsonl(Path(args.prompt_source))
    if args.sample_keys:
        wanted = set(args.sample_keys)
        source_rows = [row for row in source_rows if str(row["prompt_key"]) in wanted]
        missing = wanted - {str(row["prompt_key"]) for row in source_rows}
        if missing:
            raise RuntimeError(f"prompt source missing sample keys: {sorted(missing)}")
    else:
        source_rows = source_rows[: int(args.max_samples)]
    if not source_rows:
        raise RuntimeError("no isolated free-run prompts selected")

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    resume_path = out.with_suffix(".resume.json")
    protocol = {
        "variant": args.variant, "model_path": args.model_path,
        "phasea_root": str(Path(args.phasea_root).resolve()),
        "max_new_tokens": int(args.max_new_tokens), "temperature": 0,
        "top_p": 1, "top_k": 0, "min_p": 0,
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "build_llm_sha256": hashlib.sha256(Path(__file__).with_name("build_llm.py").read_bytes()).hexdigest(),
    }
    rows = []
    saved_runtime = None
    if args.resume and out.exists():
        saved = json.loads(resume_path.read_text()) if resume_path.exists() else {}
        if saved.get("schema_version") != 1 or saved.get("protocol") != protocol:
            raise RuntimeError("isolated resume manifest differs from requested protocol")
        saved_runtime = saved.get("runtime")
        if not isinstance(saved_runtime, dict) or not saved_runtime:
            raise RuntimeError("isolated resume is missing its actual runtime manifest")
        rows = read_jsonl(out)
        validate_completed_rows(rows, source_rows, int(args.max_new_tokens))
        existing = {str(row["prompt_key"]): row for row in rows}
        source_rows = [row for row in source_rows if str(row["prompt_key"]) not in existing]
    if not source_rows:
        # The last request can be safely published before a crash interrupts
        # final metadata. Reconstruct it from the persisted actual runtime.
        write_completion_metadata(out, rows, protocol, saved_runtime, args.prompt_source)
        print(f"[isolated {args.variant}] all requested trajectories already complete", flush=True)
        return
    validate_completed_rows(rows, source_rows, int(args.max_new_tokens))
    # A previous completed subset must not advertise completion while the
    # cohort is being expanded and additional trajectories are published.
    out.with_suffix(".meta.json").unlink(missing_ok=True)

    llm, runtime_manifest = build_real_vllm(
        args.variant,
        model_path=args.model_path,
        phasea_root=Path(args.phasea_root),
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if saved_runtime is not None and saved_runtime != runtime_manifest:
        raise RuntimeError("isolated resume actual runtime differs from previous trajectories")
    _atomic_json(resume_path, {"schema_version": 1, "protocol": protocol, "runtime": runtime_manifest})
    params = greedy_params(args.max_new_tokens)
    for index, source in enumerate(source_rows):
        input_ids = [int(x) for x in source["input_ids"]]
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=input_ids)],
            [params],
            use_tqdm=False,
        )
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise RuntimeError(f"unexpected vLLM output cardinality for {source['prompt_key']}")
        candidate = outputs[0].outputs[0]
        output_ids = [int(x) for x in candidate.token_ids]
        if not output_ids:
            raise RuntimeError(f"empty isolated output for {source['prompt_key']}")
        rows.append(
            {
                "prompt_key": str(source["prompt_key"]),
                "doc_id": source.get("doc_id"),
                "input_ids": input_ids,
                "output_ids": output_ids,
                "output_len": len(output_ids),
                "raw_text": candidate.text,
                "metric": source.get("metric"),
                "gold": source.get("gold"),
                "specific": source.get("specific"),
            }
        )
        # Persist each completed request, so process interruption loses at most
        # the active request. Publication of the full task remains manifest-gated.
        temporary = out.with_suffix(".jsonl.tmp")
        write_jsonl(temporary, rows)
        temporary.replace(out)
        print(
            f"[isolated {args.variant}] {index + 1}/{len(source_rows)} "
            f"{source['prompt_key']} output_len={len(output_ids)}",
            flush=True,
        )

    write_completion_metadata(out, rows, protocol, runtime_manifest, args.prompt_source)
    print(out)


if __name__ == "__main__":
    main()
