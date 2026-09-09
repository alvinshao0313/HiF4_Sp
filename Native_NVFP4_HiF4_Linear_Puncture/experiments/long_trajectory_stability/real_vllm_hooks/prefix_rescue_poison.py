#!/usr/bin/env python3
"""Real-vLLM token-level prefix rescue / poison (Task 14, H2 causal)."""

from __future__ import annotations

import argparse
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
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.gpu_pool import (
    available_gpus,
    cuda_env,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.build_llm import (
    build_real_vllm,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.forced_trajectory import (
    make_forced_sampling_params,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import (
    read_jsonl,
)

DEFAULT_K = (1, 4, 16, 64)
DEFAULT_FREE_TAIL = 256


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", required=True)
    p.add_argument("--isolated_root", required=True)
    p.add_argument("--drilldown_plan", required=True)
    p.add_argument("--mode", choices=["rescue", "poison", "both"], default="both")
    p.add_argument("--k_values", type=int, nargs="+", default=list(DEFAULT_K))
    p.add_argument("--free_tail", type=int, default=DEFAULT_FREE_TAIL)
    p.add_argument("--sample_keys", nargs="*", default=None)
    p.add_argument("--n_earliest", type=int, default=4)
    p.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--phasea_root", default=str(DEFAULT_PHASEA_ROOT))
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    return p.parse_args()


def load_jsonl_by_key(path: Path) -> dict[str, dict]:
    out = {}
    for row in read_jsonl(path):
        out[str(row["prompt_key"])] = row
    return out


def first_divergence(a: list[int], b: list[int]) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if int(a[i]) != int(b[i]):
            return i
    if len(a) != len(b):
        return n
    return None


def greedy_params(max_tokens: int) -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        max_tokens=int(max_tokens),
        ignore_eos=True,
    )


def generate_forced(llm, input_ids: list[int], forced_ids: list[int]) -> list[int]:
    params = make_forced_sampling_params(forced_ids, max_tokens=len(forced_ids))
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=input_ids)],
        [params],
        use_tqdm=False,
    )
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise RuntimeError("expected one forced request/output")
    generated = [int(x) for x in outputs[0].outputs[0].token_ids]
    if generated != [int(x) for x in forced_ids]:
        fd = first_divergence(generated, forced_ids)
        raise RuntimeError(f"forced prefix mismatch at {fd}")
    return generated


def generate_greedy(llm, prompt_ids: list[int], max_tokens: int) -> list[int]:
    if max_tokens <= 0:
        return []
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids)],
        [greedy_params(max_tokens)],
        use_tqdm=False,
    )
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise RuntimeError("expected one greedy request/output")
    return [int(x) for x in outputs[0].outputs[0].token_ids]


def select_samples(plan: dict, events: dict[str, dict], args: argparse.Namespace) -> list[dict]:
    if args.sample_keys:
        keys = [str(x) for x in args.sample_keys]
    else:
        keys = [
            str(x["sample_key"])
            for x in plan["selection"]["earliest_divergence"][: args.n_earliest]
        ]
    rows = []
    for key in keys:
        ev = events.get(key)
        if ev is None or ev.get("first_divergence") is None:
            raise RuntimeError(f"missing E1 divergence event for {key}")
        rows.append(
            {
                "prompt_key": key,
                "first_divergence": int(ev["first_divergence"]),
                "e0_token_at_divergence": int(ev["e0_token_at_divergence"]),
                "variant_token_at_divergence": int(ev["variant_token_at_divergence"]),
            }
        )
    return rows


def run_rescue(
    llm,
    *,
    sample: dict,
    e0: dict,
    k: int,
    free_tail: int,
) -> dict:
    key = sample["prompt_key"]
    fd = int(sample["first_divergence"])
    input_ids = [int(x) for x in e0["input_ids"]]
    e0_out = [int(x) for x in e0["output_ids"]]
    force_len = fd + int(k)
    if force_len <= 0 or force_len > len(e0_out):
        raise RuntimeError(f"{key}: invalid rescue force_len={force_len} e0_len={len(e0_out)}")
    forced = e0_out[:force_len]
    forced_gen = generate_forced(llm, input_ids, forced)
    tail_n = min(int(free_tail), max(0, len(e0_out) - force_len))
    free_gen = generate_greedy(llm, input_ids + forced_gen, tail_n)
    full = forced_gen + free_gen
    e0_cmp = e0_out[: len(full)]
    post = first_divergence(full[force_len:], e0_cmp[force_len:])
    abs_div = None if post is None else force_len + post
    rejoined = post is None and len(free_gen) == tail_n and tail_n > 0
    return {
        "kind": "rescue",
        "sample_key": key,
        "K": int(k),
        "first_divergence": fd,
        "force_len": force_len,
        "free_tail_requested": int(free_tail),
        "free_tail_actual": len(free_gen),
        "first_divergence_after_release": post,
        "absolute_first_divergence_vs_e0": abs_div,
        "rejoined_e0_for_tail": bool(rejoined),
        "matched_e0_prefix": full[:force_len] == e0_out[:force_len],
        "generated_ids": full,
        "e0_compare_ids": e0_cmp,
    }


def run_poison(
    llm,
    *,
    sample: dict,
    e0: dict,
    free_tail: int,
) -> dict:
    key = sample["prompt_key"]
    fd = int(sample["first_divergence"])
    input_ids = [int(x) for x in e0["input_ids"]]
    e0_out = [int(x) for x in e0["output_ids"]]
    e1_tok = int(sample["variant_token_at_divergence"])
    e0_tok = int(sample["e0_token_at_divergence"])
    if fd >= len(e0_out):
        raise RuntimeError(f"{key}: first_divergence={fd} beyond e0_len={len(e0_out)}")
    if int(e0_out[fd]) != e0_tok:
        raise RuntimeError(
            f"{key}: e0 token at fd mismatch: traj={e0_out[fd]} event={e0_tok}"
        )
    forced = e0_out[:fd] + [e1_tok]
    forced_gen = generate_forced(llm, input_ids, forced)
    tail_n = min(int(free_tail), max(0, len(e0_out) - len(forced)))
    free_gen = generate_greedy(llm, input_ids + forced_gen, tail_n)
    full = forced_gen + free_gen
    e0_cmp = e0_out[: len(full)]
    # divergence at fd is intentional; measure whether trajectory returns to E0 after poison
    post = first_divergence(full[fd + 1 :], e0_cmp[fd + 1 :])
    recovered = post is None and len(free_gen) == tail_n and tail_n > 0
    return {
        "kind": "poison",
        "sample_key": key,
        "first_divergence": fd,
        "poison_token": e1_tok,
        "e0_token_at_divergence": e0_tok,
        "force_len": len(forced),
        "free_tail_requested": int(free_tail),
        "free_tail_actual": len(free_gen),
        "first_divergence_after_poison": post,
        "absolute_first_divergence_vs_e0": None if post is None else fd + 1 + post,
        "recovered_e0_after_poison": bool(recovered),
        "generated_ids": full,
        "e0_compare_ids": e0_cmp,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    gpus = available_gpus()
    if len(gpus) < 2:
        raise RuntimeError(
            f"prefix rescue/poison needs 2 idle GPUs; available={gpus}"
        )
    pair = gpus[:2]
    import os

    os.environ.update(cuda_env(pair))
    print(f"[prefix rescue/poison] GPUs={pair}", flush=True)

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    isol = Path(args.isolated_root).resolve()
    plan = json.loads(Path(args.drilldown_plan).read_text(encoding="utf-8"))
    e0_by_key = load_jsonl_by_key(isol / "isolated" / "E0.jsonl")
    events = {
        str(e["prompt_key"]): e
        for e in read_jsonl(isol / "analysis" / "divergence_events.jsonl")
        if str(e.get("variant")) == "E1"
    }
    samples = select_samples(plan, events, args)
    for s in samples:
        if s["prompt_key"] not in e0_by_key:
            raise RuntimeError(f"missing isolated E0 row for {s['prompt_key']}")

    manifest: dict = {
        "schema_version": 1,
        "gpus": pair,
        "mode": args.mode,
        "k_values": [int(x) for x in args.k_values],
        "free_tail": int(args.free_tail),
        "samples": samples,
        "runtimes": {},
    }
    all_rows: list[dict] = []

    if args.mode in {"rescue", "both"}:
        llm_e1, runtime_e1 = build_real_vllm(
            "E1",
            model_path=args.model_path,
            phasea_root=Path(args.phasea_root),
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        manifest["runtimes"]["E1"] = runtime_e1
        for sample in samples:
            e0 = e0_by_key[sample["prompt_key"]]
            for k in args.k_values:
                row = run_rescue(
                    llm_e1, sample=sample, e0=e0, k=int(k), free_tail=args.free_tail
                )
                all_rows.append(row)
                print(
                    f"rescue {row['sample_key']} K={k} "
                    f"post_div={row['first_divergence_after_release']} "
                    f"rejoin={row['rejoined_e0_for_tail']}",
                    flush=True,
                )
        del llm_e1

    if args.mode in {"poison", "both"}:
        llm_e0, runtime_e0 = build_real_vllm(
            "E0",
            model_path=args.model_path,
            phasea_root=Path(args.phasea_root),
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        manifest["runtimes"]["E0"] = runtime_e0
        for sample in samples:
            e0 = e0_by_key[sample["prompt_key"]]
            row = run_poison(llm_e0, sample=sample, e0=e0, free_tail=args.free_tail)
            all_rows.append(row)
            print(
                f"poison {row['sample_key']} "
                f"post_div={row['first_divergence_after_poison']} "
                f"recovered={row['recovered_e0_after_poison']}",
                flush=True,
            )
        del llm_e0

    results_path = out_root / "prefix_rescue_poison.jsonl"
    write_jsonl(results_path, all_rows)

    rescue_rows = [r for r in all_rows if r["kind"] == "rescue"]
    poison_rows = [r for r in all_rows if r["kind"] == "poison"]
    summary = {
        "n_rescue": len(rescue_rows),
        "n_poison": len(poison_rows),
        "rescue_rejoin_rate_by_K": {
            str(k): (
                None
                if not [r for r in rescue_rows if r["K"] == k]
                else float(
                    sum(1 for r in rescue_rows if r["K"] == k and r["rejoined_e0_for_tail"])
                    / len([r for r in rescue_rows if r["K"] == k])
                )
            )
            for k in sorted({r["K"] for r in rescue_rows})
        },
        "poison_recovery_rate": (
            None
            if not poison_rows
            else float(sum(1 for r in poison_rows if r["recovered_e0_after_poison"]) / len(poison_rows))
        ),
    }
    manifest["summary"] = summary
    man_path = out_root / "prefix_rescue_poison_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(results_path)
    print(man_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
