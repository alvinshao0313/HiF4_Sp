"""Task-correctness prefix interventions using one incremental vLLM request.

``k`` is a prefix length: the first freely selected output token has index k.
Short convergence screens never carry an accuracy label. Full-budget completions
are judged only by the shared official LiveCodeBench judge adapter.
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


def contains_subsequence(values: list[int], sequence: list[int]) -> bool:
    if not sequence:
        raise ValueError("token subsequence must not be empty")
    return any(values[i:i + len(sequence)] == sequence for i in range(len(values) - len(sequence) + 1))


def coarse_prefix_lengths(t_lex: int, source_length: int, *, thinking_end: int | None = None,
                          code_start: int | None = None, total_budget: int = 38912) -> list[int]:
    if t_lex < 0 or source_length <= 0 or total_budget <= 0:
        raise ValueError("invalid frontier trajectory/budget")
    candidates = {t_lex + delta for delta in (0, 8, 16, 32, 64, 128, 256, 512)}
    for transition in (thinking_end, code_start):
        if transition is not None:
            # Transition indices refer to output tokens. Include both sides.
            candidates.update((transition - 1, transition, transition + 1))
    return sorted(k for k in candidates if 0 <= k <= source_length and k < total_budget)


def full_budget_points(screen_rows: list[dict]) -> list[int]:
    """Select both neighbors of every observed convergence-state transition."""
    rows = sorted(screen_rows, key=lambda row: int(row["k"]))
    if len({int(r["k"]) for r in rows}) != len(rows):
        raise ValueError("duplicate coarse frontier point")
    if len({(r["sample_key"], r["kind"]) for r in rows}) > 1:
        raise ValueError("frontier refinement must remain within one sample/direction")
    selected = set()
    for left, right in zip(rows, rows[1:]):
        state = lambda r: (bool(r["finished_thinking"]), bool(r["contains_code_fence"]))
        if state(left) != state(right):
            selected.update((int(left["k"]), int(right["k"])))
    return sorted(selected)


def summarize_frontier(rows: list[dict], *, endpoint: str = "pass") -> dict:
    """Preserve the full observed vector; no bisection or monotonic assumption."""
    ordered = sorted(rows, key=lambda row: int(row["k"]))
    if len({(r["sample_key"], r["kind"]) for r in ordered}) > 1:
        raise ValueError("frontier summary cannot mix samples/directions")
    if len({int(r["k"]) for r in ordered}) != len(ordered):
        raise ValueError("duplicate frontier point")
    vector = []
    for row in ordered:
        if endpoint == "pass" and (not row.get("official_judged") or not row.get("full_budget")):
            raise ValueError("accuracy frontier requires full-budget official-judge output")
        value = row.get(endpoint)
        if not isinstance(value, bool):
            raise ValueError(f"frontier endpoint {endpoint} must be a measured boolean")
        vector.append({"k": int(row["k"]), endpoint: value})
    transitions = [{"left_k": a["k"], "right_k": b["k"],
                    "left": a[endpoint], "right": b[endpoint]}
                   for a, b in zip(vector, vector[1:]) if a[endpoint] != b[endpoint]]
    status = ("UNMEASURED" if not vector else "MULTI_FRONTIER" if len(transitions) > 1
              else "SINGLE_OBSERVED_FRONTIER" if transitions else "NO_OBSERVED_TRANSITION")
    return {"status": status, "endpoint": endpoint, "vector": vector,
            "transition_intervals": transitions, "monotonicity_assumed": False}


def make_prefix_release_params(prefix: list[int], *, max_tokens: int):
    from vllm import SamplingParams
    from ..forced_trajectory import FORCED_IDS_KEY, RELEASE_AFTER_PREFIX_KEY
    if max_tokens <= len(prefix):
        raise ValueError("prefix intervention must leave at least one free token")
    extra = {FORCED_IDS_KEY: [int(x) for x in prefix], RELEASE_AFTER_PREFIX_KEY: True} if prefix else {}
    return SamplingParams(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0,
                          max_tokens=int(max_tokens), ignore_eos=False,
                          min_tokens=len(prefix), extra_args=extra)


def run_frontier_case(llm, tokenizer, cohort_row: dict, e0: dict, e1: dict, *,
                      kind: str, k: int, stage: str, total_budget: int = 38912,
                      screen_tail: int = 4096, timeout: int = 6) -> dict:
    from vllm.inputs import TokensPrompt
    if kind not in {"rescue", "poison"} or stage not in {"screen", "full"}:
        raise ValueError("invalid frontier direction/stage")
    key = str(cohort_row["prompt_key"])
    for trajectory in (e0, e1):
        if trajectory["prompt_key"] != key or trajectory["input_ids"] != cohort_row["input_ids"]:
            raise ValueError("semantic intervention requires the same isolated mechanism prompt")
    source = e0 if kind == "rescue" else e1
    if not 0 <= k <= len(source["output_ids"]) or k >= total_budget:
        raise ValueError("prefix lies outside the isolated source trajectory or generation budget")
    prefix = [int(x) for x in source["output_ids"][:k]]
    budget = total_budget if stage == "full" else min(total_budget, k + screen_tail)
    params = make_prefix_release_params(prefix, max_tokens=budget)
    outputs = llm.generate([TokensPrompt(prompt_token_ids=cohort_row["input_ids"])], [params], use_tqdm=False)
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise RuntimeError("frontier requires exactly one isolated request/output")
    out = outputs[0].outputs[0]
    ids = [int(x) for x in out.token_ids]
    if ids[:k] != prefix or len(ids) < k:
        raise RuntimeError("SEMANTIC_FRONTIER_BLOCKED: forced prefix is not exact")
    text = tokenizer.decode(ids, skip_special_tokens=False)
    think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    row = {"sample_key": key, "kind": kind, "variant": "E1" if kind == "rescue" else "E0",
           "prefix_source": "E0" if kind == "rescue" else "E1", "k": int(k),
           "stage": stage, "full_budget": stage == "full", "total_budget": total_budget,
           "max_tokens": budget, "free_tail_requested": budget - k, "forced_prefix_exact": True,
           "output_ids": ids, "raw_text": text, "generation_len": len(ids),
           "finish_reason": out.finish_reason, "stop_reason": out.stop_reason,
           "finished_thinking": contains_subsequence(ids, think_end_ids),
           "contains_code_fence": "```" in text, "official_judged": False,
           "execution_shape": "single_request_incremental_force_then_release"}
    if stage == "full":
        from .judge_isolated_lcb import judge_completion
        result = judge_completion(cohort_row, text, ids, think_end_ids=think_end_ids,
                                  max_new_tokens=total_budget, timeout=timeout)
        row.update(result)
        row["official_judged"] = True
    return row


def request_manifest(request: dict, e0: dict, e1: dict, runtime: dict) -> dict:
    payload = {"request": request, "e0": e0, "e1": e1, "runtime": runtime}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema_version": 1, "request_sha256": digest, "runtime": runtime}


def main() -> None:
    from ..build_llm import build_real_vllm, resolve_real_vllm_spec
    from ...config import DEFAULT_MODEL_PATH, DEFAULT_PHASEA_ROOT
    from ...trajectory_io import read_jsonl
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request_plan", required=True,
                   help="JSON {requests:[{sample:<cohort row>,kind,k,stage}]} for one target variant")
    p.add_argument("--isolated_root", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--phasea_root", default=str(DEFAULT_PHASEA_ROOT))
    p.add_argument("--total_budget", type=int, default=38912)
    p.add_argument("--screen_tail", type=int, default=4096)
    args = p.parse_args()
    requests = json.loads(Path(args.request_plan).read_text())["requests"]
    if not requests or len({r["kind"] for r in requests}) != 1:
        raise ValueError("run exactly one target variant in a process")
    kind = requests[0]["kind"]
    if kind not in {"rescue", "poison"}:
        raise ValueError("unsupported frontier kind")
    variant = "E1" if kind == "rescue" else "E0"
    isolated = {v: {r["prompt_key"]: r for r in read_jsonl(Path(args.isolated_root) / f"{v}.jsonl")}
                for v in ("E0", "E1")}
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # A resolved runtime contract makes resume independent of a stale PID.
    _, spec, abi = resolve_real_vllm_spec(variant, model_path=args.model_path, phasea_root=Path(args.phasea_root))
    contract = {"variant": variant, "model_path": str(spec.model_path), "abi": abi,
                "tensor_parallel_size": 2, "max_num_seqs": 1, "kv_cache_dtype": "bfloat16",
                "eager": True, "prefix_cache": False, "total_budget": args.total_budget,
                "screen_tail": args.screen_tail, "single_request_force_then_release": True}
    llm = None
    for req in requests:
        sample = req["sample"]
        key = sample["prompt_key"]
        e0, e1 = isolated["E0"][key], isolated["E1"][key]
        manifest = request_manifest(req, e0, e1, contract)
        path = root / key / f'{kind}_k{req["k"]}_{req["stage"]}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            if saved.get("manifest") == manifest and saved.get("complete") is True:
                if req["stage"] != "full" or saved["result"].get("official_judged"):
                    continue
        if llm is None:
            llm, _ = build_real_vllm(variant, model_path=args.model_path, phasea_root=Path(args.phasea_root))
        result = run_frontier_case(llm, llm.get_tokenizer(), sample, e0, e1, kind=kind,
                                   k=int(req["k"]), stage=req["stage"], total_budget=args.total_budget,
                                   screen_tail=args.screen_tail)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"manifest": manifest, "complete": True, "result": result}, ensure_ascii=False) + "\n")
        temporary.replace(path)


if __name__ == "__main__":
    if not __package__:
        __package__ = "Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion"
    main()
