"""Finite real-vLLM calibration and held-out budget stages, one process each."""
from __future__ import annotations

import argparse
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from ..ban.artifact import load_artifact
from ..ban.calibrate import calibrate
from ..ban.engine import configure, create_engine, snapshot
from ..ban.evaluation_report import _sha256
from .config import K_MIN, now, read, save, verify


def calibration(root, diagnostic):
    manifest = verify(root)
    inputs = root / ("diagnostic_inputs.pt" if diagnostic else "calibration_inputs.pt")
    samples = torch.load(inputs, map_location="cpu", weights_only=True)
    data = read(root / "windows.json")
    if diagnostic:
        data = {**data, "nsamples": 2, "domain_windows": {"math": 1, "code": 1},
                "windows": [data["windows"][0], data["windows"][256]], "token_visits": 4096,
                "unique_source_questions": 2, "unique_source_token_positions": 4096,
                "duplicate_windows": 0, "phase_note": "diagnostic subset only"}
    phase = "diagnostic" if diagnostic else "formal"
    args = SimpleNamespace(output=root / "calibration" / phase, model_path=manifest["model"],
                           hif4_runtime_spec=manifest["spec"], k_pruned=3, topn=1000,
                           phase=phase, tensor_parallel_size=2, seqlen=2048)
    engine, artifact, _ = calibrate(args, samples=samples, data=data)
    if artifact["num_samples"] != (2 if diagnostic else 512):
        raise ValueError("incomplete calibration")
    if diagnostic:
        engine.collective_rpc("ban_load_artifact", args=(str(args.output / "c4_ban.pt"),))
        configure(engine, mode="ban", lam=0.7, k_min=5, execution="compact", capture=None)
        from ..ban.diagnose import generate
        generate(engine, samples[0], max_tokens=64)
        stats = snapshot(engine)
        if (stats["tokens"] != [2111] * 48 or stats["selected"] != stats["dispatched"]
                or stats["tp_ranks"] != 2):
            raise RuntimeError("diagnostic prefill/decode or TP dispatch coverage differs")
        save(root / "diagnostic_verified.json", {"passed": True, "phase": phase, "stats": stats,
             "calibration_samples": 2, "created_at": now()})


def budget_probe(engine, rows, lam, max_tokens):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    configure(engine, mode="ban", lam=lam, k_min=K_MIN, execution="compact", capture=None)
    outputs = engine.generate([TokensPrompt(prompt_token_ids=row["input_ids"]) for row in rows],
                              SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0,
                                             max_tokens=max_tokens, n=1, seed=1234), use_tqdm=True)
    stats = snapshot(engine)
    if (len(outputs) != 16 or min(stats["tokens"]) <= 0 or stats["tp_ranks"] != 2
            or stats["selected"] != stats["dispatched"]):
        raise RuntimeError("budget generation/TP/expert coverage differs")
    requests = []
    for row, output in zip(rows, outputs):
        if output.prompt_token_ids != row["input_ids"] or len(output.outputs) != 1:
            raise RuntimeError("budget generation changed inputs/count")
        value = output.outputs[0]
        if not value.token_ids or value.finish_reason not in ("stop", "length"):
            raise RuntimeError("budget generation is empty or incomplete")
        requests.append({"id": row["id"], "domain": row["domain"],
                         "input_tokens_sha256": row["input_tokens_sha256"],
                         "output_tokens": len(value.token_ids), "output_tokens_sha256": _sha256(value.token_ids),
                         "finish_reason": value.finish_reason})
    return {"lambda": lam, "average_experts": stats["average_experts"], "stats": stats, "requests": requests}


def match_budget(root, arm, short=False):
    manifest = verify(root)
    target, tolerance = manifest["budget"]["target"], manifest["budget"]["tolerance"]
    artifact = (root / "calibration/diagnostic/c4_ban.pt" if short else
                Path(manifest["c4_artifact"]) if arm == "c4" else root / "calibration/formal/c4_ban.pt")
    parsed = load_artifact(artifact, manifest["model_identity"])
    if not short and (parsed["phase"], parsed["num_samples"]) != ("formal", 512):
        raise ValueError("budget matching requires a formal 512-window artifact")
    output = root / ("preflight_budget" if short else f"budget/{arm}")
    output.mkdir(parents=True, exist_ok=False)
    rows = read(root / "budget_inputs.json")
    max_tokens = 64 if short else 38912
    result = {"status": "running", "arm": arm, "artifact": str(artifact), "trials": [],
              "target": target, "tolerance": tolerance, "k_min": K_MIN,
              "sampling": {"max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95,
                           "top_k": 20, "min_p": 0, "seed": 1234}, "diagnostic": short,
              "started_at": now(), "test_set_tuning": False}
    save(output / "budget.json", result)
    started = time.monotonic()
    engine = create_engine(manifest["model"], manifest["spec"], artifact_path=str(artifact), k_min=5,
                           max_model_len=40960, max_num_seqs=128, max_num_batched_tokens=4096,
                           gpu_memory_utilization=0.9, seed=1234, kv_cache_dtype="bfloat16",
                           enable_chunked_prefill=None)

    def probe(lam):
        trial = budget_probe(engine, rows, lam, max_tokens)
        result["trials"].append(trial)
        best = min(result["trials"], key=lambda t: abs(t["average_experts"] - target))
        result.update(chosen_lambda=best["lambda"], chosen_average_experts=best["average_experts"],
                      budget_matched=abs(best["average_experts"] - target) <= tolerance,
                      elapsed_seconds=time.monotonic() - started)
        save(output / "budget.json", result)
        print(f"TRACE_BUDGET lambda={lam:.6f} average_experts={trial['average_experts']:.6f}", flush=True)
        return trial

    first = probe(0.7)
    if not short and not result["budget_matched"]:
        lower, upper = probe(0.0), probe(1.0)
        if lower["average_experts"] <= target <= upper["average_experts"]:
            if first["average_experts"] < target:
                lower = first
            else:
                upper = first
            for _ in range(manifest["budget"]["max_bisection_steps"]):
                if result["budget_matched"]:
                    break
                trial = probe((lower["lambda"] + upper["lambda"]) / 2)
                if trial["average_experts"] < target:
                    lower = trial
                else:
                    upper = trial
    result.update(status="complete" if short or result["budget_matched"] else "failed", ended_at=now())
    save(output / "budget.json", result)
    if not short and not result["budget_matched"]:
        raise RuntimeError(f"{arm} did not match {target} +/- {tolerance} experts; no formal evaluation allowed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", choices=["diagnostic", "calibrate", "preflight_budget", "budget"], required=True)
    parser.add_argument("--arm", choices=["c4", "trace"], default="trace")
    args = parser.parse_args()
    try:
        if args.stage in ("diagnostic", "calibrate"):
            calibration(args.run.resolve(), args.stage == "diagnostic")
        else:
            match_budget(args.run.resolve(), args.arm, args.stage == "preflight_budget")
    except Exception as error:
        if args.stage in ("budget", "preflight_budget"):
            path = args.run / ("preflight_budget" if args.stage == "preflight_budget" else f"budget/{args.arm}") / "budget.json"
            if path.exists():
                result = read(path)
                result.update(status="failed", error=str(error), ended_at=now())
                save(path, result)
        raise
