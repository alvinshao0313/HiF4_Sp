"""Count actual native NVFP4 routed projection inputs on the frozen C4 corpus."""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from ..ban_trace_calibration.processes import require_free
from ..e2e_diag_reconstruction.core import moe_semantic_hif4 as semantic
from ..e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from ..e2e_diag_reconstruction.core.moe_transforms import apply_r64_no_cross_head
from ..e2e_diag_reconstruction.training.moe_layer_runtime import build_qwen3_moe_layer_call
from ..non_equivalent_reconstruction.student import causal_mask
from ..reasoning_trace_calibration.training import context
from ..thesis_format_conversion_supplement.artifacts import process_record, read_json, run_root, sha256, verify_sources, write_json
from ..thesis_format_conversion_supplement.config import NATIVE
from ..thesis_format_conversion_supplement.gptq_calibration import load_selected


@torch.no_grad()
def scan(root, part):
    root = run_root(root)
    verify_sources(root)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if gpu not in ("0", "1"):
        raise ValueError("C4 coverage is assigned one of GPU0,1")
    require_free(gpu)
    samples, calibration = load_selected(root / "gptq_calibration")
    if (calibration.get("dataset") != "allenai/c4" or calibration.get("s1k_count") != 0
            or len(samples) != 512 or any(s.input_ids.numel() != 512 for s in samples)):
        raise ValueError("coverage requires the frozen pure C4 512x512 corpus")
    samples = samples[part::2]
    token_count = sum(s.input_ids.numel() for s in samples)
    output = root / f"qualification/c4_native_coverage_part{part}"
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, device, hidden = context(samples)
    started = time.monotonic()
    source_hashes = {str(p): sha256(p) for p in Path(__file__).parent.glob("*.py")}
    state = {"complete": False, "all_experts_activated": False, "completed_layers": [],
             "scope": "native NVFP4 causal layer replay, actual gate/up/down input rows; preconversion coverage only",
             "not_proven": "coverage after GPTQ changes the propagated inputs; formal conversion must still check every layer",
             "calibration_sha256": calibration["train_sha256"], "sequences": len(samples), "tokens": token_count,
             "part": part, "sample_ids": [s.sample_id for s in samples],
             "gpus": gpu, "pid": os.getpid(), "source_sha256": source_hashes, "layers": {}}
    for layer in range(48):
        base = load_qwen3_moe_layer_state(NATIVE, layer, device)
        runtime = semantic.NativeQwen3MoELayerRuntime(base).to(device).eval()
        names = {id(getattr(expert, name)): (i, name) for i, expert in enumerate(base.experts)
                 for name in ("gate_proj", "up_proj", "down_proj")}
        captured = {key: {"tokens": 0, "sequences": 0, "diagonal": torch.zeros(getattr(base.experts[key[0]], key[1]).shape[1], device=device)}
                    for key in names.values()}
        original = semantic.native_nvfp4_linear

        def observe(x, weight, *args, **kwargs):
            key = names.get(id(weight))
            if key is not None:
                item = captured[key]
                values = x.reshape(-1, x.shape[-1])
                item["tokens"] += values.shape[0]
                item["sequences"] += 1
                item["diagonal"] += apply_r64_no_cross_head(values.float()).square().sum(0)
            return original(x, weight, *args, **kwargs)

        counts = torch.zeros(128, dtype=torch.long, device=device)
        next_hidden = {}
        semantic.native_nvfp4_linear = observe
        try:
            for index, sample in enumerate(samples):
                if not sample.loss_mask.bool().all():
                    raise ValueError("C4 must contain complete unpadded valid sequences")
                x = hidden[sample.sample_id].to(device).unsqueeze(0)
                call = build_qwen3_moe_layer_call(str(NATIVE), x)
                result = runtime(x, attention_mask=causal_mask(x), position_embeddings=call.position_embeddings)
                if not torch.isfinite(result.output).all():
                    raise ValueError(f"nonfinite native output at layer {layer}")
                if layer == 0 and index < 2:
                    semantic.native_nvfp4_linear = original
                    plain = runtime(x, attention_mask=causal_mask(x), position_embeddings=call.position_embeddings)
                    semantic.native_nvfp4_linear = observe
                    if not torch.equal(result.output, plain.output) or not torch.equal(result.selected_experts, plain.selected_experts):
                        raise ValueError("input observation altered the native forward or routing")
                    del plain
                counts += torch.bincount(result.selected_experts.flatten(), minlength=128)
                if layer == 0 and index < 2:
                    actual = counts.cpu().tolist()
                    if any(item["tokens"] != actual[expert] for (expert, _), item in captured.items()):
                        raise ValueError("first real inputs disagree with observed routing")
                    state["instrumentation_output_unchanged"] = True
                    write_json(state, output / "coverage.json")
                    print(f"C4 part {part}: real window {index+1} observer/native output exact; gate/up/down counts agree", flush=True)
                next_hidden[sample.sample_id] = result.output[0].cpu().contiguous()
                if (index+1) % 256 == 0:
                    print(f"C4 native layer {layer}: {index+1}/{len(samples)} windows", flush=True)
        finally:
            semantic.native_nvfp4_linear = original
        observed = counts.cpu().tolist()
        projections = {}
        for (expert, name), item in captured.items():
            if item["tokens"] != observed[expert]:
                raise ValueError("router assignments and actual projection inputs disagree")
            diagonal = item.pop("diagonal")
            if not torch.isfinite(diagonal).all():
                raise ValueError("nonfinite observed input second moment")
            projections[f"expert_{expert}_{name}"] = {**item, "columns": diagonal.numel(),
                 "zero_diagonal": int((diagonal == 0).sum()),
                 "zero_diagonal_columns": torch.where(diagonal == 0)[0].cpu().tolist(),
                 "diagonal_trace": float(diagonal.sum())}
        if sum(observed) != token_count*8:
            raise ValueError("routing assignments do not cover every valid token exactly eight times")
        missing = [i for i, value in enumerate(observed) if value == 0]
        record = {"complete": True, "layer": layer, "expert_counts": observed, "unvisited": missing,
                  "minimum_tokens": min(observed), "projections": projections, "router_capture_equal": True}
        write_json(record, output / f"layer_{layer:03d}.json")
        state["completed_layers"].append(layer)
        state["layers"][str(layer)] = {"unvisited": missing, "minimum_tokens": min(observed)}
        state["seconds"] = time.monotonic()-started
        state["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        write_json(state, output / "coverage.json")
        print(f"C4 native layer {layer} complete: unvisited={missing}, minimum_tokens={min(observed)}", flush=True)
        hidden = next_hidden
        del base, runtime, captured, result, x, counts
        gc.collect(); torch.cuda.empty_cache()
    verify_sources(root)
    if source_hashes != {str(p): sha256(p) for p in Path(__file__).parent.glob("*.py")}:
        raise ValueError("C4 coverage source changed during execution")
    state["complete"] = True
    state["all_experts_activated"] = all(not value["unvisited"] for value in state["layers"].values())
    state["exit_code"] = 0
    write_json(state, output / "coverage.json")
    print(f"C4 coverage scan complete; all_experts_activated={state['all_experts_activated']}", flush=True)


def parallel(root, job_state, status_path):
    """Two finite independent window shards, then one exact count aggregation."""
    require_free("0,1")
    children = []
    for part in (0, 1):
        command = [sys.executable, "-u", "-m", __package__ + ".coverage", "--run", str(root), "--part", str(part)]
        log = root / "logs" / f"c4_native_coverage_part{part}.log"
        with log.open("w") as out:
            child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(part)},
                                     stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        job_state.setdefault("children", []).append({"pid": child.pid, "session_id": child.pid,
            "gpus": str(part), "command": command, "log": str(log), "exit_code": None})
        write_json(job_state, status_path)
    for child, entry in zip(children, job_state["children"]):
        entry["exit_code"] = child.wait()
        write_json(job_state, status_path)
    if any(entry["exit_code"] for entry in job_state["children"]):
        raise RuntimeError("C4 coverage shard failed; see child logs and status")
    paths = [root / f"qualification/c4_native_coverage_part{part}" for part in (0, 1)]
    parts = [read_json(p / "coverage.json") for p in paths]
    calibration = read_json(root / "gptq_calibration/manifest.json")
    ids = [key for part in parts for key in part["sample_ids"]]
    if (len(ids) != len(set(ids)) or set(ids) != set(calibration["ids"])
            or any(not part["complete"] or part["completed_layers"] != list(range(48))
                   or part["calibration_sha256"] != calibration["train_sha256"] for part in parts)
            or parts[0]["source_sha256"] != parts[1]["source_sha256"]):
        raise ValueError("coverage shards differ in source, data, or completeness")
    output = root / "qualification/c4_native_coverage"
    output.mkdir(exist_ok=False)
    layers = {}
    for layer in range(48):
        pair = [read_json(p / f"layer_{layer:03d}.json") for p in paths]
        counts = [a+b for a, b in zip(pair[0]["expert_counts"], pair[1]["expert_counts"])]
        projections = {}
        for name, left in pair[0]["projections"].items():
            right = pair[1]["projections"][name]
            zero = sorted(set(left["zero_diagonal_columns"]) & set(right["zero_diagonal_columns"]))
            projections[name] = {"tokens": left["tokens"]+right["tokens"], "sequences": left["sequences"]+right["sequences"],
                "columns": left["columns"], "zero_diagonal": len(zero), "zero_diagonal_columns": zero,
                "diagonal_trace": left["diagonal_trace"]+right["diagonal_trace"]}
            if projections[name]["tokens"] != counts[int(name.split("_")[1])]:
                raise ValueError("aggregated projection counts differ from routing")
        missing = [i for i, count in enumerate(counts) if not count]
        if sum(counts) != calibration["tokens"]*8:
            raise ValueError("aggregated valid routing token count differs")
        write_json({"complete": True, "layer": layer, "expert_counts": counts, "unvisited": missing,
                    "projections": projections, "router_capture_equal": True}, output / f"layer_{layer:03d}.json")
        layers[str(layer)] = {"unvisited": missing, "minimum_tokens": min(counts)}
    write_json({"complete": True, "all_experts_activated": all(not row["unvisited"] for row in layers.values()),
                "layers": layers, "completed_layers": list(range(48)), "sequences": len(ids), "tokens": calibration["tokens"],
                "calibration_sha256": calibration["train_sha256"], "parts": [str(p) for p in paths],
                "source_sha256": parts[0]["source_sha256"], "scope": parts[0]["scope"], "not_proven": parts[0]["not_proven"]},
               output / "coverage.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--part", type=int, choices=(0, 1))
    args = parser.parse_args()
    root = run_root(args.run)
    directory = root / "jobs" / ("c4_native_coverage" if args.part is None else f"c4_native_coverage_part{args.part}")
    directory.mkdir(parents=True, exist_ok=False)
    state = {**process_record(sys.argv), "status": "running", "exit_code": None}
    write_json(state, directory / "status.json")
    began = time.monotonic()
    try:
        if args.part is None:
            parallel(root, state, directory / "status.json")
        else:
            scan(root, args.part)
    except BaseException as error:
        state.update(status="failed", exit_code=1, error=repr(error))
        raise
    else:
        state.update(status="complete", exit_code=0)
    finally:
        state["seconds"] = time.monotonic()-began
        write_json(state, directory / "status.json")
