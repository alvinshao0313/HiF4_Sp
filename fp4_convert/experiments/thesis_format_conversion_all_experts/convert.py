"""Native layer-zero conversion with all-token expert calibration and direct export."""
import gc
import time

import torch
from safetensors.torch import load_file

from ..e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from ..e2e_diag_reconstruction.evaluation.moe_materialize import (
    _copy_json_without_quantization, _copy_tokenizer_assets, _copy_non_layer_tensors)
from ..reasoning_trace_calibration.training import context
from ..thesis_format_conversion_supplement.artifacts import read_json, run_root, write_json
from ..thesis_format_conversion_supplement.config import NATIVE
from ..thesis_format_conversion_supplement.export import finish, save_layer, tensor_key
from ..thesis_format_conversion_supplement.quantization import ConvertedStudent, replay
from .artifacts import verify
from .quantization import AllExpertStudent, gptq_layer
from .validation import routing_check


@torch.no_grad()
def convert(root, *, qualification=False):
    root = run_root(root)
    train = verify(root)
    samples = sorted(train, key=lambda s: s.input_ids.numel(), reverse=True)[:2] if qualification else train
    output = root / ("qualification/models" if qualification else "models") / "r64_gptq"
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    _copy_json_without_quantization(NATIVE / "config.json", output / "config.json")
    _copy_tokenizer_assets(NATIVE, output)
    weight_map = {}
    _copy_non_layer_tensors(NATIVE, output, weight_map)
    write_json({**read_json(root / "calibration.json"), "qualification": qualification,
                "actual_ids": [s.sample_id for s in samples], "actual_count": len(samples),
                "actual_tokens": sum(s.input_ids.numel() for s in samples)}, output / "calibration.json")
    _, device, hidden = context(samples)
    state = {"complete": False, "qualification": qualification, "candidate": "r64_gptq",
             "algorithm": "all_token_all_expert_calibration_sparse_output", "completed_layers": []}
    write_json(state, output / "conversion.json")
    for layer in range(1 if qualification else 48):
        layer_started = time.monotonic()
        base = load_qwen3_moe_layer_state(NATIVE, layer, device)
        directory = output / "conversion" / f"{layer:03d}"
        directory.mkdir(parents=True)
        if qualification:
            initial = AllExpertStudent(base)
            write_json(routing_check(initial, samples, hidden), root / "qualification/routing_before_gptq.json")
            del initial
            print("Real complete-sequence dense/sparse routing and gate/up/down capture checks passed", flush=True)
        student, stats = gptq_layer(base, samples, hidden, directory)
        save_started = time.monotonic()
        weight_map.update(save_layer(base, student.quantized, output))
        save_seconds = time.monotonic() - save_started
        if qualification:
            write_json(routing_check(student, samples, hidden), root / "qualification/routing_after_gptq.json")
            reloaded = load_file(str(output / "model-layer-00000-of-00048.safetensors"), device="cuda:0")
            restored = ConvertedStudent(base)
            restored.quantized = {name: reloaded[tensor_key(layer, name)] for name in student.quantized}
            before = replay(student, samples, hidden, propagate=True)
            after = replay(restored, samples, hidden, propagate=True)
            if any(not torch.equal(before[key], after[key]) for key in before):
                raise ValueError("GPTQ decoder output changed on export/reload")
            del reloaded, restored, before, after
        hidden = replay(student, samples, hidden, propagate=True)
        groups = {v["hessian_group"]: v["group_hessian_seconds"] for v in stats.values()}
        write_json({"layer_seconds": time.monotonic() - layer_started, "save_reload_seconds": save_seconds,
                    "hessian_seconds": sum(groups.values()),
                    "quantization_seconds": sum(v["quantization_seconds"] for v in stats.values()),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated()}, directory / "costs.json")
        state["completed_layers"].append(layer)
        state["seconds"] = time.monotonic() - started
        write_json(state, output / "conversion.json")
        print(f"converted all-expert r64_gptq: {layer + 1}/48, propagated {len(samples)} complete sequences", flush=True)
        del base, student, stats
        gc.collect()
        torch.cuda.empty_cache()
    coverage = {str(layer): read_json(output / "conversion" / f"{layer:03d}" / "expert_coverage.json")
                for layer in state["completed_layers"]}
    tokens = sum(int(s.loss_mask.bool().sum()) for s in samples)
    if any(not row["complete"] or row["unvisited_calibration"] or
           row["calibration_tokens_per_expert"] != [tokens] * 128 for row in coverage.values()):
        raise ValueError("export lacks complete all-token expert statistics")
    write_json({"complete": not qualification, "qualification": qualification,
                "layers": len(coverage), "experts_per_layer": 128, "all_experts_calibrated": True,
                "calibration_tokens_per_expert": tokens,
                "naturally_unvisited_by_layer": {k: v["naturally_unvisited"] for k, v in coverage.items()},
                "scope": "calibration coverage is distinct from natural router selection"}, output / "expert_coverage.json")
    finish(output, weight_map, "r64_gptq", qualification=qualification)
    verify(root)
    state.update(complete=not qualification, qualification_complete=qualification,
                 seconds=time.monotonic() - started)
    write_json(state, output / "conversion.json")
    if qualification:
        write_json({"complete": True, "layers": 1, "projections": 388,
                    "calibration_sequences": len(samples), "valid_tokens": tokens,
                    "full_sequence_lengths": [s.input_ids.numel() for s in samples],
                    "exact_decoder_output_reload": True, "dense_sparse_routing_checked": True,
                    "gate_up_down_actual_inputs_checked": True,
                    "all_experts_calibrated": True, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "seconds": state["seconds"],
                    "scope": "real full layer, longest two untruncated A20 train sequences; formal uses all 128"},
                   root / "qualification/gptq_all_experts.json")
