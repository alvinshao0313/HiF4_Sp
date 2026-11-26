"""Reproduce the failed layer's real routing and measure expanded-pool coverage."""
import argparse
import gc
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

from HiFloat4.hif4gptq.gptq.gptq_utils import GPTQ, WeightHiFxQuantizer
from ..e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from ..non_equivalent_reconstruction.transforms import projection_weights
from ..reasoning_trace_calibration.training import context
from .artifacts import read_json, run_root, sha256, write_json
from .config import NATIVE
from .export import tensor_key
from .gptq_calibration import load_pool
from .quantization import ConvertedStudent, replay, rotated


class TokenCounter:
    """Observe the actual Hessian-input site without allocating its Gram matrix."""
    def __init__(self):
        self.effective_tokens = 0
        self.calls = 0

    def add_batch(self, values, output):
        self.calls += 1


class RoutingAuditStudent(ConvertedStudent):
    def __init__(self, base):
        super().__init__(base)
        self.route_counts = torch.zeros(base.spec.num_experts, dtype=torch.long, device="cuda:0")

    def moe(self, normed, *, use_ste):
        result, logits, counts = super().moe(normed, use_ste=use_ste)
        self.route_counts += counts
        return result, logits, counts


def load_prefix(previous):
    prefix, hashes = [], {}
    for layer in range(4):
        base = load_qwen3_moe_layer_state(NATIVE, layer, "cuda:0")
        student = ConvertedStudent(base)
        path = previous / "models/r64_gptq" / f"model-layer-{layer:05d}-of-00048.safetensors"
        saved = load_file(str(path), device="cuda:0")
        student.quantized = {name: saved[tensor_key(layer, name)] for name, _ in projection_weights(base) if name != "router"}
        prefix.append(student)
        hashes[str(path)] = sha256(path)
    return prefix, hashes


@torch.no_grad()
def diagnose(root, previous):
    root, previous = run_root(root), run_root(previous)
    output = root / "qualification/failed_layer_coverage"
    output.mkdir(parents=True, exist_ok=False)
    samples, manifest = load_pool(root)
    state = read_json(previous / "models/r64_gptq/conversion.json")
    if state["complete"] or state["completed_layers"] != ["0", "1", "2", "3"]:
        raise ValueError("diagnostic requires the recorded four-layer failed conversion prefix")
    started = time.monotonic()
    prefix, prefix_hashes = load_prefix(previous)
    base = load_qwen3_moe_layer_state(NATIVE, 4, "cuda:0")
    student = RoutingAuditStudent(base)
    _, _, hidden = context(samples[:128])
    for layer, prior in enumerate(prefix):
        hidden = replay(prior, samples[:128], hidden, propagate=True)
        print(f"Replayed original calibration through saved prefix layer {layer}", flush=True)
    weights = dict(projection_weights(base))
    # Reproduce precisely the two attention GPTQ groups preceding the failure.
    for names in (("q_proj", "k_proj", "v_proj"), ("o_proj",)):
        for name in names:
            block = GPTQ(SimpleNamespace(weight=torch.nn.Parameter(rotated(name, weights[name], base.spec), requires_grad=False)))
            block.quantizer = WeightHiFxQuantizer("hifx4")
            block.effective_tokens = 0
            student.collectors[name] = block
        replay(student, samples[:128], hidden)
        for name, block in student.collectors.items():
            block.fasterquant(blocksize=128, groupsize=64, percdamp=.01)
            student.quantized[name] = block.layer.weight.detach().to(torch.bfloat16)
            block.free()
        student.collectors.clear()
        print(f"Recreated failed-layer attention GPTQ group {names}", flush=True)
    save_file({n:v.cpu().contiguous() for n,v in student.quantized.items()}, str(output / "attention.safetensors"))
    student.route_counts.zero_()
    student.collectors = {f"expert_{i}_gate_proj": TokenCounter() for i in range(base.spec.num_experts)}
    history = []
    for start in range(0, len(samples), 128):
        batch = samples[start:start+128]
        if start:
            _, _, hidden = context(batch)
            for prior in prefix:
                hidden = replay(prior, batch, hidden, propagate=True)
        replay(student, batch, hidden)
        actual = student.route_counts.cpu().tolist()
        captured = [student.collectors[f"expert_{i}_gate_proj"].effective_tokens for i in range(base.spec.num_experts)]
        if actual != captured:
            raise ValueError("router assignment counts disagree with actual GPTQ input capture")
        total = start + len(batch)
        row = {"sequences": total, "tokens": sum(s.input_ids.numel() for s in samples[:total]),
               "expert_counts": actual, "unvisited": [i for i,c in enumerate(actual) if not c],
               "expert35_tokens": actual[35], "router_capture_equal": True}
        history.append(row)
        write_json({"complete": total == len(samples), "scope": "failed layer 4 with the exact saved old GPTQ prefix and reproduced attention",
                    "not_a_full_model_coverage_certificate": True, "pool_sha256": manifest["pool_sha256"],
                    "prefix_sha256": prefix_hashes, "history": history, "seconds": time.monotonic()-started,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated()}, output / "coverage.json")
        print(f"Routing coverage {total}/{len(samples)}: unvisited={row['unvisited']}; expert35={actual[35]}", flush=True)
        del hidden
        gc.collect()
    return history


@torch.no_grad()
def scan_remaining(root, previous, start, stop):
    root, previous = run_root(root), run_root(previous)
    samples, manifest = load_pool(root)
    if not 128 <= start < stop <= len(samples):
        raise ValueError("scan interval must be within the additional complete sequences")
    output = root / "qualification/failed_layer_coverage"
    original = read_json(output / "coverage.json")
    prefix, prefix_hashes = load_prefix(previous)
    if prefix_hashes != original["prefix_sha256"] or manifest["pool_sha256"] != original["pool_sha256"]:
        raise ValueError("saved diagnostic prefix or data changed")
    student = RoutingAuditStudent(load_qwen3_moe_layer_state(NATIVE, 4, "cuda:0"))
    student.quantized = load_file(str(output / "attention.safetensors"), device="cuda:0")
    student.collectors = {f"expert_{i}_gate_proj": TokenCounter() for i in range(128)}
    path = output / f"scan_{start:04d}_{stop:04d}.json"
    if path.exists():
        raise FileExistsError(path)
    began = time.monotonic()
    history = []
    for left in range(start, stop, 128):
        right = min(left+128, stop)
        batch = samples[left:right]
        _, _, hidden = context(batch)
        for prior in prefix:
            hidden = replay(prior, batch, hidden, propagate=True)
        replay(student, batch, hidden)
        counts = student.route_counts.cpu().tolist()
        captured = [student.collectors[f"expert_{i}_gate_proj"].effective_tokens for i in range(128)]
        if counts != captured:
            raise ValueError("router assignments and actual GPTQ input capture disagree")
        history.append({"stop": right, "expert_counts": counts, "router_capture_equal": True})
        write_json({"complete": right == stop, "start": start, "stop": stop, "completed_stop": right,
                    "ids": [s.sample_id for s in samples[start:right]], "history": history,
                    "pool_sha256": manifest["pool_sha256"], "prefix_sha256": prefix_hashes,
                    "attention_sha256": sha256(output / "attention.safetensors"),
                    "seconds": time.monotonic()-began, "peak_allocated_bytes": torch.cuda.max_memory_allocated()}, path)
        print(f"Additional complete sequences [{start},{right}): expert35={counts[35]}", flush=True)
        del hidden
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    args = parser.parse_args()
    if args.start is None and args.stop is None:
        diagnose(args.run, args.previous)
    elif args.start is not None and args.stop is not None:
        scan_remaining(args.run, args.previous, args.start, args.stop)
    else:
        parser.error("--start and --stop must be given together")
