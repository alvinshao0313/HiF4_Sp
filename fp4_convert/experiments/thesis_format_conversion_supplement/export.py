"""Save chosen QDQ values directly; never invoke the standard re-quantizing exporter."""
import argparse
import gc
from pathlib import Path
import time

import torch
from safetensors.torch import load_file, save_file

from ..e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from ..e2e_diag_reconstruction.evaluation.moe_materialize import (
    _copy_json_without_quantization, _copy_tokenizer_assets, _copy_non_layer_tensors,
    _write_runtime_abi_marker, HIF4_RUNTIME_ABI_VERSION)
from ..non_equivalent_reconstruction.transforms import projection_weights
from ..reasoning_trace_calibration.training import samples_for, context
from .artifacts import *
from .config import NATIVE, SOURCE_RUN
from .quantization import rotated, search_projection, gptq_layer, replay, ConvertedStudent


def tensor_key(layer, name):
    if name.startswith("expert_"):
        _, expert, projection = name.split("_", 2)
        return f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
    return f"model.layers.{layer}.self_attn.{name}.weight"


def save_layer(base, quantized, output):
    expected = {n for n,w in projection_weights(base) if n != "router"}
    if set(quantized) != expected:
        raise ValueError("export must cover all attention and expert projections")
    prefix = f"model.layers.{base.layer_idx}"
    tensors = {f"{prefix}.{name}.weight": value.cpu().contiguous() for name,value in (
        ("input_layernorm", base.input_layernorm_weight), ("post_attention_layernorm", base.post_attention_layernorm_weight),
        ("self_attn.q_norm", base.q_norm_weight), ("self_attn.k_norm", base.k_norm_weight), ("mlp.gate", base.router_weight))}
    tensors.update({tensor_key(base.layer_idx, n): q.to(torch.bfloat16).cpu().contiguous() for n,q in quantized.items()})
    shard = f"model-layer-{base.layer_idx:05d}-of-00048.safetensors"
    save_file(tensors, str(output / shard))
    reloaded = load_file(str(output / shard))
    if any(not torch.equal(t, reloaded[n]) for n,t in tensors.items()):
        raise ValueError("chosen weights changed on save/reload")
    return {name: shard for name in tensors}


def finish(output, weight_map, candidate, *, qualification=False):
    runtime = {"runtime_schema_version": 2, "runtime_abi_version": HIF4_RUNTIME_ABI_VERSION,
               "model_type": "qwen3_moe", "variant": "fusable_r64", "algorithm_variant": "fusable_r64",
               "use_r64": True, "num_layers": 48, "rot_order": "diag_then_rot", "hidden_size": 2048,
               "head_dim": 128, "moe_intermediate_size": 768, "num_experts": 128, "top_k": 8,
               "identity_filled_layers": [], "online_activation_diag": {}, "online_activation_scale": {}}
    atomic_save(runtime, output / "hif4_runtime_spec.pt")
    _write_runtime_abi_marker(output, runtime)
    hashes = {s: sha256(output/s) for s in sorted(set(weight_map.values()))}
    write_json({"metadata": {"total_size": sum((output/s).stat().st_size for s in hashes)},
                "weight_map": weight_map}, output / "model.safetensors.index.json")
    write_json({"complete": not qualification, "qualification": qualification, "candidate": candidate,
                "native_snapshot": str(NATIVE), "shards_sha256": hashes,
                "storage": "BF16 HiF4 QDQ; selected values saved directly; no packed compression"}, output / "export.json")


def convert(root, candidate, *, qualification=False, gptq_calibration=None):
    root = run_root(root); verify_sources(root)
    if candidate not in ("r64_scale", "r64_gptq"):
        raise ValueError(candidate)
    if gptq_calibration is not None and candidate != "r64_gptq":
        raise ValueError("expanded calibration is only defined for GPTQ")
    output = root / ("qualification/models" if qualification else "models") / candidate
    output.mkdir(parents=True, exist_ok=False)
    _copy_json_without_quantization(NATIVE/"config.json", output/"config.json")
    _copy_tokenizer_assets(NATIVE, output)
    weight_map = {}; _copy_non_layer_tensors(NATIVE, output, weight_map)
    samples = hidden = None
    if candidate == "r64_gptq":
        if gptq_calibration is None:
            samples, _, hashes = samples_for(SOURCE_RUN, "matrix_norm_s1k")
            calibration_record = {"source": str(SOURCE_RUN), "sha256": hashes, "count": len(samples)}
        else:
            from .gptq_calibration import load_selected
            samples, calibration_record = load_selected(gptq_calibration)
        write_json(calibration_record, output / "calibration.json")
        _, device, hidden = context(samples)
    started = time.monotonic(); metrics = {}
    for layer in range(1 if qualification else 48):
        base = load_qwen3_moe_layer_state(NATIVE, layer, "cuda:0")
        directory = output / "conversion" / f"{layer:03d}"
        directory.mkdir(parents=True)
        if candidate == "r64_scale":
            quantized, stats, meta = {}, {}, {}
            for name, weight in projection_weights(base):
                if name == "router": continue
                q, selected, measured = search_projection(rotated(name, weight, base.spec))
                quantized[name] = q; stats[name] = measured
                meta.update({name+"."+key: value.contiguous() for key,value in selected.items()})
            save_file(meta, str(directory/"selected_scales.safetensors"))
        else:
            student, stats = gptq_layer(base, samples, hidden, directory)
            quantized = student.quantized
        save_started = time.monotonic()
        weight_map.update(save_layer(base, quantized, output))
        save_seconds = time.monotonic() - save_started
        if candidate == "r64_gptq":
            if qualification:
                # Complete input sequences, original layer input, exact selected values.
                selected=sorted(samples,key=lambda s:s.input_ids.numel(),reverse=True)[:4]
                reloaded=load_file(str(output/f"model-layer-{layer:05d}-of-00048.safetensors"),device="cuda:0")
                restored=ConvertedStudent(base)
                restored.quantized={n:reloaded[tensor_key(layer,n)] for n in quantized}
                with torch.no_grad():
                    before=replay(student,selected,hidden,propagate=True)
                    after=replay(restored,selected,hidden,propagate=True)
                if any(not torch.equal(before[k],after[k]) for k in before):
                    raise ValueError("GPTQ decoder output changed after direct export/reload")
                write_json({"complete":True,"projections":len(stats),"calibration_sequences":len(samples),
                            "valid_tokens":sum(int(s.loss_mask.bool().sum()) for s in samples),
                            "unvisited": [n for n,v in stats.items() if not v["tokens"]],
                            "exact_decoder_output_reload":True,"full_sequence_lengths":[s.input_ids.numel() for s in selected],
                            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                            "scope":"real full decoder layer, natural-column/G64 GPTQ, actual routed train inputs"},
                           root/"qualification/gptq.json")
                del restored,reloaded,before,after
            hidden = replay(student, samples, hidden, propagate=True)
            del student
        metrics[str(layer)] = stats
        write_json(stats, directory/"metrics.json")
        groups = {v["hessian_group"]:v["group_hessian_seconds"] for v in stats.values()} if candidate=="r64_gptq" else {}
        write_json({"save_reload_seconds":save_seconds,
                    "hessian_seconds":sum(groups.values()) if groups else None,
                    "quantization_seconds":sum(v.get("quantization_seconds",v.get("search_seconds",0)) for v in stats.values()),
                    "peak_allocated_bytes":max(v.get("peak_allocated_bytes",0) for v in stats.values()) if candidate=="r64_scale" else torch.cuda.max_memory_allocated(),
                    "boundary":"per-layer computation and save/reload; full process wall separately in jobs status"},directory/"costs.json")
        write_json({"complete": False, "completed_layers": list(metrics), "seconds": time.monotonic()-started,
                    "candidate": candidate}, output/"conversion.json")
        print(f"converted {candidate}: {layer+1}/48", flush=True)
        del base, quantized, stats
        if candidate == "r64_scale": del meta
        gc.collect(); torch.cuda.empty_cache()
    if candidate == "r64_gptq":
        layers = range(1 if qualification else 48)
        coverage = {str(layer): read_json(output / "conversion" / f"{layer:03d}" / "expert_coverage.json")
                    for layer in layers}
        if any(not row["complete"] or row["unvisited"] or len(row["expert_counts"]) != 128
               or min(row["expert_counts"]) <= 0 for row in coverage.values()):
            raise ValueError("full-model export requires actual coverage of every expert in every layer")
        write_json({"complete": not qualification, "qualification": qualification,
                    "layers": len(coverage), "experts_per_layer": 128,
                    "calibration_sequences": len(samples), "all_experts_activated": True,
                    "minimum_tokens_by_layer": {key: min(row["expert_counts"]) for key,row in coverage.items()},
                    "evidence": {key: f"conversion/{int(key):03d}/expert_coverage.json" for key in coverage}},
                   output / "expert_coverage.json")
    finish(output, weight_map, candidate, qualification=qualification)
    write_json({"complete": not qualification, "qualification": qualification, "seconds": time.monotonic()-started,
                "candidate": candidate, "completed_layers": list(metrics)}, output/"conversion.json")
    verify_sources(root)


if __name__ == "__main__":
    p=argparse.ArgumentParser(__doc__); p.add_argument("--run", type=Path, required=True)
    p.add_argument("--candidate", choices=("r64_scale", "r64_gptq"), required=True)
    p.add_argument("--qualification", action="store_true")
    p.add_argument("--gptq-calibration", type=Path)
    a=vars(p.parse_args()); a["root"]=a.pop("run"); convert(**a)
