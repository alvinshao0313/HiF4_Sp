"""Export full-scope W4A4 numerical-emulation weights for the paper."""
import argparse
from copy import deepcopy
from collections import defaultdict
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .checkpoint import Checkpoint
from .operands import converted_weight
from .scale_methods import METHOD_TARGET, convert_scale_weight


def export(source, output, target, method, seed, device, calibration=None):
    if output.exists():
        raise FileExistsError(output)
    if method in METHOD_TARGET and target != METHOD_TARGET[method]:
        raise ValueError(f"{method} requires target {METHOD_TARGET[method]}")
    if (method in ("brq", "brq_fused")) != (seed is not None):
        raise ValueError("Specify --seed only for the BRQ random-block baseline")
    if (method in ("gptq", "awq", "smoothquant")) != (calibration is not None):
        raise ValueError("GPTQ, AWQ and SmoothQuant require their own --calibration directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with Checkpoint(source) as checkpoint, TemporaryDirectory(dir=output.parent) as temp, ExitStack() as stack:
        scope = checkpoint.verify_headers()
        fusion = None
        if method == "brq_fused":
            from .fusion import FusedRotation
            fusion = FusedRotation(checkpoint, target, seed, device)
        folder = Path(temp) / "model"
        folder.mkdir()
        calibration_info = {}
        divisors = {}
        if calibration is not None:
            recipe_path = calibration / "recipe.json"
            recipe = json.loads(recipe_path.read_text())
            if (recipe["source"] != checkpoint.source_id() or recipe["target"] != target
                    or recipe["method"] != method or not recipe["scope_complete"]
                    or set(recipe["weights"]) != checkpoint.native):
                raise ValueError("Calibration source, target, method or full projection coverage mismatch")
            readers = {shard: stack.enter_context(safe_open(calibration / shard, framework="pt"))
                       for shard in set(recipe["weights"].values())}
            metadata = folder / "paper_metadata"
            metadata.mkdir()
            copyfile(recipe_path, metadata / "recipe.json")
            calibration_info = {"purpose": recipe["purpose"], "calibration_recipe_sha256":
                                hashlib.sha256(recipe_path.read_bytes()).hexdigest()}
        # Retain the original shard ordering. Sidecars can live in another shard.
        shards = defaultdict(list)
        for key, filename in checkpoint.weights.items():
            shards[filename].append(key)
        if fusion is not None and fusion.tied_head:
            if "lm_head.weight" in checkpoint.weights:
                raise ValueError("Tied source unexpectedly contains an independent output head")
            shards[checkpoint.weights[fusion.embedding]].append("lm_head.weight")
        index = {}
        total_bytes = 0
        matrices = {}
        for filename, keys in sorted(shards.items()):
            tensors = {}
            for key in keys:
                name, _, suffix = key.rpartition(".")
                if name in checkpoint.native:
                    if suffix in ("weight_scale", "weight_scale_2", "input_scale"):
                        continue
                    if suffix == "weight":
                        if fusion is not None:
                            tensor = fusion.tensor(key)
                        elif method in METHOD_TARGET:
                            tensor, outer, matrices[name] = convert_scale_weight(checkpoint, name, method, device)
                            if outer is not None:
                                outer_key = name + ".weight_scale_fp32"
                                tensors[outer_key] = outer
                                index[outer_key] = filename
                                total_bytes += outer.numel() * outer.element_size()
                            print(f"{method} {name} {matrices[name]['elapsed_seconds']:.2f}s", flush=True)
                        elif calibration is None:
                            tensor = converted_weight(checkpoint, name, target, method, seed, device)
                        else:
                            reader = readers[recipe["weights"][name]]
                            tensor = reader.get_tensor(key)
                            rows, packed = checkpoint.tensor(key).shape
                            if tensor.dtype != torch.bfloat16 or tensor.shape != (rows, packed * 2):
                                raise ValueError(f"Invalid fitted operand for {name}")
                            if method in ("awq", "smoothquant"):
                                scale = reader.get_tensor(name + ".input_divisor")
                                if (scale.shape != (packed * 2,) or scale.dtype != torch.float32
                                        or not torch.isfinite(scale).all() or not (scale > 0).all()):
                                    raise ValueError(f"Invalid channel scale for {name}")
                                divisors[name] = scale
                    else:
                        tensor = fusion.tensor(key) if fusion is not None else checkpoint.tensor(key)
                else:
                    tensor = fusion.tensor(key) if fusion is not None else checkpoint.tensor(key)
                tensors[key] = tensor
                index[key] = filename
                total_bytes += tensor.numel() * tensor.element_size()
            if tensors:
                save_file(tensors, folder / filename, metadata={"format": "pt"})
                print(f"{filename}: wrote {len(tensors)} tensors", flush=True)
            del tensors
        if divisors:
            save_file(divisors, metadata / "input_divisors.safetensors")
        scale_info = {}
        if method in METHOD_TARGET:
            if set(matrices) != checkpoint.native:
                raise ValueError("Incomplete scale conversion")
            rule = {
                "matrix_c": "one divisor per source matrix, 6:0.25:8, G64 hardware thresholds 4/2",
                "scale_search": "full G64 E6M2 offsets -2,-1,0,1,2 and all e8/e4 combinations",
                "uos": "G32 E2M1 RNE, E8M0 ceil(log2(amax/7.25))",
                "g256": "FP32 absmax/6 G256, normalized MXFP4 G32 floor rule, FP32 scaled partials",
                "g512": "FP32 absmax/6 G512, normalized MXFP4 G32 floor rule, FP32 scaled partials",
            }[method]
            scale_info = {"source_id": checkpoint.source_id(), "format_rule": rule,
                          "activation_rule": "standard HiF4 G64 c7 thresholds 4/2" if target == "hif4" else rule,
                          "matrices": matrices}
        spec = {"quant_method": "paper_fp4", "target": target, "method": method,
                "seed": seed, "storage": "bf16_dequantized_operands",
                "nvfp4": sorted(checkpoint.native), "fp8": sorted(checkpoint.fp8),
                "kv_cache_quant_algo": checkpoint.quant.get("kv_cache_quant_algo")}
        config = deepcopy(checkpoint.config)
        if config["model_type"] == "nemotron_h":
            # The environment has native Nemotron-H support. The release's
            # remote-code map points at a different (unquantized) repository.
            config.pop("auto_map", None)
        if fusion is not None and fusion.tied_head:
            config["tie_word_embeddings"] = False
            if "text_config" in config:
                config["text_config"]["tie_word_embeddings"] = False
        original = config.get("quantization_config", {})
        if "kv_cache_scheme" in original:
            spec["kv_cache_scheme"] = original["kv_cache_scheme"]
        config["quantization_config"] = spec
        (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (folder / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": total_bytes}, "weight_map": index}, indent=2) + "\n")
        (folder / "conversion.json").write_text(json.dumps({
            "source": str(checkpoint.root), "target": target, "method": method,
            "seed": seed, "operand_storage": "BF16 QDQ", **scope, **calibration_info,
            **scale_info,
            **({"fusion": fusion.metadata()} if fusion is not None else {})}, indent=2) + "\n")
        for file in checkpoint.root.iterdir():
            if file.suffix.lower() == ".md" and not file.name.lower().startswith("license"):
                continue  # The original release card describes a different numerical format.
            if file.is_file() and file.name not in ("config.json", "model.safetensors.index.json",
                    "hf_quant_config.json") and not file.name.startswith("model-") and file.suffix != ".safetensors":
                copyfile(file, folder / file.name)
        if output.exists():
            raise FileExistsError(output)
        folder.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target", required=True, choices=["hif4", "mxfp4"])
    parser.add_argument("--method", choices=["direct", "brq", "brq_fused", "gptq", "awq", "smoothquant", *METHOD_TARGET], default="direct")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    args = parser.parse_args()
    with torch.inference_mode():
        export(**vars(args))
