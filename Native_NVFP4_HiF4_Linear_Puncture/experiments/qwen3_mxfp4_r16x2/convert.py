"""Stream the fixed Qwen3 NVFP4 snapshot into an independently verified variant."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .artifacts import code_hashes, file_hash, save, tensor_record
from .format import FORMAT_VERSION, SCALE_RULE, VARIANTS, convert_weight
from .transform import r16x2_matrix

EXPERT = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale|weight_scale_2|input_scale)$")
REVISION = "2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3"
SOURCE = Path.home() / f".cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/{REVISION}"


def expected_prefixes():
    return {f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
            for layer in range(48) for expert in range(128)
            for projection in ("gate_proj", "up_proj", "down_proj")}


def convert(source, output, variant, threads=8):
    source, output = Path(source).absolute(), Path(output).absolute()
    if threads <= 0 or variant not in VARIANTS:
        raise ValueError("Invalid conversion threads/variant")
    if source != SOURCE:
        raise ValueError(f"This experiment pins the original snapshot: {SOURCE}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    config = json.loads((source / "config.json").read_text())
    required = {"model_type": "qwen3_moe", "num_hidden_layers": 48, "num_experts": 128,
                "num_experts_per_tok": 8, "hidden_size": 2048, "moe_intermediate_size": 768,
                "hidden_act": "silu", "norm_topk_prob": True, "mlp_only_layers": []}
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Source architecture differs from the approved Qwen3 model")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    mapping = index["weight_map"]
    prefixes = expected_prefixes()
    expert_keys = {p + "." + s for p in prefixes for s in ("weight", "weight_scale", "weight_scale_2", "input_scale")}
    if {k for k in mapping if ".mlp.experts." in k} != expert_keys:
        raise ValueError("Source expert coverage is not exactly 18,432 four-field matrices")
    if any("shared_expert" in k for k in mapping):
        raise ValueError("Unexpected shared expert")
    torch.set_num_threads(threads)
    output.mkdir(parents=True)
    started = time.monotonic()
    manifest = {"status": "converting", "created_at": datetime.now(timezone.utc).isoformat(),
                "source_snapshot": str(source), "source_revision": REVISION,
                "variant": variant, "threads": threads, "format": FORMAT_VERSION,
                "scale_rule": SCALE_RULE, "source_index_sha256": file_hash(source / "model.safetensors.index.json"),
                "source_code_sha256": code_hashes(),
                "non_expert_tensors": {}, "expert_tensors": {}, "source_shards": {},
                "output_shards": {}, "copied_files": {}, "converted_matrices": 0,
                "weight_payload_bits_per_element_including_scale": 4.25}
    if variant == "r16x2":
        manifest["r16x2"] = tensor_record(r16x2_matrix())
        manifest["transform"] = "(H2/sqrt(2)) tensor regular R4 tensor regular R4; FP32; contiguous K32; no additional signs/permutations"
    save(output / "conversion.json", manifest)
    new_mapping = {}
    try:
        with ExitStack() as stack:
            readers = {name: stack.enter_context(safe_open(str(source / name), framework="pt", device="cpu"))
                       for name in sorted(set(mapping.values()))}
            def get(name):
                return readers[mapping[name]].get_tensor(name)
            # Native ModelOpt packs gate/up together with one global scale.
            for layer in range(48):
                for expert in range(128):
                    prefix = f"model.layers.{layer}.mlp.experts.{expert}."
                    if not torch.equal(get(prefix + "gate_proj.weight_scale_2"), get(prefix + "up_proj.weight_scale_2")):
                        raise ValueError(f"Native fused gate/up global scales differ: {prefix}")
            for shard, reader in readers.items():
                manifest["source_shards"][shard] = {"bytes": (source / shard).stat().st_size, "sha256": file_hash(source / shard)}
                tensors = {}
                expected = {}
                for key in reader.keys():
                    match = EXPERT.fullmatch(key)
                    if match:
                        if match[4] != "weight":
                            continue
                        prefix = key.removesuffix(".weight")
                        expected_shape = (2048, 384) if match[3] == "down_proj" else (768, 1024)
                        if tuple(get(key).shape) != expected_shape:
                            raise ValueError(f"Unexpected source expert shape: {key}")
                        packed, scales = convert_weight(get(key), get(prefix + ".weight_scale"),
                                                        get(prefix + ".weight_scale_2"), variant)
                        for suffix, tensor in (("weight", packed), ("weight_scale", scales)):
                            name = prefix + "." + suffix
                            tensors[name] = tensor
                            expected[name] = tensor_record(tensor)
                            manifest["expert_tensors"][name] = expected[name]
                        manifest["converted_matrices"] += 1
                        if manifest["converted_matrices"] % 256 == 0:
                            print(f"{variant}: {manifest['converted_matrices']}/18432 matrices; {time.monotonic()-started:.1f}s", flush=True)
                    else:
                        tensor = reader.get_tensor(key)
                        tensors[key] = tensor
                        expected[key] = tensor_record(tensor)
                        manifest["non_expert_tensors"][key] = expected[key]
                partial = output / (shard + ".partial")
                save_file(tensors, str(partial), metadata={"format": "pt"})
                del tensors
                with safe_open(str(partial), framework="pt", device="cpu") as saved:
                    if set(saved.keys()) != set(expected):
                        raise ValueError(f"Saved tensor names differ: {shard}")
                    for name in saved.keys():
                        if tensor_record(saved.get_tensor(name)) != expected[name]:
                            raise ValueError(f"Save/reload mismatch: {name}")
                        if name in new_mapping:
                            raise ValueError(f"Duplicate output key: {name}")
                        new_mapping[name] = shard
                partial.replace(output / shard)
                manifest["output_shards"][shard] = {"bytes": (output / shard).stat().st_size,
                                                     "sha256": file_hash(output / shard)}
                save(output / "conversion.json", manifest)
        output_expert_keys = {p + "." + s for p in prefixes for s in ("weight", "weight_scale")}
        if set(new_mapping) != (set(mapping) - expert_keys) | output_expert_keys:
            raise ValueError("Output coverage/retained tensor names differ")
        if manifest["converted_matrices"] != 18432:
            raise ValueError("Incomplete conversion")
        marker = {"format": FORMAT_VERSION, "variant": variant, "group_size": 32,
                  "scale_rule": SCALE_RULE, "activation": "dynamic_mxfp4_qdq_bf16",
                  "r16x2_sha256": manifest["r16x2"]["sha256"]}
        config["quantization_config"]["qwen3_routed_mxfp4_r16x2"] = marker
        save(output / "config.json", config)
        # The alternate vLLM config source carries the same opt-in marker while
        # retaining all native non-expert ModelOpt fields.
        hf_config = json.loads((source / "hf_quant_config.json").read_text())
        hf_config["qwen3_routed_mxfp4_r16x2"] = marker
        save(output / "hf_quant_config.json", hf_config)
        for path in sorted(source.iterdir()):
            if path.is_file() and path.name not in {"config.json", "hf_quant_config.json", "model.safetensors.index.json"} and path.suffix != ".safetensors":
                shutil.copyfile(path, output / path.name)
                if file_hash(path) != file_hash(output / path.name):
                    raise ValueError(f"Auxiliary file mismatch: {path.name}")
                manifest["copied_files"][path.name] = file_hash(path)
        manifest.update(status="completed", elapsed_seconds=time.monotonic()-started,
                        non_expert_byte_exact=True, saved_reload_byte_exact=True,
                        non_expert_tensor_count=len(manifest["non_expert_tensors"]),
                        output_tensor_count=len(new_mapping),
                        actual_safetensors_bytes=sum(x["bytes"] for x in manifest["output_shards"].values()))
        save(output / "conversion.json", manifest)
        # Publication point: no loadable complete checkpoint index before ALL checks.
        total_bytes = sum(x["bytes"] for section in ("expert_tensors", "non_expert_tensors") for x in manifest[section].values())
        save(output / "model.safetensors.index.json", {"metadata": {"total_size": total_bytes}, "weight_map": new_mapping})
        print(f"{variant}: completed and verified {manifest['converted_matrices']} matrices", flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save(output / "conversion.json", manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    convert(args.source, args.output, args.variant, args.threads)
