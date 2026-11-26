"""Determine the text conversion scope from logical tensors, before TP fusion."""
from collections import defaultdict
from contextlib import ExitStack
from functools import cache
import hashlib
import json
from pathlib import Path

from safetensors import safe_open


class Checkpoint:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.config = self.read("config.json")
        self.weights = self.read("model.safetensors.index.json")["weight_map"]
        self.storage_keys = {key: key for key in self.weights}
        self.compressed = self.config.get("quantization_config", {}).get("quant_method") == "compressed-tensors"
        self.metadata_files = ["config.json", "model.safetensors.index.json"]
        if self.compressed:
            config = self.config["quantization_config"]
            groups = list(config["config_groups"].values())
            if config["format"] != "nvfp4-pack-quantized" or len(groups) != 1:
                raise ValueError("Expected a single compressed-tensors NVFP4 scheme")
            for side in ("weights", "input_activations"):
                spec = groups[0][side]
                if spec["num_bits"] != 4 or spec["group_size"] != 16 or spec["type"] != "float":
                    raise ValueError(f"Unsupported source scheme: {side}")
            if groups[0]["input_activations"]["dynamic"] != "local":
                raise ValueError("Expected static-global, dynamic-local NVFP4 activations")
            self.quant = {"quant_algo": "NVFP4", "group_size": 16}
            renamed = {}
            self.storage_keys = {}
            names = {"weight_packed": "weight", "weight_global_scale": "weight_scale_2", "input_global_scale": "input_scale"}
            for key, shard in self.weights.items():
                module, _, suffix = key.rpartition(".")
                canonical = module + "." + names.get(suffix, suffix)
                if canonical in renamed:
                    raise ValueError(f"Duplicate logical tensor {canonical}")
                renamed[canonical] = shard
                self.storage_keys[canonical] = key
            self.weights = renamed
        else:
            self.quant = self.read("hf_quant_config.json")["quantization"]
            self.metadata_files = ["config.json", "hf_quant_config.json", "model.safetensors.index.json"]
        self.native = {key.removesuffix(".weight_scale_2") for key in self.weights
                       if key.endswith(".weight_scale_2")}
        method = self.quant["quant_algo"]
        if not self.native or method not in ("NVFP4", "MIXED_PRECISION"):
            raise ValueError(f"Expected a ModelOpt NVFP4 release, found {method}")
        declared = self.quant.get("quantized_layers", {})
        if method == "MIXED_PRECISION":
            specified = {name for name, info in declared.items() if info["quant_algo"] == "NVFP4"}
            if self.native != specified:
                raise ValueError(f"NVFP4 index disagrees with release config: {self.native ^ specified}")
        self.fp8 = {name for name, info in declared.items() if info["quant_algo"] == "FP8"}
        if set(declared) - self.native - self.fp8:
            raise ValueError("Unsupported residual quantization type")
        scopes = ("model.layers.", "backbone.layers.", "model.language_model.layers.")
        for name in self.native:
            if not (name.startswith(scopes) or name == "lm_head"):
                raise ValueError(f"Determine text inference scope for {name}")
            group = declared.get(name, {}).get("group_size", self.quant.get("group_size"))
            if group != 16:
                raise ValueError(f"Unsupported NVFP4 block size for {name}: {group}")
            for suffix in ("weight", "weight_scale", "input_scale"):
                if f"{name}.{suffix}" not in self.weights:
                    raise ValueError(f"Incomplete source projection: {name}.{suffix}")

    def read(self, name):
        return json.loads((self.root / name).read_text())

    def source_id(self):
        digest = hashlib.sha256()
        for name in self.metadata_files:
            digest.update((self.root / name).read_bytes())
        return {"snapshot": self.root.name, "config_and_index_sha256": digest.hexdigest()}

    def __enter__(self):
        self.stack = ExitStack()
        self.reader = cache(lambda shard: self.stack.enter_context(
            safe_open(self.root / shard, framework="pt", device="cpu")))
        return self

    def __exit__(self, *exc):
        self.reader.cache_clear()
        return self.stack.__exit__(*exc)

    def tensor(self, key):
        value = self.reader(self.weights[key]).get_tensor(self.storage_keys[key])
        # compressed-tensors stores the global multiplier used before FP4
        # rounding. ModelOpt stores its inverse, used during reconstruction.
        if self.compressed and key.endswith((".weight_scale_2", ".input_scale")):
            value = value.reciprocal()
        return value

    def verify_headers(self):
        shards = defaultdict(set)
        for key, shard in self.weights.items():
            shards[shard].add(self.storage_keys[key])
        for shard, expected in shards.items():
            actual = set(self.reader(shard).keys())
            if actual != expected:
                raise ValueError(f"{shard}: index/header mismatch {actual ^ expected}")
        logical_values = 0
        for name in self.native:
            def header(suffix):
                key = f"{name}.{suffix}"
                sl = self.reader(self.weights[key]).get_slice(self.storage_keys[key])
                return sl.get_dtype(), sl.get_shape()
            dtype, shape = header("weight")
            if dtype != "U8" or len(shape) != 2:
                raise ValueError(f"{name}: unexpected packed tensor {dtype} {shape}")
            rows, packed_cols = shape
            sdtype, sizes = header("weight_scale")
            if sdtype != "F8_E4M3" or sizes != [rows, packed_cols // 8] or packed_cols % 8:
                raise ValueError(f"{name}: invalid NVFP4 block scales {sdtype} {sizes}")
            for suffix in ("weight_scale_2", "input_scale"):
                dtype, shape = header(suffix)
                if dtype != "F32" or shape not in ([], [1]):
                    raise ValueError(f"{name}.{suffix}: invalid scale {dtype} {shape}")
            logical_values += rows * packed_cols * 2
        return {"nvfp4_projections": len(self.native), "nvfp4_weight_values": logical_values,
                "residual_fp8_projections": len(self.fp8), "indexed_shards": len(shards),
                "indexed_tensors": len(self.weights), "nvfp4_output_head": "lm_head" in self.native}
