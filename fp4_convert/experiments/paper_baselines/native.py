"""Native ModelOpt adapter for composite projections in Nemotron and Qwen3.8."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase, register_weight_loader_v2_supported_method
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config, ModelOptNvFp4FusedMoE, ModelOptNvFp4LinearMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from .checkpoint import Checkpoint
from .runtime import PaperConfig, bind_fused_scales
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


@register_weight_loader_v2_supported_method
class PackedNativeLinear(ModelOptNvFp4LinearMethod):
    def create_weights(self, layer, *args, **kwargs):
        super().create_weights(layer, *args, **kwargs)
        bind_fused_scales(layer, ("input_scale", "weight_scale_2"))


class NativeConfig(PaperConfig):
    def __init__(self, config):
        QuantizationConfig.__init__(self)
        self._init_scope(config)
        self.nv = ModelOptNvFp4Config(True, self.kv, [], 16)

    @classmethod
    def get_name(cls):
        return "paper_native"

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, FusedMoE) and prefix in self.experts:
            return ModelOptNvFp4FusedMoE(self.nv, layer.moe_config)
        if isinstance(layer, (LinearBase, ParallelLMHead)) and self.kind(prefix) == "target":
            return PackedNativeLinear(self.nv)
        return super().get_quant_method(layer, prefix)


def native_view(source, output):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Checkpoint(source) as checkpoint, TemporaryDirectory(dir=output.parent) as tmp:
        checkpoint.verify_headers()
        folder = Path(tmp) / "model"
        folder.mkdir()
        config = checkpoint.config.copy()
        quant = {"quant_method": "paper_native", "nvfp4": sorted(checkpoint.native),
                 "fp8": sorted(checkpoint.fp8), "kv_cache_quant_algo": checkpoint.quant.get("kv_cache_quant_algo")}
        original = config.get("quantization_config", {})
        if "kv_cache_scheme" in original:
            quant["kv_cache_scheme"] = original["kv_cache_scheme"]
        config["quantization_config"] = quant
        (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        for file in checkpoint.root.iterdir():
            if file.suffix.lower() == ".md" and not file.name.lower().startswith("license"):
                continue
            if file.is_file() and file.name not in ("hf_quant_config.json", "config.json"):
                (folder / file.name).symlink_to(file.resolve())
        folder.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    native_view(**vars(parser.parse_args()))
