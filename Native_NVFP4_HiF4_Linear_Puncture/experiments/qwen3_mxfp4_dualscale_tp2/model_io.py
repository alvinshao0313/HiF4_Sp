"""Read real expert tensors using the same contiguous TP2 partition contract."""
from contextlib import ExitStack
import json
from pathlib import Path

from safetensors import safe_open
import torch


def load_experts(model, layer=0, rank=0, outer=True):
    if not 0 <= rank < 2:
        raise ValueError("Expected TP2 rank 0..1")
    model = Path(model)
    mapping = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    suffixes = ("weight", "weight_scale", "weight_scale_fp32") if outer else ("weight", "weight_scale")
    names = [f"model.layers.{layer}.mlp.experts.{eid}.{part}.{suffix}"
             for eid in range(128) for part in ("gate_proj", "up_proj", "down_proj") for suffix in suffixes]
    with ExitStack() as stack:
        readers = {shard: stack.enter_context(safe_open(str(model / shard), framework="pt")) for shard in {mapping[name] for name in names}}
        def get(eid, part, suffix):
            name = f"model.layers.{layer}.mlp.experts.{eid}.{part}.{suffix}"
            return readers[mapping[name]].get_tensor(name)
        tensors = {}
        for suffix in suffixes:
            group = 2 if suffix == "weight" else 32
            tensors["w13_" + suffix] = torch.stack([
                torch.cat([get(eid, part, suffix)[rank*384:(rank+1)*384] for part in ("gate_proj", "up_proj")])
                for eid in range(128)])
            if suffix == "weight_scale_fp32":
                tensors["w2_" + suffix] = torch.stack([get(eid, "down_proj", suffix) for eid in range(128)])
                continue
            tensors["w2_" + suffix] = torch.stack([
                get(eid, "down_proj", suffix)[:, rank*(384//group):(rank+1)*(384//group)]
                for eid in range(128)]).contiguous()
        return tensors
