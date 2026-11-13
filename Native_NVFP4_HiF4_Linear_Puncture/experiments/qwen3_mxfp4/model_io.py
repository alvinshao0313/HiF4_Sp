"""Read real expert tensors using the same contiguous TP4 partition contract."""
from contextlib import ExitStack
import json
from pathlib import Path

from safetensors import safe_open
import torch


def load_experts(model, layer=0, rank=0, native=False):
    if not 0 <= rank < 4:
        raise ValueError("Expected TP4 rank 0..3")
    model = Path(model)
    mapping = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    suffixes = ("weight", "weight_scale", "weight_scale_2", "input_scale") if native else ("weight", "weight_scale")
    names = [f"model.layers.{layer}.mlp.experts.{eid}.{part}.{suffix}"
             for eid in range(128) for part in ("gate_proj", "up_proj", "down_proj") for suffix in suffixes]
    with ExitStack() as stack:
        readers = {shard: stack.enter_context(safe_open(str(model / shard), framework="pt")) for shard in {mapping[name] for name in names}}
        def get(eid, part, suffix):
            name = f"model.layers.{layer}.mlp.experts.{eid}.{part}.{suffix}"
            return readers[mapping[name]].get_tensor(name)
        tensors = {}
        for suffix in ("weight", "weight_scale"):
            group = 2 if suffix == "weight" else (16 if native else 32)
            tensors["w13_" + suffix] = torch.stack([
                torch.cat([get(eid, part, suffix)[rank*192:(rank+1)*192] for part in ("gate_proj", "up_proj")])
                for eid in range(128)])
            tensors["w2_" + suffix] = torch.stack([
                get(eid, "down_proj", suffix)[:, rank*(192//group):(rank+1)*(192//group)]
                for eid in range(128)]).contiguous()
        if native:
            tensors["g1"] = torch.stack([get(eid, "gate_proj", "weight_scale_2").reshape(()) for eid in range(128)])
            tensors["g2"] = torch.stack([get(eid, "down_proj", "weight_scale_2").reshape(()) for eid in range(128)])
            tensors["a1"] = torch.stack([get(eid, part, "input_scale").reshape(()) for eid in range(128) for part in ("gate_proj", "up_proj")]).amax().reciprocal().reshape(1)
            tensors["a2"] = torch.stack([get(eid, "down_proj", "input_scale").reshape(()) for eid in range(128)]).amax().reciprocal().reshape(1)
    return tensors
