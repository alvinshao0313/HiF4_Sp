"""Fold the selected last layer and BF16 head, sharing all other B shards."""
import argparse
import os
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file, save_file

from ..e2e_diag_reconstruction.evaluation.moe_materialize import _state_to_tensors, _write_runtime_abi_marker
from ..module_design_ablation.parameters import transformed_state
from .artifacts import load, read_json, sha256, state_digest, verify, write_json
from .config import CANDIDATE
from .model import LastHead, non_layer


@torch.no_grad()
def materialize(root, *, qualification=False, zero_step=False):
    root, m = verify(root)
    torch.backends.cuda.matmul.allow_tf32 = False
    origin = Path(m["initialization"])
    name = "B_zero_step" if zero_step else CANDIDATE
    output = root / ("qualification/models" if qualification else "models") / name
    output.mkdir(parents=True, exist_ok=False)
    index = read_json(origin / "model.safetensors.index.json")
    layer_shard = index["weight_map"]["model.layers.47.input_layernorm.weight"]
    head_shard = index["weight_map"]["lm_head.weight"]
    rewritten = {layer_shard, head_shard}
    for shard in sorted(set(index["weight_map"].values()) - rewritten):
        os.link(origin / shard, output / shard)
    model = LastHead(m, torch.device("cuda:0"))
    if not zero_step:
        directory = root / ("qualification/training" if qualification else "training")
        if not read_json(directory / "manifest.json")["complete"]:
            raise ValueError("completed joint training is required")
        selected = directory / ("formal/last_updated.pt" if qualification else "selected.pt")
        model.restore(load(selected)["parameters"])
    tensors = _state_to_tensors(transformed_state(model.base, model.layer.learned))
    gate = tensors.pop("model.layers.47.mlp.gate.weight")
    gates = load_file(str(origin / index["weight_map"]["model.layers.47.mlp.gate.weight"]))
    if not torch.equal(gate, gates["model.layers.47.mlp.gate.weight"]):
        raise RuntimeError("frozen router changed during folding")
    previous = load_file(str(origin / layer_shard))
    for key in previous:
        if key.endswith(("self_attn.q_norm.weight", "self_attn.k_norm.weight")) or zero_step:
            if not torch.equal(previous[key], tensors[key]):
                raise RuntimeError(f"frozen/zero-step last-layer tensor changed: {key}")
    save_file(tensors, str(output / layer_shard))
    values = non_layer(origin)
    embedding_before = state_digest({k: v for k, v in values.items() if k not in {"model.norm.weight", "lm_head.weight"}})
    values["model.norm.weight"] = model.final_norm.to(torch.bfloat16).cpu().contiguous()
    values["lm_head.weight"] = model.fused_head().cpu().contiguous()
    if zero_step:
        old = non_layer(origin)
        if any(not torch.equal(old[k], v) for k, v in values.items()):
            raise RuntimeError("unit head or final norm changed zero-step B")
    if embedding_before != state_digest({k: v for k, v in values.items() if k not in {"model.norm.weight", "lm_head.weight"}}):
        raise RuntimeError("frozen non-layer tensor changed")
    save_file(values, str(output / head_shard))
    for shard, expected in ((layer_shard, tensors), (head_shard, values)):
        if state_digest(load_file(str(output / shard))) != state_digest(expected):
            raise RuntimeError("exported tensors changed on save/reload")
    for asset in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                  "generation_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
                  "hif4_runtime_spec.pt", "model.safetensors.index.json"):
        if (origin / asset).is_file():
            shutil.copyfile(origin / asset, output / asset)
    _write_runtime_abi_marker(output, load(origin / "hif4_runtime_spec.pt"))
    hashes = read_json(origin / "export.json")["shards_sha256"]
    hashes.update({name: sha256(output / name) for name in rewritten})
    write_json({"complete": True, "candidate": CANDIDATE, "qualification": qualification,
                "zero_step": zero_step, "zero_step_tensors_identical": zero_step,
                "initialization": str(origin), "rewritten_shards": sorted(rewritten),
                "frozen_prefix_router_QKnorm_embedding_identical": True, "shards_sha256": hashes,
                "storage": "BF16 QDQ ordinary projections (logical HiF4 W4A4); BF16 unquantized folded lm_head; not packed storage"}, output / "export.json")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--qualification", action="store_true")
    parser.add_argument("--zero-step", action="store_true")
    args = parser.parse_args()
    if args.zero_step and not args.qualification:
        parser.error("zero-step is a qualification model")
    materialize(args.run, qualification=args.qualification, zero_step=args.zero_step)
