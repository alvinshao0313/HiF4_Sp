"""Share B's frozen shards and write only the aligned BF16 router shard."""
import argparse
import os
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file, save_file

from .artifacts import load, read_json, sha256, state_digest, verify, write_json
from .config import CANDIDATE


def materialize(root, *, qualification=False, zero_step=False):
    root, manifest = verify(root)
    origin = Path(manifest["initialization"])
    name = "B_zero_step" if zero_step else CANDIDATE
    output = root / ("qualification/models" if qualification else "models") / name
    output.mkdir(parents=True, exist_ok=False)
    exported = read_json(origin / "export.json")
    index = read_json(origin / "model.safetensors.index.json")
    router_shard = "model-routers.safetensors"
    for shard in sorted(set(index["weight_map"].values()) - {router_shard}):
        os.link(origin / shard, output / shard)
        if not os.path.samefile(origin / shard, output / shard):
            raise RuntimeError("frozen shard is not identical to B")
    routers = load_file(str(origin / router_shard))
    initial = state_digest(routers)
    if not zero_step:
        directory = root / ("qualification/training" if qualification else "training")
        trained = read_json(directory / "manifest.json")
        expected = list(range(2 if qualification else 48))
        if not trained["complete"] or trained["completed_layers"] != expected:
            raise ValueError("all requested router layers must be complete")
        for layer in expected:
            value = load(directory / f"layers/{layer:03d}/selected.pt")["router_weight"]
            key = f"model.layers.{layer}.mlp.gate.weight"
            if value.dtype != torch.bfloat16 or value.shape != routers[key].shape:
                raise RuntimeError("invalid selected BF16 router weight")
            routers[key] = value.contiguous()
    save_file(routers, str(output / router_shard))
    reloaded = load_file(str(output / router_shard))
    if state_digest(routers) != state_digest(reloaded):
        raise RuntimeError("exported router tensors changed")
    if zero_step and state_digest(reloaded) != initial:
        raise RuntimeError("zero-step router export differs from B")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "generation_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
                 "hif4_runtime_spec.pt", "model.safetensors.index.json"):
        if (origin / name).is_file():
            shutil.copyfile(origin / name, output / name)
    # The ABI marker is source-owned metadata, copied verbatim with the runtime.
    from ..e2e_diag_reconstruction.evaluation.moe_materialize import _write_runtime_abi_marker
    _write_runtime_abi_marker(output, load(origin / "hif4_runtime_spec.pt"))
    write_json({"complete": True, "candidate": CANDIDATE, "qualification": qualification,
                "zero_step": zero_step, "initialization": str(origin),
                "non_router_shards_identical": True, "router_tensor_sha256": state_digest(routers),
                "shards_sha256": {**exported["shards_sha256"], router_shard: sha256(output / router_shard)},
                "storage": "BF16 HiF4 QDQ values; logical W4A4; not packed storage"}, output / "export.json")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--qualification", action="store_true")
    parser.add_argument("--zero-step", action="store_true")
    args = parser.parse_args()
    if args.zero_step and not args.qualification:
        parser.error("zero-step is an isolated qualification only")
    materialize(args.run, qualification=args.qualification, zero_step=args.zero_step)
