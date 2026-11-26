"""Independent model views, exact TP2 expert hashes and unchanged GPQA inputs."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from ..qwen3_aisbench_gpqa.common import check_hashes, file_hash, now, read, save
from ..qwen3_aisbench_gpqa.report import canonical
from ..qwen3_mxfp4.artifacts import tensor_record
from ..qwen3_mxfp4.model_io import load_experts as load_tp4
from ..qwen3_mxfp4_dualscale_tp2.model_io import load_experts


def prepare(root):
    root = Path(root).resolve()
    if (root / "preparation.json").exists():
        raise FileExistsError("Preparation already exists")
    parent = root.parent
    auth = read(root / "authorization.json")
    previous = Path(auth["old_g512_root"])
    torch.set_num_threads(8)
    original = read(previous / "preparation.json")
    check_hashes(original["source_sha256"])
    for name in ("communication.json", "operators.json", "dualscale_g512/probe/runtime_validation.json"):
        if read(previous / name)["status"] != "passed":
            raise ValueError(f"Missing original G512 TP2 evidence: {name}")
    (root / "protocol_inputs.json").write_bytes((parent / "protocol_inputs.json").read_bytes())
    sources = dict(original["source_sha256"])
    uos_original = read(parent / "direct_uos_c7p25/manifest.json")
    check_hashes(uos_original["source_sha256"])
    sources.update(uos_original["source_sha256"])
    sources.update({str(p): file_hash(p) for p in Path(__file__).parent.glob("*.py")})
    expected = {"dualscale_g512": read(previous / "expected_runtime.json")}
    uos_model = Path(uos_original["source_snapshot"])
    old = canonical(parent / "direct_uos_c7p25/probe/audit")
    expected["direct_uos_c7p25"] = {"ranks": {"0": {}, "1": {}}, "non_expert_methods": old[0]["non_experts"]}
    for layer in range(48):
        for rank in range(2):
            tensors = load_experts(uos_model, layer, rank, outer=False)
            if layer in (0, 47):
                left, right = (load_tp4(uos_model, layer, rank * 2 + i) for i in range(2))
                for key, value in tensors.items():
                    merged = (torch.cat([left[key][:, :192], right[key][:, :192], left[key][:, 192:], right[key][:, 192:]], dim=1)
                              if key.startswith("w13_") else torch.cat([left[key], right[key]], dim=2))
                    if not torch.equal(merged, value):
                        raise ValueError(f"UOS TP2 slice mismatch: {layer}/{rank}/{key}")
            expected["direct_uos_c7p25"]["ranks"][str(rank)][f"model.layers.{layer}.mlp.experts"] = {
                key: tensor_record(value) for key, value in tensors.items()}
        if layer % 8 == 7:
            print(f"UOS TP2 expected checkpoint hashes: {layer+1}/48 layers", flush=True)
    save(root / "expected_runtime.json", expected)
    for file in (root / "expected_runtime.json", root / "protocol_inputs.json"):
        sources[str(file)] = file_hash(file)
    save(root / "preparation.json", {"status": "passed", "at": now(), "source_sha256": sources,
        "previous_g512": str(previous), "source_parent": str(parent), "first_last_tp2_merge_exact": True,
        "uos_runtime_changes": "none", "all_48_layers_expected_hashes": True})


def candidate(root, variant, kv_gib):
    root = Path(root).resolve()
    if kv_gib not in (32, 56) or variant not in ("direct_uos_c7p25", "dualscale_g512"):
        raise ValueError("Unexpected candidate or qualified KV budget")
    target = root / variant
    if target.exists():
        raise FileExistsError(target)
    prep = read(root / "preparation.json")
    check_hashes(prep["source_sha256"])
    decision = read(root / "budget_decision.json")
    if kv_gib != decision["kv_gib"]:
        raise ValueError("KV budget differs from measured decision")
    original = (Path(prep["previous_g512"]) if variant == "dualscale_g512" else root.parent) / variant
    manifest = deepcopy(read(original / "manifest.json"))
    checkpoint = Path(manifest["source_snapshot"])
    manifest.update(created_at=now(), validation_phase="probe", source_sha256=prep["source_sha256"],
                    authorization="TP2, concurrent GPU pairs; increased KV only after throughput qualification")
    devices = "0,1" if variant == "dualscale_g512" else "2,3"
    manifest["environment"]["CUDA_VISIBLE_DEVICES"] = devices
    manifest["protocol"].update(tensor_parallel_size=2, cuda_visible_devices=devices,
                                 kv_cache_memory_bytes=kv_gib*1024**3,
                                 kv_cache_budget_mode=f"Explicit {kv_gib} GiB per GPU, BF16")
    for phase in ("probe", "formal"):
        model = target / phase / "model"
        model.mkdir(parents=True)
        for path in checkpoint.iterdir():
            if path.is_file() and path.name != "conversion.json":
                (model / path.name).symlink_to(path.resolve())
        command = manifest["commands"][phase]
        for flag, value in (("--model_path", model), ("--output_dir", target / phase / "evaluation"),
                            ("--tensor_parallel_size", "2"), ("--kv_cache_memory_bytes", str(kv_gib*1024**3))):
            command[command.index(flag) + 1] = str(value)
        if variant == "direct_uos_c7p25":
            command[2] = str(Path(__file__).parent.parent / "qwen3_mxfp4_uos/entry.py")
    save(target / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--variant", choices=("direct_uos_c7p25", "dualscale_g512"))
    parser.add_argument("--kv-gib", type=int)
    args = parser.parse_args()
    if args.variant:
        candidate(args.root, args.variant, args.kv_gib)
    else:
        prepare(args.root)
