"""Reuse the checkpoint and frozen AISBench inputs in independent TP2 views."""
import argparse
from copy import deepcopy
from pathlib import Path

import torch

from .artifacts import code_hashes, save, tensor_record
from .model_io import load_experts
from ..qwen3_aisbench_gpqa.common import check_hashes, file_hash, now, read
from ..qwen3_mxfp4_dualscale.model_io import load_experts as load_tp4


def prepare(root):
    root = Path(root).resolve()
    parent = root.parent
    if (root / "preparation.json").exists():
        raise FileExistsError("TP2 preparation already exists")
    authorization = read(root / "authorization.json")
    if authorization["status"] != "old_controller_stopped" or authorization["new_devices"] != "0,1":
        raise ValueError("TP2 authorization is missing")
    original = read(parent / "dualscale_g512/manifest.json")
    check_hashes(original["source_sha256"])
    torch.set_num_threads(8)
    checkpoint = Path(original["source_snapshot"])
    (root / "protocol_inputs.json").write_bytes((parent / "protocol_inputs.json").read_bytes())
    config = deepcopy(original)
    config.update(created_at=now(), authorization=authorization["instruction"], validation_phase="probe")
    config["environment"]["CUDA_VISIBLE_DEVICES"] = "0,1"
    config["protocol"].update(tensor_parallel_size=2, cuda_visible_devices="0,1")
    here = Path(__file__).resolve().parent
    for phase in ("probe", "formal"):
        target = root / "dualscale_g512" / phase
        model = target / "model"
        model.mkdir(parents=True)
        for path in checkpoint.iterdir():
            if path.is_file() and path.name != "conversion.json":
                (model / path.name).symlink_to(path.resolve())
        command = config["commands"][phase]
        command[2] = str(here / "entry.py")
        for flag, value in (("--tensor_parallel_size", "2"), ("--model_path", model),
                            ("--output_dir", target / "evaluation")):
            command[command.index(flag) + 1] = str(value)
        restored = list(command)
        restored[2] = original["commands"][phase][2]
        for flag in ("--tensor_parallel_size", "--model_path", "--output_dir"):
            index = restored.index(flag)
            restored[index + 1] = original["commands"][phase][index + 1]
        if restored != original["commands"][phase]:
            raise ValueError("An unauthorized CLI change was introduced")
    sources = dict(original["source_sha256"])
    sources.update(code_hashes())
    sources[str(root / "protocol_inputs.json")] = file_hash(root / "protocol_inputs.json")
    config["source_sha256"] = sources
    save(root / "dualscale_g512/manifest.json", config)
    # Compare TP2 slices against independently loaded pairs of the old TP4 slices.
    for layer in (0, 47):
        for rank in range(2):
            two = load_experts(checkpoint, layer, rank)
            left, right = (load_tp4(checkpoint, layer, rank * 2 + offset) for offset in (0, 1))
            for key, value in two.items():
                if key.startswith("w13_"):
                    expected = torch.cat([left[key][:, :192], right[key][:, :192],
                                          left[key][:, 192:], right[key][:, 192:]], dim=1)
                elif key == "w2_weight_scale_fp32":
                    assert torch.equal(left[key], right[key])
                    expected = left[key]
                else:
                    expected = torch.cat([left[key], right[key]], dim=2)
                if not torch.equal(value, expected):
                    raise ValueError(f"TP2/TP4 partition disagreement: {layer}/{rank}/{key}")
            del two, left, right
    expected = {"ranks": {"0": {}, "1": {}}}
    for layer in range(48):
        for rank in range(2):
            tensors = load_experts(checkpoint, layer, rank)
            expected["ranks"][str(rank)][f"model.layers.{layer}.mlp.experts"] = {
                name: tensor_record(value) for name, value in tensors.items()}
            del tensors
        if (layer + 1) % 8 == 0:
            print(f"Verified TP2 checkpoint slices: {layer + 1}/48 layers", flush=True)
    from ..qwen3_aisbench_gpqa.report import canonical
    old_methods = canonical(parent / "dualscale_g512/probe/audit")
    if any(item["non_experts"] != old_methods[0]["non_experts"] for item in old_methods.values()):
        raise ValueError("Prior TP4 non-expert methods differ across ranks")
    expected["non_expert_methods"] = old_methods[0]["non_experts"]
    save(root / "expected_runtime.json", expected)
    sources[str(root / "expected_runtime.json")] = file_hash(root / "expected_runtime.json")
    config["source_sha256"] = sources
    save(root / "dualscale_g512/manifest.json", config)
    save(root / "preparation.json", {
        "status": "passed", "at": now(), "parent_run": str(parent), "checkpoint": str(checkpoint),
        "devices": "0,1", "tensor_parallel_size": 2, "source_sha256": sources,
        "input_hashes_equal": True, "weight_conversion": "none; original G512 checkpoint reused",
        "all_48_layers_two_rank_expected_hashes": True, "first_last_layer_tp4_merge_exact": True,
        "protocol_changes": ["tensor_parallel_size: 4 -> 2", "CUDA_VISIBLE_DEVICES: 0,1,2,3 -> 0,1"],
    })
    print(f"Prepared G512 TP2: {root}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    prepare(parser.parse_args().root)
