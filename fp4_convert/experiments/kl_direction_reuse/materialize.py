"""Shared immutable R64 baseline and a single replaced layer; no E4 loader."""
import os
from pathlib import Path
import shutil

import torch
from safetensors.torch import save_file

from ..e2e_diag_reconstruction.evaluation.moe_materialize import materialize_moe_identity_checkpoint, _state_to_tensors
from ..e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from ..non_equivalent_reconstruction.transforms import LayerParameters, transformed_state
from .artifacts import load, parameters_hash, read_json, sha256, write_json
from .data import Dataset


def model_identity(directory):
    directory = Path(directory)
    names = sorted(p.name for p in directory.glob("*.safetensors"))
    names += ["config.json", "model.safetensors.index.json"]
    if (directory / "hif4_runtime_spec.pt").exists():
        names.append("hif4_runtime_spec.pt")
    for name in ("hf_quant_config.json", "hif4_runtime_abi.json"):
        if (directory / name).exists():
            names.append(name)
    return {name: sha256(directory / name) for name in names}


def baseline(root):
    ds = Dataset(root)
    out = ds.root / "baseline"
    if out.exists():
        raise FileExistsError("baseline already exists; incomplete exports must be inspected explicitly")
    materialize_moe_identity_checkpoint(source_snapshot=ds.snapshot, output_dir=out, use_r64=True)
    write_json(dict(status="COMPLETE", kind="identity_diag_r64_hif4", source=str(ds.snapshot),
                    source_files=model_identity(ds.snapshot), files=model_identity(out),
                    protocol_sha256=sha256(ds.root / "protocol.json")), out / "export.json")
    return out


def check_baseline(root, *, verify_files=True):
    root = Path(root)
    out = root / "baseline"
    manifest = read_json(out / "export.json")
    if manifest["status"] != "COMPLETE" or manifest["kind"] != "identity_diag_r64_hif4" or manifest["protocol_sha256"] != sha256(root / "protocol.json"):
        raise RuntimeError("baseline provenance mismatch")
    if verify_files and manifest["files"] != model_identity(out):
        raise RuntimeError("frozen baseline changed")
    runtime = load(out / "hif4_runtime_spec.pt")
    if not runtime["use_r64"] or runtime["runtime_abi_version"] != 3:
        raise RuntimeError("baseline must use R64 and HiF4 ABI 3")
    return manifest


def candidate(root, checkpoint_path, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    ds = Dataset(root)
    manifest = check_baseline(root)
    if manifest["source_files"] != model_identity(ds.snapshot):
        raise RuntimeError("source checkpoint changed after baseline conversion")
    cp = load(checkpoint_path)
    if cp["protocol_sha256"] != sha256(root / "protocol.json"):
        raise RuntimeError("checkpoint belongs to another protocol")
    layer, params = cp["layer"], cp["parameters"]
    if layer not in ds.protocol["protocol"]["layers"]:
        raise ValueError("checkpoint layer outside the experiment")
    output.mkdir(parents=True, exist_ok=False)
    replaced = f"model-layer-{layer:05d}-of-00048.safetensors"
    for p in (root / "baseline").iterdir():
        if p.name == replaced or p.name == "export.json":
            continue
        if p.suffix == ".safetensors":
            os.link(p, output / p.name)
        else:
            shutil.copy2(p, output / p.name)
    state = load_qwen3_moe_layer_state(ds.snapshot, layer, "cpu")
    learned = LayerParameters(state, "group")
    learned.load_state_dict(params, strict=True)
    with torch.no_grad():
        tensors = _state_to_tensors(transformed_state(state, learned))
    save_file(tensors, str(output / replaced))
    files = dict(manifest["files"])
    files[replaced] = sha256(output / replaced)
    write_json(dict(status="COMPLETE", kind="single_layer_non_equivalent", layer=layer,
                    checkpoint=str(Path(checkpoint_path).resolve()), checkpoint_sha256=sha256(checkpoint_path),
                    parameters_sha256=parameters_hash(params), files=files,
                    protocol_sha256=sha256(root / "protocol.json")), output / "export.json")
    return output
