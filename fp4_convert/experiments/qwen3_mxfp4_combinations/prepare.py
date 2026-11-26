"""Freeze four converted checkpoints and the existing AISBench GPQA protocol."""
import argparse
from copy import deepcopy
from pathlib import Path

import torch

from ..qwen3_aisbench_gpqa.common import now, read
from ..qwen3_mxfp4_dualscale_tp2.model_io import load_experts
from .artifacts import code_hashes, check_hashes, file_hash, save, tensor_record
from .format import VARIANTS, metadata, options


def prepare(root):
    root = Path(root).resolve()
    if (root / "preparation.json").exists():
        raise FileExistsError("Preparation already exists")
    auth = read(root / "authorization.json")
    reference = Path(auth["reference_run"]) / "direct_uos_c7p25"
    original = read(reference / "manifest.json")
    check_hashes(original["source_sha256"])
    if read(reference / "formal/runtime_validation.json")["status"] != "passed":
        raise ValueError("Missing reference TP2 runtime verification")
    previous_expected = read(reference.parent / "expected_runtime.json")["direct_uos_c7p25"]
    (root / "protocol_inputs.json").write_bytes((Path(auth["protocol_parent"]) / "protocol_inputs.json").read_bytes())
    # Preserve the root entry, lighteval and prompt dependencies frozen by the
    # completed reference, in addition to this experiment's implementation.
    sources = dict(original["source_sha256"])
    sources.update(code_hashes())
    reference_result = Path(read(reference / "probe/status.json")["results_file"])
    sources[str(reference_result)] = file_hash(reference_result)
    sources[str(root / "protocol_inputs.json")] = file_hash(root / "protocol_inputs.json")
    expected = {}
    torch.set_num_threads(8)
    for variant in VARIANTS:
        model = root / "weights" / variant
        conversion = read(model / "conversion.json")
        if (conversion["status"] != "completed" or conversion["converted_matrices"] != 18432
                or not conversion["non_expert_byte_exact"] or not conversion["saved_reload_byte_exact"]
                or conversion["quantization"] != metadata(variant)):
            raise ValueError(f"Unverified conversion: {variant}")
        check_hashes(conversion["source_code_sha256"])
        for name in ("conversion.json", "config.json", "hf_quant_config.json", "model.safetensors.index.json"):
            sources[str(model / name)] = file_hash(model / name)
        expected[variant] = {"ranks": {"0": {}, "1": {}},
                             "non_expert_methods": previous_expected["non_expert_methods"]}
        for layer in range(48):
            for rank in range(2):
                tensors = load_experts(model, layer, rank, outer=options(variant)[1])
                expected[variant]["ranks"][str(rank)][f"model.layers.{layer}.mlp.experts"] = {
                    key: tensor_record(value) for key, value in tensors.items()}
            if layer % 16 == 15:
                print(f"{variant}: TP2 slice hashes {layer+1}/48", flush=True)
    save(root / "expected_runtime.json", expected)
    sources[str(root / "expected_runtime.json")] = file_hash(root / "expected_runtime.json")
    for index, variant in enumerate(VARIANTS):
        target, checkpoint = root / variant, root / "weights" / variant
        manifest = deepcopy(original)
        devices = auth["devices"][index % 2]
        manifest.update(created_at=now(), variant=variant, source_snapshot=str(checkpoint),
                        source_sha256=sources, authorization="User approved four combinations, TP2 GPU0-3, BF16 KV56GiB",
                        checks={"cpu_tests": str(root / "cpu_validation.json"), "gpu_inference": "not_run"})
        manifest["environment"]["CUDA_VISIBLE_DEVICES"] = devices
        manifest["protocol"].update(cuda_visible_devices=devices, quantization=metadata(variant))
        manifest["probe_definition"]["purpose"] = "TP2 batch256 execution only, not accuracy"
        for phase in ("probe", "formal"):
            view = target / phase / "model"
            view.mkdir(parents=True)
            for path in checkpoint.iterdir():
                if path.is_file() and path.name != "conversion.json":
                    (view / path.name).symlink_to(path.resolve())
            command = manifest["commands"][phase]
            command[2] = str(Path(__file__).parent / "entry.py")
            for flag, value in (("--model_path", view), ("--output_dir", target / phase / "evaluation")):
                command[command.index(flag)+1] = str(value)
        save(target / "manifest.json", manifest)
    save(root / "preparation.json", {"status": "passed", "at": now(), "source_sha256": sources,
        "reference_candidate": str(reference), "all_48_layers_expected_hashes": True,
        "protocol_inputs_identical": True, "variants": list(VARIANTS)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    prepare(parser.parse_args().root)
