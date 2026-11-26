"""Reuse frozen input identities and verify the two existing checkpoints."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil

import torch

from ..qwen3_mxfp4.artifacts import check_hashes, tensor_record
from ..qwen3_mxfp4_dualscale_tp2.model_io import load_experts
from .common import GPQA_REFERENCE, HERE, LANES, REFERENCE, REPO, VARIANTS, file_hash, now, read, save


def prepare(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    previous = read(REFERENCE / "manifest.json")
    check_hashes(previous["source_sha256"])
    old = read(GPQA_REFERENCE / "workflow_manifest.json")
    models = {v: old["models"][v]["checkpoint"] for v in VARIANTS}
    direct = Path(models["direct"])
    conversion = read(direct / "conversion.json")
    if (conversion["status"] != "completed" or conversion["variant"] != "direct"
            or conversion["converted_matrices"] != 18432
            or not conversion["non_expert_byte_exact"] or not conversion["saved_reload_byte_exact"]
            or conversion["source_snapshot"] != models["native"]):
        raise ValueError("Missing original native/direct checkpoint evidence")
    shard_hashes = {}
    for variant, field in (("native", "source_shards"), ("direct", "output_shards")):
        for name, record in conversion[field].items():
            path = Path(models[variant]) / name
            shard_hashes[str(path)] = record["sha256"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for path, actual in zip(shard_hashes, pool.map(file_hash, shard_hashes)):
            if actual != shard_hashes[path]:
                raise ValueError(f"Checkpoint bytes changed: {path}")
    for name, expected in conversion["copied_files"].items():
        if any(file_hash(Path(model) / name) != expected for model in models.values()):
            raise ValueError(f"Native/direct tokenizer or generation file changed: {name}")
        reference_model = Path(next(iter(previous["models"].values()))["checkpoint"])
        if file_hash(reference_model / name) != expected:
            raise ValueError(f"Tokenizer differs from combination comparison: {name}")
    print("Native and direct checkpoint bytes and tokenizer identities verified", flush=True)
    for name in ("datasets.json", "lcb_inputs.json", "mmlu_pro_inputs.json",
                 "lcb_prompt_example.json", "mmlu_pro_prompt_example.json"):
        shutil.copyfile(REFERENCE / name, root / name)
    from ..qwen3_mxfp4_rotation_eval.common import install_datasets
    install_datasets(root / "datasets.json")
    expected = {"non_expert_methods": next(iter(read(REFERENCE / "expected_runtime.json").values()))["non_expert_methods"],
                "direct_ranks": {"0": {}, "1": {}}}
    torch.set_num_threads(8)
    for layer in range(48):
        for rank in range(2):
            tensors = load_experts(direct, layer, rank, outer=False)
            expected["direct_ranks"][str(rank)][f"model.layers.{layer}.mlp.experts"] = {
                key: tensor_record(value) for key, value in tensors.items()}
        if layer % 16 == 15:
            print(f"Direct TP2 packed slice identities: {layer+1}/48 layers", flush=True)
    save(root / "expected_runtime.json", expected)
    source_hashes = dict(previous["source_sha256"])
    paths = list(HERE.glob("*.py")) + list(root.glob("*.json")) + [direct / "conversion.json"]
    for model in models.values():
        paths += [p for p in Path(model).iterdir() if p.is_file() and p.suffix in (".json", ".jinja")]
    source_hashes.update({str(path): file_hash(path) for path in paths})
    config_reference = REFERENCE / "dualscale_g512_uos_c7p25/formal/mmlu_pro/summary.json"
    config = read(config_reference)["protocol"]
    save(root / "manifest.json", {"status": "prepared", "created_at": now(), "models": models,
         "reference": str(REFERENCE), "source_sha256": source_hashes, "checkpoint_sha256": shard_hashes,
         "protocol": config, "inputs": previous["inputs"], "aisbench_revision": previous["aisbench_revision"],
         "aisbench_sources": previous["aisbench_sources"], "lanes": LANES,
         "authorization": "User requested native NVFP4 and direct MXFP4 LCB/MMLU-Pro, GPU2-7 in TP2 pairs, AISBench prompts and BF16 KV",
         "sampling": "Five actual responses per question; arithmetic mean, no trimming or voting",
         "kv_gib_per_card": 56, "tp": 2, "no_weight_or_quantizer_changes": True})
    print(f"Prepared {root}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    prepare(parser.parse_args().output)
