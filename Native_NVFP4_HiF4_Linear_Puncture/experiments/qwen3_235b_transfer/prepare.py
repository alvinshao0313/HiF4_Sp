"""Prepare fresh full-text calibration and real longest-batch smoke inputs."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from transformers import AutoTokenizer

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_model_spec import Qwen3MoeModelSpec
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.calibration import (
    _write_split, build_or_load_calibration, load_s1k_dataset, split_length_stats,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.artifact import sha256, write_json
from .config import diag_config, group_config, source_snapshot
from .fetch_model import MODEL, REVISION


def prepare(run_root: Path) -> None:
    if (run_root / "preparation.json").exists():
        raise FileExistsError("preparation already exists; inspect its recorded inputs")
    snapshot = source_snapshot()
    spec = Qwen3MoeModelSpec.from_snapshot(snapshot, source_model=MODEL)
    if (spec.num_layers, spec.hidden_size) != (94, 4096):
        raise ValueError("unexpected source geometry")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
        raise ValueError("source must have a nonempty chat template")
    cfg = diag_config(run_root)
    train, val = build_or_load_calibration(cfg, tokenizer, None, cfg.calib_cache_dir)
    ds = load_s1k_dataset()
    token_hash = hashlib.sha256()
    text_hash = hashlib.sha256()
    for sample in train + val:
        expected = tokenizer(ds[sample.source_index]["text"], add_special_tokens=False,
                             truncation=False)["input_ids"]
        if sample.input_ids.tolist() != expected or not sample.loss_mask.eq(1).all():
            raise ValueError(f"full-text calibration mismatch: {sample.sample_id}")
        token_hash.update(sample.input_ids.numpy().tobytes())
        text_hash.update(ds[sample.source_index]["text"].encode("utf-8") + b"\0")
    smoke_cfg = diag_config(run_root, smoke=True)
    longest = {}
    for split, samples in (("train", train), ("val", val)):
        batch = sorted(samples, key=lambda item: item.input_ids.numel(), reverse=True)[:4]
        _write_split(Path(smoke_cfg.calib_cache_dir), split, batch, smoke_cfg,
                     {"selection": "four_longest_full_text_samples_from_formal_split",
                      "formal_cache": cfg.calib_cache_dir})
        longest[split] = {"sample_ids": [s.sample_id for s in batch],
                          "lengths": [s.input_ids.numel() for s in batch]}
    write_json(cfg.to_dict(), run_root / "configs/e4.json")
    write_json(smoke_cfg.to_dict(), run_root / "configs/smoke_e4.json")
    write_json(group_config(run_root).to_dict(), run_root / "configs/group_top_mass.json")
    record = {
        "model": MODEL, "revision": REVISION, "source_snapshot": str(snapshot),
        "dataset": "simplescaling/s1K-1.1_tokenized", "dataset_fingerprint": ds._fingerprint,
        "train": split_length_stats(train), "validation": split_length_stats(val),
        "ordered_tokens_sha256": token_hash.hexdigest(), "ordered_texts_sha256": text_hash.hexdigest(),
        "source_metadata_sha256": {name: sha256(snapshot / name) for name in (
            "config.json", "hf_quant_config.json", "model.safetensors.index.json",
            "tokenizer.json", "tokenizer_config.json")},
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest(),
        "longest_batch_smoke": longest, "weights_validated": False,
    }
    write_json(record, run_root / "preparation.json")
    print(record, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True, type=Path)
    prepare(parser.parse_args().run_root.resolve())
