"""The existing matrix_norm path with an independent corpus and output root."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from ..e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from ..e2e_diag_reconstruction.core.moe_semantic_hif4 import NativeQwen3MoELayerRuntime
from ..e2e_diag_reconstruction.data.batching import DynamicCalibrationCollator, build_length_bucket_batches
from ..e2e_diag_reconstruction.training.moe_trainer import build_initial_moe_hidden_cache
from ..module_design_ablation.student import Student
from ..non_equivalent_reconstruction.train import propagate, run_epoch, teacher_targets
from .artifacts import atomic_save, load, read_json, sha256, state_digest, validate_root, verify_sources, write_json
from .config import MODELS, NATIVE
from .prepare import require_isolation


def samples_for(root, candidate):
    if candidate not in MODELS[:2]:
        raise ValueError(candidate)
    directory = Path(root) / "calibration" / ("s1k_clean" if candidate == MODELS[0] else "native_traces")
    samples = tuple(load(directory / f"{split}.pt") for split in ("train", "val"))
    if [len(s) for s in samples] != [128, 32]:
        raise ValueError("expected 128 train and 32 validation samples")
    ids = [s.sample_id for split in samples for s in split]
    if len(set(ids)) != 160:
        raise ValueError("calibration sample IDs overlap")
    hashes = {split: sha256(directory / f"{split}.pt") for split in ("train", "val")}
    if candidate == MODELS[0]:
        cleaned = read_json(directory / "manifest.json")
        if not cleaned["complete"] or hashes != cleaned["calibration_sha256"]:
            raise ValueError("cleaned S1K changed or is incomplete")
        audited = read_json(Path(root) / "isolation/audit.json")["s1k"]
        if any(hashes[split] != audited[split]["sha256"] for split in hashes):
            raise ValueError("S1K changed after isolation audit")
    if candidate == MODELS[1]:
        generated = read_json(Path(root) / "generation/manifest.json")
        if not generated["complete"] or hashes != generated["calibration_sha256"]:
            raise ValueError("native trace calibration changed or is incomplete")
        attempts = {row["id"]: row for row in generated["attempts"] if row.get("accepted")}
        for split, values in zip(("train", "val"), samples):
            for domain in ("math", "code"):
                if sum(s.meta["domain"] == domain for s in values) != {"train": 64, "val": 16}[split]:
                    raise ValueError("native trace domain balance is incorrect")
            for sample in values:
                attempt = attempts[sample.sample_id]
                path = Path(root) / "generation" / attempt["path"]
                record = read_json(path)
                ids = torch.tensor(record["prompt_token_ids"] + record["generated_token_ids"], dtype=torch.long)
                if (attempt["split"] != split or sha256(path) != attempt["sha256"]
                        or not torch.equal(ids, sample.input_ids) or not sample.loss_mask.bool().all()):
                    raise ValueError("native calibration no longer preserves its complete generated token IDs")
    return *samples, hashes


def context(samples):
    tokenizer = AutoTokenizer.from_pretrained(NATIVE, local_files_only=True)
    collator = DynamicCalibrationCollator(tokenizer.pad_token_id)
    device = torch.device("cuda:0")
    cache = build_initial_moe_hidden_cache(NATIVE, samples, collator, device, 4)
    hidden = {s.sample_id: cache.get(s.sample_id) for s in samples}
    return collator, device, hidden


def config():
    return SimpleNamespace(batch_size=4, calib_seed=42, router_loss="top_mass", router_top_k=8,
                           router_temperature=1., router_loss_weight=0.)


def train(root, candidate, *, qualification=False):
    root = validate_root(root)
    if not qualification:
        require_isolation(root)
    verify_sources(root)
    training, validation, hashes = samples_for(root, candidate)
    output = root / ("qualification" if qualification else "training") / candidate
    output.mkdir(parents=True, exist_ok=False)
    if not qualification and candidate == MODELS[0]:
        write_json({"reused": False, "reason": "user_authorized_cleaned_s1k_requires_fresh_training"}, output / "reuse.json")
    if qualification:
        training = sorted(training, key=lambda s: s.input_ids.numel(), reverse=True)[:4]
        validation = sorted(validation, key=lambda s: s.input_ids.numel(), reverse=True)[:4]
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    samples = training + validation
    collator, device, hidden = context(samples)
    cfg = config()
    manifest = {"complete": False, "candidate": candidate, "qualification": qualification,
                "calibration_sha256": hashes, "completed_layers": [], "initialization": str(NATIVE),
                "source_hashes": read_json(root / "source_hashes.json"),
                "sample_ids": {"train": [s.sample_id for s in training], "val": [s.sample_id for s in validation]}}
    write_json(manifest, output / "manifest.json")
    started = time.monotonic()
    for layer in range(2 if qualification else 48):
        base = load_qwen3_moe_layer_state(NATIVE, layer, device)
        teacher = NativeQwen3MoELayerRuntime(base).to(device).eval()
        targets, logits = teacher_targets(teacher, NATIVE, samples, collator, hidden, device, 4)
        del teacher
        student = Student(base, "matrix_norm").to(device)
        names = [n for n, p in student.named_parameters() if p.requires_grad]
        if not names or any(not n.startswith(("learned.matrices.", "learned.norms.")) or "router" in n for n in names):
            raise ValueError(f"wrong stage-one trainable parameters: {names}")
        frozen_before = state_digest({"router": base.router_weight, **dict(student.learned.named_buffers())})
        epochs = 2 if qualification else 20
        optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=0.)
        batches = len(build_length_bucket_batches(training, 4, 42))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * batches, eta_min=0.)
        initial_state = state_digest(student.learned.snapshot())
        best = None
        for epoch in range(epochs):
            t = run_epoch(student, cfg, NATIVE, training, collator, hidden, targets, logits, device,
                          epoch=epoch, optimizer=optimizer, scheduler=scheduler)
            v = run_epoch(student, cfg, NATIVE, validation, collator, hidden, targets, logits, device, epoch=epoch)
            if best is None or v["reconstruction_nmse"] < best["metrics"]["reconstruction_nmse"]:
                best = {"epoch": epoch + 1, "metrics": v, "parameters": student.learned.snapshot()}
            write_json({"train": t, "validation": v}, output / f"layers/{layer:03d}/epochs/{epoch + 1:03d}.json")
        selected = output / f"layers/{layer:03d}/selected.pt"
        atomic_save(best, selected)
        student.learned.load_state_dict(load(selected)["parameters"], strict=True)
        if state_digest(student.learned.snapshot()) != state_digest(best["parameters"]):
            raise RuntimeError("selected state differs after reload")
        if initial_state == state_digest(student.learned.snapshot()):
            raise RuntimeError("stage-one parameters did not update")
        frozen_after = state_digest({"router": base.router_weight, **dict(student.learned.named_buffers())})
        if frozen_before != frozen_after:
            raise RuntimeError("stage-one frozen state changed")
        hidden = propagate(student, NATIVE, samples, hidden, device, 4)
        if any(not torch.isfinite(x).all() for x in hidden.values()):
            raise RuntimeError("invalid propagated layer output")
        manifest["completed_layers"].append(layer)
        manifest["seconds"] = time.monotonic() - started
        write_json(manifest, output / "manifest.json")
        del student, base, targets, logits, optimizer, scheduler, best
        gc.collect()
        torch.cuda.empty_cache()
    verify_sources(root)
    manifest["complete"] = True
    write_json(manifest, output / "manifest.json")


def training_dir(root, candidate, *, qualification=False):
    directory = Path(root) / ("qualification" if qualification else "training") / candidate
    manifest = read_json(directory / "manifest.json")
    if not manifest["complete"] or manifest.get("qualification", False) != qualification:
        raise ValueError("training checkpoint is incomplete or diagnostic")
    return Path(manifest.get("training_dir", directory))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidate", choices=MODELS[:2], required=True)
    parser.add_argument("--qualification", action="store_true")
    args = parser.parse_args()
    train(args.run, args.candidate, qualification=args.qualification)
