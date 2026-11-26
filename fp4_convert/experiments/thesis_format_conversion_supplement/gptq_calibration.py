"""Expand GPTQ calibration within the pinned, previously clustered clean S1K pool."""
import argparse
from pathlib import Path
import random

import torch

from ..e2e_diag_reconstruction.data.calibration import build_s1k_original_sample
from ..reasoning_trace_calibration.clean_s1k import source
from ..reasoning_trace_calibration.config import ENGINE, SEED
from ..reasoning_trace_calibration.isolation import Clusters
from ..reasoning_trace_calibration.prepare import require_isolation
from ..reasoning_trace_calibration.training import samples_for
from .artifacts import atomic_save, load, read_json, run_root, sha256, write_json
from .config import SOURCE_RUN


def eligible_indices(questions, edges, pools):
    """Reuse the complete 1000-row cluster graph, including transitive matches."""
    positions = {row["id"]: i for i, row in enumerate(questions)}
    if len(positions) != len(questions):
        raise ValueError("duplicate question identifiers")
    clusters = Clusters(len(questions))
    for edge in edges:
        clusters.union(positions[edge["left"]], positions[edge["right"]])
    held_out = {row["id"] for domain in pools.values() for split, rows in domain.items()
                if split in ("dev", "val") for row in rows}
    blocked = {clusters.find(i) for i, row in enumerate(questions)
               if row["role"] == "test" or row.get("known_test_origin")
               or row.get("split") == "val" or row["id"] in held_out}
    return [row["source_index"] for i, row in enumerate(questions)
            if row["id"].startswith("s1k_original_") and clusters.find(i) not in blocked]


def prepare(root):
    root = run_root(root)
    directory = root / "gptq_calibration"
    directory.mkdir(parents=True, exist_ok=False)
    require_isolation(SOURCE_RUN)
    training, validation, hashes = samples_for(SOURCE_RUN, "matrix_norm_s1k")
    questions = read_json(SOURCE_RUN / "isolation/questions.json")
    audit = read_json(SOURCE_RUN / "isolation/audit.json")
    pools = read_json(SOURCE_RUN / "isolation/pools.json")
    data, tokenizer, _, records, evidence = source()
    recorded = {row["source_index"]: row for row in questions if row["id"].startswith("s1k_original_")}
    if len(recorded) != len(data):
        raise ValueError("prior isolation graph does not cover the complete S1K source")
    for row in records:
        prior = recorded[row["source_index"]]
        if any(row[key] != prior[key] for key in ("question", "identities", "known_test_origin")):
            raise ValueError("pinned source differs from the previously audited question graph")
    eligible = set(eligible_indices(questions, audit["edges"], pools))
    old_indices = [s.source_index for s in training]
    if not set(old_indices) <= eligible:
        raise ValueError("the expanded isolation rules exclude original training rows")
    ordered = old_indices + [i for i in random.Random(SEED).sample(range(len(data)), len(data))
                             if i in eligible and i not in old_indices]
    samples = training + [build_s1k_original_sample(tokenizer, data[i], i) for i in ordered[len(training):]]
    if any(s.input_ids.numel() > ENGINE["max_model_len"] or not s.loss_mask.bool().all() for s in samples):
        raise ValueError("expanded calibration must preserve complete valid sequences within context")
    if {s.sample_id for s in samples} & {s.sample_id for s in validation}:
        raise ValueError("expanded calibration overlaps validation")
    atomic_save(samples, directory / "pool.pt")
    write_json({"complete": True, "source": evidence, "source_run": str(SOURCE_RUN),
                "original_calibration_sha256": hashes,
                "isolation_sha256": {name: sha256(SOURCE_RUN / "isolation" / name)
                                     for name in ("questions.json", "audit.json", "pools.json", "manifest.json")},
                "policy": "retain original 128 then append unused seed42 order; exclude test, known-test, validation and development clusters",
                "seed": SEED, "original_count": len(training), "count": len(samples),
                "ids": [s.sample_id for s in samples], "indices": ordered,
                "tokens": sum(s.input_ids.numel() for s in samples),
                "max_length": max(s.input_ids.numel() for s in samples),
                "pool_sha256": sha256(directory / "pool.pt"),
                "authorization": "2026-11-17 user permits larger GPTQ calibration to cover every expert",
                "scope": "GPTQ only; A20/B20/orthogonal calibration and downstream protocols unchanged"},
               directory / "pool.json")
    print(f"Prepared {len(samples)} complete clean S1K sequences", flush=True)


def load_pool(root):
    directory = run_root(root) / "gptq_calibration"
    manifest = read_json(directory / "pool.json")
    if not manifest["complete"] or sha256(directory / "pool.pt") != manifest["pool_sha256"]:
        raise ValueError("expanded calibration pool changed")
    samples = load(directory / "pool.pt")
    if [s.sample_id for s in samples] != manifest["ids"]:
        raise ValueError("expanded sample IDs changed")
    return samples, manifest


def select_all(root):
    """Freeze one enlarged corpus for all layers; never add held-out rows mid-run."""
    root = run_root(root)
    samples, pool = load_pool(root)
    directory = root / "gptq_calibration"
    atomic_save(samples, directory / "train.pt")
    write_json({"complete": True, "count": len(samples), "tokens": pool["tokens"],
                "max_length": pool["max_length"], "ids": pool["ids"],
                "train_sha256": sha256(directory / "train.pt"), "pool_sha256": pool["pool_sha256"],
                "policy": pool["policy"], "user_amendment": pool["authorization"],
                "comparison_boundary": "GPTQ uses more calibration than A20; report 128 versus expanded count and full preparation/conversion cost"},
               directory / "manifest.json")


def load_selected(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if not manifest["complete"] or sha256(directory / "train.pt") != manifest["train_sha256"]:
        raise ValueError("selected expanded GPTQ calibration changed")
    samples = load(directory / "train.pt")
    if len(samples) != manifest["count"] or [s.sample_id for s in samples] != manifest["ids"]:
        raise ValueError("selected expanded GPTQ sample identity mismatch")
    return samples, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    prepare(parser.parse_args().run)
