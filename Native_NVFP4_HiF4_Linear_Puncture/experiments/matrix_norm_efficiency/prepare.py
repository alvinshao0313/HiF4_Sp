"""Reuse verified native traces, audit AIME24, and select complete sequences."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time

from ..reasoning_trace_calibration.config import MODELS
from ..reasoning_trace_calibration.isolation import cluster_questions
from ..reasoning_trace_calibration.prepare import require_isolation
from ..reasoning_trace_calibration.training import samples_for as source_samples
from .artifacts import atomic_save, digest, load, read_json, sha256, validate_root, write_json
from .config import NATIVE, SEED, SOURCE_RUN, TOKEN_FRACTION
from .selection import Candidate, select


SELECTIONS = ("full", "random50", "coverage50")


def isolation_audit(records, tests):
    """Check the fixed training, validation, and development membership together."""
    groups, edges = cluster_questions(list(records) + list(tests))
    overlaps, cross_split = [], []
    for group in groups:
        selected = [r for r in group if r["role"] != "test"]
        formal = [r for r in group if r["role"] == "test"]
        if selected and formal:
            overlaps.append({"members": [r["id"] for r in group],
                             "affected_splits": sorted({r["split"] for r in selected})})
        if len({r["split"] for r in selected}) > 1:
            cross_split.append([r["id"] for r in selected])
    return {"status": "blocked_test_overlap" if overlaps else "blocked_cross_split" if cross_split else "passed",
            "selected_questions": len(records), "test_questions": len(tests), "clusters": len(groups),
            "test_overlap": overlaps, "cross_split_overlap": cross_split, "edges": edges,
            "rule": {"character_ngram": 5, "jaccard_minimum": .8, "numeric_masking": True},
            "replacement_policy": "no_automatic_replacement"}


def _whole_sequences(samples):
    if not samples or len({s.sample_id for s in samples}) != len(samples):
        raise ValueError("calibration split is empty or has duplicate IDs")
    for sample in samples:
        if (sample.input_ids.ndim != 1 or sample.input_ids.numel() == 0
                or sample.loss_mask.shape != sample.input_ids.shape or not sample.loss_mask.eq(1).all()):
            raise ValueError(f"complete-sequence loss coverage changed: {sample.sample_id}")


def _aime24(root):
    from datasets import Dataset

    manifest_path = root / "evaluation/data/manifest.json"
    manifest = read_json(manifest_path)
    entry = manifest["tasks"]["aime24"]
    if entry["repo"] != "HuggingFaceH4/aime_2024" or entry["count"] != 30 or not entry["source_revision"]:
        raise ValueError("AIME24 must bind the official complete 30-question source")
    split = entry["splits"]["train"]
    path = (root / split["path"]).resolve()
    if not path.is_relative_to(root) or sha256(path) != split["sha256"]:
        raise ValueError("AIME24 Arrow differs from the pinned run input")
    dataset = Dataset.from_file(str(path))
    if len(dataset) != entry["count"]:
        raise ValueError("AIME24 question coverage differs from its manifest")
    rows = [{"id": f"evaluation:aime24:{i}", "domain": "math", "role": "test",
             "question": row["problem"], "identities": []} for i, row in enumerate(dataset)]
    return rows, {"manifest_path": str(manifest_path.relative_to(root)), "manifest_sha256": sha256(manifest_path),
                  "repo": entry["repo"], "source_revision": entry["source_revision"],
                  "path": str(path.relative_to(root)), "sha256": split["sha256"], "count": len(rows)}


def prepare(root):
    root = validate_root(root)
    if (root / "prepare.inputs.json").exists() or (root / "calibration").exists():
        raise FileExistsError("prepared calibration already exists; use a new run")
    started = time.monotonic()
    source = Path(SOURCE_RUN).resolve()
    require_isolation(source)
    training, validation, source_hashes = source_samples(source, MODELS[1])
    for samples in (training, validation):
        _whole_sequences(samples)
    generation = read_json(source / "generation/manifest.json")
    if Path(generation["source_snapshot"]).resolve() != Path(NATIVE).resolve():
        raise ValueError("native trace teacher differs from the experiment's initial native model")
    pools = read_json(source / "isolation/pools.json")
    questions = {r["id"]: r for domain in pools.values() for split in domain.values() for r in split}
    dev = [dict(row) for domain in ("math", "code") for row in pools[domain]["dev"]]
    if len(dev) != 128 or any(len(pools[d]["dev"]) != 64 for d in ("math", "code")):
        raise ValueError("fixed development questions must remain 64 math and 64 code")
    records = []
    for split, samples in (("train", training), ("val", validation)):
        for sample in samples:
            original = questions[sample.sample_id]
            if original["question"] != sample.meta["question"] or original["domain"] != sample.meta["domain"]:
                raise ValueError(f"calibration question metadata differs from its fixed pool: {sample.sample_id}")
            records.append({**original, "split": split, "role": "selected"})
    records.extend({**row, "split": "dev", "role": "selected"} for row in dev)
    tests, aime24 = _aime24(root)
    audit = isolation_audit(records, tests)
    write_json({**audit, "aime24": aime24}, root / "prepare.aime24_audit.json")
    if audit["status"] != "passed":
        raise ValueError(f"new AIME24 isolation failed: {audit['status']}; fixed val/dev were not changed")

    candidates = [Candidate(s.sample_id, s.meta["domain"], s.input_ids.numel(), s.meta["question"]) for s in training]
    selections = {name: select(candidates, name, fraction=TOKEN_FRACTION, seed=SEED) for name in SELECTIONS}
    destination = root / "calibration"
    destination.mkdir(exist_ok=False)
    write_json(dev, destination / "development.json")
    outputs = {"calibration/development.json": sha256(destination / "development.json"),
               "prepare.aime24_audit.json": sha256(root / "prepare.aime24_audit.json")}
    reports = {}
    for name, (ids, report) in selections.items():
        directory = destination / name
        directory.mkdir()
        if name == "full":
            shutil.copyfile(source / "calibration/native_traces/train.pt", directory / "train.pt")
        else:
            chosen = set(ids)
            atomic_save([s for s in training if s.sample_id in chosen], directory / "train.pt")
        shutil.copyfile(source / "calibration/native_traces/val.pt", directory / "val.pt")
        hashes = {split: sha256(directory / f"{split}.pt") for split in ("train", "val")}
        if hashes["val"] != source_hashes["val"]:
            raise ValueError("fixed validation bytes changed while preparing selections")
        if name == "full" and hashes["train"] != source_hashes["train"]:
            raise ValueError("full native calibration bytes changed")
        report.update(calibration_sha256=hashes, validation_ids=[s.sample_id for s in validation],
                      validation_tokens=sum(s.input_ids.numel() for s in validation))
        write_json(report, directory / "selection.json")
        reports[name] = {k: report[k] for k in ("source_tokens", "selected_count", "selected_tokens", "budget_tokens", "unused_tokens")}
        for filename in ("train.pt", "val.pt", "selection.json"):
            path = directory / filename
            outputs[str(path.relative_to(root))] = sha256(path)
    source_files = ["generation/manifest.json", "isolation/manifest.json", "isolation/audit.json", "isolation/pools.json",
                    "calibration/native_traces/train.pt", "calibration/native_traces/val.pt"]
    source_evidence = {str(source / relative): sha256(source / relative) for relative in source_files}
    inputs = {"complete": True, "source_run": str(source), "source_corpus": "native_traces",
              "source_files": source_evidence, "aime24": aime24, "files": outputs, "selections": reports,
              "development_ids": [r["id"] for r in dev], "development_used_for_selection": False,
              "clean_s1k_evidence_only": str(source / "calibration/s1k_clean/manifest.json"),
              "seconds": time.monotonic() - started}
    inputs["content_sha256"] = digest(inputs)
    write_json(inputs, root / "prepare.inputs.json")
    return inputs


def samples_for(root, selection):
    root = validate_root(root)
    if selection not in SELECTIONS:
        raise ValueError(f"unknown calibration selection: {selection}")
    inputs = read_json(root / "prepare.inputs.json")
    if not inputs["complete"] or digest({k: v for k, v in inputs.items() if k != "content_sha256"}) != inputs["content_sha256"]:
        raise ValueError("prepared input manifest changed or is incomplete")
    directory = root / "calibration" / selection
    names = ("train.pt", "val.pt", "selection.json")
    for filename in names:
        path = directory / filename
        if sha256(path) != inputs["files"][str(path.relative_to(root))]:
            raise ValueError(f"prepared calibration changed: {path}")
    for relative in ("calibration/development.json", "prepare.aime24_audit.json"):
        if sha256(root / relative) != inputs["files"][relative]:
            raise ValueError(f"prepared isolation or development input changed: {relative}")
    for key, expected in (("manifest_path", "manifest_sha256"), ("path", "sha256")):
        if sha256(root / inputs["aime24"][key]) != inputs["aime24"][expected]:
            raise ValueError("pinned AIME24 input changed after data isolation")
    report = read_json(directory / "selection.json")
    training, validation = (load(directory / f"{split}.pt") for split in ("train", "val"))
    for samples in (training, validation):
        _whole_sequences(samples)
    if ([s.sample_id for s in training] != report["selected_ids"]
            or [s.sample_id for s in validation] != report["validation_ids"]
            or len({s.sample_id for s in training + validation}) != len(training) + len(validation)):
        raise ValueError("prepared calibration membership or isolation changed")
    return training, validation, report["calibration_sha256"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    print(prepare(parser.parse_args().run))
