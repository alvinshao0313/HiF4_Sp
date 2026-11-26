"""Use audited, original trace tokens without re-tokenizing generated text."""
from __future__ import annotations

from collections import Counter
import random
from pathlib import Path

import torch

from ..ban.evaluate import file_sha256
from ..ban.evaluation_report import _sha256
from .config import read


def load_traces(root):
    root = Path(root).resolve()
    generation = read(root / "generation/manifest.json")
    isolation = read(root / "isolation/manifest.json")
    if not generation["complete"]:
        raise ValueError("source trace generation is incomplete")
    inputs = {}
    for key, filename in (("pools_sha256", "pools.json"), ("audit_sha256", "audit.json"),
                          ("questions_sha256", "questions.json")):
        path = root / "isolation" / filename
        if file_sha256(path) != isolation[key]:
            raise ValueError(f"source isolation evidence changed: {filename}")
        inputs[str(path)] = isolation[key]
    audit = read(root / "isolation/audit.json")
    if audit["status"] != "passed" or not audit["includes_official_source_tests"]:
        raise ValueError("audited isolation from official tests is required")
    clean = root / "calibration/s1k_clean/manifest.json"
    if file_sha256(clean) != isolation["clean_s1k_manifest_sha256"]:
        raise ValueError("source isolation binding changed")
    inputs[str(clean)] = isolation["clean_s1k_manifest_sha256"]
    records = {"train": [], "val": []}
    seen = set()
    for split, expected in (("train", 128), ("val", 32)):
        path = root / "calibration/native_traces" / f"{split}.pt"
        if file_sha256(path) != generation["calibration_sha256"][split]:
            raise ValueError(f"source trace tensors changed: {split}")
        inputs[str(path)] = generation["calibration_sha256"][split]
        samples = torch.load(path, map_location="cpu", weights_only=False)
        by_id = {sample.sample_id: sample for sample in samples}
        if len(samples) != expected or len(by_id) != expected:
            raise ValueError(f"incomplete source split: {split}")
        for attempt in generation["attempts"]:
            if attempt["split"] != split or not attempt["accepted"]:
                continue
            path = root / "generation" / attempt["path"]
            if file_sha256(path) != attempt["sha256"]:
                raise ValueError(f"source generated record changed: {path}")
            inputs[str(path)] = attempt["sha256"]
            row = read(path)
            if (row["id"] in seen or row["id"] not in by_id or row["split"] != split
                    or not row["accepted"] or not row["generated_token_ids"]):
                raise ValueError("trace split/ID/completion mismatch")
            ids = row["prompt_token_ids"] + row["generated_token_ids"]
            sample = by_id[row["id"]]
            if (not torch.equal(sample.input_ids, torch.tensor(ids, dtype=torch.long))
                    or not sample.loss_mask.bool().all() or sample.meta["domain"] != row["domain"]):
                raise ValueError("generated tokens differ from saved calibration tokens")
            seen.add(row["id"])
            records[split].append({"id": row["id"], "domain": row["domain"], "tokens": ids,
                                   "prompt_tokens": len(row["prompt_token_ids"]),
                                   "question_sha256": row["question_sha256"], "source": str(path),
                                   "source_sha256": attempt["sha256"]})
        if Counter(r["domain"] for r in records[split]) != {"math": expected // 2, "code": expected // 2}:
            raise ValueError("source domain balance differs from 64/64 train and 16/16 val")
    if ({r["question_sha256"] for r in records["train"]}
            & {r["question_sha256"] for r in records["val"]}):
        raise ValueError("training and validation questions overlap")
    for path in (root / "generation/manifest.json", root / "isolation/manifest.json"):
        inputs[str(path)] = file_sha256(path)
    return records, inputs


def sample_windows(training, nsamples=512, seqlen=2048, seed=0):
    if nsamples < 2 or nsamples % 2 or seqlen < 1:
        raise ValueError("balanced sampling requires an even number of windows and positive length")
    generator = random.Random(seed)
    tensors, windows, excluded, eligible_counts = [], [], [], {}
    for domain in ("math", "code"):
        eligible = []
        for row in training:
            if row["domain"] != domain:
                continue
            if len(row["tokens"]) < seqlen:
                excluded.append({"id": row["id"], "domain": domain, "tokens": len(row["tokens"]),
                                 "reason": "shorter than calibration window"})
            else:
                eligible.append(row)
        if not eligible:
            raise ValueError(f"no {domain} training trace can supply a full window")
        eligible_counts[domain] = len(eligible)
        for _ in range(nsamples // 2):
            row = generator.choice(eligible)
            end = generator.randint(max(seqlen, row["prompt_tokens"] + 1), len(row["tokens"]))
            start = end - seqlen
            ids = row["tokens"][start:end]
            tensors.append(torch.tensor(ids, dtype=torch.long))
            windows.append({"id": row["id"], "domain": domain, "start": start, "end_exclusive": end,
                            "source": row["source"], "source_sha256": row["source_sha256"],
                            "prompt_tokens": row["prompt_tokens"], "input_tokens_sha256": _sha256(ids)})
    covered = {}
    for row in windows:
        covered.setdefault(row["id"], set()).update(range(row["start"], row["end_exclusive"]))
    metadata = {"dataset": "audited_native_generated_traces", "split": "train", "seed": seed,
                "nsamples": nsamples, "seqlen": seqlen, "domain_windows": {"math": nsamples // 2, "code": nsamples // 2},
                "sampling": "uniform eligible trace with replacement, then uniform window end in generated response",
                "eligible_traces": eligible_counts, "excluded": excluded, "windows": windows,
                "unique_source_questions": len(covered), "token_visits": nsamples * seqlen,
                "unique_source_token_positions": sum(len(v) for v in covered.values()),
                "duplicate_windows": len(windows) - len({(w["id"], w["start"]) for w in windows})}
    return torch.stack(tensors), metadata


def budget_inputs(validation):
    rows = []
    for domain in ("math", "code"):
        selected = [r for r in validation if r["domain"] == domain][:8]
        if len(selected) != 8:
            raise ValueError("budget matching requires eight held-out prompts per domain")
        for row in selected:
            ids = row["tokens"][:row["prompt_tokens"]]
            if not ids or len(ids) + 38912 > 40960:
                raise ValueError("budget prompt does not fit the formal context")
            rows.append({"id": row["id"], "domain": domain, "input_ids": ids,
                         "input_tokens_sha256": _sha256(ids), "source": row["source"]})
    return rows
