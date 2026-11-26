"""Freeze the user-selected 512 C4 training windows of 512 tokens."""
import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path
import random
import time

import torch
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer

from ..e2e_diag_reconstruction.data.calibration import CalibrationSample
from ..reasoning_trace_calibration.isolation import Clusters, grams, normalize
from ..reasoning_trace_calibration.prepare import require_isolation
from ..thesis_format_conversion_supplement.artifacts import (
    atomic_save, digest, read_json, run_root, sha256, write_json,
)
from ..thesis_format_conversion_supplement.config import NATIVE, SOURCE_RUN

REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
C4 = Path("/home/shaoyuantian/.cache/huggingface/datasets/allenai___c4/default-b04fc8a0b8562884/0.0.0") / REVISION


def held_out_questions():
    require_isolation(SOURCE_RUN)
    rows = read_json(SOURCE_RUN / "isolation/questions.json")
    positions = {row["id"]: i for i, row in enumerate(rows)}
    groups = Clusters(len(rows))
    for edge in read_json(SOURCE_RUN / "isolation/audit.json")["edges"]:
        groups.union(positions[edge["left"]], positions[edge["right"]])
    pools = read_json(SOURCE_RUN / "isolation/pools.json")
    ids = {row["id"] for domain in pools.values() for split, items in domain.items()
           if split in ("dev", "val") for row in items}
    blocked = {groups.find(i) for i, row in enumerate(rows)
               if row["role"] == "test" or row.get("known_test_origin")
               or row.get("split") == "val" or row["id"] in ids}
    return [row for i, row in enumerate(rows) if groups.find(i) in blocked]


class ExclusionIndex:
    """Existing 5-gram Jaccard >= 0.8 rule, plus embedded exact questions.

    Length filtering is lossless for the set-Jaccard threshold. The additional
    substring check detects a test question embedded inside a longer webpage.
    """
    def __init__(self, rows):
        self.views = []
        for masked in (False, True):
            entries = {}
            for row in rows:
                text = normalize(row["question"], mask_numbers=masked)
                entries.setdefault(text, row["id"])
            by_size, anchors, short = defaultdict(list), defaultdict(list), []
            for text, identity in entries.items():
                group = grams(text)
                by_size[len(group)].append((group, identity))
                if len(text) < 16:
                    short.append((text, identity))
                else:
                    anchors[text[:16]].append((text, identity))
            self.views.append((masked, by_size, sorted(by_size), anchors, short))

    def match(self, text):
        for masked, by_size, sizes, anchors, short in self.views:
            normalized = normalize(text, mask_numbers=masked)
            for question, identity in short:
                if question in normalized:
                    return {"id": identity, "kind": "embedded_exact", "numeric_masked": masked}
            for start in range(len(normalized)):
                for question, identity in anchors.get(normalized[start:start+16], ()):
                    if normalized.startswith(question, start):
                        return {"id": identity, "kind": "embedded_exact", "numeric_masked": masked}
            group = grams(normalized)
            low, high = (4*len(group)+4)//5, 5*len(group)//4
            for size in sizes[bisect_left(sizes, low):bisect_right(sizes, high)]:
                for other, identity in by_size[size]:
                    overlap = len(group & other)
                    if 5*overlap >= 4*(len(group)+len(other)-overlap):
                        return {"id": identity, "kind": "5gram_jaccard", "numeric_masked": masked}
        return None


def prepare(root):
    count, sequence_length = 512, 512
    root = run_root(root)
    output = root / "gptq_calibration"
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    paths = [C4 / f"c4-train-{i:05d}-of-00002.arrow" for i in range(2)]
    info = read_json(C4 / "dataset_info.json")
    if list(info["download_checksums"]) != [f"hf://datasets/allenai/c4@{REVISION}/en/c4-train.00000-of-01024.json.gz"]:
        raise ValueError("cached C4 does not have the pinned English train provenance")
    data = concatenate_datasets([Dataset.from_file(str(p)) for p in paths])
    tokenizer = AutoTokenizer.from_pretrained(NATIVE, local_files_only=True)
    excluded = held_out_questions()
    print(f"Building exclusion index for {len(excluded)} held-out/connected questions", flush=True)
    index = ExclusionIndex(excluded)
    samples, records, rejected, seen = [], [], [], set()
    windows = random.Random(42)
    for row_index in random.Random(42).sample(range(len(data)), len(data)):
        row = data[row_index]
        text = row["text"]
        if not isinstance(text, str) or not text.strip():
            rejected.append({"row": row_index, "reason": "empty"}); continue
        text_hash = digest(text)
        if text_hash in seen:
            rejected.append({"row": row_index, "reason": "duplicate_document"}); continue
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if len(tokens) < sequence_length:
            rejected.append({"row": row_index, "reason": "shorter_than_window", "tokens": len(tokens)}); continue
        match = index.match(text)
        if match:
            rejected.append({"row": row_index, "reason": "held_out_overlap", "match": match}); continue
        start = windows.randint(0, len(tokens)-sequence_length)
        sample_id = f"c4_en_train_{row_index}"
        ids = torch.tensor(tokens[start:start+sequence_length], dtype=torch.long)
        samples.append(CalibrationSample(sample_id, row_index, ids, torch.ones_like(ids),
                                        {"dataset": "allenai/c4", "revision": REVISION, "split": "train", "full_document": False,
                                         "window_start": start, "document_tokens": len(tokens)}))
        records.append({"id": sample_id, "row": row_index, "tokens": sequence_length, "document_tokens": len(tokens),
                        "window_start": start, "text_sha256": text_hash, "input_ids_sha256": digest(ids.tolist()), "url": row.get("url")})
        seen.add(text_hash)
        if len(samples) % 128 == 0:
            print(f"Prepared {len(samples)}/{count} C4 windows of {sequence_length} tokens", flush=True)
        if len(samples) == count:
            break
    if len(samples) != count:
        raise ValueError("not enough isolated C4 training documents for distinct windows")
    atomic_save(samples, output / "train.pt")
    manifest = {"complete": True, "dataset": "allenai/c4", "revision": REVISION, "subset": "en", "split": "train",
                "count": count, "seed": 42, "s1k_count": 0, "full_documents": False,
                "tokens_per_sequence": sequence_length, "document_boundaries_preserved": True,
                "tokens": sum(s.input_ids.numel() for s in samples),
                "max_length": max(s.input_ids.numel() for s in samples),
                "ids": [s.sample_id for s in samples], "train_sha256": sha256(output / "train.pt"),
                "sources": {str(p): sha256(p) for p in paths}, "dataset_info_sha256": sha256(C4 / "dataset_info.json"),
                "isolation_sources": {str(SOURCE_RUN / "isolation" / name): sha256(SOURCE_RUN / "isolation" / name)
                                      for name in ("questions.json", "audit.json", "pools.json")},
                "held_out_questions": len(excluded), "rejected": rejected, "documents": records,
                "policy": "user-selected 512x512; seed42 shuffled fixed local train shard, one seeded 512-token window per distinct isolated document; no concatenation or chat template",
                "comparison_boundary": "pure C4 replaces S1K for GPTQ only; source and token budgets differ from A20",
                "preparation_seconds": time.monotonic()-began,
                "source_sha256": {str(p): sha256(p) for p in Path(__file__).parent.glob("*.py")}}
    write_json(manifest, output / "manifest.json")
    print({k: manifest[k] for k in ("count", "tokens", "max_length", "train_sha256", "preparation_seconds")}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    prepare(parser.parse_args().run)
