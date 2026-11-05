"""User-authorized S1K cleaning; preserve clean slots and refill from unused rows."""
from __future__ import annotations

import ast
import random

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from ..e2e_diag_reconstruction.data.calibration import build_s1k_original_sample
from .artifacts import atomic_save, load, read_json, sha256, write_json
from .config import ENGINE, NATIVE, SEED, S1K, S1K_ARROW, S1K_REVISION


def source():
    data = Dataset.from_file(str(S1K_ARROW))
    if len(data) != 1000:
        raise ValueError("the pinned S1K source must contain exactly 1000 rows")
    info = read_json(S1K_ARROW.parent / "dataset_info.json")
    expected = f"hf://datasets/simplescaling/s1K-1.1_tokenized@{S1K_REVISION}/data/train-00000-of-00001.parquet"
    if set(info["download_checksums"]) != {expected}:
        raise ValueError("S1K Arrow does not come from the pinned revision")
    tokenizer = AutoTokenizer.from_pretrained(NATIVE, local_files_only=True)
    originals = {split: load(S1K / "calibration" / f"{split}.pt") for split in ("train", "val")}
    order = random.Random(SEED).sample(range(len(data)), k=len(data))
    if {split: [s.source_index for s in values] for split, values in originals.items()} != {
            "train": order[:128], "val": order[128:160]}:
        raise ValueError("the existing S1K split differs from its seed42 source order")
    for values in originals.values():
        for sample in values:
            row = data[sample.source_index]
            reconstructed = build_s1k_original_sample(tokenizer, row, sample.source_index)
            if (row["question"] != sample.meta["question"] or not torch.equal(sample.input_ids, reconstructed.input_ids)
                    or not torch.equal(sample.loss_mask, reconstructed.loss_mask)):
                raise ValueError(f"pinned source fails exact original-token verification: {sample.sample_id}")
    records = []
    original_split = {s.source_index: split for split, values in originals.items() for s in values}
    for index, row in enumerate(data):
        metadata = ast.literal_eval(row["metadata"])
        if not isinstance(metadata, dict):
            raise ValueError(f"S1K metadata is not a dictionary: {index}")
        original_id = metadata.get("unique_id")
        known_test = ("math" in row["source_type"].lower() and isinstance(original_id, str)
                      and original_id.startswith("test/"))
        records.append({"id": f"s1k_original_{index}", "domain": "math", "role": "s1k_pool",
                        "question": row["question"], "identities": ["math-original:" + original_id] if original_id else [],
                        "source_index": index, "source_type": row["source_type"], "original_id": original_id,
                        "known_test_origin": known_test, "original_split": original_split.get(index)})
    evidence = {"repo": "simplescaling/s1K-1.1_tokenized", "revision": S1K_REVISION,
                "arrow": str(S1K_ARROW), "sha256": sha256(S1K_ARROW), "count": len(data),
                "fingerprint": data._fingerprint, "original_tokens_verified": 160,
                "original_sha256": {split: sha256(S1K / "calibration" / f"{split}.pt") for split in originals}}
    return data, tokenizer, originals, records, evidence


def choose(original_indices, cluster_by_index, blocked_clusters, source_count):
    """Keep eligible original slots; fill train then val in one fixed unused order."""
    original_set = {i for values in original_indices.values() for i in values}
    unused = [i for i in random.Random(SEED).sample(range(source_count), k=source_count) if i not in original_set]
    selected, dropped, used = {}, [], {}
    for split in ("train", "val"):
        selected[split] = []
        for index in original_indices[split]:
            cluster = cluster_by_index[index]
            if cluster in blocked_clusters or (cluster in used and used[cluster] != split):
                selected[split].append(None)
                dropped.append({"split": split, "source_index": index, "cluster": cluster,
                                "reason": "test_cluster" if cluster in blocked_clusters else "cross_split_cluster"})
            else:
                selected[split].append(index)
                used[cluster] = split
    replacements, cursor = [], 0
    for split in ("train", "val"):
        for slot, value in enumerate(selected[split]):
            if value is not None:
                continue
            while cursor < len(unused):
                index = unused[cursor]
                cursor += 1
                cluster = cluster_by_index[index]
                if cluster not in blocked_clusters and cluster not in used:
                    selected[split][slot] = index
                    used[cluster] = split
                    replacements.append({"split": split, "slot": slot, "removed": original_indices[split][slot],
                                         "inserted": index, "cluster": cluster})
                    break
            else:
                raise ValueError("unused S1K pool cannot fill the approved 128/32 split")
    return selected, dropped, replacements


def materialize(root, data, tokenizer, originals, groups, evidence):
    output = root / "calibration/s1k_clean"
    output.mkdir(parents=True, exist_ok=False)
    cluster_by_index, blocked, witnesses = {}, set(), {}
    for number, group in enumerate(groups):
        forbidden = [r for r in group if r["role"] == "test" or r.get("known_test_origin")]
        if forbidden:
            blocked.add(number)
            witnesses[number] = [{"id": r["id"], "known_test_origin": r.get("known_test_origin", False),
                                  "original_id": r.get("original_id")} for r in forbidden]
        for row in group:
            if row["role"] == "s1k_pool":
                cluster_by_index[row["source_index"]] = number
    # A must keep complete trajectories within the native model's actual context.
    too_long = []
    for index in range(len(data)):
        if len(tokenizer.encode(data[index]["text"], add_special_tokens=False)) > ENGINE["max_model_len"]:
            blocked.add(cluster_by_index[index])
            too_long.append(index)
    original_indices = {split: [s.source_index for s in values] for split, values in originals.items()}
    selected, dropped, replacements = choose(original_indices, cluster_by_index, blocked, len(data))
    old = {s.source_index: s for values in originals.values() for s in values}
    stats = {}
    for split, indices in selected.items():
        samples = [old[i] if i in old else build_s1k_original_sample(tokenizer, data[i], i) for i in indices]
        atomic_save(samples, output / f"{split}.pt")
        lengths = sorted(s.input_ids.numel() for s in samples)
        stats[split] = {"count": len(samples), "tokens": sum(lengths), "min": lengths[0], "max": lengths[-1],
                        "median": lengths[len(lengths) // 2], "lengths": lengths}
    selected_split = {i: split for split, indices in selected.items() for i in indices}
    for group in groups:
        for row in group:
            if "source_index" in row and row["id"].startswith("s1k_original_"):
                if row["source_index"] in selected_split:
                    row.update(role="s1k", split=selected_split[row["source_index"]])
                elif row.get("known_test_origin"):
                    row["role"] = "test"
    write_json({"complete": True, "policy": "retain_clean_original_slots_fill_unused_seed42_order", "seed": SEED,
                "source": evidence, "original_indices": original_indices, "selected_indices": selected,
                "dropped": [{**row, "witnesses": witnesses.get(row["cluster"], [])} for row in dropped],
                "replacements": replacements, "source_rows_over_context": too_long, "stats": stats,
                "calibration_sha256": {s: sha256(output / f"{s}.pt") for s in selected}}, output / "manifest.json")
    print(f"cleaned S1K: removed {len(dropped)}, kept {160 - len(dropped)}, filled {len(replacements)}", flush=True)
    return selected
