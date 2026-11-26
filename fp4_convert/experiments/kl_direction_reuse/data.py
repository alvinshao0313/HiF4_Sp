from pathlib import Path
import random
import json

from .artifacts import checked_load, read_json, save, tensor_hash, write_json
from .config import PROTOCOL, recipes


def fixed_batches(samples, size=4):
    if len(samples) % size:
        raise ValueError("training samples must form complete fixed batches")
    ordered = sorted(samples, key=lambda s: (len(s["input_ids"]), s["id"]))
    return [[s["id"] for s in ordered[i:i+size]] for i in range(0, len(ordered), size)]


def epoch_order(count, epoch, seed=42):
    ids = list(range(count))
    random.Random(seed + epoch).shuffle(ids)
    return ids


def prepare(root, model=PROTOCOL.model):
    from transformers import AutoTokenizer
    from ..e2e_diag_reconstruction.data.calibration import load_s1k_dataset, build_s1k_original_sample
    from ...src.checkpoint import resolve_local_snapshot
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("prepare requires a new, empty experiment directory")
    snapshot = Path(model).resolve() if Path(model).is_dir() else Path(resolve_local_snapshot(model))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
    ds = load_s1k_dataset()
    perm = random.Random(PROTOCOL.seed).sample(range(len(ds)), k=len(ds))
    sizes = dict(train=32, val=16, test=32)
    if len(perm) < sum(sizes.values()):
        raise ValueError("not enough S1K samples")
    root.mkdir(parents=True, exist_ok=True)
    entries, splits, selected, offset, hashes = {}, {}, [], 0, set()
    for split, count in sizes.items():
        splits[split] = []
        for index in perm[offset:offset+count]:
            sample = build_s1k_original_sample(tokenizer, ds[index], index)
            if len(sample.input_ids) < 2:
                raise ValueError("NLL requires at least two tokens")
            digest = tensor_hash(sample.input_ids)
            if digest in hashes:
                raise ValueError("duplicate token sequence across the selected samples")
            hashes.add(digest)
            item = dict(id=sample.sample_id, source_index=index, split=split,
                        input_ids=sample.input_ids, token_sha256=digest)
            entries[sample.sample_id] = {**save(item, root / "data" / f"{sample.sample_id}.pt"),
                                        "tokens": len(sample.input_ids), "token_sha256": digest}
            splits[split].append(sample.sample_id)
            if split == "train":
                selected.append(item)
        offset += count
    payload = dict(status="COMPLETE", schema_version=1, protocol=PROTOCOL.to_dict(),
                   snapshot=str(snapshot.resolve()), splits=splits, samples=entries,
                   batches=fixed_batches(selected), recipes=recipes())
    write_json(payload, root / "protocol.json")
    return payload


class Dataset:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.protocol = read_json(self.root / "protocol.json")
        if self.protocol["status"] != "COMPLETE":
            raise RuntimeError("dataset preparation is incomplete")
        if self.protocol["protocol"] != json.loads(json.dumps(PROTOCOL.to_dict())) or self.protocol["recipes"] != recipes():
            raise RuntimeError("prepared protocol differs from this implementation")
        self.samples = {sid: checked_load(e) for sid, e in self.protocol["samples"].items()}
        for sid, item in self.samples.items():
            entry = self.protocol["samples"][sid]
            if item["id"] != sid or len(item["input_ids"]) != entry["tokens"] or tensor_hash(item["input_ids"]) != entry["token_sha256"]:
                raise RuntimeError("sample identity or token coverage mismatch")
        flattened = sum(self.protocol["splits"].values(), [])
        if {key: len(value) for key, value in self.protocol["splits"].items()} != {"train": 32, "val": 16, "test": 32}:
            raise RuntimeError("unexpected experiment split sizes")
        if len(flattened) != len(set(flattened)) or set(flattened) != set(self.samples):
            raise RuntimeError("split coverage or disjointness failure")
        if fixed_batches([self.samples[s] for s in self.ids("train")]) != self.protocol["batches"]:
            raise RuntimeError("fixed batch membership changed")

    def ids(self, split):
        return self.protocol["splits"][split]

    @property
    def snapshot(self):
        return Path(self.protocol["snapshot"])
