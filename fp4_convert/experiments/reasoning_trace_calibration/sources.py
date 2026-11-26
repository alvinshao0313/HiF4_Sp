"""Pinned question-only source projection; references remain in judging storage."""
from __future__ import annotations

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from .artifacts import read_json, sha256, write_json
from .config import COHORT, MMLU_DATA, SOURCES, S1K


def source_identity(domain, row):
    if domain == "math":
        return []
    source = row.get("source")
    if source == 2 and row.get("cf_contest_id", 0) > 0 and row.get("cf_index"):
        return [f"codeforces:{row['cf_contest_id']}:{row['cf_index'].lower()}"]
    return [f"contest:{source}:{row['name'].strip().lower()}"]


def read_projection(domain, filename, columns):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    source = SOURCES[domain]
    fs = HfFileSystem(endpoint="https://huggingface.co")
    path = f"datasets/{source['repo']}@{source['revision']}/{filename}"
    with fs.open(path, "rb", block_size=256 * 1024) as stream:
        table = pq.read_table(stream, columns=columns)
    return table.to_pylist()


def fetch_sources(root):
    from huggingface_hub import HfApi
    destination = Path(root) / "data_sources"
    if (destination / "manifest.json").exists():
        evidence = read_json(destination / "manifest.json")
        for relative, entry in evidence.items():
            if (entry["revision"] != SOURCES[relative.split("/")[1]]["revision"]
                    or sha256(Path(root) / relative) != entry["sha256"]):
                raise ValueError(f"pinned source projection changed: {relative}")
        if any("source_file_sha256" not in entry for entry in evidence.values()):
            for domain, source in SOURCES.items():
                info = HfApi(endpoint="https://huggingface.co").dataset_info(source["repo"], revision=source["revision"], files_metadata=True)
                if info.sha != source["revision"]:
                    raise ValueError("source metadata revision changed")
                remote = {item.rfilename: item for item in info.siblings}
                for relative, entry in evidence.items():
                    if relative.split("/")[1] == domain:
                        file = remote[entry["file"]]
                        if file.lfs is None:
                            raise ValueError("source parquet has no content SHA256")
                        entry.update(source_file_sha256=file.lfs.sha256, source_file_bytes=file.size)
            write_json(evidence, destination / "manifest.json")
        return read_json(destination / "questions.json")
    destination.mkdir(exist_ok=False)
    evidence, records = {}, []
    for domain, source in SOURCES.items():
        api = HfApi(endpoint="https://huggingface.co")
        info = api.dataset_info(source["repo"], revision=source["revision"], files_metadata=True)
        if info.sha != source["revision"]:
            raise ValueError(f"dataset revision mismatch: {domain}")
        filenames = sorted(item.rfilename for item in info.siblings if item.rfilename.startswith("data/") and item.rfilename.endswith(".parquet"))
        if not filenames:
            raise ValueError(f"no parquet sources: {domain}")
        remote = {item.rfilename: item for item in info.siblings}
        columns = ["problem", "solution", "level", "type"] if domain == "math" else [
            "name", "description", "source", "cf_contest_id", "cf_index"]
        def fetch(filename):
            return filename, read_projection(domain, filename, columns)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for filename, rows in pool.map(fetch, filenames):
                split = Path(filename).name.split("-")[0]
                if split not in {"train", "test", "valid", "validation"}:
                    raise ValueError(f"unrecognized split: {filename}")
                local = destination / domain / (Path(filename).name + ".json")
                write_json(rows, local)
                evidence[str(local.relative_to(root))] = {"sha256": sha256(local), "count": len(rows),
                                                         "repo": source['repo'], "revision": source['revision'], "file": filename,
                                                         "source_file_sha256": remote[filename].lfs.sha256,
                                                         "source_file_bytes": remote[filename].size}
                for index, row in enumerate(rows):
                    question = row["problem" if domain == "math" else "description"]
                    if not isinstance(question, str) or not question.strip():
                        raise ValueError(f"missing question: {domain}/{filename}/{index}")
                    records.append({"id": f"{domain}:{split}:{Path(filename).stem}:{index}", "domain": domain,
                                    "role": "candidate" if split == "train" else "test", "question": question,
                                    "identities": source_identity(domain, row),
                                    "locator": {"domain": domain, "file": filename, "row": index}})
                print(json.dumps({"event": "source", "domain": domain, "file": filename, "rows": len(rows)}), flush=True)
    write_json(evidence, destination / "manifest.json")
    write_json(records, destination / "questions.json")
    return records


def _arrow(path, expected_count):
    from datasets import Dataset
    dataset = Dataset.from_file(str(path))
    if len(dataset) != expected_count:
        raise ValueError(f"wrong pinned task count: {path}: {len(dataset)}")
    return dataset


def evaluation_questions():
    """Read the actual pinned task content, not a possibly drifting Hub cache."""
    cache = Path("/home/shaoyuantian/.cache/huggingface/datasets")
    paths = [
        ("arc_easy", cache / "allenai___ai2_arc/ARC-Easy/0.0.0/210d026faf9955653af8916fad021475a3f00453/ai2_arc-test.arrow", 2376),
        ("arc_challenge", cache / "allenai___ai2_arc/ARC-Challenge/0.0.0/210d026faf9955653af8916fad021475a3f00453/ai2_arc-test.arrow", 1172),
        ("gsm8k", cache / "openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-test.arrow", 1319),
        ("lcb", cache / "lighteval___code_generation_lite/v6/0.0.0/89e5fc5c2a8e748f50e95bc7235fab2372d49bfa/code_generation_lite-test.arrow", 175),
        ("mmlu_pro", MMLU_DATA / "data/test.arrow", 12032),
    ]
    aime_directory = cache / "yentinglin___aime_2025/default/0.0.0/6f71d77b0b89b9dabe07ab466c51df33f514df7f"
    aime_paths = list(aime_directory.glob("*.arrow"))
    if len(aime_paths) != 1:
        raise ValueError(f"expected one pinned AIME arrow file: {aime_paths}")
    paths.append(("aime25", aime_paths[0], 30))
    mmlu_manifest = read_json(MMLU_DATA / "dataset_manifest.json")
    if sha256(MMLU_DATA / "data/test.arrow") != mmlu_manifest["splits"]["test"]["sha256"]:
        raise ValueError("canonical MMLU content changed")
    records, evidence = [], {}
    for task, path, count in paths:
        dataset = _arrow(path, count)
        evidence[task] = {"path": str(path), "sha256": sha256(path), "count": count}
        for index, row in enumerate(dataset):
            question = row.get("question_content", row.get("question", row.get("problem")))
            identities = []
            if task == "lcb":
                platform, qid = row["platform"].lower(), row["question_id"]
                if platform == "codeforces":
                    identities = [f"codeforces:{row['contest_id']}:{qid.split('_')[-1].lower()}"]
                else:
                    identities = [f"{platform}:{qid.lower()}"]
            records.append({"id": f"evaluation:{task}:{index}", "domain": "code" if task == "lcb" else "math",
                            "role": "test", "question": question, "identities": identities})
    return records, evidence


def s1k_questions(calibration_dir=None):
    from ..non_equivalent_reconstruction.artifact import load
    records, evidence = [], {}
    for split in ("train", "val"):
        directory = S1K / "calibration" if calibration_dir is None else Path(calibration_dir)
        path = directory / f"{split}.pt"
        samples = load(path)
        if len(samples) != {"train": 128, "val": 32}[split]:
            raise ValueError("S1K split count changed")
        evidence[split] = {"path": str(path), "sha256": sha256(path), "ids": [s.sample_id for s in samples]}
        for sample in samples:
            records.append({"id": sample.sample_id, "domain": "math", "role": "s1k", "split": split,
                            "question": sample.meta["question"], "identities": []})
    return records, evidence


def reference(root, row):
    """Only judging callers read solutions/hidden tests; generation never does."""
    locator = row["locator"]
    domain, filename, index = locator["domain"], locator["file"], locator["row"]
    local = Path(root) / "data_sources" / domain / (Path(filename).name + ".json")
    evidence = read_json(Path(root) / "data_sources/manifest.json")[str(local.relative_to(root))]
    if sha256(local) != evidence["sha256"]:
        raise ValueError(f"source projection changed: {local}")
    if domain == "math":
        return {"solution": read_json(local)[index]["solution"]}
    # Fetch only test columns, never the multi-GB reference solution column.
    cached = Path(root) / "judging/source_tests" / (Path(filename).name + ".json")
    prepared = Path(root) / "judging/reference_manifest.json"
    if prepared.exists():
        bound = read_json(prepared)["files"]
        expected = bound[str(cached.relative_to(root))]
        if not cached.exists() or sha256(cached) != expected["sha256"]:
            raise ValueError("prepared code reference changed or is missing")
    if not cached.exists():
        write_json(read_projection("code", filename, ["name", "public_tests", "private_tests", "generated_tests"]), cached)
    rows = read_json(cached)
    selected = rows[index]
    if selected["name"] != read_json(local)[index]["name"]:
        raise ValueError("code judging row does not match question projection")
    return {key: selected[key] for key in ("public_tests", "private_tests", "generated_tests")}


def prepare_references(root):
    """Fetch pinned judging columns before compute, so later scoring is offline."""
    root = Path(root)
    pools = read_json(root / "isolation/pools.json")
    generated = read_json(root / "generation/manifest.json")
    if not generated["complete"]:
        raise ValueError("completed calibration generation is required")
    questions = {r["id"]: r for split in pools["code"].values() for r in split}
    selected = [questions[r["id"]] for r in generated["attempts"] if r["domain"] == "code" and r["accepted"]]
    selected += pools["code"]["dev"]
    representatives = {r["locator"]["file"]: r for r in selected}
    destination = root / "judging/reference_manifest.json"
    if destination.exists():
        raise FileExistsError(destination)
    def fetch(row):
        reference(root, row)
        path = root / "judging/source_tests" / (Path(row["locator"]["file"]).name + ".json")
        print(json.dumps({"event": "reference_cached", "file": path.name}), flush=True)
        return str(path.relative_to(root)), {"sha256": sha256(path), "bytes": path.stat().st_size,
                "source_file": row["locator"]["file"], "revision": SOURCES["code"]["revision"]}
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = dict(pool.map(fetch, representatives.values()))
    write_json({"complete": True, "files": files, "selected_code_questions": len(selected),
                "references_used_for_generation": False}, destination)
