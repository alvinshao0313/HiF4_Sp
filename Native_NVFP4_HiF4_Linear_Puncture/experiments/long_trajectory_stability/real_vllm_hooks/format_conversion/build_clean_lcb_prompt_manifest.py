#!/usr/bin/env python3
"""Build clean_lcb_prompt_manifest.jsonl from audited E0 raw cache exact input_ids."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyarrow.parquet as pq

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import write_jsonl


def prompt_key(ids: list[int]) -> str:
    digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()[:12]
    return f"n{len(ids)}_c{int(digest[:8], 16)}"


def _one(value):
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("expected exactly one generation of input/output tokens")
        return value[0]
    return value


def _token_ids(value):
    """Accept flat token IDs or a single nested generation list."""
    if not isinstance(value, list) or not value:
        raise ValueError("empty token id list")
    if isinstance(value[0], list):
        return [int(x) for x in _one(value)]
    return [int(x) for x in value]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_e0_details(phasea_root: Path) -> list[dict]:
    root = Path(phasea_root) / "E0_native_nvfp4/eval/livecodebench"
    paths = sorted(root.glob("**/details_lcb:codegeneration_v6|0_*.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected exactly one E0 LCB details file, got {paths}")
    return json.loads(paths[0].read_text()), paths[0]


def build_manifest(phasea_root: Path, audit_json: Path, out_path: Path) -> dict:
    audit = json.loads(audit_json.read_text())
    if audit.get("status") != "FORMAL_PROMPT_PROTOCOL_MISMATCH":
        raise RuntimeError(f"unexpected audit status: {audit.get('status')}")
    e0_matches = audit["raw_cache_sources"]["E0"]
    if len(e0_matches) != 1:
        raise RuntimeError(f"expected exactly one E0 raw cache lineage, got {e0_matches}")
    cache_path = Path(e0_matches[0]["path"])
    if not cache_path.exists():
        raise RuntimeError(f"E0 raw cache missing: {cache_path}")
    cache_sha = sha256_file(cache_path)
    if cache_sha != e0_matches[0]["sha256"]:
        raise RuntimeError(
            f"E0 raw cache sha changed: got={cache_sha} expected={e0_matches[0]['sha256']}"
        )

    e0_details, e0_details_path = load_e0_details(phasea_root)
    e0_index = {str(row["doc"]["id"]): row for row in e0_details}
    if len(e0_index) != len(e0_details):
        raise RuntimeError("duplicate E0 formal doc IDs")

    table = pq.read_table(cache_path).to_pylist()
    cached = {
        str(row["sample_id"]): (
            json.loads(row["sample"]) if isinstance(row["sample"], str) else row["sample"]
        )
        for row in table
    }
    if set(cached) != set(e0_index):
        raise RuntimeError("raw cache doc IDs do not match formal E0 details")

    audit_by_doc = {str(r["doc_id"]): r for r in audit["rows"]}
    rows = []
    for doc_id in sorted(e0_index, key=int):
        item = e0_index[doc_id]
        raw = cached[doc_id]
        ids = _token_ids(raw["input_tokens"])
        if not ids:
            raise RuntimeError(f"empty raw input_ids for doc {doc_id}")
        if raw["input"] != item["model_response"]["input"]:
            raise RuntimeError(f"raw input text mismatch for doc {doc_id}")
        text_hash = hashlib.sha256(raw["input"].encode()).hexdigest()
        id_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        audit_row = audit_by_doc[str(doc_id)]
        expected = audit_row["variants"]["E0"]["raw_input_ids_sha256"]
        if id_hash != expected:
            raise RuntimeError(
                f"raw input_ids sha mismatch doc={doc_id}: got={id_hash} expected={expected}"
            )
        if ids[:8] != audit_row["variants"]["E0"]["raw_input_ids_head"]:
            raise RuntimeError(f"raw input_ids head mismatch doc={doc_id}")
        if ids[-12:] != audit_row["variants"]["E0"]["raw_input_ids_tail"]:
            raise RuntimeError(f"raw input_ids tail mismatch doc={doc_id}")
        if not raw["input"].startswith("<|im_start|>user\n"):
            raise RuntimeError(f"E0 prompt missing chat wrapper doc={doc_id}")
        rows.append(
            {
                "doc_id": str(doc_id),
                "prompt_key": prompt_key(ids),
                "input_ids": ids,
                "prompt_text_sha256": text_hash,
                "raw_input_ids_sha256": id_hash,
                "prompt_source": "E0_formal_raw_cache_exact_input_ids",
                "raw_cache_path": str(cache_path.resolve()),
                "gold": item.get("gold"),
                "specific": {**item["doc"]["specific"], "task": "livecodebench"},
            }
        )

    if len({row["prompt_key"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate prompt_key in clean manifest")
    if len(rows) != 175:
        raise RuntimeError(f"expected 175 prompts, got {len(rows)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, rows)
    meta = {
        "schema_version": 1,
        "n_prompts": len(rows),
        "source": "E0_formal_raw_cache_exact_input_ids",
        "raw_cache_path": str(cache_path.resolve()),
        "raw_cache_sha256": cache_sha,
        "e0_details_path": str(e0_details_path.resolve()),
        "policy": "E0_and_E1_must_consume_identical_input_ids_without_retokenize",
        "output": str(out_path.resolve()),
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phasea_root", type=Path, required=True)
    parser.add_argument("--audit_json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = build_manifest(args.phasea_root, args.audit_json, args.output)
    print(json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
