from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def as_int_list(value) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise TypeError(f"expected list-like ids, got {type(value)}")
    return [int(x) for x in value]


def first_generation_ids(value) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise TypeError(f"expected generation id list, got {type(value)}")
    if value and isinstance(value[0], list):
        return as_int_list(value[0])
    return as_int_list(value)


def prompt_key(input_ids: list[int]) -> str:
    checksum = 0
    for i, value in enumerate(input_ids):
        checksum = (checksum + (i + 1) * (int(value) + 1)) % 1000000007
    return f"n{len(input_ids)}_c{checksum}"


@dataclass(frozen=True)
class Trajectory:
    prompt_key: str
    doc_id: str
    input_ids: list[int]
    output_ids: list[int]
    raw_text: str | None
    metric: object
    gold: object
    specific: object

    def to_json(self) -> dict:
        return {
            "prompt_key": self.prompt_key,
            "doc_id": self.doc_id,
            "input_ids": self.input_ids,
            "output_ids": self.output_ids,
            "output_len": len(self.output_ids),
            "raw_text": self.raw_text,
            "metric": self.metric,
            "gold": self.gold,
            "specific": self.specific,
        }


def find_latest_detail_json(capture_dir: Path) -> Path:
    matches = sorted(capture_dir.rglob("details_mmlu_pro|0_*.json"))
    if not matches:
        raise FileNotFoundError(f"no MMLU-Pro detail JSON under {capture_dir}")
    return matches[-1]


def load_detail_trajectories(detail_path: Path) -> list[Trajectory]:
    records = json.loads(detail_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError(f"expected list in {detail_path}")
    out: list[Trajectory] = []
    for index, row in enumerate(records):
        response = row.get("model_response") or {}
        input_ids = as_int_list(response.get("input_tokens"))
        output_ids = first_generation_ids(response.get("output_tokens"))
        if not input_ids or not output_ids:
            raise ValueError(f"detail row {index} is missing exact ids; use capture_main.py")
        raw_text = response.get("text")
        if isinstance(raw_text, list):
            raw_text = raw_text[0] if raw_text else None
        doc = row.get("doc") or {}
        out.append(
            Trajectory(
                prompt_key=prompt_key(input_ids),
                doc_id=str(doc.get("id", index)),
                input_ids=input_ids,
                output_ids=output_ids,
                raw_text=raw_text,
                metric=row.get("metric"),
                gold=row.get("gold"),
                specific=doc.get("specific"),
            )
        )
    keys = [x.prompt_key for x in out]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate prompt keys; variant alignment would be ambiguous")
    return out


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_capture(capture_dir: Path, output_path: Path) -> Path:
    detail = find_latest_detail_json(capture_dir)
    trajectories = load_detail_trajectories(detail)
    write_jsonl(output_path, (x.to_json() for x in trajectories))
    lengths = [len(x.output_ids) for x in trajectories]
    meta = {
        "detail_source": str(detail.resolve()),
        "num_trajectories": len(trajectories),
        "min_output_len": min(lengths),
        "max_output_len": max(lengths),
    }
    output_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_path
