"""Atomic run_state / manifest helpers."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import RUN_SUBDIRS, RUNS_ROOT, STAGE_ORDER, STATUS_RUNNING


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def run_root(run_id: str) -> Path:
    return RUNS_ROOT / str(run_id)


def ensure_run_dirs(run_id: str) -> Path:
    root = run_root(run_id)
    for name in RUN_SUBDIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def default_run_state(run_id: str, *, through_stage: str, pid: int | None = None) -> dict[str, Any]:
    if through_stage not in STAGE_ORDER:
        raise ValueError(f"unknown through_stage={through_stage}")
    return {
        "run_id": str(run_id),
        "status": STATUS_RUNNING,
        "current_stage": None,
        "completed_stages": [],
        "pid": pid,
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "exit_code": None,
        "failure_or_gate_reason": None,
        "next_allowed_stage": STAGE_ORDER[0],
        "through_stage": through_stage,
        "waiting_review_gate": None,
    }


def load_run_state(run_id: str) -> dict[str, Any]:
    path = run_root(run_id) / "run_state.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return read_json(path)


def save_run_state(run_id: str, state: dict[str, Any]) -> Path:
    state = dict(state)
    state["updated_at"] = utc_now()
    path = run_root(run_id) / "run_state.json"
    atomic_write_json(path, state)
    return path


def stage_index(stage: str) -> int:
    if stage not in STAGE_ORDER:
        raise ValueError(f"unknown stage={stage}")
    return STAGE_ORDER.index(stage)


def stages_inclusive(from_stage: str, through_stage: str) -> list[str]:
    lo = stage_index(from_stage)
    hi = stage_index(through_stage)
    if lo > hi:
        raise ValueError(f"from_stage {from_stage} after through_stage {through_stage}")
    return list(STAGE_ORDER[lo : hi + 1])


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
