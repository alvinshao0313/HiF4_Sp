"""Atomic IO helpers for JSON/CSV/gzip JSONL."""

from __future__ import annotations

import csv
import gzip
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_json(path: Path | str, data: Any, *, indent: int = 2) -> None:
    """Write JSON atomically (temp file + fsync + os.replace)."""
    path = Path(path)
    ensure_dir(path.parent)
    payload = json.dumps(data, indent=indent, ensure_ascii=False, allow_nan=False)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def read_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path | str, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    rows_list = list(rows)
    if fieldnames is None:
        if not rows_list:
            fieldnames = []
        else:
            # Union of keys so later rows with extra fields are not silently dropped.
            seen: list[str] = []
            for row in rows_list:
                for k in row.keys():
                    if k not in seen:
                        seen.append(k)
            fieldnames = seen
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows_list:
                writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


class GzipJsonlWriter:
    """Streaming gzip JSONL writer with periodic flush."""

    def __init__(self, path: Path | str, flush_every: int = 256) -> None:
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self.flush_every = flush_every
        self._count = 0
        self._fh = gzip.open(self.path, "wt", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        self._fh.write(line)
        self._fh.write("\n")
        self._count += 1
        if self._count % self.flush_every == 0:
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "GzipJsonlWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def iter_gzip_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_text(path: Path | str, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
