"""Explicit local C4 sampling matching the pinned upstream training loader.

The text stream and window boundaries follow ``calib_utils/data_utils.py``
at SOURCE_COMMIT. Local Arrow files are required so dataset cache fallback
cannot silently choose a different input corpus.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Iterable

import torch

from .algorithm import SOURCE_COMMIT


def _sample_windows(
    input_ids: torch.Tensor, nsamples: int, seqlen: int, seed: int
) -> tuple[torch.Tensor, list[int]]:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("C4 token stream must have shape [1, tokens]")
    if input_ids.shape[1] < seqlen:
        raise ValueError(
            f"Tokenized sample is shorter than seqlen: {input_ids.shape[1]} < {seqlen}"
        )
    generator = random.Random(seed)
    starts = [generator.randint(0, input_ids.shape[1] - seqlen) for _ in range(nsamples)]
    samples = torch.cat([input_ids[:, start : start + seqlen] for start in starts], dim=0)
    return samples.to(dtype=torch.long).contiguous(), starts


def sample_c4(
    model_path: str | Path,
    arrow_paths: Iterable[str | Path],
    nsamples: int,
    seqlen: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Sample C4 train windows without downloading or discovering datasets.

    Arrow shards are sorted lexically, matching upstream's explicit cached
    Arrow loader. Tokenization checks each 256 nonempty documents, preserving
    the upstream prefix and its inclusive random-window upper bound.
    """
    from datasets import Dataset, concatenate_datasets
    from transformers import AutoTokenizer

    for name, value in (("nsamples", nsamples), ("seqlen", seqlen)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"C4 {name} must be an integer >= 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("C4 seed must be an integer")
    model_dir = Path(model_path).expanduser().resolve(strict=True)
    if not model_dir.is_dir():
        raise NotADirectoryError(f"Local tokenizer model directory required: {model_dir}")
    paths = sorted(Path(path).expanduser() for path in arrow_paths)
    if not paths:
        raise ValueError("At least one explicit C4 train Arrow file is required")
    sources = []
    for path in paths:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"C4 Arrow source is not a file: {resolved}")
        stat = resolved.stat()
        sources.append(
            {"path": str(resolved), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), use_fast=False, local_files_only=True
    )
    if not callable(tokenizer):
        raise TypeError(f"Local AutoTokenizer is not callable: {type(tokenizer).__name__}")
    pieces = [Dataset.from_file(source["path"]) for source in sources]
    dataset = pieces[0] if len(pieces) == 1 else concatenate_datasets(pieces)
    if "text" not in dataset.column_names:
        raise ValueError("C4 Arrow dataset must contain a text column")
    minimum = nsamples * seqlen + 1
    chunks: list[str] = []
    scanned = 0
    stream = None
    for row in dataset.select_columns(["text"]):
        scanned += 1
        text = row["text"]
        if text is None or text == "":
            continue
        if not isinstance(text, str):
            raise TypeError(f"C4 text at row {scanned - 1} must be a string or null")
        chunks.append(text)
        if len(chunks) % 256 == 0:
            stream = tokenizer(" ".join(chunks), return_tensors="pt").input_ids
            if stream.shape[1] >= minimum:
                break
    else:
        stream = tokenizer(" ".join(chunks), return_tensors="pt").input_ids
    if stream is None or stream.shape[1] < minimum:
        found = 0 if stream is None else stream.shape[1]
        raise ValueError(f"Only found {found} C4 tokens, need at least {minimum}.")
    samples, starts = _sample_windows(stream, nsamples, seqlen, seed)
    provenance = {
        "source_commit": SOURCE_COMMIT,
        "source_path": "calib_utils/data_utils.py",
        "dataset": "allenai/c4",
        "split": "train",
        "sources": sources,
        "model_path": str(model_dir),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_use_fast_requested": False,
        "documents_available": len(dataset),
        "documents_scanned": scanned,
        "documents_used": len(chunks),
        "stream_tokens": int(stream.shape[1]),
        "minimum_stream_tokens": minimum,
        "document_join_separator": " ",
        "tokenize_check_documents": 256,
        "nsamples": nsamples,
        "seqlen": seqlen,
        "seed": seed,
        "starts": starts,
        "window_start_max_inclusive": int(stream.shape[1]) - seqlen,
        "input_ids_dtype": "int64",
        "input_ids_byte_order": "little",
        "input_ids_sha256": hashlib.sha256(
            samples.cpu().numpy().astype("<i8", copy=False).tobytes()
        ).hexdigest(),
    }
    return samples, provenance
