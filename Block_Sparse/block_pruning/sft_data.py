"""S1K reasoning chat dataset + SFT collator (Block_Sparse self-contained)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "simplescaling/s1K-1.1_tokenized"

__all__ = [
    "DEFAULT_DATASET_NAME",
    "ReasoningChatDataset",
    "ChatSFTCollator",
    "load_reasoning_dataset",
    "row_to_messages",
    "build_reasoning_train_dataset",
]


def _as_text(value: Any, *, field: str) -> str:
    if value is None:
        raise ValueError(f"Missing required field: {field}")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"Empty string for field: {field}")
        return text
    if isinstance(value, (list, tuple)):
        for item in value:
            if item is None:
                continue
            if isinstance(item, str) and item.strip():
                return item.strip()
        raise ValueError(f"No non-empty string in sequence field: {field}")
    raise TypeError(f"Unsupported type for field {field}: {type(value)}")


def row_to_messages(row: dict[str, Any], *, trace_source: str = "deepseek") -> list[dict[str, str]]:
    """Convert an S1K-1.1 row into Qwen chat messages."""
    source = str(trace_source).strip().lower()
    if source != "deepseek":
        raise ValueError(f"Only trace_source='deepseek' is implemented, got {trace_source!r}")

    question = _as_text(row.get("question"), field="question")
    thinking = _as_text(
        row.get("deepseek_thinking_trajectory"),
        field="deepseek_thinking_trajectory",
    )
    attempt = _as_text(row.get("deepseek_attempt"), field="deepseek_attempt")
    assistant = f"<think>\n{thinking}\n</think>\n{attempt}"
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant},
    ]


def load_reasoning_dataset(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_path: str | None = None,
    split: str = "train",
):
    """Load HF datasets; local path takes precedence over hub name."""
    from datasets import load_dataset

    if dataset_path:
        path = Path(str(dataset_path).strip())
        if not path.exists():
            raise FileNotFoundError(f"dataset_path does not exist: {path}")
        logger.info("Loading reasoning dataset from local path: %s", path)
        if path.is_dir() and (path / "dataset_info.json").exists():
            from datasets import load_from_disk

            ds = load_from_disk(str(path))
            if split and hasattr(ds, "keys") and split in ds:
                ds = ds[split]
        elif path.suffix in {".parquet", ".json", ".jsonl"}:
            ds = load_dataset(
                "parquet" if path.suffix == ".parquet" else "json",
                data_files=str(path),
                split=split,
            )
        else:
            ds = load_dataset(str(path), split=split)
    else:
        name = str(dataset_name).strip()
        if not name:
            raise ValueError("dataset_name must be non-empty when dataset_path is not set")
        logger.info("Loading reasoning dataset from cache/hub: %s split=%s", name, split)
        ds = load_dataset(name, split=split)

    if len(ds) == 0:
        raise RuntimeError("Reasoning dataset is empty")
    return ds


class ReasoningChatDataset(Dataset):
    """HF Dataset → chat messages wrapper."""

    def __init__(self, hf_dataset, *, trace_source: str = "deepseek") -> None:
        self._ds = hf_dataset
        self.trace_source = str(trace_source)

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._ds[int(index)]
        if not isinstance(row, dict):
            row = dict(row)
        messages = row_to_messages(row, trace_source=self.trace_source)
        return {"messages": messages}


@dataclass
class ChatSFTCollator:
    """chat messages → input_ids / labels / attention_mask; prompt labels=-100."""

    tokenizer: Any
    max_length: int = 8192
    allow_truncate: bool = False

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Empty batch")

        input_ids_list: list[list[int]] = []
        labels_list: list[list[int]] = []
        lengths: list[int] = []

        for feat in features:
            messages = feat.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"Expected messages with >=2 turns, got {messages!r}")

            full_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=False,
            )
            if isinstance(full_ids, dict) or hasattr(full_ids, "input_ids"):
                full_ids = list(full_ids["input_ids"])
            elif not isinstance(full_ids, list):
                full_ids = list(full_ids)
            if full_ids and not isinstance(full_ids[0], int):
                raise TypeError(
                    f"apply_chat_template must return List[int], got {type(full_ids[0])}"
                )

            prompt_ids = self.tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            if isinstance(prompt_ids, dict) or hasattr(prompt_ids, "input_ids"):
                prompt_ids = list(prompt_ids["input_ids"])
            elif not isinstance(prompt_ids, list):
                prompt_ids = list(prompt_ids)

            if len(full_ids) > int(self.max_length):
                if not self.allow_truncate:
                    raise ValueError(
                        f"Sequence length {len(full_ids)} exceeds max_length={self.max_length}; "
                        "set allow_truncate=true to truncate"
                    )
                full_ids = full_ids[: int(self.max_length)]
            prompt_len = min(len(prompt_ids), len(full_ids))

            labels = list(full_ids)
            for i in range(prompt_len):
                labels[i] = -100

            input_ids_list.append(full_ids)
            labels_list.append(labels)
            lengths.append(len(full_ids))

        max_len = max(lengths)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id is None")

        batch_input_ids = torch.full((len(features), max_len), int(pad_id), dtype=torch.long)
        batch_labels = torch.full((len(features), max_len), -100, dtype=torch.long)
        batch_attention = torch.zeros((len(features), max_len), dtype=torch.long)

        for i, (ids, labs) in enumerate(zip(input_ids_list, labels_list)):
            n = len(ids)
            batch_input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            batch_labels[i, :n] = torch.tensor(labs, dtype=torch.long)
            batch_attention[i, :n] = 1

        return {
            "input_ids": batch_input_ids,
            "labels": batch_labels,
            "attention_mask": batch_attention,
        }


def build_reasoning_train_dataset(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_path: Optional[str] = None,
    split: str = "train",
    trace_source: str = "deepseek",
) -> ReasoningChatDataset:
    hf_ds = load_reasoning_dataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        split=split,
    )
    dataset = ReasoningChatDataset(hf_ds, trace_source=trace_source)
    sample = dataset[0]
    logger.info(
        "Reasoning dataset ready: size=%d sample_roles=%s",
        len(dataset),
        [m["role"] for m in sample["messages"]],
    )
    return dataset
