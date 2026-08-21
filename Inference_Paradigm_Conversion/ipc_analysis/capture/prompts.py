"""Fixed prompt families for root-cause activation capture."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

ANALYSIS_SEED = 20260809


@dataclass(frozen=True)
class PromptItem:
    sample_id: str
    family: str
    text: str
    split: str  # discovery | validation


def _sid(family: str, idx: int) -> str:
    return f"{family}_{idx:03d}"


def build_prompt_bank(samples_per_family: int = 32) -> list[PromptItem]:
    """Deterministic prompt bank; first half discovery, second half validation."""
    assert samples_per_family % 2 == 0
    half = samples_per_family // 2
    families = {
        "natural": [
            f"Write a short paragraph about daily life in a coastal city, version {i}."
            for i in range(samples_per_family)
        ],
        "arc_mmlu": [
            (
                f"Question {i}: A ball is thrown upward. Which force acts on it after leaving the hand?\n"
                "A) Gravity only\nB) Hand force only\nC) No forces\nD) Magnetic force\n"
                "Answer:"
            )
            for i in range(samples_per_family)
        ],
        "mmlu_pro": [
            (
                f"Reasoning problem {i}: If a store sells apples at $2 each and oranges at $3 each, "
                "and a customer buys 4 apples and 5 oranges with a $2 discount, what is the total? "
                "Explain step by step."
            )
            for i in range(samples_per_family)
        ],
        "math": [
            (
                f"Math {i}: Solve for x: 3x + {i+1} = {2*i+10}. Show your reasoning and put the "
                "final answer after the step-by-step solution."
            )
            for i in range(samples_per_family)
        ],
    }
    items: list[PromptItem] = []
    for family, texts in families.items():
        for i, text in enumerate(texts):
            split = "discovery" if i < half else "validation"
            items.append(
                PromptItem(
                    sample_id=_sid(family, i),
                    family=family,
                    text=text,
                    split=split,
                )
            )
    # Stable order by hash of sample_id with seed salt
    items.sort(
        key=lambda p: hashlib.sha1(f"{ANALYSIS_SEED}:{p.sample_id}".encode()).hexdigest()
    )
    return items


def discovery_items(samples_per_family: int = 32) -> list[PromptItem]:
    return [p for p in build_prompt_bank(samples_per_family) if p.split == "discovery"]


def validation_items(samples_per_family: int = 32) -> list[PromptItem]:
    return [p for p in build_prompt_bank(samples_per_family) if p.split == "validation"]
