"""Fixed 4-family × 8 prompt bank (cal = first 4, val = last 4 per family)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase


FAMILIES = (
    "natural_language",
    "knowledge_reasoning",
    "math_reasoning",
    "instruction_code",
)


@dataclass(frozen=True)
class PromptItem:
    prompt_id: str
    family: str
    text: str
    split: str  # cal | val
    index_in_family: int


def _natural_language(i: int) -> str:
    topics = [
        "morning market in a mountain town",
        "a rainy evening on the subway",
        "grandparents tending a small garden",
        "students preparing for a school festival",
        "a bookstore owner arranging new arrivals",
        "fishermen returning to harbor at dusk",
        "a teacher reflecting after class",
        "neighbors sharing food after a power outage",
    ]
    return (
        f"Write a short, concrete paragraph about {topics[i]}. "
        "Use ordinary vocabulary and keep it under 120 words."
    )


def _knowledge_reasoning(i: int) -> str:
    qs = [
        (
            "Which force keeps Earth in orbit around the Sun?\n"
            "A) Magnetism\nB) Gravity\nC) Friction\nD) Buoyancy\nExplain briefly and give the letter."
        ),
        (
            "Photosynthesis primarily occurs in which plant organelle?\n"
            "A) Mitochondrion\nB) Chloroplast\nC) Nucleus\nD) Ribosome\nExplain briefly and give the letter."
        ),
        (
            "If water freezes at 0°C at standard pressure, what happens to its density compared with liquid water near 4°C?\n"
            "Answer with reasoning."
        ),
        (
            "A ball is thrown upward. After it leaves the hand, which force acts continuously (ignore air)?\n"
            "A) Hand force\nB) Gravity\nC) No forces\nD) Magnetic force\nExplain and answer."
        ),
        (
            "Why does a metal spoon feel colder than a wooden spoon at the same room temperature?\n"
            "Give a concise physics explanation."
        ),
        (
            "In a closed ecosystem jar with plants and snails, what role do plants play for oxygen?\n"
            "Explain in a few sentences."
        ),
        (
            "Which statement about atoms is correct?\n"
            "A) Electrons are in the nucleus\nB) Protons have negative charge\n"
            "C) Neutrons are uncharged\nD) Nucleus contains only electrons\nExplain and answer."
        ),
        (
            "Why do coastal areas often have milder temperatures than inland deserts at similar latitudes?\n"
            "Explain using heat capacity of water."
        ),
    ]
    return f"Knowledge question {i}:\n{qs[i]}"


def _math_reasoning(i: int) -> str:
    problems = [
        "Solve for x: 3x + 5 = 20. Show steps and box the final answer.",
        "A train travels 120 km in 1.5 hours. What is its average speed in km/h? Show steps.",
        "Compute (17 + 8) × 3 - 11. Show steps.",
        "If 2/5 of a number is 18, what is the number? Show steps.",
        "A rectangle has length 12 and width 7. What is its area and perimeter? Show steps.",
        "Solve: 4(y - 3) = 2y + 10. Show steps.",
        "There are 3 red and 5 blue balls. Probability of drawing one blue at random? Show steps.",
        "Compound interest: principal 1000, rate 5% per year, 2 years compounded annually. Final amount? Show steps.",
    ]
    return f"Math problem {i}:\n{problems[i]}"


def _instruction_code(i: int) -> str:
    tasks = [
        "Write a Python function `is_palindrome(s: str) -> bool` that ignores spaces and case.",
        "Write a Python function that returns the top-k frequent words from a list of strings.",
        "Write a bash one-liner to count lines containing ERROR in app.log.",
        "Write a Python function to merge two sorted integer lists into one sorted list.",
        "Write a SQL query to select users with more than 3 orders from tables users(id) and orders(user_id).",
        "Write a Python generator that yields Fibonacci numbers up to n inclusive.",
        "Write a short Python snippet using pathlib to list all `.yaml` files recursively under `./configs`.",
        "Write a Python function `clamp(x, lo, hi)` and a unit-test style assert example.",
    ]
    return (
        f"Instruction {i}:\n{tasks[i]}\n"
        "Return only the code (or query) with a one-line comment describing complexity if relevant."
    )


_BUILDERS = {
    "natural_language": _natural_language,
    "knowledge_reasoning": _knowledge_reasoning,
    "math_reasoning": _math_reasoning,
    "instruction_code": _instruction_code,
}


def build_prompt_bank() -> list[PromptItem]:
    items: list[PromptItem] = []
    for family in FAMILIES:
        builder = _BUILDERS[family]
        for i in range(8):
            split = "cal" if i < 4 else "val"
            items.append(
                PromptItem(
                    prompt_id=f"{family}_{i:02d}",
                    family=family,
                    text=builder(i),
                    split=split,
                    index_in_family=i,
                )
            )
    return items


def prompts_for_split(split: str) -> list[PromptItem]:
    if split not in {"cal", "val"}:
        raise ValueError(f"split must be cal|val, got {split}")
    return [p for p in build_prompt_bank() if p.split == split]


def tokenize_prompt(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    *,
    max_seq_len: int = 256,
) -> dict[str, torch.Tensor]:
    """Tokenize one prompt for prefill capture."""
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": text}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        rendered = text

    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
        padding=False,
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded.get("attention_mask"),
    }
