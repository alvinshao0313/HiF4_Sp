"""Whole-sequence selection under shared domain and length-stratum budgets."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import random
import re

from ..reasoning_trace_calibration.isolation import grams, normalize


@dataclass(frozen=True)
class Candidate:
    sample_id: str
    domain: str
    tokens: int
    question: str


def _strata(candidates, fraction):
    result = []
    for domain in ("math", "code"):
        rows = sorted((r for r in candidates if r.domain == domain), key=lambda r: (r.tokens, r.sample_id))
        if len(rows) < 6:
            raise ValueError(f"at least six {domain} sequences are required for three length strata")
        size, remainder = divmod(len(rows), 3)
        groups, start = [], 0
        for index in range(3):
            stop = start + size + (index < remainder)
            groups.append(rows[start:stop])
            start = stop
        exact = [sum(r.tokens for r in group) * fraction for group in groups]
        budgets = [int(value) for value in exact]
        left = int(sum(exact)) - sum(budgets)
        for index in sorted(range(3), key=lambda i: (-(exact[i] - budgets[i]), i))[:left]:
            budgets[index] += 1
        for index, (group, budget) in enumerate(zip(groups, budgets)):
            if min(r.tokens for r in group) > budget:
                raise ValueError(f"budget cannot retain a complete sequence in {domain}/length_{index}")
            result.append((f"{domain}/length_{index}", group, budget))
    return result


def _similarities(rows):
    text = [normalize(row.question) for row in rows]
    words = [frozenset(re.findall(r"\w+", value)) for value in text]
    characters = [grams(value) for value in text]

    def jaccard(a, b):
        return len(a & b) / len(a | b) if a or b else 0.

    return [[.5 * jaccard(words[i], words[j]) + .5 * jaccard(characters[i], characters[j])
             for j in range(len(rows))] for i in range(len(rows))]


def _choose(rows, budget, method, seed, stratum):
    remaining = list(range(len(rows)))
    selected, used = [], 0
    if method == "random50":
        derived = int.from_bytes(hashlib.sha256(f"{seed}:{stratum}".encode()).digest()[:8], "big")
        random.Random(derived).shuffle(remaining)
        for index in remaining:
            if used + rows[index].tokens <= budget:
                selected.append(index)
                used += rows[index].tokens
    else:
        similarities = _similarities(rows)
        covered = [0.] * len(rows)
        while True:
            feasible = [j for j in remaining if used + rows[j].tokens <= budget]
            if not feasible:
                break
            # Facility-location coverage has diminishing returns for near duplicates.
            # The cost is the complete sequence, not just tokens used in the loss.
            gains = {j: sum(max(covered[i], similarities[i][j]) - covered[i] for i in range(len(rows)))
                     / rows[j].tokens for j in feasible}
            chosen = min(feasible, key=lambda j: (-gains[j], rows[j].sample_id))
            selected.append(chosen)
            used += rows[chosen].tokens
            covered = [max(covered[i], similarities[i][chosen]) for i in range(len(rows))]
            remaining.remove(chosen)
    if not selected:
        raise ValueError(f"selection left an empty length stratum: {stratum}")
    return [rows[i].sample_id for i in selected], used


def select(candidates, method, *, fraction=.5, seed=42):
    """Return source-ordered IDs and an explicit actual-cost report.

    Budgets are never borrowed between length strata. Random and coverage use
    identical caps; indivisible sequences can leave different unused balances.
    Coverage uses training question text only, not answers or held-out tasks.
    """
    candidates = list(candidates)
    if method not in {"full", "random50", "coverage50"}:
        raise ValueError(f"unknown selection: {method}")
    fraction = Fraction(str(fraction))
    if not 0 < fraction <= 1:
        raise ValueError("token fraction must be in (0, 1]")
    ids = [r.sample_id for r in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate IDs")
    for row in candidates:
        if row.domain not in {"math", "code"} or type(row.tokens) is not int or row.tokens <= 0:
            raise ValueError(f"invalid candidate domain or token count: {row.sample_id}")
        normalize(row.question)
    effective_fraction = Fraction(1) if method == "full" else fraction
    strata = _strata(candidates, effective_fraction)
    chosen, reports = [], []
    for name, rows, budget in strata:
        if method == "full":
            selected, used = [r.sample_id for r in rows], sum(r.tokens for r in rows)
        else:
            selected, used = _choose(rows, budget, method, seed, name)
        chosen.extend(selected)
        reports.append({"name": name, "source_ids": [r.sample_id for r in rows],
                        "source_tokens": sum(r.tokens for r in rows), "budget_tokens": budget,
                        "selected_ids_by_rank": selected, "selected_tokens": used,
                        "unused_tokens": budget - used, "length_min": min(r.tokens for r in rows),
                        "length_max": max(r.tokens for r in rows)})
    chosen_set = set(chosen)
    selected_ids = [row.sample_id for row in candidates if row.sample_id in chosen_set]
    report = {"method": method, "seed": seed, "fraction": float(effective_fraction),
              "source_count": len(candidates), "source_tokens": sum(r.tokens for r in candidates),
              "selected_count": len(selected_ids), "selected_ids": selected_ids,
              "budget_tokens": sum(r["budget_tokens"] for r in reports),
              "selected_tokens": sum(r["selected_tokens"] for r in reports),
              "unused_tokens": sum(r["unused_tokens"] for r in reports), "strata": reports,
              "coverage_kind": "question_text_facility_location" if method == "coverage50" else None,
              "no_token_truncation": True, "no_cross_stratum_budget_transfer": True}
    return selected_ids, report
