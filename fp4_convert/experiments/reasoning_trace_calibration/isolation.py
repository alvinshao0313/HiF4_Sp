"""Exact and near-duplicate question clusters before any teacher generation."""
from __future__ import annotations

from collections import Counter, defaultdict
import html
import re
import unicodedata


def normalize(text, *, mask_numbers=False):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("question must be a nonempty string")
    text = html.unescape(unicodedata.normalize("NFKC", text)).lower()
    text = re.sub(r"\\(?:left|right|displaystyle|textstyle)\b", "", text)
    text = re.sub(r"\\[,;!]", " ", text).replace("$", "")
    if mask_numbers:
        text = re.sub(r"\d+(?:[.,]\d+)*", "#", text)
    return re.sub(r"\s+", " ", text).strip()


def grams(text):
    return frozenset(text[i:i + 5] for i in range(max(1, len(text) - 4)))


class Clusters:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, index):
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        self.parent[max(a, b)] = min(a, b)


def near_edges(texts):
    """Exact Jaccard>=4/5 set join using the lossless prefix-filter property.

    Global rare-first ordering keeps postings small; it changes candidate cost,
    never the acceptance rule. Shared boilerplate alone is not a match.
    """
    sets = [grams(text) for text in texts]
    frequency = Counter(g for group in sets for g in group)
    postings = defaultdict(list)
    for index, group in enumerate(sets):
        ordered = sorted(group, key=lambda g: (frequency[g], g))
        prefix = ordered[:len(group) - (4 * len(group) + 4) // 5 + 1]
        candidates = set(other for g in prefix for other in postings[g])
        for other in sorted(candidates):
            previous = sets[other]
            if 5 * min(len(group), len(previous)) < 4 * max(len(group), len(previous)):
                continue
            intersection = len(group & previous)
            union = len(group) + len(previous) - intersection
            if 5 * intersection >= 4 * union:
                yield other, index, intersection / union
        for g in prefix:
            postings[g].append(index)


def cluster_questions(records):
    identifiers = [row["id"] for row in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate source record IDs")
    clusters, edges = Clusters(len(records)), []
    identity = {}
    for index, row in enumerate(records):
        for key in row.get("identities", []):
            if key in identity:
                clusters.union(index, identity[key])
                edges.append({"left": identifiers[identity[key]], "right": row["id"], "kind": "source_id", "value": key})
            else:
                identity[key] = index
    for masked in (False, True):
        texts = [normalize(row["question"], mask_numbers=masked) for row in records]
        for left, right, similarity in near_edges(texts):
            clusters.union(left, right)
            edges.append({"left": identifiers[left], "right": identifiers[right],
                          "kind": "numeric_masked_5gram" if masked else "5gram", "jaccard": similarity})
    grouped = defaultdict(list)
    for index, row in enumerate(records):
        grouped[clusters.find(index)].append(row)
    return list(grouped.values()), edges


def split_pools(groups, *, seed=42):
    import random
    pools, exclusions, s1k_leaks = {}, [], []
    accepted = {"math": [], "code": []}
    for group in groups:
        roles = {row["role"] for row in group}
        forbidden = "test" in roles or "s1k" in roles
        if "test" in roles and "s1k" in roles:
            s1k_leaks.append([row["id"] for row in group])
        if forbidden:
            exclusions.append({"members": [row["id"] for row in group], "roles": sorted(roles)})
            continue
        # Keep a single source representative so no renamed/numeric variant can
        # cross train, validation or the shared development pool.
        candidates = sorted((row for row in group if row["role"] == "candidate"), key=lambda row: row["id"])
        if candidates:
            accepted[candidates[0]["domain"]].append(candidates[0])
    for domain in ("math", "code"):
        rows = sorted(accepted[domain], key=lambda row: row["id"])
        random.Random(seed).shuffle(rows)
        if len(rows) < 144:
            raise ValueError(f"too few isolated {domain} questions: {len(rows)}")
        # The full disjoint replacement pools are fixed before generation.
        dev, rest = rows[:64], rows[64:]
        pools[domain] = {"dev": dev, "val": rest[::5], "train": [row for i, row in enumerate(rest) if i % 5]}
    return pools, exclusions, s1k_leaks
