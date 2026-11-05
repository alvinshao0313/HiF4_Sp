"""Cluster before generation, materialize authorized clean S1K, and verify isolation."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from .artifacts import initialize, read_json, sha256, validate_root, write_json
from .isolation import cluster_questions, normalize, split_pools
from .sources import evaluation_questions, fetch_sources, s1k_questions


def audit(root, *, include_sources=True):
    root = Path(root)
    validate_root(root)
    tests, test_evidence = evaluation_questions()
    if include_sources:
        from .clean_s1k import source, materialize
        candidates = fetch_sources(root)
        data, tokenizer, originals, s1k, source_evidence = source()
    else:
        candidates = []
        s1k, s1k_evidence = s1k_questions()
    rows = candidates + tests + s1k
    print(f"clustering {len(rows)} question records before any generation", flush=True)
    groups, edges = cluster_questions(rows)
    if include_sources:
        materialize(root, data, tokenizer, originals, groups, source_evidence)
        _, s1k_evidence = s1k_questions(root / "calibration/s1k_clean")
    write_json(rows, root / "isolation/questions.json")
    leaks = [[row for row in group] for group in groups
             if {"test", "s1k"}.issubset({r["role"] for r in group})]
    cross_split = [group for group in groups if len({r["split"] for r in group if r["role"] == "s1k"}) > 1]
    report = {"s1k": s1k_evidence, "evaluation": test_evidence, "questions": len(rows),
              "clusters": len(groups), "edges": edges, "s1k_test_overlap": leaks,
              "s1k_cross_split_overlap": cross_split,
              "status": "blocked_s1k_overlap" if leaks or cross_split else "passed",
              "includes_official_source_tests": include_sources}
    write_json(report, root / "isolation/audit.json")
    if leaks or cross_split:
        raise ValueError(f"S1K contains {len(leaks)} test-overlapping clusters; see {root / 'isolation/audit.json'}. "
                         "Existing A samples were not modified. Dependent training must stop.")
    if include_sources:
        from transformers import AutoTokenizer
        from .config import ENGINE, GENERATION, NATIVE
        from .generate import prompt
        tokenizer = AutoTokenizer.from_pretrained(NATIVE, local_files_only=True)
        budget_excluded = []
        eligible = []
        for group in groups:
            if any(r["role"] in {"test", "s1k"} for r in group):
                eligible.append(group)
                continue
            options = [r for r in group if r["role"] == "candidate"]
            if not options:
                continue
            chosen = min(options, key=lambda r: r["id"])
            if len(prompt(tokenizer, chosen)) + GENERATION["max_tokens"] > ENGINE["max_model_len"]:
                budget_excluded.append([r["id"] for r in group])
            else:
                eligible.append(group)
        write_json(budget_excluded, root / "isolation/prompt_budget_exclusions.json")
        pools, exclusions, _ = split_pools(eligible)
        for domain in pools:
            for split in pools[domain]:
                for row in pools[domain][split]:
                    row["question_sha256"] = hashlib.sha256(normalize(row["question"]).encode()).hexdigest()
        write_json(pools, root / "isolation/pools.json")
        write_json(exclusions, root / "isolation/exclusions.json")
        write_json({"pools_sha256": sha256(root / "isolation/pools.json"),
                    "audit_sha256": sha256(root / "isolation/audit.json"),
                    "questions_sha256": sha256(root / "isolation/questions.json"),
                    "clean_s1k_manifest_sha256": sha256(root / "calibration/s1k_clean/manifest.json")}, root / "isolation/manifest.json")
    return report


def require_isolation(root):
    manifest = read_json(Path(root) / "isolation/manifest.json")
    for key, filename in (("pools_sha256", "pools.json"), ("audit_sha256", "audit.json"),
                          ("questions_sha256", "questions.json")):
        if sha256(Path(root) / "isolation" / filename) != manifest[key]:
            raise ValueError(f"isolation artifact changed: {filename}")
    audit = read_json(Path(root) / "isolation/audit.json")
    if audit["status"] != "passed" or not audit["includes_official_source_tests"]:
        raise ValueError("complete test-isolation evidence is required")
    clean = Path(root) / "calibration/s1k_clean/manifest.json"
    if sha256(clean) != manifest["clean_s1k_manifest_sha256"]:
        raise ValueError("cleaned S1K manifest changed after isolation")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--s1k-only", action="store_true", help="Early read-only S1K/formal-test audit; does not authorize generation")
    args = parser.parse_args()
    if args.initialize:
        initialize(args.run)
    audit(args.run, include_sources=not args.s1k_only)


if __name__ == "__main__":
    main()
