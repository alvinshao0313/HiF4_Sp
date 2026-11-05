"""One fixed math/code development set chooses the sole router initialization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import read_json, sha256, validate_root, verify_sources, write_json
from .config import ENGINE, GENERATION, MODELS, NATIVE
from .generate import close_engine, engine, prompt
from .prepare import require_isolation
from .scoring import code_score, math_score
from .sources import reference


def evaluate(root, candidate):
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    root = validate_root(root)
    require_isolation(root)
    verify_sources(root)
    if candidate not in MODELS[:2]:
        raise ValueError(candidate)
    model = root / "models" / candidate
    exported = read_json(model / "export.json")
    if not exported["complete"] or exported["qualification"]:
        raise ValueError("development requires the complete trained checkpoint")
    output = root / "development" / candidate
    output.mkdir(parents=True, exist_ok=False)
    pools = read_json(root / "isolation/pools.json")
    tokenizer = AutoTokenizer.from_pretrained(NATIVE, local_files_only=True)
    llm = engine(model)
    rows = []
    try:
        requests = []
        for domain in ("math", "code"):
            questions = pools[domain]["dev"]
            if len(questions) != 64:
                raise ValueError("development domain requires 64 questions")
            for row in questions:
                ids = prompt(tokenizer, row)
                if len(ids) + GENERATION["max_tokens"] > ENGINE["max_model_len"]:
                    raise ValueError(f"development question exceeds the fixed prompt budget: {row['id']}")
                requests.append((domain, row, ids))
        results = llm.generate([{"prompt_token_ids": ids} for _, _, ids in requests], SamplingParams(**GENERATION), use_tqdm=False)
        if len(results) != 128:
            raise ValueError("incomplete shared development generation")
        for (domain, row, ids), generation in zip(requests, results):
            result = generation.outputs[0]
            saved = {"id": row["id"], "domain": domain, "prompt_token_ids": ids,
                     "generated_token_ids": list(result.token_ids), "text": result.text,
                     "finish_reason": result.finish_reason}
            write_json(saved, output / f"responses/{len(rows):03d}.json")
            rows.append(saved)
    finally:
        close_engine(llm)
    # Reference material is opened only after the inference engine has closed.
    sources = {r["id"]: r for d in pools.values() for r in d["dev"]}
    for row in rows:
        answer = reference(root, sources[row["id"]])
        row["score"] = (math_score(row["text"], answer["solution"]) if row["domain"] == "math"
                        else code_score(row["text"], answer))
        print(json.dumps({"event": "development_score", "id": row["id"], "score": row["score"]}), flush=True)
    means = {d: sum(r["score"] for r in rows if r["domain"] == d) / 64 for d in ("math", "code")}
    write_json({"candidate": candidate, "complete": True, "per_question": rows, "domain_scores": means,
                "equal_mean": sum(means.values()) / 2, "export_sha256": sha256(model / "export.json")}, output / "metrics.json")


def select(root):
    root = validate_root(root)
    target = root / "development/selection.json"
    if target.exists():
        raise FileExistsError("router initialization was already fixed")
    rows = {name: read_json(root / "development" / name / "metrics.json") for name in MODELS[:2]}
    for name, row in rows.items():
        if not row["complete"] or len(row["per_question"]) != 128:
            raise ValueError("incomplete development evaluation")
        if sha256(root / "models" / name / "export.json") != row["export_sha256"]:
            raise ValueError("development checkpoint changed")
    a, b = (rows[name] for name in MODELS[:2])
    if {r["id"] for r in a["per_question"]} != {r["id"] for r in b["per_question"]}:
        raise ValueError("development question sets differ")
    winner = MODELS[1] if b["equal_mean"] > a["equal_mean"] else MODELS[0]
    write_json({"winner": winner, "scores": {n: r["equal_mean"] for n, r in rows.items()},
                "development_sha256": {n: sha256(root / "development" / n / "metrics.json") for n in rows},
                "export_sha256": {n: r["export_sha256"] for n, r in rows.items()},
                "rule": "equal mean math accuracy/code pass@1; tie chooses A"}, target)
    return winner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidate", choices=MODELS[:2])
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args()
    select(args.run) if args.select else evaluate(args.run, args.candidate)
