"""Check the isolated common math/code pool and lock each final candidate before testing."""
import argparse
import gc
import os
from pathlib import Path
import shutil

import torch

from ..reasoning_trace_calibration.generate import close_engine, engine, prompt
from ..reasoning_trace_calibration.scoring import code_score, math_score
from ..reasoning_trace_calibration.sources import reference
from .artifacts import atomic_save, digest, load, read_json, sha256, state_digest, verify, write_json
from .config import CANDIDATES, ENGINE, GENERATION, NATIVE, ORIGIN
from .export import materialize


def requests_for(m):
    from transformers import AutoTokenizer
    source = Path(m["source_run"])
    pools = read_json(source / "isolation/pools.json")
    tokenizer = AutoTokenizer.from_pretrained(NATIVE, local_files_only=True)
    requests = []
    for domain in ("math", "code"):
        if len(pools[domain]["dev"]) != 64:
            raise ValueError("the shared development pool must contain 64 math and 64 code")
        for row in pools[domain]["dev"]:
            ids = prompt(tokenizer, row)
            if len(ids) + GENERATION["max_tokens"] > ENGINE["max_model_len"]:
                raise ValueError("development prompt exceeds original generation budget")
            requests.append((domain, row, ids))
    baseline = read_json(source / "development" / ORIGIN / "metrics.json")
    expected = {row["id"]: row for row in baseline["per_question"]}
    if not baseline["complete"] or len(expected) != 128:
        raise ValueError("complete original B development results required")
    for domain, row, ids in requests:
        old = expected[row["id"]]
        if old["domain"] != domain or old["prompt_token_ids"] != ids:
            raise ValueError("shared development prompt/token identity differs from original B")
    return requests, baseline


def evaluate(root, m, model, output, requests):
    from vllm import SamplingParams
    output.mkdir(parents=True, exist_ok=False)
    llm = engine(model)
    rows = []
    try:
        results = llm.generate([{"prompt_token_ids": ids} for _, _, ids in requests],
                               SamplingParams(**GENERATION), use_tqdm=False)
        if len(results) != 128:
            raise ValueError("incomplete common development inference")
        for (domain, question, ids), result in zip(requests, results):
            answer = result.outputs[0]
            row = {"id": question["id"], "domain": domain, "prompt_token_ids": ids,
                   "generated_token_ids": list(answer.token_ids), "text": answer.text,
                   "finish_reason": answer.finish_reason}
            if answer.finish_reason not in ("stop", "length"):
                raise ValueError("unexpected development completion status")
            write_json(row, output / f"responses/{len(rows):03d}.json")
            rows.append(row)
    finally:
        close_engine(llm)
    # Gold solutions and hidden code tests are opened only after generation ends.
    sources = {r["id"]: r for _, r, _ in requests}
    for row in rows:
        gold = reference(Path(m["source_run"]), sources[row["id"]])
        row["score"] = (math_score(row["text"], gold["solution"]) if row["domain"] == "math"
                        else code_score(row["text"], gold))
        print({"event": "development_score", "id": row["id"], "score": row["score"]}, flush=True)
    means = {d: sum(r["score"] for r in rows if r["domain"] == d) / 64 for d in ("math", "code")}
    result = {"complete": True, "per_question": rows, "domain_scores": means, "equal_mean": sum(means.values()) / 2,
              "export_sha256": sha256(model / "export.json"),
              "prompt_token_sha256": digest([(r["id"], ids) for _, r, ids in requests])}
    write_json(result, output / "metrics.json")
    return result


def choose(options):
    # Prefer B on a development tie; otherwise earlier epoch, then smaller LR.
    return min(options, key=lambda r: (-r["equal_mean"], not r["is_original_B"], r["epoch"], r["lr"], r["tag"]))


def select(root, objective):
    root, m = verify(root)
    requests, baseline = requests_for(m)
    directory = root / "training" / objective
    trained = read_json(directory / "manifest.json")
    if not trained["complete"] or trained["qualification"]:
        raise ValueError("complete formal calibration training required")
    output = root / "development" / objective
    output.mkdir(parents=True, exist_ok=False)
    pool = (trained["pool"] if m["selection_policy"] == "development" else [trained["calibration_selected"]])
    options, seen = [], {}
    if m["selection_policy"] == "development":
        options.append({"tag": "epoch_zero", "is_original_B": True, "epoch": 0, "lr": 0.,
                        "equal_mean": baseline["equal_mean"], "domain_scores": baseline["domain_scores"],
                        "checkpoint": str(directory / "epoch_zero.pt"), "model": m["initialization"]})
    for index, row in enumerate(pool):
        checkpoint = Path(row["checkpoint"])
        values = load(checkpoint)
        fingerprint = state_digest(values["parameters"])
        tag = f"candidate_{index:02d}"
        if fingerprint == trained["initial_parameters_sha256"]:
            metrics, model = baseline, Path(m["initialization"])
        elif fingerprint in seen:
            metrics, model = seen[fingerprint]
        else:
            model = materialize(root, objective, checkpoint=checkpoint, output=root / "development_models" / objective / tag)
            gc.collect()
            torch.cuda.empty_cache()
            metrics = evaluate(root, m, model, output / tag, requests)
            seen[fingerprint] = (metrics, model)
        options.append({**row, "tag": tag, "is_original_B": fingerprint == trained["initial_parameters_sha256"],
                        "equal_mean": metrics["equal_mean"], "domain_scores": metrics["domain_scores"], "model": str(model)})
    selected = choose(options) if m["selection_policy"] == "development" else options[0]
    atomic_save(load(selected["checkpoint"]), directory / "selected.pt")
    model = Path(selected["model"])
    final = root / "models" / CANDIDATES[objective]
    final.mkdir(parents=True, exist_ok=False)
    index = read_json(model / "model.safetensors.index.json")
    for name in set(index["weight_map"].values()):
        os.link(model / name, final / name)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json",
                 "vocab.json", "merges.txt", "chat_template.jinja", "hif4_runtime_spec.pt", "hif4_runtime_abi.json",
                 "model.safetensors.index.json"):
        if (model / name).is_file():
            shutil.copyfile(model / name, final / name)
    metadata = read_json(model / "export.json")
    metadata.update(candidate=CANDIDATES[objective], objective=objective, selected_checkpoint=selected["checkpoint"],
                    selected_checkpoint_sha256=sha256(selected["checkpoint"]), selection_policy=m["selection_policy"],
                    selected_model=str(model), qualification=False, complete=True)
    write_json(metadata, final / "export.json")
    write_json({"complete": True, "policy": m["selection_policy"], "baseline_B": {k: baseline[k] for k in ("domain_scores", "equal_mean")},
                "options": options, "selected": selected, "formal_tests_used_for_selection": False,
                "prompt_token_sha256": digest([(r["id"], ids) for _, r, ids in requests]),
                "export_sha256": sha256(final / "export.json")}, output / "selection.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--objective", choices=CANDIDATES, required=True)
    args = parser.parse_args()
    select(args.run, args.objective)
