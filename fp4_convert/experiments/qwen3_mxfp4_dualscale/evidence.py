"""Verify saved inference evidence and build repeat-preserving GPQA metrics."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from ..nvfp4_gpqa.task import extract_final_answer
from .artifacts import save, tensor_record
from .model_io import load_experts


def summarize_phase(root, phase, protocol_inputs, output):
    root = Path(root)
    status = json.loads((root / phase / "status.json").read_text())
    if status["status"] != "completed" or status["returncode"] != 0:
        raise ValueError(f"Incomplete evaluation: {root}/{phase}")
    expected_questions = 198 if phase == "formal" else 26
    limit = 32768 if phase == "formal" else 64
    protocol = {str(row["id"]): row for row in json.loads(Path(protocol_inputs).read_text())}
    details = json.loads(Path(status["details_file"]).read_text())
    records = {str(row["doc"]["id"]): row for row in details}
    cached = [row for path in status["raw_caches"] for row in pq.read_table(path).to_pylist()]
    if len(records) != expected_questions or len(details) != expected_questions or len(cached) != expected_questions:
        raise ValueError("Raw cache or scoring coverage is incomplete/duplicated")
    if {str(row["sample_id"]) for row in cached} != set(records):
        raise ValueError("Raw cache IDs disagree with details")
    if phase == "formal" and set(records) != set(protocol):
        raise ValueError("Formal evaluation does not contain all original question IDs")
    totals = [0] * 10
    lengths = []
    parse_counts = Counter()
    input_truncated = 0
    for row in cached:
        sample_id, sample = str(row["sample_id"]), row["sample"]
        reference = protocol[sample_id]
        if hashlib.sha256(sample["input"].encode()).hexdigest() != reference["prompt_sha256"]:
            raise ValueError(f"Prompt/options changed: {sample_id}")
        token_hash = hashlib.sha256(json.dumps(sample["input_tokens"], separators=(",", ":")).encode()).hexdigest()
        if token_hash != reference["input_tokens_sha256"]:
            raise ValueError(f"Actual input tokens changed: {sample_id}")
        specific = records[sample_id]["doc"]["specific"]
        if specific["question_sha256"] != reference["question_sha256"]:
            raise ValueError(f"Question identity changed: {sample_id}")
        parsed = [extract_final_answer(text) for text in sample["text"]]
        scores = [int(answer == reference["gold"]) for answer, _ in parsed]
        if len(parsed) != 10 or len(sample["output_tokens"]) != 10:
            raise ValueError(f"Incomplete ten-sample response: {sample_id}")
        if scores != specific["gpqa_scoring"]["scores"] or [s for _, s in parsed] != specific["gpqa_scoring"]["parse_status"]:
            raise ValueError(f"Automatic raw-text score disagrees with recorded score: {sample_id}")
        for index, score in enumerate(scores):
            totals[index] += score
        parse_counts.update(s for _, s in parsed)
        lengths.extend(len(tokens) for tokens in sample["output_tokens"])
        input_truncated += sample["truncated_tokens_count"]
    if any(length <= 0 or length > limit for length in lengths) or input_truncated:
        raise ValueError("Unexpected output length or truncated benchmark prompt")
    accuracy = sum(totals) / len(lengths)
    if abs(accuracy - status["accuracy"]) > 1e-12:
        raise ValueError("Summary accuracy disagrees with evaluation status")
    report = {"status": "verified", "accuracy_is_benchmark": phase == "formal",
              "questions": expected_questions, "generations": len(lengths),
              "accuracy": accuracy, "correct": sum(totals),
              "accuracy_by_sampling_index": [value / expected_questions for value in totals],
              "correct_by_sampling_index": totals, "parse_status": dict(parse_counts),
              "answer_parse_failure_rate": 1 - parse_counts["ok"] / len(lengths),
              "truncation_rate": sum(length == limit for length in lengths) / len(lengths),
              "truncation_definition": "output length reaches max_new_tokens; cache does not retain finish_reason",
              "input_truncated_tokens": input_truncated, "max_new_tokens": limit,
              "input_prompt_tokens_options_verified": True, "raw_caches": status["raw_caches"],
              "details_file": status["details_file"], "returncode": status["returncode"]}
    save(output, report)
    return report


def verify_runtime_audit(model, directory, output):
    variant = json.loads((Path(model) / "conversion.json").read_text())["variant"]
    loaded, methods = {}, []
    for path in sorted(Path(directory).glob("worker-*.jsonl")):
        events = [json.loads(line) for line in path.read_text().splitlines()]
        expert_events = [event for event in events if event["event"] == "loaded_experts"]
        if not expert_events:
            continue
        if any(event["variant"] != variant or event["method"] != "Qwen3ModelOptMXFP4FusedMoE" for event in expert_events):
            raise ValueError(f"Unexpected expert implementation/variant: {path}")
        ranks = {event["tp_rank"] for event in expert_events}
        if len(ranks) != 1:
            raise ValueError(f"Worker changed TP rank: {path}")
        rank = ranks.pop()
        if rank in loaded:
            raise ValueError("Duplicate expert worker/rank")
        loaded[rank] = {event["prefix"]: event for event in expert_events}
        if len(expert_events) != 48 or len(loaded[rank]) != 48:
            raise ValueError(f"Worker does not cover all 48 layers: {path}")
        selections = {event["prefix"]: event["method"] for event in events if event["event"] == "non_expert_method"}
        for layer in range(48):
            prefix = f"model.layers.{layer}"
            expected = {prefix + ".mlp.gate": "UnquantizedLinearMethod",
                        prefix + ".self_attn.qkv_proj": "ModelOptNvFp4LinearMethod",
                        prefix + ".self_attn.o_proj": "ModelOptNvFp4LinearMethod",
                        prefix + ".self_attn.attn": "ModelOptFp8KVCacheMethod"}
            for name, method in expected.items():
                if selections.get(name) != method:
                    raise ValueError(f"Non-expert method changed/missing: rank={rank} {name}: {selections.get(name)}")
        # ModelOpt returns None for VocabParallelEmbedding/ParallelLMHead;
        # their constructor selects the native UnquantizedEmbeddingMethod.
        for prefix in ("lm_head", "model.embed_tokens"):
            if selections.get(prefix) != "NoneType":
                raise ValueError(f"Embedding/LM head selection changed/missing: {prefix}")
        methods.append({"rank": rank, "selections": selections})
    if set(loaded) != {0, 1, 2, 3}:
        raise ValueError("Missing TP4 worker audit")
    for layer in range(48):
        for rank in range(4):
            event = loaded[rank][f"model.layers.{layer}.mlp.experts"]
            expected = {name: tensor_record(tensor) for name, tensor in load_experts(model, layer, rank).items()}
            if event["tensors"] != expected:
                raise ValueError(f"Packed weight/scale TP slice mismatch: layer={layer} rank={rank}")
        if (layer + 1) % 8 == 0:
            print(f"TP4 packed-byte audit: {layer+1}/48 layers", flush=True)
    save(output, {"status": "passed", "variant": variant, "layers_per_rank": 48, "ranks": [0, 1, 2, 3],
                  "packed_weight_and_scale_bytes_exact": True, "non_expert_methods": methods})


def comparison(root):
    root = Path(root)
    summaries = {name: json.loads((root / f"{name}_summary.json").read_text()) for name in ("native", "direct", "r64")}
    if any(not item["accuracy_is_benchmark"] or item["generations"] != 1980 for item in summaries.values()):
        raise ValueError("All three formal results must be complete")
    save(root / "comparison.json", {"status": "completed", "results": summaries,
        "r64_minus_direct_percentage_points": 100 * (summaries["r64"]["accuracy"] - summaries["direct"]["accuracy"]),
        "direct_minus_native_percentage_points": 100 * (summaries["direct"]["accuracy"] - summaries["native"]["accuracy"]),
        "r64_minus_native_percentage_points": 100 * (summaries["r64"]["accuracy"] - summaries["native"]["accuracy"])})
