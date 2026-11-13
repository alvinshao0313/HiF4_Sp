"""Five real lm_eval ARC passes; no sampling or cached score replication."""
import argparse
import json
from pathlib import Path
import time

# Imported by spawn too, before vLLM loads the checkpoint quantization config.
from . import entry as registration
from .common import digest, observe_generation, read, save


def install_arc_data(manifest):
    from datasets import Dataset, DatasetDict
    from ..e2e_diag_reconstruction.evaluation.vllm_runner import _patch_lm_eval_transformers
    _patch_lm_eval_transformers()
    from lm_eval.api.task import ConfigurableTask
    from .common import file_hash

    routes = read(manifest)
    for route in routes.values():
        for item in route["splits"].values():
            if file_hash(item["path"]) != item["sha256"]:
                raise ValueError("Pinned ARC data changed")

    def download(task, dataset_kwargs=None):
        if task.DATASET_PATH != "allenai/ai2_arc" or task.DATASET_NAME not in routes:
            raise ValueError("Only the pinned ARC tasks are allowed in this entry")
        task.dataset = DatasetDict({split: Dataset.from_file(record["path"])
                                    for split, record in routes[task.DATASET_NAME]["splits"].items()})

    ConfigurableTask.download = download


def run(args):
    from ..e2e_diag_reconstruction.evaluation.lm_eval_vllm import patch_lm_eval_vllm_generate_compat
    install_arc_data(args.dataset_manifest)
    patch_lm_eval_vllm_generate_compat()
    from lm_eval import simple_evaluate
    from lm_eval.models.vllm_causallms import VLLM

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    observe_generation(output)
    lm = VLLM(pretrained=args.model, trust_remote_code=True, dtype="auto",
              tensor_parallel_size=4, batch_size="auto", max_model_len=40960,
              max_num_batched_tokens=8192, max_num_seqs=256, seed=1234,
              gpu_memory_utilization=0.55, kv_cache_memory_bytes=34359738368,
              kv_cache_dtype="bfloat16", enforce_eager=True,
              linear_backend="emulation", moe_backend="emulation")
    core = lm.model.llm_engine.engine_core
    output_thread = core.output_queue_thread
    rounds = []
    try:
        for index in range(1, args.rounds + 1):
            started = time.monotonic()
            result = simple_evaluate(model=lm, tasks=["arc_easy", "arc_challenge"],
                                     num_fewshot=0, batch_size="auto", limit=args.limit,
                                     log_samples=True, random_seed=0, numpy_random_seed=1234,
                                     torch_random_seed=1234, fewshot_random_seed=1234)
            target = output / f"round_{index:02d}"
            target.mkdir()
            samples = result.pop("samples")
            evidence = {}
            for task, rows in samples.items():
                expected = args.limit if args.limit is not None else {"arc_easy": 2376, "arc_challenge": 1172}[task]
                if len(rows) != expected or len({row["doc_id"] for row in rows}) != expected:
                    raise ValueError(f"Incomplete ARC coverage: {task}")
                with (target / f"{task}_samples.jsonl").open("w", encoding="utf-8") as stream:
                    for row in rows:
                        stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                scores = {key: [float(row[key]) for row in rows] for key in ("acc", "acc_norm")}
                for key, values in scores.items():
                    if any(value not in (0, 1) for value in values):
                        raise ValueError("ARC individual score is not binary")
                    if abs(sum(values) / expected - result["results"][task][key + ",none"]) > 1e-12:
                        raise ValueError("ARC aggregate differs from saved per-question scores")
                evidence[task] = {"count": expected, "scores": scores,
                                  "input_sha256": digest([{key: row[key] for key in ("doc_id", "doc", "arguments")}
                                                          for row in rows])}
            # lm_eval includes objects in configs. Its own JSON display convention
            # converts these to strings; all numerical results and samples are retained.
            (target / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n")
            record = {"round": index, "elapsed_seconds": time.monotonic() - started,
                      "tasks": evidence, "results": str(target / "results.json")}
            save(target / "summary.json", record)
            rounds.append(record)
            print(f"ARC round {index}/{args.rounds} completed", flush=True)
    finally:
        core.shutdown(timeout=60)
        output_thread.join(timeout=60)
        if output_thread.is_alive():
            raise RuntimeError("ARC engine output thread did not terminate")
    for task in ("arc_easy", "arc_challenge"):
        if len({row["tasks"][task]["input_sha256"] for row in rounds}) != 1:
            raise ValueError("ARC repeats have different inputs")
    save(output / "summary.json", {"status": "completed", "rounds": rounds,
         "short_test": args.limit is not None,
         "protocol": "lm_eval zero-shot option loglikelihood; acc primary, acc_norm also saved; five actual forward passes"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset_manifest", required=True)
    parser.add_argument("--rounds", type=int, choices=(2, 5), default=5)
    parser.add_argument("--limit", type=int)
    run(parser.parse_args())
