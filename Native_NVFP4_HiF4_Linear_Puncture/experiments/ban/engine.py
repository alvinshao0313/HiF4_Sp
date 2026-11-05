"""Explicit vLLM configuration and checked worker RPCs."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .algorithm import validate_parameters
from .artifact import load_artifact, model_identity, write_json

WORKER = "Native_NVFP4_HiF4_Linear_Puncture.experiments.ban.worker.Worker"


def ban_config(model, spec, artifact_path=None, lam=0.7, k_min=3, execution="compact"):
    identity = model_identity(model, spec)
    validate_parameters(lam, k_min, identity["top_k"])
    if artifact_path:
        load_artifact(artifact_path, identity)
    return {"model_path": str(Path(model).resolve()), "spec_path": str(Path(spec).resolve()),
            "artifact_path": str(Path(artifact_path).resolve()) if artifact_path else None,
            "lam": lam, "k_min": k_min, "execution": execution}


def create_engine(model, spec, tensor_parallel_size=2, max_model_len=4096,
                  max_num_seqs=1, max_num_batched_tokens=None,
                  gpu_memory_utilization=0.8, seed=0, kv_cache_dtype="auto",
                  enable_chunked_prefill=False, **settings):
    from vllm import LLM
    config = ban_config(model, spec, **settings)
    # Match main.py: nested HiF4 helpers also resolve the sidecar without an
    # explicit path, outside model-construction's current-config context.
    os.environ["HIF4_RUNTIME_SPEC_PATH"] = config["spec_path"]
    return LLM(model=str(model), dtype="bfloat16", tensor_parallel_size=tensor_parallel_size,
               max_model_len=max_model_len, max_num_seqs=max_num_seqs,
               max_num_batched_tokens=(max_model_len if max_num_batched_tokens is None else max_num_batched_tokens),
               gpu_memory_utilization=gpu_memory_utilization, enforce_eager=True, enable_prefix_caching=False,
               enable_chunked_prefill=enable_chunked_prefill, kv_cache_dtype=kv_cache_dtype,
               moe_backend="triton", seed=seed, worker_cls=WORKER,
               additional_config={"hif4_runtime_spec_path": config["spec_path"], "ban": config})


def configure(engine, **kwargs):
    return engine.collective_rpc("ban_configure", kwargs=kwargs)


def snapshot(engine):
    ranks = engine.collective_rpc("ban_snapshot")
    for other in ranks[1:]:
        for key in ("histogram", "tokens", "selected", "dispatched"):
            if other[key] != ranks[0][key]:
                raise RuntimeError(f"TP ranks disagree on Ban {key}")
    result = dict(ranks[0])
    captures = [row["capture"] for row in ranks if row["capture"] is not None]
    result["capture"] = captures[0] if captures else None
    result["average_experts"] = sum(result["selected"]) / max(1, sum(result["tokens"]))
    result["tp_ranks"] = len(ranks)
    return result


def install_engine_hook(settings, stats_dir):
    from vllm import LLM
    initialize, generate = LLM.__init__, LLM.generate
    root = Path(stats_dir)

    def init(self, *args, **kwargs):
        if not kwargs.get("enforce_eager", False):
            raise ValueError("Ban requires --enforce_eager")
        if kwargs.get("worker_cls", "auto") != "auto":
            raise ValueError("Ban cannot replace another custom worker")
        additional = dict(kwargs.get("additional_config") or {})
        spec = additional.get("hif4_runtime_spec_path")
        if not spec:
            raise ValueError("Ban requires --hif4_runtime_spec")
        config = ban_config(kwargs["model"], spec, **settings)
        additional["ban"] = config
        os.environ["HIF4_RUNTIME_SPEC_PATH"] = config["spec_path"]
        kwargs.update(additional_config=additional, worker_cls=WORKER, enable_prefix_caching=False)
        initialize(self, *args, **kwargs)
        self._ban_batch = 0
        configure(self, mode="ban")
        write_json(root / "engine.json", {**config, "tensor_parallel_size": self.llm_engine.vllm_config.parallel_config.tensor_parallel_size})

    def run(self, *args, **kwargs):
        configure(self, mode="ban")
        result = generate(self, *args, **kwargs)
        stats = snapshot(self)
        if not sum(stats["tokens"]) or (settings["execution"] == "compact" and stats["selected"] != stats["dispatched"]):
            raise RuntimeError("Ban routing was not observed or dispatch differs from selection")
        write_json(root / f"routing_{self._ban_batch:04d}.json", stats)
        self._ban_batch += 1
        print("BAN_ROUTING " + json.dumps({k: stats[k] for k in ("average_experts", "execution", "tp_ranks")}), flush=True)
        return result

    LLM.__init__, LLM.generate = init, run
