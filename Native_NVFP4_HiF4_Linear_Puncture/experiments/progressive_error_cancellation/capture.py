"""Actual vLLM TP2 teacher-forced distribution capture for this protocol."""
from __future__ import annotations

import os
import time
from pathlib import Path

import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.losses import distribution_metrics
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.materialize import model_identity
from .artifact import atomic_save, read_json, sha256, tensor_sha256, write_json
from .data import Dataset

ATTR = "_progressive_error_capture"


class Install:
    def __call__(self, model):
        from vllm.distributed import get_tensor_model_parallel_rank
        state = {"rank": get_tensor_model_parallel_rank(), "sid": None, "positions": None,
                 "final": None}
        setattr(model, ATTR, state)

        def pre(_module, args, kwargs):
            if state["sid"] is None:
                return
            positions = kwargs.get("positions", args[1] if len(args) > 1 else None)
            if not torch.is_tensor(positions):
                raise RuntimeError("vLLM forward positions missing")
            state["positions"] = positions.detach().cpu().reshape(-1)

        def norm(_module, _args, output):
            if state["sid"] is None:
                return
            value = output[0] if isinstance(output, tuple) else output
            if state["positions"] is None or value.shape[0] != len(state["positions"]):
                raise RuntimeError("final hidden position coverage mismatch")
            state["final"] = value.detach().cpu().clone()

        state["handles"] = [model.register_forward_pre_hook(pre, with_kwargs=True),
                             model.model.norm.register_forward_hook(norm)]
        return state["rank"]


class Begin:
    def __init__(self, sid, ids):
        self.sid, self.ids = sid, ids

    def __call__(self, model):
        state = getattr(model, ATTR)
        state.update(sid=self.sid, ids=self.ids, positions=None, final=None)


class Finish:
    def __init__(self, output, teacher, save_logits):
        self.output, self.teacher, self.save_logits = Path(output), teacher, save_logits

    def __call__(self, model):
        state = getattr(model, ATTR)
        sid, ids, rank = state["sid"], state["ids"], state["rank"]
        final = state["final"]
        if final is None or len(final) != len(ids):
            raise RuntimeError("final hidden capture incomplete")
        row_dir = self.output / sid
        row_dir.mkdir(parents=True, exist_ok=True)
        row = {"rank": rank, "length": len(ids), "token_sha256": tensor_sha256(ids),
               "final_hidden": None, "logits": [], "metrics": {"kl_sum": 0., "kl_tokens": 0,
               "nll_sum": 0., "nll_tokens": 0}}
        hidden_meta = {"path": str(row_dir / "final_hidden.pt"), "shape": list(final.shape),
                       "sha256": tensor_sha256(final)}
        if rank == 0:
            atomic_save(final, row_dir / "final_hidden.pt")
        row["final_hidden"] = hidden_meta
        teacher_logits = None
        if self.teacher is not None and rank == 0:
            teacher_logits = []
            for entry in self.teacher["logits"]:
                teacher_logits.append(torch.load(entry["path"], map_location="cpu", weights_only=False))
            teacher_logits = torch.cat(teacher_logits)
        chunk = 256
        for start in range(0, len(ids), chunk):
            stop = min(start + chunk, len(ids))
            logits = model.compute_logits(final[start:stop].to(model.lm_head.weight.device))
            if rank == 0:
                z = logits.detach().cpu()
                path = row_dir / f"logits_{start:06d}.pt"
                atomic_save(z, path)
                row["logits"].append({"path": str(path), "start": start, "stop": stop,
                                      "vocab": int(z.shape[-1]), "sha256": tensor_sha256(z)})
                reference = z if teacher_logits is None else teacher_logits[start:stop]
                metrics = distribution_metrics(z, reference, ids, start)
                for key, value in metrics.items():
                    row["metrics"][key] += value
            del logits
        state.update(sid=None, positions=None, final=None)
        return row


def capture(root, model_dir=None, output=None, *, split="holdout", native=False, sample_ids=None):
    if output is None:
        raise ValueError("capture output is required")
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.config import require_gpus
    require_gpus(2)
    ds = Dataset(root)
    ids = list(ds.ids(split)) if sample_ids is None else list(sample_ids)
    if not ids or not set(ids) <= set(ds.ids(split)):
        raise ValueError("invalid capture sample IDs")
    model_dir = Path(ds.snapshot if model_dir is None else model_dir).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"capture output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    teacher_manifest = None
    if not native:
        teacher_manifest = read_json(Path(root) / "captures/native/manifest.json")
    kwargs = dict(model=str(model_dir), trust_remote_code=True, tensor_parallel_size=2,
                  gpu_memory_utilization=.85, max_model_len=max(len(ds.samples[s].input_ids) for s in ids) + 1,
                  dtype="auto", enforce_eager=True, kv_cache_dtype="bfloat16",
                  enable_prefix_caching=False, max_num_seqs=1, max_num_batched_tokens=2048, seed=42)
    if native:
        os.environ.pop("HIF4_RUNTIME_SPEC_PATH", None)
        kwargs.update(linear_backend="emulation", moe_backend="emulation")
    else:
        export = read_json(model_dir / "export.json")
        if export.get("status") != "COMPLETE" or export.get("files") != model_identity(model_dir):
            raise RuntimeError("candidate export identity mismatch")
        spec = str(model_dir / "hif4_runtime_spec.pt")
        os.environ["HIF4_RUNTIME_SPEC_PATH"] = spec
        kwargs.update(moe_backend="triton", additional_config={"hif4_runtime_spec_path": spec})
    llm = LLM(**kwargs)
    if sorted(llm.apply_model(Install())) != [0, 1]:
        raise RuntimeError("capture requires TP2 ranks")
    rows = {}
    started = time.monotonic()
    for sid in ids:
        token_ids = ds.samples[sid].input_ids
        llm.apply_model(Begin(sid, token_ids))
        llm.generate([TokensPrompt(prompt_token_ids=token_ids.tolist())],
                     SamplingParams(max_tokens=1, temperature=0, ignore_eos=True), use_tqdm=False)
        teacher = None if native else teacher_manifest["samples"][sid]
        replies = sorted(llm.apply_model(Finish(output, teacher, True)), key=lambda x: x["rank"])
        if [r["rank"] for r in replies] != [0, 1]:
            raise RuntimeError("missing TP rank capture")
        if replies[0]["final_hidden"]["sha256"] != replies[1]["final_hidden"]["sha256"]:
            raise RuntimeError("TP replicated final hidden differs")
        rows[sid] = replies[0]
        write_json({"completed": list(rows), "total": len(ids)}, output / "progress.json")
    payload = {"status": "COMPLETE", "execution": "actual_vllm_tp2_teacher_forced_prefill",
               "native": native, "model_dir": str(model_dir), "model_files": model_identity(model_dir),
               "samples": rows, "split": split, "wall_seconds": time.monotonic() - started,
               "source_snapshot": str(ds.snapshot), "dataset_protocol_sha256": sha256(Path(root) / "protocol.json")}
    write_json(payload, output / "manifest.json")
    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--model_dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="holdout")
    parser.add_argument("--native", action="store_true")
    capture(**vars(parser.parse_args()))
