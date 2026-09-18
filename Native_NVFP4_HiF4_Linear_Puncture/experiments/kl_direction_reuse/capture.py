"""Real vLLM TP2 teacher-forced captures and streaming KL/NLL evaluation."""
from pathlib import Path
import os
import time

import torch

from .artifacts import checked_load, read_json, save, sha256, source_fingerprint, tensor_hash, write_json
from .config import PROTOCOL, require_gpus
from .data import Dataset
from .losses import distribution_metrics
from .materialize import model_identity

ATTR = "_kl_direction_capture"


class Logits:
    def __init__(self, entries, length):
        self.entries, self.current, self.tensor = entries, None, None
        if not entries:
            raise RuntimeError("empty teacher logits")
        self.shape = (length, entries[0]["vocab"])
        cursor = 0
        for e in entries:
            if e["start"] != cursor or e["stop"] <= cursor or e["vocab"] != self.shape[1]:
                raise RuntimeError("logit chunk coverage mismatch")
            cursor = e["stop"]
        if cursor != length:
            raise RuntimeError("incomplete logit coverage")

    def __getitem__(self, item):
        if not isinstance(item, slice) or item.step not in (None, 1):
            raise ValueError("logits require contiguous token slices")
        start, stop, _ = item.indices(self.shape[0])
        values = []
        for entry in self.entries:
            a, b = max(start, entry["start"]), min(stop, entry["stop"])
            if a >= b:
                continue
            if self.current != entry["path"]:
                self.tensor = checked_load(entry)
                self.current = entry["path"]
                if self.tensor.shape != (entry["stop"]-entry["start"], self.shape[1]) or not torch.isfinite(self.tensor).all():
                    raise RuntimeError("invalid teacher logit tensor")
            values.append(self.tensor[a-entry["start"]:b-entry["start"]])
        if not values:
            raise ValueError("empty teacher logit slice")
        return values[0] if len(values) == 1 else torch.cat(values)


class CaptureStore:
    def __init__(self, directory, ds):
        self.directory = Path(directory)
        self.manifest = read_json(self.directory / "manifest.json")
        m = self.manifest
        if m["status"] != "COMPLETE" or m["execution"] != "actual_vllm_tp2_teacher_forced_prefill" or m["protocol_sha256"] != sha256(ds.root / "protocol.json"):
            raise RuntimeError("capture provenance mismatch")
        if m["source_sha256"] != source_fingerprint():
            from .artifacts import capture_fingerprint
            proof = read_json(ds.root / 'capture_reuse.json')
            entry = proof['manifests'].get(str((self.directory / 'manifest.json').resolve()))
            if (proof['status'] != 'CERTIFIED' or proof['producer_sha256'] != capture_fingerprint()
                    or entry != sha256(self.directory / 'manifest.json')):
                raise RuntimeError("capture producer changed or reuse certificate does not bind this manifest")
        for sid, row in m["samples"].items():
            if row["token_sha256"] != ds.samples[sid]["token_sha256"] or row["length"] != len(ds.samples[sid]["input_ids"]):
                raise RuntimeError("capture tokens changed")

    def layer(self, sid, layer):
        value = checked_load(self.manifest["samples"][sid]["layers"][str(layer)])
        n = self.manifest["samples"][sid]["length"]
        for name in ("input", "output", "router_logits"):
            if value[name].shape[0] != n or not torch.isfinite(value[name]).all():
                raise RuntimeError(f"invalid {name} capture")
        return value

    def logits(self, sid):
        row = self.manifest["samples"][sid]
        return Logits(row["logits"], row["length"])


class Install:
    def __call__(self, model):
        from vllm.distributed import get_tensor_model_parallel_rank
        state = dict(rank=get_tensor_model_parallel_rank(), sid=None, positions=None, fields={}, handles=[])
        if hasattr(model, ATTR):
            raise RuntimeError("capture hooks already installed")
        setattr(model, ATTR, state)
        def capture(name, value):
            if state["sid"] is None:
                return
            positions = state["positions"]
            if positions is None or value.shape[0] != len(positions):
                raise RuntimeError(f"capture row/position mismatch: {name}")
            state["fields"].setdefault(name, []).append((positions.clone(), value.detach().cpu().clone()))
        def pre(module, args, kwargs):
            if state["sid"] is not None:
                p = kwargs.get("positions", args[1] if len(args) > 1 else None)
                if not torch.is_tensor(p):
                    raise RuntimeError("vLLM forward positions missing")
                state["positions"] = p.detach().cpu().reshape(-1)
        state["handles"].append(model.register_forward_pre_hook(pre, with_kwargs=True))
        for layer in PROTOCOL.layers:
            block = model.model.layers[layer]
            def inp(module, args, output, index=layer):
                capture(f"{index}/input", output[1])
            def out(module, args, output, index=layer):
                capture(f"{index}/output", output[1])
            def router(module, args, output, index=layer):
                capture(f"{index}/router_logits", output[0] if isinstance(output, tuple) else output)
            state["handles"].extend([block.input_layernorm.register_forward_hook(inp),
                                    model.model.layers[layer+1].input_layernorm.register_forward_hook(out),
                                    block.mlp.gate.register_forward_hook(router)])
        def norm(module, args, output):
            capture("final_hidden", output[0])
        state["handles"].append(model.model.norm.register_forward_hook(norm))
        return state["rank"]


class Begin:
    def __init__(self, sid, ids):
        self.sid, self.ids = sid, ids

    def __call__(self, model):
        state = getattr(model, ATTR)
        if state["sid"] is not None:
            raise RuntimeError("previous capture not finished")
        state.update(sid=self.sid, ids=self.ids, fields={}, positions=None)


class Finish:
    def __init__(self, directory, teacher_row, save_logits):
        self.directory, self.teacher_row, self.save_logits = str(directory), teacher_row, save_logits

    def __call__(self, model):
        state = getattr(model, ATTR)
        sid, ids, rank = state["sid"], state["ids"], state["rank"]
        n = len(ids)
        values = {}
        for name, parts in state["fields"].items():
            pos = torch.cat([p for p, _ in parts])
            order = pos.argsort()
            if not torch.equal(pos[order], torch.arange(n)):
                raise RuntimeError(f"missing/duplicate capture position at {sid}: {name}")
            values[name] = torch.cat([v for _, v in parts])[order].contiguous()
            if not torch.isfinite(values[name]).all():
                raise RuntimeError("nonfinite production capture")
        out = Path(self.directory) / sid
        row = dict(rank=rank, length=n, token_sha256=tensor_hash(ids), layers={}, logits=[], logits_hashes=[],
                   replica_hashes={k: tensor_hash(v) for k, v in values.items()})
        if rank == 0:
            for layer in PROTOCOL.layers:
                payload = {name: values[f"{layer}/{name}"] for name in ("input", "output", "router_logits")}
                row["layers"][str(layer)] = save(payload, out / f"L{layer}.pt")
            row["final_hidden"] = save(values["final_hidden"], out / "final_hidden.pt")
        teacher = Logits(self.teacher_row["logits"], n) if self.teacher_row is not None and rank == 0 else None
        metrics = dict(kl_sum=0., kl_tokens=0, nll_sum=0., nll_tokens=0)
        for start in range(0, n, PROTOCOL.capture_chunk):
            stop = min(start + PROTOCOL.capture_chunk, n)
            # Every rank must enter the real vocabulary collective.
            z = model.compute_logits(values["final_hidden"][start:stop].to(model.lm_head.weight.device))
            if rank == 0:
                if z is None or len(z) != stop-start:
                    raise RuntimeError("production LM head did not return all predictor logits")
                row["logits_hashes"].append(dict(start=start, stop=stop, sha256=tensor_hash(z)))
                if self.save_logits:
                    row["logits"].append({**save(z.detach().cpu(), out / f"logits_{start:06d}.pt"),
                                          "start": start, "stop": stop, "vocab": z.shape[-1]})
                q = z.detach() if teacher is None else teacher[start:stop].to(z.device)
                delta = distribution_metrics(z, q, ids, start)
                for key, value in delta.items():
                    metrics[key] += value
            del z
        row["metrics"] = metrics
        state.update(sid=None, fields={}, positions=None)
        return row


def capture(root, variant, *, model_dir=None, output=None, split="all", sample_ids=None):
    # Device discovery can initialize the CUDA runtime without setting PyTorch's
    # initialized flag. vLLM's automatic detection then misses the poisoned fork.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    require_gpus(2)
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    ds = Dataset(root)
    ids = list(ds.samples) if split == "all" else ds.ids(split)
    if sample_ids is not None:
        if len(set(sample_ids)) != len(sample_ids) or not set(sample_ids) <= set(ids):
            raise ValueError("capture sample subset invalid")
        ids = list(sample_ids)
    native = variant == "native"
    if variant not in {"native", "baseline", "candidate"}:
        raise ValueError(variant)
    if variant == "candidate" and (model_dir is None or output is None):
        raise ValueError("candidate capture requires explicit model-dir and output")
    if not ids:
        raise ValueError("empty capture sample list")
    model_dir = ds.snapshot if native else Path(model_dir or ds.root / "baseline").resolve()
    output = Path(output or ds.root / "captures" / variant).resolve()
    if output.exists():
        raise FileExistsError(f"capture output already exists: {output}")
    teacher = None if native else CaptureStore(ds.root / "captures/native", ds)
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    kwargs = dict(model=str(model_dir), trust_remote_code=True, tensor_parallel_size=2,
                  gpu_memory_utilization=.85, max_model_len=max(len(ds.samples[s]["input_ids"]) for s in ids)+1,
                  dtype="auto", enforce_eager=True, kv_cache_dtype="bfloat16", enable_prefix_caching=False,
                  max_num_seqs=1, max_num_batched_tokens=PROTOCOL.prefill_chunk, seed=42)
    if native:
        os.environ.pop("HIF4_RUNTIME_SPEC_PATH", None)
        kwargs.update(linear_backend="emulation", moe_backend="emulation")
    else:
        export = read_json(model_dir / "export.json")
        if export["status"] != "COMPLETE" or export["files"] != model_identity(model_dir):
            raise RuntimeError("incomplete or modified export")
        spec = str(model_dir / "hif4_runtime_spec.pt")
        os.environ["HIF4_RUNTIME_SPEC_PATH"] = spec
        kwargs.update(moe_backend="triton", additional_config={"hif4_runtime_spec_path": spec})
    output.mkdir(parents=True)
    started = time.monotonic()
    llm = LLM(**kwargs)
    if sorted(llm.apply_model(Install())) != [0, 1]:
        raise RuntimeError("capture requires two TP ranks")
    rows = {}
    for sid in ids:
        tokens = ds.samples[sid]["input_ids"]
        llm.apply_model(Begin(sid, tokens))
        llm.generate([TokensPrompt(prompt_token_ids=tokens.tolist())],
                     SamplingParams(max_tokens=1, temperature=0, ignore_eos=True), use_tqdm=False)
        ref = None if teacher is None else teacher.manifest["samples"][sid]
        replies = sorted(llm.apply_model(Finish(output, ref, native)), key=lambda r: r["rank"])
        if [r["rank"] for r in replies] != [0, 1] or replies[0]["replica_hashes"] != replies[1]["replica_hashes"]:
            raise RuntimeError("TP replicated layer boundaries disagree")
        rows[sid] = replies[0]
        write_json(dict(completed=list(rows), total=len(ids)), output / "progress.json")
        print(f"{variant}: {sid} {len(rows)}/{len(ids)}", flush=True)
    payload = dict(status="COMPLETE", execution="actual_vllm_tp2_teacher_forced_prefill", variant=variant,
                   model_dir=str(model_dir), model_files=model_identity(model_dir), runtime=kwargs,
                   protocol_sha256=sha256(ds.root / "protocol.json"), source_sha256=source_fingerprint(),
                   samples=rows, wall_seconds=time.monotonic()-started)
    write_json(payload, output / "manifest.json")
    return payload
