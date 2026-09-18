"""Single-GPU, fixed-budget training. Validation never chooses or rolls back parameters."""
import fcntl
import ctypes
import json
import math
from pathlib import Path
import time

import torch

from .artifacts import (ActiveClock, BudgetExpired, check_budget, checked_load, move,
                        parameters_hash, read_json, save, sha256, source_fingerprint, write_json)
from .capture import CaptureStore
from .config import PROTOCOL, recipes, require_gpus
from .data import Dataset, epoch_order
from .directions import Directions, build as build_directions
from .materialize import check_baseline, model_identity
from .optimization import backward_sample, finite_gradients
from .runtime import Runtime


def _heap_live_bytes():
    """Return live libc heap bytes for smoke resource accounting, when glibc exposes it."""
    try:
        class Mallinfo2(ctypes.Structure):
            _fields_ = [(name, ctypes.c_size_t) for name in
                        ("arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks",
                         "fsmblks", "uordblks", "fordblks", "keepcost")]
        libc = ctypes.CDLL(None)
        libc.mallinfo2.restype = Mallinfo2
        value = libc.mallinfo2()
        return int(value.uordblks + value.hblkhd)
    except (AttributeError, OSError):
        return None


class _TrainingView:
    """Restrict only the training batches while preserving full capture provenance."""

    def __init__(self, dataset, train_sample_ids):
        ids = tuple(train_sample_ids)
        expected = set(dataset.ids("train"))
        if len(ids) != len(set(ids)) or not set(ids) <= expected:
            raise ValueError("training sample subset must be distinct members of the train split")
        if not ids:
            raise ValueError("training sample subset cannot be empty")
        self.root = dataset.root
        self.samples = dataset.samples
        self.snapshot = dataset.snapshot
        self.protocol = dict(dataset.protocol)
        self.protocol["splits"] = dict(dataset.protocol["splits"])
        self.protocol["splits"]["train"] = list(ids)
        ordered = sorted(ids, key=lambda sid: (len(self.samples[sid]["input_ids"]), sid))
        if len(ordered) % 4:
            raise ValueError("training sample subset must form complete batch-size-4 groups")
        self.protocol["batches"] = [ordered[i:i + 4] for i in range(0, len(ordered), 4)]

    def ids(self, split):
        return self.protocol["splits"][split]


def training_provenance(root):
    root = Path(root)
    return dict(protocol=sha256(root / "protocol.json"), baseline=sha256(root / "baseline/export.json"),
                native=sha256(root / "captures/native/manifest.json"),
                baseline_capture=sha256(root / "captures/baseline/manifest.json"), source=source_fingerprint())


def require_verification(root, layer, provenance):
    gate = read_json(Path(root) / "verification" / f"L{layer:02d}" / "report.json")
    if gate["status"] != "PASS" or gate["provenance"] != provenance or gate["layer"] != layer:
        raise RuntimeError("current-code production/gradient verification has not passed")
    return gate


def train(root, recipe_name, *, resume=False, budget_seconds=None, train_sample_ids=None):
    require_gpus(1)
    choices = {r["name"]: r for r in recipes()}
    if recipe_name not in choices:
        raise ValueError(f"unknown recipe {recipe_name}; expected {list(choices)}")
    ds = Dataset(root)
    smoke = (ds.root / "smoke_config.json").exists()
    if smoke != (budget_seconds is not None and train_sample_ids is not None):
        raise ValueError("budget/sample overrides require an isolated smoke root and both options")
    if not smoke and (budget_seconds is not None or train_sample_ids is not None):
        raise ValueError("formal training cannot use smoke overrides")
    if smoke:
        settings = read_json(ds.root / "smoke_config.json")
        if budget_seconds != settings["budget_seconds"] or train_sample_ids != settings["train_sample_ids"]:
            raise ValueError("smoke invocation differs from its frozen configuration")
    if train_sample_ids is not None:
        ds = _TrainingView(ds, train_sample_ids)
    budget_seconds = PROTOCOL.budget_seconds if budget_seconds is None else float(budget_seconds)
    if not torch.isfinite(torch.tensor(budget_seconds)) or budget_seconds <= 0:
        raise ValueError("budget_seconds must be finite and positive")
    recipe = choices[recipe_name]
    out = ds.root / "runs" / recipe_name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "train.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train(ds, recipe, out, resume, budget_seconds=budget_seconds,
                      train_sample_ids=tuple(train_sample_ids) if train_sample_ids is not None else None)


def _train(ds, recipe, out, resume, *, budget_seconds=None, train_sample_ids=None):
    budget_seconds = PROTOCOL.budget_seconds if budget_seconds is None else budget_seconds
    thresholds = (1800, 3600, 5400) if train_sample_ids is None else (150, 300, 450)
    check_baseline(ds.root)
    native = CaptureStore(ds.root / "captures/native", ds)
    baseline = CaptureStore(ds.root / "captures/baseline", ds)
    if native.manifest["model_files"] != model_identity(ds.snapshot):
        raise RuntimeError("native teacher model changed")
    if baseline.manifest["model_files"] != read_json(ds.root / "baseline/export.json")["files"]:
        raise RuntimeError("baseline capture belongs to another model")
    provenance = training_provenance(ds.root)
    require_verification(ds.root, recipe["layer"], provenance)
    torch.manual_seed(PROTOCOL.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    runtime = Runtime(ds.snapshot, ds.root / "baseline", recipe["layer"])
    optimizer = torch.optim.AdamW(runtime.student.learned.parameters(), lr=PROTOCOL.lr, weight_decay=0.)
    prior, phase, cache, saved_times = 0., "RUNNING", None, []
    if resume:
        state = read_json(out / "state.json")
        if state["status"] not in {"INTERRUPTED", "FAILED"}:
            raise RuntimeError("resume requires an accounted, interrupted/failed attempt; a hard-killed RUNNING attempt needs inspection")
        if state["provenance"] != provenance or state["recipe"] != recipe:
            raise RuntimeError("resume configuration or provenance changed")
        if float(state["budget_seconds"]) != budget_seconds or tuple(state.get("train_sample_ids") or ()) != tuple(train_sample_ids or ()):
            raise RuntimeError("resume budget or training sample subset changed")
        committed = checked_load(state["checkpoint"])
        prior, saved_times = state["charged_seconds"], state["saved_times"]
        runtime.restore(committed["parameters"])
        optimizer.load_state_dict(committed["optimizer"])
        if committed["cache"] is not None:
            cache = Directions(committed["cache"], provenance=provenance, ds=ds, layer=runtime.layer)
    else:
        if (out / "state.json").exists():
            raise FileExistsError("run exists; use --resume only for an accounted interruption")
        committed = dict(parameters=runtime.snapshot_parameters(), optimizer=move(optimizer.state_dict(), "cpu"),
                         step=0, epoch=0, cursor=0, cache=None, elapsed=0., tokens_seen=0,
                         budget_seconds=budget_seconds, train_sample_ids=list(train_sample_ids) if train_sample_ids is not None else None)
        committed["initial_parameters_sha256"] = parameters_hash(committed["parameters"])
    if prior >= budget_seconds:
        raise RuntimeError("the training budget is already exhausted")
    committed.update(layer=runtime.layer, recipe=recipe, protocol_sha256=provenance["protocol"], provenance=provenance)
    clock = ActiveClock(prior)
    torch.cuda.reset_peak_memory_stats()
    durations = dict(refresh=0., updates=0.)
    attempts = len(list((out / "directions").glob("refresh_*"))) if (out / "directions").exists() else 0
    manifest = dict(status="RUNNING", recipe=recipe, provenance=provenance, budget_seconds=budget_seconds,
                    train_sample_ids=list(train_sample_ids) if train_sample_ids is not None else None)

    def persist(status):
        checkpoint_entry = save(committed, out / "resume.pt")
        write_json(dict(status=status, recipe=recipe, provenance=provenance, checkpoint=checkpoint_entry,
                        budget_seconds=budget_seconds,
                        train_sample_ids=list(train_sample_ids) if train_sample_ids is not None else None,
                        charged_seconds=clock.elapsed, saved_times=saved_times), out / "state.json")

    def save_due():
        for threshold in thresholds:
            if clock.elapsed >= threshold and threshold not in saved_times:
                # Called before a new state is committed, so this is the last
                # completed update available at the crossing of the time boundary.
                save({**committed, "evaluation_seconds": threshold}, out / "checkpoints" / f"time_{threshold}.pt")
                saved_times.append(threshold)

    try:
        persist("RUNNING")
        while True:
            save_due()
            check_budget(clock, budget_seconds)
            epoch, cursor = committed["epoch"], committed["cursor"]
            if recipe["objective"] == "cached_kl" and (cache is None or epoch >= cache.manifest["epoch"] + recipe["reuse_epochs"]):
                t = time.monotonic()
                cache = build_directions(out / "directions" / f"refresh_{attempts:04d}", runtime=runtime,
                    ds=ds, native=native, baseline=baseline, epoch=epoch, step=committed["step"],
                    provenance=provenance, clock=clock, budget=budget_seconds, previous=cache)
                attempts += 1
                durations["refresh"] += time.monotonic() - t
                committed["cache"] = str(cache.path)
                save_due()
                persist("RUNNING")
            check_budget(clock, budget_seconds)
            order = epoch_order(len(ds.protocol["batches"]), epoch)
            batch = ds.protocol["batches"][order[cursor]]
            denominator = sum(len(ds.samples[s]["input_ids"]) for s in batch)
            optimizer.zero_grad(set_to_none=True)
            t = time.monotonic()
            losses = dict(main=0., router=0.)
            for sid in batch:
                check_budget(clock, budget_seconds)
                x = baseline.layer(sid, runtime.layer)["input"].unsqueeze(0).to(runtime.device)
                target = native.layer(sid, runtime.layer)
                target = {k: target[k].unsqueeze(0).to(runtime.device) for k in ("output", "router_logits")}
                record = cache.sample(sid, denominator) if recipe["objective"] == "cached_kl" else None
                stats = backward_sample(runtime, x, target, native.logits(sid) if recipe["objective"] == "direct_kl" else None,
                                        denominator, recipe["objective"], record)
                for key in losses:
                    losses[key] += stats[key]
            if not all(math.isfinite(value) for value in losses.values()):
                raise RuntimeError("nonfinite training objective")
            norm = finite_gradients(runtime.student.learned.parameters())
            check_budget(clock, budget_seconds)
            optimizer.step()
            torch.cuda.synchronize()
            candidate_state = dict(parameters=runtime.snapshot_parameters(), optimizer=move(optimizer.state_dict(), "cpu"))
            durations["updates"] += time.monotonic()-t
            save_due()
            # An update that finishes beyond the deadline is not the primary result.
            check_budget(clock, budget_seconds)
            next_cursor = cursor + 1
            committed = {**committed, **candidate_state, "step": committed["step"]+1,
                         "epoch": epoch + (next_cursor == len(order)), "cursor": next_cursor % len(order),
                         "cache": str(cache.path) if cache is not None else None,
                         "elapsed": clock.elapsed, "tokens_seen": committed["tokens_seen"]+denominator}
            if committed["step"] & (committed["step"]-1) == 0:
                save(committed, out / "checkpoints" / f"step_{committed['step']:06d}.pt")
            persist("RUNNING")
            row = dict(step=committed["step"], epoch=committed["epoch"], elapsed=clock.elapsed,
                       tokens_seen=committed["tokens_seen"], gradient_norm=norm, **losses)
            if train_sample_ids is not None:
                rss = next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                           if line.startswith('VmRSS:'))
                row.update(cuda_allocated=torch.cuda.memory_allocated(),
                           cuda_reserved=torch.cuda.memory_reserved(), rss_bytes=int(rss)*1024)
                heap_live = _heap_live_bytes()
                if heap_live is not None:
                    row["heap_live"] = heap_live
            with (out / "training.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
    except BudgetExpired:
        phase = "COMPLETE" if committed["step"] else "NO_UPDATE_WITHIN_BUDGET"
    except KeyboardInterrupt:
        phase = "INTERRUPTED"
        raise
    except BaseException:
        phase = "FAILED"
        raise
    finally:
        torch.cuda.synchronize()
        save_due()
        if phase in {"COMPLETE", "NO_UPDATE_WITHIN_BUDGET"}:
            manifest["final_checkpoint"] = save({**committed, "evaluation_seconds": budget_seconds}, out / "final.pt")
        persist(phase)
        manifest.update(status=phase, charged_seconds=clock.elapsed, committed_seconds=committed["elapsed"],
                        overshoot_seconds=max(0., clock.elapsed-budget_seconds), steps=committed["step"],
                        tokens_seen=committed["tokens_seen"], peak_memory_bytes=torch.cuda.max_memory_allocated(),
                        durations=durations, parameters_sha256=parameters_hash(committed["parameters"]),
                        initial_parameters_sha256=committed["initial_parameters_sha256"],
                        checkpoint_thresholds=list(thresholds))
        write_json(manifest, out / "manifest.json")
    if phase == "NO_UPDATE_WITHIN_BUDGET":
        raise RuntimeError("no parameter update completed within the training budget; inspect the saved manifest")
    return manifest
