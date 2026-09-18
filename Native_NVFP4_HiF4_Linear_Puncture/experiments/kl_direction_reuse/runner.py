"""Process-isolated stages; GPU identifiers are always supplied by the caller."""
import os
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

from .artifacts import read_json, source_fingerprint, write_json
from .config import idle_devices, recipes
from .data import Dataset

MODULE = "Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse"


def invoke(root, command, *args, gpu_ids=None):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu_ids is None else gpu_ids
    env["PYTHONUNBUFFERED"] = "1"
    expected_source = env.get("KLD_EXPECTED_SOURCE")
    if expected_source and source_fingerprint() != expected_source:
        raise RuntimeError("source code changed after background launch; inspect before continuing")
    argv = [sys.executable, "-u", "-m", MODULE, command, "--root", str(root), *map(str, args)]
    control = Path(env["KLD_CONTROL_DIR"]) if env.get("KLD_CONTROL_DIR") else None
    started = time.monotonic()
    row = dict(command=command, args=list(map(str, args)), devices=gpu_ids,
               started_at=datetime.now(timezone.utc).isoformat(), status="STARTING", pid=None)
    stream, process = None, None
    if control:
        logs = control / "steps"
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / f"{len(list(logs.glob('*.log'))):03d}_{command}.log"
        row["log"] = str(log)
        stream = log.open("x")
        write_json(row, control / "current.json")
    print(json.dumps(row), flush=True)
    try:
        if gpu_ids is not None:
            row["gpu_inventory"] = idle_devices(gpu_ids)
        process = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT if stream else None,
                                   start_new_session=True)
        row.update(pid=process.pid, status="RUNNING")
        if control:
            write_json(row, control / "current.json")
        code = process.wait()
        row.update(returncode=code, status="COMPLETE" if code == 0 else "FAILED")
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
        row.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                   returncode=process.returncode if process is not None else None, error=repr(error))
        raise
    finally:
        if stream:
            stream.close()
        row.update(wall_seconds=time.monotonic()-started, finished_at=datetime.now(timezone.utc).isoformat())
        Path(root).mkdir(parents=True, exist_ok=True)
        with (Path(root) / "stages.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        if control:
            write_json(row, control / "current.json")
        print(json.dumps(row), flush=True)
    if code:
        raise subprocess.CalledProcessError(code, argv)


def suite(root, stage, train_gpu, eval_gpus, *, after_baseline=False, after_captures=False):
    train_ids, eval_ids = train_gpu.split(","), eval_gpus.split(",")
    if len(train_ids) != 1 or len(eval_ids) != 2 or len(set(eval_ids)) != 2 or not all(x.strip() for x in train_ids+eval_ids):
        raise ValueError("one explicit training GPU and two distinct evaluation GPUs are required")
    if (after_baseline or after_captures) and stage != "all" or after_baseline and after_captures:
        raise ValueError("select one continuation boundary and run every remaining stage")
    stages = ["prepare", "verify", "train", "evaluate"] if stage == "all" else [stage]
    for current in stages:
        if current == "prepare":
            if after_captures:
                validate_after_captures(root)
                continue
            if after_baseline:
                validate_after_baseline(root)
            else:
                invoke(root, "prepare")
                invoke(root, "baseline")
            invoke(root, "capture", "--variant", "native", gpu_ids=eval_gpus)
            invoke(root, "capture", "--variant", "baseline", gpu_ids=eval_gpus)
        elif current == "verify":
            for layer in (8, 24, 40):
                invoke(root, "verify-prepare", "--layer", layer, gpu_ids=train_gpu)
                folder = Path(root).resolve() / "verification" / f"L{layer:02d}"
                preparation = read_json(folder / "prepare.json")
                invoke(root, "capture", "--variant", "candidate", "--model-dir", preparation["probe_model"],
                       "--output", folder / "actual_probe", "--sample-ids", ",".join(preparation["samples"]),
                       gpu_ids=eval_gpus)
                invoke(root, "verify-finish", "--layer", layer, "--capture-dir", folder / "actual_probe", gpu_ids=train_gpu)
        elif current == "train":
            for recipe in recipes():
                invoke(root, "train", "--recipe", recipe["name"], gpu_ids=train_gpu)
        elif current == "evaluate":
            evaluate_suite(root, eval_gpus)
            invoke(root, "report")
        else:
            raise ValueError(current)


def validate_after_baseline(root):
    """Explicit restart boundary: complete preparation, no later valid outputs."""
    from .materialize import check_baseline, model_identity
    ds = Dataset(root)
    manifest = check_baseline(ds.root)
    if manifest["source_files"] != model_identity(ds.snapshot):
        raise RuntimeError("native model changed since the preserved baseline was converted")
    for name in ("captures/native", "captures/baseline", "verification", "runs", "report.json"):
        if (ds.root / name).exists():
            raise FileExistsError(f"inspect existing downstream output before restarting: {name}")
    return manifest


def validate_after_captures(root):
    from .capture import CaptureStore
    from .materialize import check_baseline, model_identity
    ds = Dataset(root)
    base = check_baseline(ds.root)
    native = model_identity(ds.snapshot)
    if base['source_files'] != native:
        raise RuntimeError('native model changed after baseline conversion')
    for variant, files in (('native', native), ('baseline', base['files'])):
        store = CaptureStore(ds.root / 'captures' / variant, ds)
        if store.manifest['model_files'] != files or set(store.manifest['samples']) != set(ds.samples):
            raise RuntimeError(f'incomplete or mismatched preserved capture: {variant}')
    for name in ('verification', 'runs', 'report.json'):
        if (ds.root / name).exists():
            raise FileExistsError(f'inspect and preserve failed downstream output before restarting: {name}')


def evaluate_suite(root, eval_gpus):
    ds = Dataset(root)
    common_steps = {}
    for layer in (8, 24, 40):
        steps = []
        for r in [r for r in recipes() if r["layer"] == layer]:
            m = read_json(ds.root / "runs" / r["name"] / "manifest.json")
            if m["status"] != "COMPLETE":
                raise RuntimeError(f"cannot evaluate incomplete recipe {r['name']}")
            steps.append(m["steps"])
        common_steps[layer] = 1 << (min(steps).bit_length()-1)
    write_json(common_steps, ds.root / "common_steps.json")
    for r in recipes():
        run = ds.root / "runs" / r["name"]
        points = [("final", run / "final.pt", ds.ids("val") + ds.ids("test"))]
        points.extend((f"time_{t}", run / "checkpoints" / f"time_{t}.pt", ds.ids("val")) for t in (1800, 3600, 5400))
        step = common_steps[r["layer"]]
        points.append((f"step_{step:06d}", run / "checkpoints" / f"step_{step:06d}.pt", ds.ids("val")))
        for name, cp, ids in points:
            model = run / "exports" / name
            output = run / "evaluation" / ("final" if name == "final" else f"curves/{name}")
            invoke(root, "export", "--checkpoint", cp, "--output", model)
            invoke(root, "capture", "--variant", "candidate", "--model-dir", model, "--output", output,
                   "--sample-ids", ",".join(ids), gpu_ids=eval_gpus)
