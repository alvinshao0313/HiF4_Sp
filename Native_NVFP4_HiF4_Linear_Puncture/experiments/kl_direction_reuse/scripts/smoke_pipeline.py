"""Run the finite L24 smoke gate for the four-card train-then-TP2 schedule."""
import argparse
import fcntl
import json
import math
import signal
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.artifacts import (
    checked_load, parameters_hash, read_json, sha256, write_json,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.config import idle_devices, require_environment
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.data import Dataset
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.restart import reuse_inputs
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.smoke import (
    SMOKE_RECIPES, controller_hashes, resource_check, settings, summarize_smoke,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.scripts.long_pipeline import (
    Queue, Task, reuse_verified_phase, verification_tasks,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.config import recipes

MODULE = "Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse"


def smoke_tasks(root, ds, config):
    """Build the verification, training, and evaluation phases separately."""
    train_tasks = []
    eval_tasks = []
    selected = set(SMOKE_RECIPES) | {'L08_mse', 'L08_direct_kl'}
    for recipe in recipes():
        if recipe["name"] not in selected:
            continue
        train = Task(recipe["name"], "train", (
            "--recipe", recipe["name"], "--budget-seconds", "600",
            "--train-sample-ids", ",".join(config["train_sample_ids"])), 1,
            str(root / "runs" / recipe["name"] / "manifest.json"),
            pool="train")
        train_tasks.append(train)
        if recipe["name"] not in SMOKE_RECIPES:
            continue
        run = root / "runs" / recipe["name"]
        model = run / "exports" / "final"
        output = run / "evaluation" / "final"
        export_key = recipe["name"] + "_final_export"
        eval_key = recipe["name"] + "_final_evaluate"
        eval_tasks.extend((
            Task(export_key, "export", ("--checkpoint", str(run / "final.pt"),
                "--output", str(model)), 0, str(model / "export.json"),
                pool="cpu"),
            Task(eval_key, "capture", ("--variant", "candidate", "--model-dir", str(model),
                "--output", str(output), "--sample-ids", ",".join(config["eval_sample_ids"])),
                2, str(output / "manifest.json"), dependencies=(export_key,), pool="eval"),
        ))
    return dict(train=train_tasks, evaluate=eval_tasks)


def schedule_evidence(queue, ds, config):
    rows = [json.loads(line) for line in (ds.root / 'stages.jsonl').read_text().splitlines()]
    trains = [r for r in rows if r['command'] == 'train']
    evaluations = [r for r in rows if r['command'] == 'capture' and not r.get('reused')]
    reservations = [r for r in rows if r['command'] == 'reserve-gpu']
    if (len(trains) != 5 or len(evaluations) != len(SMOKE_RECIPES) or not reservations or
            any(r['returncode'] != 0 for r in trains + evaluations + reservations)):
        raise RuntimeError('four-card scheduling paths did not complete')
    train_events = sorted({event for r in trains for event in (r['started_at'], r['finished_at'])})
    if not any(sum(r['started_at'] <= event < r['finished_at'] for r in trains) >= 4
               for event in train_events):
        raise RuntimeError('four training processes never overlapped')
    if min(r['started_at'] for r in evaluations) < max(r['finished_at'] for r in trains):
        raise RuntimeError('evaluation started before all training completed')
    pairs = {','.join(queue.eval_gpus[:2]), ','.join(queue.eval_gpus[2:])}
    if {r['devices'] for r in evaluations} != pairs:
        raise RuntimeError('both fixed TP2 pairs were not exercised')
    if not any(a['devices'] != b['devices'] and
               max(a['started_at'], b['started_at']) < min(a['finished_at'], b['finished_at'])
               for a in evaluations for b in evaluations):
        raise RuntimeError('two TP2 evaluation processes never overlapped')
    evidence = [str((ds.root / 'stages.jsonl').resolve())]
    for r in reservations:
        ready = read_json(r['ready_file'])
        if ready['pid'] != r['pid'] or r['status'] != 'RELEASED':
            raise RuntimeError('reservation did not initialize and release correctly')
        evidence.append(r['ready_file'])
    # The fourth training job stresses shared CPU/RAM/IO concurrency. It is not
    # an extra formal experiment or an extra comparison method.
    run = ds.root / 'runs' / config['concurrency_probe']
    m = read_json(run / 'manifest.json')
    cp = checked_load(m['final_checkpoint'])
    if (m['status'] != 'COMPLETE' or m['steps'] < 2 or m['budget_seconds'] != 600 or
            m['train_sample_ids'] != config['train_sample_ids'] or
            parameters_hash(cp['parameters']) != m['parameters_sha256'] or
            m['parameters_sha256'] == m['initial_parameters_sha256']):
        raise RuntimeError('fourth concurrent training probe is incomplete')
    updates = [json.loads(line) for line in (run / 'training.jsonl').read_text().splitlines()]
    if len(updates) != m['steps'] or any(not math.isfinite(r[k]) for r in updates
                                        for k in ('main', 'router', 'gradient_norm')):
        raise RuntimeError('fourth concurrent training probe has invalid updates')
    resource_check(updates)
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.capture import CaptureStore
    cap = CaptureStore(run / 'evaluation/final', ds)
    export = read_json(run / 'exports/final/export.json')
    if (export['checkpoint_sha256'] != sha256(run / 'final.pt') or
            cap.manifest['model_files'] != export['files'] or
            set(cap.manifest['samples']) != set(config['eval_sample_ids'])):
        raise RuntimeError('fourth probe export/capture mismatch')
    for sid, row in cap.manifest['samples'].items():
        metric = row['metrics']
        n = len(ds.samples[sid]['input_ids'])
        if (metric['kl_tokens'] != n or metric['nll_tokens'] != n-1 or
                any(not math.isfinite(v) for v in metric.values())):
            raise RuntimeError('fourth probe TP2 metrics incomplete')
    for name in ('manifest.json', 'training.jsonl', 'final.pt',
                 'exports/final/export.json', 'evaluation/final/manifest.json'):
        evidence.append(str((run / name).resolve()))
    return dict(status='PASS', training_concurrency=4, tp2_concurrency=2,
                pairs=sorted(pairs), reserver_sha256=queue.reserver,
                reservation_count=len(reservations), evidence_paths=evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--train-gpus", required=True)
    parser.add_argument("--eval-gpus", required=True)
    args = parser.parse_args()
    require_environment()
    train_gpus, eval_gpus = args.train_gpus.split(","), args.eval_gpus.split(",")
    if len(train_gpus) != 4 or len(set(train_gpus)) != 4 or len(eval_gpus) != 4 or len(set(eval_gpus)) != 4:
        raise ValueError("smoke requires four training GPUs and four evaluation GPUs")
    if set(train_gpus) != set(eval_gpus):
        raise ValueError("four-card smoke must reuse the same GPUs for training and evaluation phases")
    idle_devices(",".join(train_gpus))
    control = args.control.resolve()
    control.mkdir(parents=True, exist_ok=True)
    root = reuse_inputs(args.source_root, args.smoke_root, verification=True)
    ds = Dataset(root)
    config = settings(ds, train_gpus, eval_gpus)
    config['concurrency_probe'] = 'L24_mse'
    controllers = controller_hashes()
    write_json(config, root / "smoke_config.json")
    control.mkdir(parents=True, exist_ok=True)
    queue = Queue(root, control, train_gpus=train_gpus, eval_gpus=eval_gpus)
    def terminate(signum, frame):
        raise KeyboardInterrupt("smoke controller received SIGTERM")
    signal.signal(signal.SIGTERM, terminate)
    with (control / "host.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue.status()
        try:
            tasks = smoke_tasks(root, ds, config)
            for phase, group in zip(("verify_prepare", "verify_probe", "verify_finish"), verification_tasks(ds)):
                reuse_verified_phase(queue, phase, group, ds)
            queue.run_training_batches(tasks["train"])
            queue.run_phase("smoke_evaluate", tasks["evaluate"])
            queue.stop_reservations()
            queue.check_source()
            if controllers != controller_hashes():
                raise RuntimeError('smoke controllers changed while running')
            schedule = schedule_evidence(queue, ds, config)
            result = summarize_smoke(root)
            result['schedule'] = schedule
            for artifact in schedule['evidence_paths']:
                result['evidence'][artifact] = sha256(artifact)
            write_json(result, root / 'smoke_report.json')
            write_json(result, control / "smoke_report.json")
            queue.status("COMPLETE", report=str(control / "smoke_report.json"))
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        except BaseException as error:
            try:
                queue.stop_owned_jobs()
            finally:
                queue.stop_reservations()
            queue.status("INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                         error=repr(error))
            raise


if __name__ == "__main__":
    main()
