"""Four single-GPU training slots, followed by two fixed TP2 evaluation pairs."""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.artifacts import (
    read_json, sha256, source_fingerprint, write_json,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.config import (
    PROTOCOL, idle_devices, recipes, require_environment,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.data import Dataset
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.capture import CaptureStore
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.materialize import (
    check_baseline, model_identity,
)

MODULE = "Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse"


def timestamp():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Task:
    key: str
    command: str
    args: tuple
    slots: int
    output: str
    dependencies: tuple = ()
    pool: str | None = None


def task_pool(task):
    if task.pool is not None:
        return task.pool
    if task.slots == 0:
        return "cpu"
    if task.slots == 2:
        return "eval"
    return "train"


def verification_tasks(ds):
    ordered = sorted(ds.ids("train"), key=lambda s: (len(ds.samples[s]["input_ids"]), s))
    samples = ",".join((ordered[0], ordered[-1]))
    prep, capture, finish = [], [], []
    for layer in PROTOCOL.layers:
        folder = ds.root / "verification" / f"L{layer:02d}"
        key = f"L{layer:02d}"
        prep.append(Task(key + "_prepare", "verify-prepare", ("--layer", str(layer)), 1,
                         str(folder / "prepare.json"), pool="train"))
        capture.append(Task(key + "_probe", "capture", ("--variant", "candidate", "--model-dir",
            str(folder / "probe_model"), "--output", str(folder / "actual_probe"),
            "--sample-ids", samples), 2, str(folder / "actual_probe/manifest.json"), pool="eval"))
        finish.append(Task(key + "_finish", "verify-finish", ("--layer", str(layer),
            "--capture-dir", str(folder / "actual_probe")), 1,
            str(folder / "report.json"), pool="train"))
    return prep, capture, finish


def training_tasks(root):
    choices = recipes()
    by_layer = {layer: [r for r in choices if r["layer"] == layer] for layer in PROTOCOL.layers}
    ordered = [by_layer[layer][index] for index in range(5) for layer in PROTOCOL.layers]
    return [Task(r["name"], "train", ("--recipe", r["name"]), 1,
                 str(root / "runs" / r["name"] / "manifest.json"), pool="train") for r in ordered]


def _evaluation_points(ds, recipe, *, include_common_step=None):
    run = ds.root / "runs" / recipe["name"]
    points = [("final", run / "final.pt", ds.ids("val") + ds.ids("test"))]
    points += [(f"time_{t}", run / "checkpoints" / f"time_{t}.pt", ds.ids("val"))
               for t in (1800, 3600, 5400)]
    if include_common_step is not None:
        points.append((f"step_{include_common_step:06d}",
                       run / "checkpoints" / f"step_{include_common_step:06d}.pt", ds.ids("val")))
    return points


def evaluation_tasks_for_recipe(ds, recipe, *, include_common_step=None):
    tasks = []
    run = ds.root / "runs" / recipe["name"]
    for name, checkpoint, ids in _evaluation_points(ds, recipe,
                                                     include_common_step=include_common_step):
        model = run / "exports" / name
        output = run / "evaluation" / ("final" if name == "final" else f"curves/{name}")
        key = recipe["name"] + "_" + name
        tasks.append(Task(key + "_export", "export", ("--checkpoint", str(checkpoint),
            "--output", str(model)), 0, str(model / "export.json"), pool="cpu"))
        tasks.append(Task(key + "_evaluate", "capture", ("--variant", "candidate",
            "--model-dir", str(model), "--output", str(output), "--sample-ids", ",".join(ids)), 2,
            str(output / "manifest.json"), (key + "_export",), pool="eval"))
    return tasks


def common_steps(ds):
    values = {}
    for layer in PROTOCOL.layers:
        manifests = [read_json(ds.root / "runs" / r["name"] / "manifest.json")
                     for r in recipes() if r["layer"] == layer]
        if any(m["status"] != "COMPLETE" or m["steps"] <= 0 for m in manifests):
            raise RuntimeError("common step selection requires all training configurations to complete")
        values[str(layer)] = 1 << (min(m["steps"] for m in manifests).bit_length() - 1)
    path = ds.root / "common_steps.json"
    if path.exists() and read_json(path) != values:
        raise RuntimeError("common step selection changed since evaluation preparation")
    write_json(values, path)
    return {int(k): v for k, v in values.items()}


def evidence(task):
    path = Path(task.output)
    value = read_json(path)
    expected = {"verify-prepare": "AWAITING_ACTUAL_NONZERO_CAPTURE", "verify-finish": "PASS"}
    if task.command == "report":
        if len(value["candidates"]) != 15 or not value["cost"]["stage_ledger_complete"]:
            raise RuntimeError("final report is incomplete")
    elif value["status"] != expected.get(task.command, "COMPLETE"):
        raise RuntimeError(f"incomplete output for {task.key}")
    if task.command == "verify-prepare" and (value["fresh_gradient"] != "PASS" or
                                               value["router_gradient_scope"] != "PASS"):
        raise RuntimeError("required gradient checks did not pass")
    if task.command == "train" and value["steps"] <= 0:
        raise RuntimeError("training completed without an update")
    result = {str(path): sha256(path)}
    if task.command == "train":
        result[str(path.parent / "final.pt")] = sha256(path.parent / "final.pt")
    return result


def reuse_verified_phase(queue, phase, tasks, ds):
    """Account for verified artifacts without rerunning or overwriting them."""
    queue.phase = phase
    write_json([dict(key=t.key, command=t.command, args=t.args, slots=t.slots,
                     output=t.output, dependencies=t.dependencies, pool=task_pool(t))
                for t in tasks], queue.control / f"plan_{phase}.json")
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.train import training_provenance
    provenance = training_provenance(queue.root)
    for task in tasks:
        if queue.resume and queue._admit_completed(task, set()):
            continue
        if task.command == "verify-prepare":
            value = read_json(task.output)
            if value["status"] != "AWAITING_ACTUAL_NONZERO_CAPTURE" or value["provenance"] != provenance:
                raise RuntimeError(f"verified preparation is stale: {task.key}")
        elif task.command == "verify-finish":
            value = read_json(task.output)
            if value["status"] != "PASS" or value["provenance"] != provenance:
                raise RuntimeError(f"verified finish is stale: {task.key}")
        elif task.command == "capture":
            value = read_json(task.output)
            if value["status"] != "COMPLETE":
                raise RuntimeError(f"verified capture is incomplete: {task.key}")
        else:
            raise ValueError(f"cannot reuse task type: {task.command}")
        folder = queue.control / "jobs" / task.key
        folder.mkdir(parents=True, exist_ok=True)
        now = timestamp()
        row = dict(task=task.key, command=task.command, args=list(task.args), devices=None,
                   pool=task_pool(task), status="COMPLETE", reused=True, started_at=now,
                   finished_at=now, returncode=0, wall_seconds=0., original_wall_seconds=0.,
                   evidence=evidence(task))
        write_json(row, folder / "status.json")
        with (queue.root / "stages.jsonl").open("a") as ledger:
            fcntl.flock(ledger, fcntl.LOCK_EX)
            ledger.write(json.dumps(row) + "\n")
        queue.completed.append(task.key)
    queue.status()


class Queue:
    def __init__(self, root, control, *, train_gpus, eval_gpus, resume=False):
        self.root, self.control = Path(root), Path(control)
        self.train_gpus = tuple(train_gpus)
        self.eval_gpus = tuple(eval_gpus)
        self.resume = resume
        self.source, self.driver = source_fingerprint(), sha256(__file__)
        self.reserver = sha256(Path(__file__).with_name('reserve_gpus.py'))
        self.running = {}
        self.reservations = {}
        self.completed = []
        self.phase = "STARTING"

    @property
    def all_gpus(self):
        return tuple(dict.fromkeys(self.train_gpus + self.eval_gpus))

    def check_source(self):
        if (source_fingerprint() != self.source or sha256(__file__) != self.driver or
                sha256(Path(__file__).with_name('reserve_gpus.py')) != self.reserver):
            raise RuntimeError("source or controller changed after launch")

    def status(self, state="RUNNING", **extra):
        write_json(dict(status=state, pid=os.getpid(), root=str(self.root), phase=self.phase,
                        gpus=self.all_gpus, train_gpus=self.train_gpus, eval_gpus=self.eval_gpus,
                        one_gpu_per_training=True, updated_at=timestamp(), completed=self.completed,
                        active=[dict(key=k, pid=v["process"].pid, devices=v["gpus"], pool=v["pool"],
                                     log=str(v["log"])) for k, v in self.running.items()],
                        reservations=[dict(gpu=gpu, pid=v["process"].pid, log=str(v["log"]))
                                     for gpu, v in self.reservations.items()], **extra),
                   self.control / "status.json")

    def _reservation_status(self, row):
        write_json(row, self.control / "reservations" / f"gpu_{row['gpu']}.json")

    def start_reservation(self, gpu):
        """Hold an unused final-batch GPU with a small, owned CUDA allocation."""
        gpu = str(gpu)
        if gpu in self.reservations:
            raise RuntimeError(f"GPU {gpu} is already reserved")
        self.check_source()
        inventory = idle_devices(gpu)
        folder = self.control / "reservations"
        folder.mkdir(parents=True, exist_ok=True)
        attempt = len(list(folder.glob(f'gpu_{gpu}_*.log')))
        log = folder / f"gpu_{gpu}_{attempt:03d}.log"
        ready = log.with_suffix('.ready.json')
        stream = log.open("x")
        script = Path(__file__).with_name("reserve_gpus.py")
        argv = [sys.executable, "-u", str(script), "--devices", "1", "--memory-mib", "256",
                '--ready-file', str(ready)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
               "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1"}
        try:
            process = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        except BaseException:
            stream.close()
            raise
        row = dict(gpu=gpu, pid=process.pid, status="RUNNING", started_at=timestamp(),
                   log=str(log), gpu_inventory=inventory, memory_mib=256,
                   command='reserve-gpu', args=argv[3:], devices=gpu, ready_file=str(ready))
        self.reservations[gpu] = dict(process=process, stream=stream, row=row, log=log,
                                      started=time.monotonic())
        self._reservation_status(row)
        deadline = time.monotonic() + 60
        while not ready.exists():
            self.check_reservations()
            if time.monotonic() >= deadline:
                raise RuntimeError(f'CUDA reservation did not become ready: {gpu}')
            time.sleep(.1)
        if read_json(ready)['pid'] != process.pid:
            raise RuntimeError('reservation readiness belongs to another process')
        print(json.dumps(dict(reservation=True, **row)), flush=True)
        self.status()

    def stop_reservations(self, gpus=None):
        for gpu, value in list(self.reservations.items()):
            if gpus is not None and gpu not in gpus:
                continue
            process = value["process"]
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            code = process.wait()
            value["stream"].close()
            row = dict(value["row"], status="RELEASED", returncode=code,
                       finished_at=timestamp(), wall_seconds=time.monotonic()-value["started"])
            self._reservation_status(row)
            with (self.root / 'stages.jsonl').open('a') as ledger:
                fcntl.flock(ledger, fcntl.LOCK_EX)
                ledger.write(json.dumps(row) + '\n')
            del self.reservations[gpu]
        self.status()

    def check_reservations(self):
        for gpu, value in self.reservations.items():
            if value["process"].poll() is not None:
                raise RuntimeError(f"GPU reservation process exited unexpectedly: {gpu}")

    def _pool_gpus(self, pool):
        if pool == "train":
            return self.train_gpus
        if pool == "eval":
            return self.eval_gpus
        return ()

    def _assign(self, task):
        pool = task_pool(task)
        if pool == "cpu":
            return [] if not any(v["pool"] == "cpu" for v in self.running.values()) else None
        devices = self._pool_gpus(pool)
        used = {gpu for v in self.running.values() for gpu in v["gpus"]}
        if pool == 'eval':
            # Preserve pair membership when the two engines finish at different times.
            for offset in range(0, len(devices), 2):
                pair = list(devices[offset:offset + 2])
                if len(pair) == 2 and not used.intersection(pair):
                    return pair
            return None
        used.update(self.reservations)
        free = [gpu for gpu in devices if gpu not in used]
        if len(free) < task.slots:
            return None
        return free[:task.slots]

    def start(self, task, gpus):
        self.check_source()
        folder = self.control / "jobs" / task.key
        previous = read_json(folder / "status.json") if (folder / "status.json").exists() else None
        if previous and not self.resume:
            raise FileExistsError(f"job already exists: {task.key}")
        args = list(task.args)
        if previous and task.command == "train":
            args.append("--resume")
        elif previous and task.command in ("capture", "export", "verify-prepare") and Path(task.output).parent.exists():
            raise RuntimeError(f"inspect and archive partial output before explicit retry: {Path(task.output).parent}")
        if task_pool(task) == 'eval':
            self.stop_reservations(gpus)
        inventory = idle_devices(",".join(gpus)) if gpus else []
        folder.mkdir(parents=True, exist_ok=True)
        attempt = len(list(folder.glob("attempt_*.log")))
        log = folder / f"attempt_{attempt:03d}.log"
        stream = log.open("x")
        argv = [sys.executable, "-u", "-m", MODULE, task.command, "--root", str(self.root), *args]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpus), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
               "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
               "PYTHONUNBUFFERED": "1"}
        row = dict(task=task.key, command=task.command, args=args, devices=",".join(gpus) or None,
                   pool=task_pool(task), started_at=timestamp(), status="RUNNING", log=str(log),
                   gpu_inventory=inventory, attempt=attempt)
        try:
            process = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        except BaseException:
            stream.close()
            raise
        row["pid"] = process.pid
        self.running[task.key] = dict(task=task, process=process, stream=stream, row=row,
                                      gpus=gpus, pool=task_pool(task), started=time.monotonic(),
                                      folder=folder, log=log)
        write_json(row, folder / "status.json")
        print(json.dumps(row), flush=True)
        self.status()

    def finish(self, key, *, interrupted=False):
        live = self.running.pop(key)
        task, process, row = live["task"], live["process"], live["row"]
        code = process.wait()
        live["stream"].close()
        row.update(returncode=130 if interrupted and code == 0 else code, finished_at=timestamp(),
                   wall_seconds=time.monotonic() - live["started"],
                   status="INTERRUPTED" if interrupted else ("COMPLETE" if code == 0 else "FAILED"))
        error = None
        if row["status"] == "COMPLETE":
            try:
                row["evidence"] = evidence(task)
            except BaseException as exc:
                row.update(status="FAILED", error=repr(exc), returncode=1)
                error = exc
        with (self.root / "stages.jsonl").open("a") as ledger:
            fcntl.flock(ledger, fcntl.LOCK_EX)
            ledger.write(json.dumps(row) + "\n")
        write_json(row, live["folder"] / f"attempt_{row['attempt']:03d}.json")
        write_json(row, live["folder"] / "status.json")
        print(json.dumps(row), flush=True)
        if row["status"] == "COMPLETE":
            self.completed.append(key)
        self.status()
        if not interrupted and row["status"] != "COMPLETE":
            raise RuntimeError(f"{key} failed; see {live['log']}") from error

    def stop_owned_jobs(self):
        for live in self.running.values():
            if live["process"].poll() is None:
                os.killpg(live["process"].pid, signal.SIGTERM)
        for key in list(self.running):
            self.finish(key, interrupted=self.running[key]["process"].poll() != 0)

    def _admit_completed(self, task, done):
        path = self.control / "jobs" / task.key / "status.json"
        if not path.exists():
            return False
        saved = read_json(path)
        if not self.resume:
            raise FileExistsError(path)
        if saved["status"] != "COMPLETE":
            return False
        if evidence(task) != saved["evidence"]:
            raise RuntimeError(f"completed output changed: {task.key}")
        done.add(task.key)
        self.completed.append(task.key)
        return True

    def run_phase(self, phase, tasks, *, hold_finished=False):
        self.phase = phase
        write_json([dict(key=t.key, command=t.command, args=t.args, slots=t.slots,
                         output=t.output, dependencies=t.dependencies, pool=task_pool(t))
                    for t in tasks], self.control / f"plan_{phase}.json")
        pending, done = list(tasks), set()
        keys = {t.key for t in tasks}
        if len(keys) != len(tasks) or any(not set(t.dependencies) <= keys for t in tasks):
            raise ValueError('phase contains duplicate tasks or dependencies outside the phase')
        for task in list(pending):
            if self._admit_completed(task, done):
                pending.remove(task)
        while pending or self.running:
            self.check_reservations()
            for key, live in list(self.running.items()):
                if live["process"].poll() is not None:
                    released = live['gpus']
                    self.finish(key)
                    done.add(key)
                    if hold_finished:
                        for gpu in released:
                            self.start_reservation(gpu)
            for task in list(pending):
                if not set(task.dependencies) <= done:
                    continue
                assigned = self._assign(task)
                if assigned is None:
                    continue
                self.start(task, assigned)
                pending.remove(task)
            if pending and not self.running:
                raise RuntimeError("unsatisfied task dependency or invalid resource request")
            if self.running:
                time.sleep(1)

    def run_training_batches(self, tasks):
        """Run one four-single-GPU batch at a time and reserve a short final batch's gap."""
        batch_size = len(self.train_gpus)
        if batch_size != 4:
            raise RuntimeError("the four-card schedule requires exactly four training GPUs")
        for batch_index in range(0, len(tasks), batch_size):
            batch = tasks[batch_index:batch_index + batch_size]
            missing = self.train_gpus[len(batch):]
            for gpu in missing:
                self.start_reservation(gpu)
            last = batch_index + batch_size >= len(tasks)
            self.run_phase(f"train_batch_{batch_index // batch_size + 1:02d}", batch,
                           hold_finished=last)

    def run_train_evaluate(self, ds):
        """Run all training batches first, then use four cards as two TP2 pairs."""
        self.run_training_batches(training_tasks(self.root))

        tasks = []
        for recipe in recipes():
            for task in evaluation_tasks_for_recipe(ds, recipe):
                tasks.append(task)
        self.run_phase("evaluate", tasks)
        selected = common_steps(ds)
        tasks = [task for recipe in recipes()
                 for task in evaluation_tasks_for_recipe(ds, recipe,
                     include_common_step=selected[recipe["layer"]]) if "_step_" in task.key]
        self.run_phase("common_steps", tasks)
        self.stop_reservations()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--train-gpus", required=True)
    parser.add_argument("--eval-gpus", required=True)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-verified", action="store_true",
                        help="reuse current PASS verification artifacts and start at training")
    args = parser.parse_args()
    require_environment()
    train_gpus, eval_gpus = args.train_gpus.split(","), args.eval_gpus.split(",")
    if len(train_gpus) != 4 or len(set(train_gpus)) != 4 or not all(g.isdigit() for g in train_gpus):
        raise ValueError("provide exactly four distinct physical training GPU indices")
    if len(eval_gpus) != 4 or len(set(eval_gpus)) != 4 or not all(g.isdigit() for g in eval_gpus):
        raise ValueError("provide exactly four distinct physical evaluation GPU indices")
    if set(train_gpus) != set(eval_gpus):
        raise ValueError("four-card train/evaluation schedules must use the same GPU set")
    root, control = args.root.resolve(), args.control.resolve()
    if control == root or root in control.parents:
        raise ValueError("controller must be outside the experiment root")
    control.mkdir(parents=True, exist_ok=True)
    original = root.with_name(root.name + ".control")
    original.mkdir(parents=True, exist_ok=True)
    with (original / "host.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        q = Queue(root, control, train_gpus=train_gpus, eval_gpus=eval_gpus, resume=args.resume)
        def terminate(signum, frame):
            raise KeyboardInterrupt("controller received SIGTERM")
        signal.signal(signal.SIGTERM, terminate)
        launch_path = control / "launch.json"
        if launch_path.exists():
            launch = read_json(launch_path)
            if not args.resume or launch["source"] != q.source or launch["driver"] != q.driver or \
                    launch["train_gpus"] != train_gpus or launch["eval_gpus"] != eval_gpus or \
                    launch['reserver'] != q.reserver:
                raise RuntimeError("explicit resume requires identical code and allocation")
        else:
            if args.resume:
                raise FileNotFoundError("no previous long queue to resume")
            write_json(dict(root=str(root), train_gpus=train_gpus, eval_gpus=eval_gpus,
                            source=q.source, driver=q.driver, reserver=q.reserver, started_at=timestamp(),
                            schedule='four_train_then_two_tp2',
                            automatic_retry=False, training_slots=len(train_gpus),
                            training_gpus_per_job=1, tp2_exclusive=True,
                            budget_seconds=PROTOCOL.budget_seconds, recipes=recipes()), launch_path)
        q.status()
        try:
            idle_devices(",".join(q.all_gpus))
            from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.smoke import require_smoke
            require_smoke(args.smoke_report, root, train_gpus, eval_gpus)
            schedule = read_json(args.smoke_report)['schedule']
            if schedule['status'] != 'PASS' or schedule['reserver_sha256'] != q.reserver:
                raise RuntimeError('missing current four-training/two-TP2 scheduling evidence')
            ds = Dataset(root)
            base = check_baseline(root)
            native = model_identity(ds.snapshot)
            if base["source_files"] != native:
                raise RuntimeError("native model differs from preserved baseline source")
            for variant, files in (("native", native), ("baseline", base["files"])):
                capture = CaptureStore(root / "captures" / variant, ds)
                if capture.manifest["model_files"] != files or set(capture.manifest["samples"]) != set(ds.samples):
                    raise RuntimeError("incomplete preserved captures")
            prep, capture, finish = verification_tasks(ds)
            if args.reuse_verified:
                reuse_verified_phase(q, "verify_prepare", prep, ds)
                reuse_verified_phase(q, "verify_tp2", capture, ds)
                reuse_verified_phase(q, "verify_finish", finish, ds)
            else:
                q.run_phase("verify_prepare", prep)
                q.run_phase("verify_tp2", capture)
                q.run_phase("verify_finish", finish)
            q.run_train_evaluate(ds)
            q.run_phase("report", [Task("report", "report", (), 0, str(root / "report.json"), pool="cpu")])
        except BaseException as error:
            try:
                q.stop_owned_jobs()
            finally:
                q.stop_reservations()
            q.status("INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", error=repr(error))
            raise
        q.status("COMPLETE", report=str(root / "report.json"))


if __name__ == "__main__":
    main()
